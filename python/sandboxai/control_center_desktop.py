"""Tk desktop Control Center backed exclusively by :mod:`sandboxai.adapter`.

Architecture (kept intentionally unchanged from the first version):

    Desktop Control Center (this file)
        -> SandboxAIAdapter (python/sandboxai/adapter.py)
            -> existing CLI/training/evaluation/benchmark infrastructure
                -> Godot workers

This module owns *presentation only*: widget layout, polling cadence and
turning user input into calls on :class:`~sandboxai.adapter.SandboxAIAdapter`.
Every dict-shaping/formatting rule lives in
:mod:`sandboxai.control_center_viewmodel` so it is unit-testable without Tk
and cannot drift between pages. No page invents a metric: a value the
adapter did not return is shown as "n/a", never estimated.

Tk is a small, supported local desktop dependency that ships with CPython on
Windows; no additional GUI toolkit is required. All adapter calls that touch
disk or a subprocess run on a background thread pool
(:class:`BackgroundRunner`) and hand their result back to the Tk thread
through a queue, so a slow poll (a big run directory, a stuck process list)
never freezes the window.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .adapter import SandboxAIAdapter
from . import control_center_viewmodel as vm

# ---------------------------------------------------------------------------
# Background work: keeps every adapter call off the Tk event loop thread.
# ---------------------------------------------------------------------------


class BackgroundRunner:
    """Runs adapter calls off the Tk thread; delivers results back on it.

    Tk widgets may only be touched from the thread running ``mainloop()``.
    Every page submits its adapter/viewmodel calls here instead of calling
    them directly, so a page's ``refresh()`` always returns immediately and
    the window never freezes while training/benchmark/evaluation artifacts
    are being read from disk or a process list is being walked.
    """

    def __init__(self, root: tk.Misc, workers: int = 3, poll_ms: int = 100) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="control-center-bg")
        self._queue: "queue.Queue[tuple[Callable[[Any, BaseException | None], None], Any, BaseException | None]]" = (
            queue.Queue()
        )
        self._closed = False
        self._poll_ms = poll_ms
        self._pump()

    def submit(self, fn: Callable[[], Any], callback: Callable[[Any, BaseException | None], None]) -> None:
        if self._closed:
            return

        def _run() -> None:
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to the GUI, never crashes a bg thread
                self._queue.put((callback, None, exc))
            else:
                self._queue.put((callback, result, None))

        self._executor.submit(_run)

    def _pump(self) -> None:
        try:
            while True:
                callback, result, error = self._queue.get_nowait()
                try:
                    callback(result, error)
                except Exception:
                    # A callback bug must not take down the polling loop or
                    # the rest of the GUI.
                    import traceback

                    traceback.print_exc()
        except queue.Empty:
            pass
        if not self._closed:
            self._root.after(self._poll_ms, self._pump)

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------

_FONT_FAMILY = "Segoe UI"
COLOR_BG = "#0b1120"
COLOR_SURFACE = "#111827"
COLOR_SURFACE_RAISED = "#151f32"
COLOR_HOVER = "#1c2940"
COLOR_TEXT = "#e7edf7"
COLOR_ACCENT = "#38bdf8"
COLOR_OK = "#4ade80"
COLOR_WARN = "#fbbf24"
COLOR_ERROR = "#fb7185"
COLOR_MUTED = "#94a3b8"
COLOR_BORDER = "#2b3a52"


class StatCard(ttk.Frame):
    """One labelled value in a Dashboard/System stat row."""

    def __init__(self, parent: tk.Misc, label: str) -> None:
        super().__init__(parent, style="Card.TFrame", padding=(10, 8))
        ttk.Label(self, text=label, style="CardLabel.TLabel").pack(anchor="w")
        self._value = ttk.Label(self, text="n/a", style="CardValue.TLabel")
        self._value.pack(anchor="w")

    def set(self, text: str, color: str | None = None) -> None:
        self._value.configure(text=text, foreground=color or "")


class StatRow(ttk.Frame):
    """A horizontal row of :class:`StatCard` built from an ordered mapping."""

    def __init__(self, parent: tk.Misc, labels: tuple[str, ...]) -> None:
        super().__init__(parent)
        self._cards: dict[str, StatCard] = {}
        columns = min(5, max(1, len(labels)))
        for index, label in enumerate(labels):
            card = StatCard(self, label)
            row, column = divmod(index, columns)
            card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            self.columnconfigure(column, weight=1, uniform="stats")
            self._cards[label] = card

    def update_values(self, values: dict[str, tuple[str, str | None]]) -> None:
        for label, (text, color) in values.items():
            if label in self._cards:
                self._cards[label].set(text, color)


class LineChart(tk.Canvas):
    """A minimal, dependency-free bounded line chart.

    Draws one series of ``(x, y)`` points already bounded by the adapter
    (``deque(maxlen=...)``) and again decimated to the canvas width by
    :func:`control_center_viewmodel.downsample_series`, so render cost never
    grows with run length. No animation, no decoration beyond axis labels
    and a light grid - this is a measurement instrument, not a dashboard
    graphic.
    """

    def __init__(self, parent: tk.Misc, title: str, height: int = 140) -> None:
        super().__init__(parent, height=height, background=COLOR_SURFACE_RAISED, highlightthickness=1,
                          highlightbackground=COLOR_BORDER)
        self._title = title
        self._points: list[tuple[float, float]] = []
        self.bind("<Configure>", lambda _event: self._redraw())

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = points
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        width = max(int(self.winfo_width()), 1)
        height = max(int(self.winfo_height()), 1)
        pad_left, pad_right, pad_top, pad_bottom = 46, 10, 16, 18
        self.create_text(8, 6, anchor="nw", text=self._title, font=(_FONT_FAMILY, 9, "bold"), fill=COLOR_MUTED)
        points = vm.downsample_series(self._points, max_points=max(width - pad_left - pad_right, 10))
        if len(points) < 2:
            self.create_text(width / 2, height / 2, text="not enough data yet", fill=COLOR_MUTED,
                              font=(_FONT_FAMILY, 9))
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        if x_min == x_max:
            x_min, x_max = x_min - 1.0, x_max + 1.0
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        def to_canvas(x: float, y: float) -> tuple[float, float]:
            cx = pad_left + (x - x_min) / (x_max - x_min) * plot_w
            cy = pad_top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h
            return cx, cy

        for fraction in (0.0, 0.5, 1.0):
            gy = pad_top + fraction * plot_h
            self.create_line(pad_left, gy, width - pad_right, gy, fill="#243044")
            value = y_max - fraction * (y_max - y_min)
            self.create_text(pad_left - 6, gy, anchor="e", text=vm.format_number(value, 2), fill=COLOR_MUTED,
                              font=(_FONT_FAMILY, 8))
        coords: list[float] = []
        for x, y in points:
            cx, cy = to_canvas(x, y)
            coords.extend((cx, cy))
        self.create_line(*coords, fill=COLOR_ACCENT, width=2, smooth=False)
        self.create_text(width - pad_right, height - 4, anchor="se",
                          text=f"latest: {vm.format_number(ys[-1], 3)}", fill=COLOR_MUTED, font=(_FONT_FAMILY, 8))


class LogPanel(ttk.Frame):
    """Bounded, non-blocking stdout/stderr viewer for one process.

    - stdout is plain text; stderr lines are tagged red, so failures are
      visually obvious without any separate error-scanning logic.
    - "Auto-scroll" is on by default; unchecking it (or scrolling up
      manually) pauses the jump-to-bottom behavior without pausing polling.
    - The Text widget itself is capped at ``max_lines`` (old lines are
      deleted), independent of the adapter's own bounded buffer, so memory
      cannot grow across a very long session even if a process runs for
      hours.
    """

    def __init__(self, parent: tk.Misc, max_lines: int = 4000) -> None:
        super().__init__(parent)
        self._max_lines = max_lines
        self._stdout_after = -1
        self._stderr_after = -1
        self._user_scrolled_up = False
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        self._autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Auto-scroll", variable=self._autoscroll).pack(side="left")
        ttk.Button(toolbar, text="Clear view", command=self.clear).pack(side="left", padx=(8, 0))
        self._truncated_label = ttk.Label(toolbar, text="", foreground=COLOR_WARN)
        self._truncated_label.pack(side="right")
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.text = tk.Text(text_frame, height=16, wrap="none", state="disabled", background="#0d1117",
                             foreground="#c9d1d9", insertbackground="#c9d1d9", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.text.tag_configure("stderr", foreground="#ff7b72")
        self.text.tag_configure("meta", foreground="#8b949e")
        self.text.bind("<MouseWheel>", self._on_manual_scroll)
        self.text.bind("<Button-4>", self._on_manual_scroll)
        self.text.bind("<Button-5>", self._on_manual_scroll)

    def _on_manual_scroll(self, _event: object) -> None:
        self._user_scrolled_up = True

    def reset_cursor(self) -> None:
        """Call when switching to a different process id."""
        self._stdout_after = -1
        self._stderr_after = -1
        self._user_scrolled_up = False
        self.clear()

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def apply_log(self, log: dict[str, Any]) -> None:
        if "error" in log and not log.get("stdout") and not log.get("stderr"):
            self._append_meta(f"[log unavailable: {log['error']}]")
            return
        self.text.configure(state="normal")
        for line in log.get("stdout", []):
            self.text.insert("end", _safe_line(line) + "\n")
        for line in log.get("stderr", []):
            self.text.insert("end", _safe_line(line) + "\n", ("stderr",))
        overflow = int(self.text.index("end-1c").split(".")[0]) - self._max_lines
        if overflow > 0:
            self.text.delete("1.0", f"{overflow + 1}.0")
        self.text.configure(state="disabled")
        if log.get("stdout_cursor") is not None:
            self._stdout_after = log["stdout_cursor"]
        if log.get("stderr_cursor") is not None:
            self._stderr_after = log["stderr_cursor"]
        truncated = log.get("stdout_truncated") or log.get("stderr_truncated")
        self._truncated_label.configure(text="older output was trimmed" if truncated else "")
        if self._autoscroll.get() and not self._user_scrolled_up:
            self.text.see("end")

    def _append_meta(self, message: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message + "\n", ("meta",))
        self.text.configure(state="disabled")

    @property
    def stdout_after(self) -> int:
        return self._stdout_after

    @property
    def stderr_after(self) -> int:
        return self._stderr_after


def _safe_line(line: Any) -> str:
    """Defends the log view against non-string/undecodable process output."""
    try:
        return str(line)
    except Exception:  # pragma: no cover - defensive
        return "<unprintable output>"


def _sortable_table(parent: tk.Misc, columns: tuple[tuple[str, str, int], ...]) -> ttk.Treeview:
    """Builds a Treeview with click-to-sort columns (ascending/descending)."""
    tree = ttk.Treeview(parent, columns=tuple(c[0] for c in columns), show="headings", selectmode="extended")
    for key, title, width in columns:
        tree.heading(key, text=title, command=lambda k=key: _sort_tree(tree, k, False))
        tree.column(key, width=width, anchor="w", stretch=True)
    return tree


def _sort_tree(tree: ttk.Treeview, column: str, descending: bool) -> None:
    def sort_key(item_id: str) -> Any:
        value = tree.set(item_id, column)
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    rows = sorted(tree.get_children(""), key=sort_key, reverse=descending)
    for index, item_id in enumerate(rows):
        tree.move(item_id, "", index)
    tree.heading(column, command=lambda: _sort_tree(tree, column, not descending))


def _open_in_file_manager(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        messagebox.showwarning("Could not open folder", f"{path}\n\n{exc}")


STATE_COLORS = {
    "Running": COLOR_OK, "Starting": COLOR_WARN, "Stopping": COLOR_WARN, "Paused": COLOR_WARN,
    "Finished": COLOR_MUTED, "Error": COLOR_ERROR, "running": COLOR_OK, "finished": COLOR_MUTED,
    "failed": COLOR_ERROR,
}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class Page(ttk.Frame):
    title = ""
    subtitle = ""

    def __init__(self, parent: tk.Misc, app: "ControlCenter") -> None:
        super().__init__(parent, padding=14)
        self.app = app
        self.adapter = app.adapter
        self._built = False

    def show(self) -> None:
        if not self._built:
            self._heading()
            self.build()
            self._built = True
        self.refresh()

    def _heading(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text=self.title, style="PageTitle.TLabel").pack(anchor="w")
        if self.subtitle:
            ttk.Label(header, text=self.subtitle, style="PageSubtitle.TLabel").pack(anchor="w")

    def build(self) -> None:
        raise NotImplementedError

    def refresh(self) -> None:
        """Called on every poll tick while this page is visible. Must only
        submit background work; it must never block."""

    def report_error(self, context: str, error: BaseException) -> None:
        self.app.set_status(f"{context}: {error}", error=True)


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Live overview of the most recent training run - real, persisted/live data only."

    STAT_LABELS = (
        "state", "run id", "progress", "fps", "elapsed / eta", "envs / workers",
        "rollout length", "ppo updates", "device", "reward",
    )

    def build(self) -> None:
        self.warning_banner = ttk.Label(self, text="", style="Warning.TLabel", wraplength=900, justify="left")
        self.stats = StatRow(self, self.STAT_LABELS)
        self.stats.pack(fill="x")
        self.warning_banner.pack(fill="x", pady=(8, 0))

        checkpoints_frame = ttk.LabelFrame(self, text="Checkpoints & latest evaluation", padding=10)
        checkpoints_frame.pack(fill="x", pady=(12, 0))
        self.checkpoints_label = ttk.Label(checkpoints_frame, text="n/a", justify="left")
        self.checkpoints_label.pack(anchor="w")

        charts_frame = ttk.LabelFrame(self, text="Live telemetry (bounded history)", padding=10)
        charts_frame.pack(fill="both", expand=True, pady=(12, 0))
        charts_grid = ttk.Frame(charts_frame)
        charts_grid.pack(fill="both", expand=True)
        self.reward_chart = LineChart(charts_grid, "mean episode reward vs. timesteps")
        self.fps_chart = LineChart(charts_grid, "steps/second vs. timesteps")
        self.kl_chart = LineChart(charts_grid, "PPO approx. KL vs. timesteps")
        for index, chart in enumerate((self.reward_chart, self.fps_chart, self.kl_chart)):
            chart.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
        charts_grid.columnconfigure(0, weight=1)
        charts_grid.columnconfigure(1, weight=1)
        charts_grid.rowconfigure(0, weight=1)
        charts_grid.rowconfigure(1, weight=1)

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Open run folder", command=self._open_run_folder).pack(side="left")
        ttk.Button(actions, text="View in Runs / Checkpoints", command=self._open_in_runs).pack(side="left", padx=(8, 0))

        self._last_run_dir: str | None = None

    def refresh(self) -> None:
        self.app.background.submit(self.adapter.dashboard_snapshot, self._on_snapshot)

    def _on_snapshot(self, snapshot: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or snapshot is None:
            self.report_error("Dashboard refresh failed", error or RuntimeError("unknown error"))
            return
        view = vm.dashboard_view(snapshot)
        self._last_run_dir = view.get("run_dir")
        self.stats.update_values({
            "state": (view["state"] or "no runs yet", STATE_COLORS.get(view["state"] or "")),
            "run id": (view["run_id"] or "n/a", None),
            "progress": (
                f"{vm.format_fraction_as_percent((view['progress_percent'] or 0) / 100.0)}"
                f" ({vm.format_number(view['timesteps'])}/{vm.format_number(view['target_timesteps'])})"
                if view["progress_percent"] is not None else "n/a", None,
            ),
            "fps": (vm.format_number(view["fps"], 1), None),
            "elapsed / eta": (f"{vm.format_duration(view['elapsed_seconds'])} / {vm.format_duration(view['eta_seconds'])}", None),
            "envs / workers": (f"{vm.format_number(view['environment_count'])} / {vm.format_number(view['env_workers'])}", None),
            "rollout length": (vm.format_number(view["rollout_length"]), None),
            "ppo updates": (vm.format_number(view["ppo_updates"]), None),
            "device": (view["device"] or "n/a", None),
            "reward": (vm.format_number(view["reward"], 3), None),
        })
        if view["stale"]:
            self.warning_banner.configure(text="\n".join(view["warnings"]), style="Error.TLabel")
        elif view["warnings"] or view["problems"]:
            self.warning_banner.configure(text="\n".join(view["warnings"] + view["problems"]), style="Warning.TLabel")
        else:
            self.warning_banner.configure(text="")

        checkpoint_lines = [
            f"current: {view['current_checkpoint'] or 'n/a'}",
            f"final: {view['final_checkpoint'] or 'n/a'}",
        ]
        if view["best_checkpoint"]:
            best = view["best_checkpoint"]
            checkpoint_lines.append(
                "best evaluation: reward=" + vm.format_number(best.get("mean_episode_reward"), 3)
                + f", win rate={vm.format_fraction_as_percent(best.get('win_rate'))}"
            )
        else:
            checkpoint_lines.append("best evaluation: n/a")
        if view["ppo_diagnostics"]:
            diag = view["ppo_diagnostics"]
            checkpoint_lines.append(
                "PPO diagnostics: approx_kl=" + vm.format_number(diag.get("approx_kl"), 4)
                + f", clip_fraction={vm.format_number(diag.get('clip_fraction'), 3)}"
                + f", explained_variance={vm.format_number(diag.get('explained_variance'), 3)}"
                + f", entropy={vm.format_number(diag.get('entropy'), 3)}"
            )
        self.checkpoints_label.configure(text="\n".join(checkpoint_lines))

        if view["run_dir"]:
            self.app.background.submit(lambda: self.adapter.telemetry_series(view["run_dir"]), self._on_telemetry)
        else:
            for chart in (self.reward_chart, self.fps_chart, self.kl_chart):
                chart.set_points([])

    def _on_telemetry(self, series: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or series is None or not series.get("available"):
            return
        data = series.get("series", {})
        self.reward_chart.set_points(data.get("mean_episode_reward", []))
        self.fps_chart.set_points(data.get("steps_per_second", []))
        self.kl_chart.set_points(data.get("approx_kl", []))

    def _open_run_folder(self) -> None:
        if self._last_run_dir:
            _open_in_file_manager(Path(self._last_run_dir))
        else:
            messagebox.showinfo("No run yet", "No training run has been found under the output root.")

    def _open_in_runs(self) -> None:
        self.app.show_page("Runs / Checkpoints")
        page = self.app.pages["Runs / Checkpoints"]
        if self._last_run_dir:
            page.select_run(self._last_run_dir)


class TrainingPage(Page):
    title = "Training"
    subtitle = "Launches python -m sandboxai train with a validated TrainingConfig - no RL logic lives here."

    def build(self) -> None:
        self.field_vars: dict[str, tk.StringVar] = {}
        defaults = vm.default_training_values()
        groups = vm.training_field_groups()

        form_container = ttk.Frame(self)
        form_container.pack(fill="x")

        basic_frame = ttk.LabelFrame(form_container, text="Basic configuration", padding=10)
        basic_frame.pack(fill="x")
        self._build_fields(basic_frame, groups["basic"], defaults)

        self._advanced_visible = tk.BooleanVar(value=False)
        toggle = ttk.Checkbutton(form_container, text="Show advanced options", variable=self._advanced_visible,
                                  command=self._toggle_advanced)
        toggle.pack(anchor="w", pady=(8, 0))
        self.advanced_frame = ttk.LabelFrame(form_container, text="Advanced (PPO / curriculum / checkpoints)", padding=10)
        self._build_fields(self.advanced_frame, groups["advanced"], defaults)

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(10, 4))
        self.start_button = ttk.Button(actions, text="Start training", command=self._start, style="Primary.TButton")
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop safely", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.force_stop_button = ttk.Button(actions, text="Force stop", command=self._force_stop, state="disabled", style="Danger.TButton")
        self.force_stop_button.pack(side="left", padx=(8, 0))
        self.state_label = ttk.Label(actions, text="idle", foreground=COLOR_MUTED)
        self.state_label.pack(side="left", padx=(16, 0))

        info_frame = ttk.Frame(self)
        info_frame.pack(fill="x")
        self.run_dir_label = ttk.Label(info_frame, text="run directory: n/a", foreground=COLOR_MUTED)
        self.run_dir_label.pack(anchor="w")

        log_frame = ttk.LabelFrame(self, text="Process output", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log_panel = LogPanel(log_frame)
        self.log_panel.pack(fill="both", expand=True)

        self.process_id: str | None = None

    def _build_fields(self, parent: ttk.Frame, specs: list[vm.TrainingFieldSpec], defaults: dict[str, str]) -> None:
        for row_index, spec in enumerate(specs):
            ttk.Label(parent, text=spec.label, width=26).grid(row=row_index, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=defaults.get(spec.name, ""))
            self.field_vars[spec.name] = var
            if spec.kind == "choice" and spec.choices:
                widget: tk.Widget = ttk.Combobox(parent, textvariable=var, values=spec.choices, state="readonly", width=28)
            elif spec.kind == "bool":
                widget = ttk.Checkbutton(parent, variable=var, onvalue="true", offvalue="false")
            elif spec.name in ("bc_checkpoint", "godot_executable"):
                widget = ttk.Frame(parent)
                ttk.Entry(widget, textvariable=var, width=30).pack(side="left")
                ttk.Button(widget, text="Browse", width=8,
                           command=lambda v=var: self._browse_file(v)).pack(side="left", padx=(4, 0))
            else:
                widget = ttk.Entry(parent, textvariable=var, width=30)
            widget.grid(row=row_index, column=1, sticky="w", pady=2, padx=(6, 16))
            if spec.help:
                ttk.Label(parent, text=spec.help, foreground=COLOR_MUTED, wraplength=360).grid(
                    row=row_index, column=2, sticky="w"
                )

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _toggle_advanced(self) -> None:
        if self._advanced_visible.get():
            self.advanced_frame.pack(fill="x", pady=(8, 0))
        else:
            self.advanced_frame.pack_forget()

    def refresh(self) -> None:
        if self.process_id:
            self.app.background.submit(lambda: self.adapter.process_status(self.process_id), self._on_process_status)

    def _start(self) -> None:
        values = {name: var.get() for name, var in self.field_vars.items()}
        try:
            config = vm.parse_training_form(values)
        except ValueError as exc:
            messagebox.showerror("Invalid training configuration", str(exc))
            return
        self.start_button.configure(state="disabled")

        def _launch() -> dict[str, Any]:
            return self.adapter.start_training(config)

        self.app.background.submit(_launch, self._on_started)

    def _on_started(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        self.start_button.configure(state="normal")
        if error is not None or result is None:
            messagebox.showerror("Training could not start", str(error))
            return
        self.process_id = result["process_id"]
        self.log_panel.reset_cursor()
        self.stop_button.configure(state="normal")
        self.force_stop_button.configure(state="normal")
        self.run_dir_label.configure(text=f"run directory: {result.get('run_dir', 'n/a')}")
        self.app.set_status(f"Training started ({self.process_id[:8]})")

    def _stop(self) -> None:
        if not self.process_id:
            return
        self.app.background.submit(lambda: self.adapter.cancel(self.process_id), lambda *_: None)
        self.app.set_status("Stop requested - the trainer will save a final checkpoint before exiting")

    def _force_stop(self) -> None:
        if not self.process_id:
            return
        if not messagebox.askyesno("Force stop", "This skips the cooperative shutdown; the final checkpoint will "
                                                  "NOT be saved. Continue?"):
            return
        self.app.background.submit(lambda: self.adapter.force_stop(self.process_id), lambda *_: None)

    def _on_process_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            return
        state = status.get("state", "unknown")
        color = STATE_COLORS.get(state, COLOR_MUTED)
        backend_state = (status.get("backend") or {}).get("state")
        label = backend_state or state
        self.state_label.configure(text=label, foreground=color)
        if state in ("finished", "failed"):
            self.stop_button.configure(state="disabled")
            self.force_stop_button.configure(state="disabled")
            if state == "failed" and status.get("error"):
                self.app.set_status(f"Training process failed: {status['error']}", error=True)
        self.app.background.submit(
            lambda: self.adapter.process_log(self.process_id, self.log_panel.stdout_after, self.log_panel.stderr_after),
            self._on_log,
        )

    def _on_log(self, log: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or log is None:
            return
        self.log_panel.apply_log(log)


class AgentsPage(Page):
    title = "Agents"
    subtitle = "Every training/evaluation/benchmark process this Control Center has launched or is tracking."

    COLUMNS = (
        ("kind", "Type", 90), ("run_id", "Run", 140), ("status", "Status", 90), ("pid", "PID", 70),
        ("environment_count", "Envs", 60), ("env_workers", "Workers", 70), ("progress_percent", "Progress", 80),
        ("started_at", "Started", 140), ("error", "Error", 220),
    )

    def build(self) -> None:
        paned = ttk.Panedwindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)
        top = ttk.Frame(paned)
        paned.add(top, weight=2)
        self.tree = _sortable_table(top, self.COLUMNS)
        self.tree.pack(fill="both", expand=True, side="left")
        scroll = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        bottom = ttk.Frame(paned, padding=(0, 8, 0, 0))
        paned.add(bottom, weight=1)
        actions = ttk.Frame(bottom)
        actions.pack(fill="x")
        self.stop_button = ttk.Button(actions, text="Stop safely", command=self._stop, state="disabled")
        self.stop_button.pack(side="left")
        self.force_stop_button = ttk.Button(actions, text="Force stop", command=self._force_stop, state="disabled", style="Danger.TButton")
        self.force_stop_button.pack(side="left", padx=(8, 0))
        self.log_panel = LogPanel(bottom)
        self.log_panel.pack(fill="both", expand=True, pady=(6, 0))

        self._row_to_process: dict[str, str] = {}
        self._selected_process_id: str | None = None

    def refresh(self) -> None:
        self.app.background.submit(self.adapter.list_processes, self._on_processes)
        if self._selected_process_id:
            self.app.background.submit(
                lambda: self.adapter.process_log(
                    self._selected_process_id, self.log_panel.stdout_after, self.log_panel.stderr_after
                ),
                self._on_log,
            )

    def _on_processes(self, processes: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or processes is None:
            self.report_error("Agents refresh failed", error or RuntimeError("unknown"))
            return
        rows = vm.process_table_rows(processes)
        selected = self._selected_process_id
        self.tree.delete(*self.tree.get_children())
        self._row_to_process.clear()
        for row in rows:
            item_id = self.tree.insert("", "end", values=(
                row["kind"], row["run_id"], row["status"] or "n/a", row["pid"] if row["pid"] is not None else "n/a",
                vm.format_number(row["environment_count"]), vm.format_number(row["env_workers"]),
                vm.format_fraction_as_percent((row["progress_percent"] or 0) / 100.0) if row["progress_percent"] is not None else "n/a",
                vm.format_timestamp(row["started_at"]), row["error"] or "",
            ))
            self._row_to_process[item_id] = row["id"]
            if row["id"] == selected:
                self.tree.selection_set(item_id)

    def _on_select(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            self._selected_process_id = None
            self.stop_button.configure(state="disabled")
            self.force_stop_button.configure(state="disabled")
            return
        process_id = self._row_to_process.get(selection[0])
        if process_id != self._selected_process_id:
            self._selected_process_id = process_id
            self.log_panel.reset_cursor()
        self.stop_button.configure(state="normal")
        self.force_stop_button.configure(state="normal")
        if process_id:
            self.app.background.submit(lambda: self.adapter.process_log(process_id), self._on_log)

    def _on_log(self, log: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or log is None:
            return
        self.log_panel.apply_log(log)

    def _stop(self) -> None:
        if self._selected_process_id:
            self.app.background.submit(lambda: self.adapter.cancel(self._selected_process_id), lambda *_: None)

    def _force_stop(self) -> None:
        if self._selected_process_id and messagebox.askyesno(
            "Force stop", "Skip cooperative shutdown and kill this process now?"
        ):
            self.app.background.submit(lambda: self.adapter.force_stop(self._selected_process_id), lambda *_: None)


class BenchmarkPage(Page):
    title = "Benchmarks"
    subtitle = "Runs python -m sandboxai benchmark and compares measured throughput across configurations."

    RESULT_COLUMNS = (
        ("source", "Sweep", 160), ("environments", "Envs", 60), ("workers", "Workers", 70),
        ("total_steps", "Total steps", 90), ("steps_per_second", "Steps/s", 90),
        ("episodes_per_second", "Episodes/s", 90), ("p50_ms", "p50 ms", 70), ("p95_ms", "p95 ms", 70),
        ("elapsed_seconds", "Elapsed", 80), ("info_mode", "Info mode", 90),
    )

    def build(self) -> None:
        form = ttk.LabelFrame(self, text="New benchmark sweep", padding=10)
        form.pack(fill="x")
        self.env_counts_var = tk.StringVar(value="1,2,4,8")
        self.worker_counts_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="2000")
        self.enemy_count_var = tk.StringVar(value="1")
        self.compact_var = tk.BooleanVar(value=True)
        for row, (label, var, width) in enumerate((
            ("Environment counts (comma-separated)", self.env_counts_var, 24),
            ("Worker counts (comma-separated)", self.worker_counts_var, 24),
            ("Steps per configuration", self.steps_var, 10),
            ("Enemy count", self.enemy_count_var, 10),
        )):
            ttk.Label(form, text=label, width=32).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(form, textvariable=var, width=width).grid(row=row, column=1, sticky="w", pady=2)
        ttk.Checkbutton(form, text="Compact training-path infos (uncheck for full diagnostic infos)",
                         variable=self.compact_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=8)
        self.run_button = ttk.Button(actions, text="Run benchmark", command=self._start, style="Primary.TButton")
        self.run_button.pack(side="left")
        self.state_label = ttk.Label(actions, text="idle", foreground=COLOR_MUTED)
        self.state_label.pack(side="left", padx=(12, 0))

        history_frame = ttk.LabelFrame(self, text="Result history (click a column to sort; select rows to compare)",
                                        padding=8)
        history_frame.pack(fill="both", expand=True)
        self.tree = _sortable_table(history_frame, self.RESULT_COLUMNS)
        self.tree.pack(fill="both", expand=True, side="left")
        scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        self.scaling_label = ttk.Label(self, text="", justify="left", foreground=COLOR_MUTED)
        self.scaling_label.pack(fill="x", pady=(6, 0))

        self.process_id: str | None = None

    def refresh(self) -> None:
        self.app.background.submit(self.adapter.benchmark_history, self._on_history)
        if self.process_id:
            self.app.background.submit(lambda: self.adapter.process_status(self.process_id), self._on_process_status)

    def _start(self) -> None:
        try:
            environment_counts = [int(v.strip()) for v in self.env_counts_var.get().split(",") if v.strip()]
            worker_counts = [int(v.strip()) for v in self.worker_counts_var.get().split(",") if v.strip()]
            steps = int(self.steps_var.get())
            enemy_count = int(self.enemy_count_var.get())
        except ValueError:
            messagebox.showerror("Invalid benchmark configuration", "Environment/worker counts, steps and enemy "
                                                                      "count must be integers.")
            return

        def _launch() -> dict[str, Any]:
            return self.adapter.start_benchmark(
                environment_counts=environment_counts, worker_counts=worker_counts, steps=steps,
                enemy_count=enemy_count, compact_infos=self.compact_var.get(),
            )

        self.run_button.configure(state="disabled")
        self.app.background.submit(_launch, self._on_started)

    def _on_started(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        self.run_button.configure(state="normal")
        if error is not None or result is None:
            messagebox.showerror("Benchmark could not start", str(error))
            return
        self.process_id = result["process_id"]
        self.app.set_status(f"Benchmark started ({self.process_id[:8]})")

    def _on_process_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            return
        self.state_label.configure(text=status.get("state", "unknown"), foreground=STATE_COLORS.get(status.get("state"), COLOR_MUTED))
        if status.get("state") in ("finished", "failed"):
            self.process_id = None

    def _on_history(self, history: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or history is None:
            self.report_error("Benchmark history refresh failed", error or RuntimeError("unknown"))
            return
        rows = vm.benchmark_history_rows(history)
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert("", "end", values=(
                row["source"], row["environments"], row["workers"], vm.format_number(row["total_steps"]),
                vm.format_number(row["steps_per_second"], 1), vm.format_number(row["episodes_per_second"], 2),
                vm.format_number(row["p50_ms"], 2), vm.format_number(row["p95_ms"], 2),
                vm.format_duration(row["elapsed_seconds"]), row["info_mode"] or "n/a",
            ))
        if history:
            scaling = history[0].get("scaling") or {}
            if scaling:
                lines = [
                    f"Most recent sweep ({history[0]['directory']}):",
                    f"  best throughput at {scaling.get('best_environment_count')} environments "
                    f"({vm.format_number(scaling.get('best_steps_per_second'), 1)} steps/s)",
                ]
                if scaling.get("diminishing_returns_at_environment_count") is not None:
                    lines.append(
                        f"  diminishing returns from {scaling['diminishing_returns_at_environment_count']} "
                        "environments onward"
                    )
                if scaling.get("worker_scaling"):
                    lines.append("  worker sharding speed-up measured vs. single-process baseline (see table)")
                self.scaling_label.configure(text="\n".join(lines))
            else:
                self.scaling_label.configure(text="")


class EvaluationPage(Page):
    title = "Evaluations"
    subtitle = "Evaluates a frozen checkpoint with the existing evaluator; action-head diagnostics included."

    CHECKPOINT_COLUMNS = (("run_id", "Run", 140), ("kind", "Kind", 80), ("path", "Path", 320),
                           ("modified_utc", "Modified", 160))
    EVAL_COLUMNS = (("path", "Path", 260), ("timesteps", "Timesteps", 90), ("episodes", "Episodes", 80),
                     ("win_rate", "Win rate", 80), ("loss_rate", "Loss rate", 80),
                     ("mean_episode_reward", "Reward", 80))

    def build(self) -> None:
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left, weight=1)
        ttk.Label(left, text="Checkpoints", style="Section.TLabel").pack(anchor="w")
        self.checkpoint_tree = _sortable_table(left, self.CHECKPOINT_COLUMNS)
        self.checkpoint_tree.pack(fill="both", expand=True)

        form = ttk.LabelFrame(left, text="Run evaluation on the selected checkpoint", padding=10)
        form.pack(fill="x", pady=(8, 0))
        self.episodes_var = tk.StringVar(value="20")
        self.env_count_var = tk.StringVar(value="1")
        self.device_var = tk.StringVar(value="auto")
        for row, (label, var, kind) in enumerate((
            ("Episodes", self.episodes_var, "entry"), ("Environment count", self.env_count_var, "entry"),
            ("Device", self.device_var, "choice"),
        )):
            ttk.Label(form, text=label, width=16).grid(row=row, column=0, sticky="w", pady=2)
            if kind == "choice":
                ttk.Combobox(form, textvariable=var, values=("auto", "cpu", "cuda"), state="readonly",
                             width=16).grid(row=row, column=1, sticky="w")
            else:
                ttk.Entry(form, textvariable=var, width=18).grid(row=row, column=1, sticky="w")
        self.run_button = ttk.Button(form, text="Start evaluation", command=self._start, style="Primary.TButton")
        self.run_button.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.selected_checkpoint_label = ttk.Label(left, text="selected checkpoint: none", foreground=COLOR_MUTED,
                                                    wraplength=380)
        self.selected_checkpoint_label.pack(anchor="w")
        self.state_label = ttk.Label(left, text="", foreground=COLOR_MUTED)
        self.state_label.pack(anchor="w")

        right = ttk.Frame(paned, padding=(8, 0, 0, 0))
        paned.add(right, weight=1)
        ttk.Label(right, text="Evaluation results (select multiple rows to compare)", style="Section.TLabel").pack(anchor="w")
        self.eval_tree = _sortable_table(right, self.EVAL_COLUMNS)
        self.eval_tree.pack(fill="both", expand=False)
        self.eval_tree.configure(selectmode="extended")
        self.eval_tree.bind("<<TreeviewSelect>>", self._on_eval_select)

        detail_frame = ttk.LabelFrame(right, text="Structured result / comparison", padding=8)
        detail_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.detail_text = tk.Text(detail_frame, wrap="word", state="disabled", height=18, font=("Consolas", 9))
        self.detail_text.pack(fill="both", expand=True)

        self._checkpoint_paths: dict[str, str] = {}
        self._eval_paths: dict[str, str] = {}
        self.selected_checkpoint: str | None = None
        self.checkpoint_tree.bind("<<TreeviewSelect>>", self._on_checkpoint_select)
        self.process_id: str | None = None

    def select_checkpoint(self, path: str) -> None:
        self.selected_checkpoint = path
        self.selected_checkpoint_label.configure(text=f"selected checkpoint: {path}")

    def refresh(self) -> None:
        self.app.background.submit(self.adapter.discover_checkpoints, self._on_checkpoints)
        self.app.background.submit(self.adapter.discover_evaluations, self._on_evaluations)
        if self.process_id:
            self.app.background.submit(lambda: self.adapter.process_status(self.process_id), self._on_process_status)

    def _on_checkpoints(self, entries: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or entries is None:
            self.report_error("Checkpoint list refresh failed", error or RuntimeError("unknown"))
            return
        self.checkpoint_tree.delete(*self.checkpoint_tree.get_children())
        self._checkpoint_paths.clear()
        for entry in entries:
            item_id = self.checkpoint_tree.insert("", "end", values=(
                entry["run_id"], entry["kind"], entry["path"], entry.get("modified_utc") or "n/a",
            ))
            self._checkpoint_paths[item_id] = entry["path"]

    def _on_checkpoint_select(self, _event: object) -> None:
        selection = self.checkpoint_tree.selection()
        if selection:
            self.selected_checkpoint = self._checkpoint_paths.get(selection[0])

    def _start(self) -> None:
        if not self.selected_checkpoint:
            messagebox.showwarning("Checkpoint required", "Select a checkpoint from the list on the left.")
            return
        try:
            episodes = int(self.episodes_var.get())
            environment_count = int(self.env_count_var.get())
        except ValueError:
            messagebox.showerror("Invalid evaluation configuration", "Episodes and environment count must be integers.")
            return
        checkpoint = self.selected_checkpoint
        device = self.device_var.get()

        def _launch() -> dict[str, Any]:
            return self.adapter.start_evaluation(checkpoint, episodes=episodes, environment_count=environment_count,
                                                  device=device)

        self.run_button.configure(state="disabled")
        self.app.background.submit(_launch, self._on_started)

    def _on_started(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        self.run_button.configure(state="normal")
        if error is not None or result is None:
            messagebox.showerror("Evaluation could not start", str(error))
            return
        self.process_id = result["process_id"]
        self.app.set_status(f"Evaluation started ({self.process_id[:8]})")

    def _on_process_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            return
        self.state_label.configure(text=f"evaluation process: {status.get('state')}",
                                    foreground=STATE_COLORS.get(status.get("state"), COLOR_MUTED))
        if status.get("state") in ("finished", "failed"):
            self.process_id = None

    def _on_evaluations(self, entries: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or entries is None:
            self.report_error("Evaluation list refresh failed", error or RuntimeError("unknown"))
            return
        self.eval_tree.delete(*self.eval_tree.get_children())
        self._eval_paths.clear()
        for entry in entries:
            item_id = self.eval_tree.insert("", "end", values=(
                entry["path"], vm.format_number(entry.get("timesteps")), vm.format_number(entry.get("episodes")),
                vm.format_fraction_as_percent(entry.get("win_rate")), vm.format_fraction_as_percent(entry.get("loss_rate")),
                vm.format_number(entry.get("mean_episode_reward"), 3),
            ))
            self._eval_paths[item_id] = entry["path"]

    def _on_eval_select(self, _event: object) -> None:
        selection = self.eval_tree.selection()
        paths = [self._eval_paths[item_id] for item_id in selection if item_id in self._eval_paths]
        if not paths:
            return
        self.app.background.submit(lambda: [self.adapter.evaluation_detail(path) for path in paths], self._on_details)

    def _on_details(self, details: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or details is None:
            self.report_error("Evaluation detail failed", error or RuntimeError("unknown"))
            return
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        if len(details) == 1:
            self.detail_text.insert("end", _render_evaluation_detail(vm.evaluation_view(details[0])))
        else:
            rows = vm.evaluation_comparison_rows(details)
            self.detail_text.insert("end", _render_evaluation_comparison(rows))
        self.detail_text.configure(state="disabled")


def _render_evaluation_detail(view: dict[str, Any]) -> str:
    if not view.get("available"):
        return f"Evaluation summary unavailable: {view.get('error')}"
    lines = [
        f"path: {view['path']}",
        f"episodes: {vm.format_number(view['episodes'])}   timesteps: {vm.format_number(view['timesteps'])}",
        "",
        "Outcomes",
        f"  win rate: {vm.format_fraction_as_percent(view['outcomes']['win_rate'])}"
        f"   loss rate: {vm.format_fraction_as_percent(view['outcomes']['loss_rate'])}"
        f"   timeout rate: {vm.format_fraction_as_percent(view['outcomes']['timeout_rate'])}",
        "",
        "Combat",
        f"  kills: {vm.format_number(view['combat']['mean_kills'], 2)}"
        f"   deaths: {vm.format_number(view['combat']['mean_deaths'], 2)}"
        f"   damage dealt: {vm.format_number(view['combat']['mean_damage_dealt'], 1)}"
        f"   damage received: {vm.format_number(view['combat']['mean_damage_received'], 1)}",
        "",
        "Accuracy",
        f"  accuracy: {vm.format_fraction_as_percent(view['accuracy']['mean_accuracy'])}"
        f"   shots fired: {vm.format_number(view['accuracy']['mean_shots_fired'], 1)}"
        f"   shots hit: {vm.format_number(view['accuracy']['mean_shots_hit'], 1)}",
        "",
        "Action-head diagnostics (zero-shot / policy discharge behavior)",
        f"  shoot request rate: {vm.format_fraction_as_percent(view['action_head_diagnostics']['policy_shoot_request_rate'])}",
        f"  discharge rate: {vm.format_fraction_as_percent(view['action_head_diagnostics']['discharge_rate'])}",
        f"  action-pipeline localization: {view['action_head_diagnostics']['localization'] or 'n/a'}",
    ]
    return "\n".join(lines)


def _render_evaluation_comparison(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No comparable evaluation summaries in this selection."
    header = f"{'path':40} {'reward':>8} {'win%':>7} {'loss%':>7} {'acc%':>7} {'discharge%':>11}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{Path(row['path']).name:40} {vm.format_number(row['reward'], 2):>8} "
            f"{vm.format_fraction_as_percent(row['win_rate']):>7} {vm.format_fraction_as_percent(row['loss_rate']):>7} "
            f"{vm.format_fraction_as_percent(row['accuracy']):>7} {vm.format_fraction_as_percent(row['discharge_rate']):>11}"
        )
    return "\n".join(lines)


class RunsPage(Page):
    title = "Runs / Checkpoints"
    subtitle = "Read-only inventory from run_inspection.py - selecting a run shows its full detail."

    COLUMNS = (("run_id", "Run", 160), ("state", "State", 90), ("progress_percent", "Progress", 80),
               ("device", "Device", 70), ("environment_count", "Envs", 55), ("env_workers", "Workers", 65),
               ("checkpoints", "Checkpoints", 90), ("reward", "Reward", 80), ("win_rate", "Win rate", 80),
               ("modified_utc", "Modified", 160))

    def build(self) -> None:
        paned = ttk.Panedwindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)
        top = ttk.Frame(paned)
        paned.add(top, weight=1)
        self.tree = _sortable_table(top, self.COLUMNS)
        self.tree.pack(fill="both", expand=True, side="left")
        scroll = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        bottom = ttk.LabelFrame(paned, text="Run detail", padding=8)
        paned.add(bottom, weight=1)
        self.detail_text = tk.Text(bottom, wrap="word", state="disabled", font=("Consolas", 9))
        self.detail_text.pack(fill="both", expand=True)
        actions = ttk.Frame(bottom)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Open run folder", command=self._open_folder).pack(side="left")
        ttk.Button(actions, text="Evaluate latest checkpoint", command=self._evaluate).pack(side="left", padx=(8, 0))

        self._row_to_dir: dict[str, str] = {}
        self._selected_run_dir: str | None = None
        self._pending_run_selection: str | None = None

    def select_run(self, run_dir: str) -> None:
        """Selects ``run_dir`` in the table once it is populated.

        Called right after navigating here (e.g. from the Dashboard's "View
        in Runs / Checkpoints"), before the first background
        ``list_runs()`` fetch may have completed - so the target is
        remembered and applied by ``_on_runs`` as soon as the row exists,
        instead of silently doing nothing on a timing race.
        """
        if self._select_existing_row(run_dir):
            return
        self._pending_run_selection = run_dir

    def _select_existing_row(self, run_dir: str) -> bool:
        for item_id, directory in self._row_to_dir.items():
            if directory == run_dir:
                self.tree.selection_set(item_id)
                self.tree.see(item_id)
                self._on_select(None)
                return True
        return False

    def refresh(self) -> None:
        self.app.background.submit(self.adapter.list_runs, self._on_runs)

    def _on_runs(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or result is None:
            self.report_error("Run list refresh failed", error or RuntimeError("unknown"))
            return
        rows = vm.runs_table_rows(result)
        selected = self._selected_run_dir
        self.tree.delete(*self.tree.get_children())
        self._row_to_dir.clear()
        for row in rows:
            item_id = self.tree.insert("", "end", values=(
                row["run_id"], row["state"] or "n/a",
                vm.format_fraction_as_percent((row["progress_percent"] or 0) / 100.0) if row["progress_percent"] is not None else "n/a",
                row["device"] or "n/a", vm.format_number(row["environment_count"]), vm.format_number(row["env_workers"]),
                vm.format_number(row["checkpoints"]), vm.format_number(row["reward"], 3),
                vm.format_fraction_as_percent(row["win_rate"]), row["modified_utc"] or "n/a",
            ))
            self._row_to_dir[item_id] = row["run_dir"]
            if row["run_dir"] == selected:
                self.tree.selection_set(item_id)
        if self._pending_run_selection is not None:
            if self._select_existing_row(self._pending_run_selection):
                self._pending_run_selection = None

    def _on_select(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        run_dir = self._row_to_dir.get(selection[0])
        self._selected_run_dir = run_dir
        if run_dir:
            self.app.background.submit(lambda: self.adapter.inspect_run(run_dir), self._on_detail)

    def _on_detail(self, report: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or report is None:
            return
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", _render_run_detail(report))
        self.detail_text.configure(state="disabled")

    def _open_folder(self) -> None:
        if self._selected_run_dir:
            _open_in_file_manager(Path(self._selected_run_dir))

    def _evaluate(self) -> None:
        if not self._selected_run_dir:
            return
        latest = Path(self._selected_run_dir) / "checkpoints" / "latest.zip"
        if not latest.is_file():
            messagebox.showinfo("No checkpoint yet", "This run has no checkpoints/latest.zip yet.")
            return
        self.app.show_page("Evaluations")
        self.app.pages["Evaluations"].select_checkpoint(str(latest))


def _render_run_detail(report: dict[str, Any]) -> str:
    manifest = report.get("manifest") or {}
    config = report.get("config") or {}
    checkpoints = report.get("checkpoints") or {}
    evaluation = report.get("evaluation") or {}
    warnings = report.get("warnings") or []
    problems = report.get("problems") or []
    lines = [
        f"run_id: {report.get('run_id')}    experiment: {report.get('experiment_id') or 'n/a'}",
        f"state: {(report.get('status') or {}).get('state')}    created: {manifest.get('created_utc', 'n/a')}",
        f"device: {config.get('device', 'n/a')}    seed: {manifest.get('seed', config.get('seed', 'n/a'))}",
        f"envs: {config.get('environment_count', 'n/a')}    workers: {config.get('env_workers', 'n/a')}",
        f"total timesteps: {config.get('total_training_steps', 'n/a')}    "
        f"rollout length: {config.get('resolved_rollout_length', config.get('rollout_length', 'n/a'))}",
        f"godot: {(manifest.get('godot') or {}).get('version', 'n/a')}",
        "",
        f"checkpoints: {checkpoints.get('count', 0)} "
        f"(latest={checkpoints.get('has_latest')}, best={checkpoints.get('has_best')})",
    ]
    latest_eval = evaluation.get("latest")
    if latest_eval:
        lines.append(
            f"latest evaluation: reward={latest_eval.get('mean_episode_reward')} "
            f"win_rate={latest_eval.get('win_rate')}"
        )
    training_profile = report.get("summary", {}).get("training_profile")
    if training_profile:
        lines.append(f"profiling artifact: {training_profile}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    if problems:
        lines.append("")
        lines.append("Problems:")
        lines.extend(f"  - {problem}" for problem in problems)
    return "\n".join(lines)


class SystemPage(Page):
    title = "System / Telemetry"
    subtitle = "Real, measured host/runtime status. Unavailable metrics are shown as such, never estimated."

    STAT_LABELS = ("cpu", "process memory", "python", "godot", "torch", "cuda")

    def build(self) -> None:
        self.stats = StatRow(self, self.STAT_LABELS)
        self.stats.pack(fill="x")
        deps_frame = ttk.LabelFrame(self, text="Optional dependencies", padding=10)
        deps_frame.pack(fill="x", pady=(12, 0))
        self.deps_label = ttk.Label(deps_frame, text="n/a", justify="left")
        self.deps_label.pack(anchor="w")

        chart_frame = ttk.LabelFrame(self, text="Host resource usage (sampled each refresh, bounded to the last "
                                                  "300 samples)", padding=10)
        chart_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.cpu_chart = LineChart(chart_frame, "CPU percent (this process)")
        self.cpu_chart.pack(fill="both", expand=True, pady=(0, 4))
        self.rss_chart = LineChart(chart_frame, "process RSS (MB)")
        self.rss_chart.pack(fill="both", expand=True)

        from collections import deque

        self._cpu_series: "deque[tuple[float, float]]" = deque(maxlen=300)
        self._rss_series: "deque[tuple[float, float]]" = deque(maxlen=300)
        self._sample_index = 0.0

    def refresh(self) -> None:
        self.app.background.submit(self.adapter.system_status, self._on_status)

    def _on_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            self.report_error("System status refresh failed", error or RuntimeError("unknown"))
            return
        godot_text = "not found"
        if status.get("godot_available"):
            version = status.get("godot_version")
            godot_text = f"available ({version})" if version else "available (version unknown)"
        self.stats.update_values({
            "cpu": (vm.format_fraction_as_percent((status.get("cpu_percent") or 0) / 100.0), None),
            "process memory": (vm.format_bytes((status.get("process_rss_mb") or 0) * 1024 * 1024)
                                if status.get("process_rss_mb") is not None else "n/a", None),
            "python": (status.get("python_version", "n/a"), None),
            "godot": (godot_text, COLOR_OK if status.get("godot_available") else COLOR_WARN),
            "torch": ("available" if status.get("torch_available") else "not installed",
                      COLOR_OK if status.get("torch_available") else COLOR_MUTED),
            "cuda": ("available" if status.get("cuda_available") else "not available",
                     COLOR_OK if status.get("cuda_available") else COLOR_MUTED),
        })
        deps = status.get("dependencies", {})
        self.deps_label.configure(text="   ".join(
            f"{name}: {'yes' if available else 'no'}" for name, available in deps.items()
        ))
        self._sample_index += 1.0
        if status.get("cpu_percent") is not None:
            self._cpu_series.append((self._sample_index, float(status["cpu_percent"])))
        if status.get("process_rss_mb") is not None:
            self._rss_series.append((self._sample_index, float(status["process_rss_mb"])))
        self.cpu_chart.set_points(list(self._cpu_series))
        self.rss_chart.set_points(list(self._rss_series))


class SettingsPage(Page):
    title = "Settings"
    subtitle = "Project and output roots this Control Center reads from and writes runs into."

    def build(self) -> None:
        self.project_root_label = ttk.Label(self, text="")
        self.project_root_label.pack(anchor="w", pady=2)
        self.output_root_label = ttk.Label(self, text="")
        self.output_root_label.pack(anchor="w", pady=2)
        ttk.Button(self, text="Change output root...", command=self._change_output_root).pack(anchor="w", pady=(8, 0))
        ttk.Label(self, text="Changing the output root points this session's Runs/Checkpoints/Evaluations/"
                              "Benchmarks pages at a different directory; it does not move existing runs.",
                  foreground=COLOR_MUTED, wraplength=700).pack(anchor="w", pady=(4, 0))

    def refresh(self) -> None:
        self.project_root_label.configure(text=f"project root: {self.adapter.project_root}")
        self.output_root_label.configure(text=f"output root: {self.adapter.output_root}")

    def _change_output_root(self) -> None:
        directory = filedialog.askdirectory(initialdir=str(self.adapter.output_root))
        if directory:
            self.app.set_output_root(directory)
            self.refresh()


PAGE_CLASSES: tuple[type[Page], ...] = (
    DashboardPage, TrainingPage, AgentsPage, BenchmarkPage, EvaluationPage, RunsPage, SystemPage, SettingsPage,
)


# ---------------------------------------------------------------------------
# Application shell
# ---------------------------------------------------------------------------


class ControlCenter(tk.Tk):
    POLL_MS = 1500

    def __init__(self, adapter: SandboxAIAdapter | None = None) -> None:
        super().__init__()
        self.title("SandboxAI Control Center")
        self.geometry("1360x860")
        self.minsize(1080, 680)
        self.adapter = adapter or SandboxAIAdapter()
        self.background = BackgroundRunner(self)
        self._configure_style()
        self.pages: dict[str, Page] = {}
        self._current: Page | None = None
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._build_shell()
        self.after(self.POLL_MS, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(background=COLOR_BG)
        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT,
                        fieldbackground=COLOR_SURFACE_RAISED, bordercolor=COLOR_BORDER,
                        lightcolor=COLOR_BORDER, darkcolor=COLOR_BORDER,
                        font=(_FONT_FAMILY, 10))
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("PageTitle.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                        font=(_FONT_FAMILY, 20, "bold"))
        style.configure("PageSubtitle.TLabel", background=COLOR_BG, foreground=COLOR_MUTED,
                        font=(_FONT_FAMILY, 10))
        style.configure("Section.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                        font=(_FONT_FAMILY, 12, "bold"))
        style.configure("Card.TFrame", background=COLOR_SURFACE_RAISED, relief="flat", borderwidth=0)
        style.configure("CardLabel.TLabel", background=COLOR_SURFACE_RAISED, foreground=COLOR_MUTED,
                        font=(_FONT_FAMILY, 9))
        style.configure("CardValue.TLabel", background=COLOR_SURFACE_RAISED, foreground=COLOR_TEXT,
                        font=(_FONT_FAMILY, 15, "bold"))
        style.configure("TLabelframe", background=COLOR_BG, bordercolor=COLOR_BORDER, relief="solid")
        style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_MUTED,
                        font=(_FONT_FAMILY, 10, "bold"))
        style.configure("TButton", background=COLOR_SURFACE_RAISED, foreground=COLOR_TEXT,
                        borderwidth=0, padding=(12, 8))
        style.map("TButton", background=[("active", COLOR_HOVER), ("pressed", "#243b53")],
                  foreground=[("disabled", COLOR_MUTED)])
        style.configure("Primary.TButton", background=COLOR_ACCENT, foreground="#06131d",
                        borderwidth=0, padding=(14, 9), font=(_FONT_FAMILY, 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#7dd3fc"),
                                                 ("pressed", "#0ea5e9"),
                                                 ("disabled", COLOR_BORDER)])
        style.configure("Danger.TButton", background=COLOR_SURFACE_RAISED, foreground=COLOR_ERROR,
                        borderwidth=0, padding=(12, 8))
        style.map("Danger.TButton", background=[("active", "#4c1d2a"),
                                                ("pressed", "#881337")])
        style.configure("NavGroup.TLabel", background=COLOR_BG, foreground=COLOR_MUTED,
                        font=(_FONT_FAMILY, 9, "bold"))
        style.configure("Nav.TButton", anchor="w", padding=(14, 10), background=COLOR_BG)
        style.map("Nav.TButton", background=[("active", COLOR_HOVER)])
        style.configure("NavSelected.TButton", anchor="w", padding=(14, 10),
                        background=COLOR_HOVER, foreground=COLOR_ACCENT,
                        font=(_FONT_FAMILY, 10, "bold"))
        style.map("NavSelected.TButton", background=[("active", COLOR_HOVER)])
        style.configure("Warning.TLabel", background=COLOR_BG, foreground=COLOR_WARN)
        style.configure("Error.TLabel", background=COLOR_BG, foreground=COLOR_ERROR)
        style.configure("Treeview", background=COLOR_SURFACE, fieldbackground=COLOR_SURFACE,
                        foreground=COLOR_TEXT, rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background=COLOR_SURFACE_RAISED,
                        foreground=COLOR_MUTED, relief="flat", padding=(8, 8))
        style.map("Treeview", background=[("selected", "#164e63")],
                  foreground=[("selected", COLOR_TEXT)])
        style.configure("TEntry", fieldbackground=COLOR_SURFACE_RAISED, foreground=COLOR_TEXT,
                        insertcolor=COLOR_TEXT, padding=7)
        style.configure("TCombobox", fieldbackground=COLOR_SURFACE_RAISED, foreground=COLOR_TEXT,
                        padding=6)
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=COLOR_SURFACE, foreground=COLOR_MUTED,
                        padding=(12, 8))
        style.map("TNotebook.Tab", background=[("selected", COLOR_HOVER)],
                  foreground=[("selected", COLOR_ACCENT)])

    def _build_shell(self) -> None:
        outer = ttk.Frame(self, padding=0)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="SandboxAI Control Center", style="PageTitle.TLabel").pack(side="left")
        self.status_label = ttk.Label(header, text="ready", foreground=COLOR_MUTED)
        self.status_label.pack(side="right")
        ttk.Separator(outer).pack(fill="x")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        nav = ttk.Frame(body, width=210, padding=(12, 18))
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        self.content = ttk.Frame(body, padding=(8, 14, 18, 18))
        self.content.pack(side="left", fill="both", expand=True)

        nav_groups = {0: "Overview", 1: "Operations", 3: "Analysis", 6: "System"}
        for index, page_class in enumerate(PAGE_CLASSES):
            if index in nav_groups:
                if index:
                    ttk.Separator(nav).pack(fill="x", pady=(12, 8))
                ttk.Label(nav, text=nav_groups[index], style="NavGroup.TLabel").pack(
                    fill="x", padx=8, pady=(0, 4)
                )
            button = ttk.Button(nav, text=page_class.title, style="Nav.TButton",
                                 command=lambda name=page_class.title: self.show_page(name))
            button.pack(fill="x", pady=2)
            self._nav_buttons[page_class.title] = button
            page = page_class(self.content, self)
            self.pages[page_class.title] = page

        ttk.Separator(nav).pack(fill="x", pady=10)
        ttk.Label(nav, text="Real backend only.\nNo simulated data.", foreground=COLOR_MUTED,
                  justify="left").pack(anchor="w")

        self.show_page(PAGE_CLASSES[0].title)

    def show_page(self, name: str) -> None:
        if self._current is not None:
            self._current.pack_forget()
        page = self.pages[name]
        page.pack(fill="both", expand=True)
        self._current = page
        for title, button in self._nav_buttons.items():
            button.configure(style="NavSelected.TButton" if title == name else "Nav.TButton")
        page.show()

    def set_status(self, message: str, error: bool = False) -> None:
        self.status_label.configure(text=message, foreground=COLOR_ERROR if error else COLOR_MUTED)

    def set_output_root(self, output_root: str) -> None:
        self.background.close()
        self.adapter = SandboxAIAdapter(project_root=self.adapter.project_root, output_root=output_root)
        self.background = BackgroundRunner(self)
        for page in self.pages.values():
            page.adapter = self.adapter

    def _tick(self) -> None:
        if self._current is not None:
            try:
                self._current.refresh()
            except Exception as exc:  # noqa: BLE001 - a page bug must not stop polling entirely
                self.set_status(f"internal error refreshing {self._current.title}: {exc}", error=True)
        self.after(self.POLL_MS, self._tick)

    def _on_close(self) -> None:
        self.background.close()
        self.adapter.close()
        self.destroy()


def main(project_root: str | None = None, output_root: str = "training") -> int:
    app = ControlCenter(adapter=SandboxAIAdapter(project_root=project_root, output_root=output_root))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

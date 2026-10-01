"""Desktop Control Center pages.

Each page translates UI intent into adapter calls; formatting and validation
remain in the independently tested viewmodel.
"""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from . import control_center_viewmodel as vm
from .control_center_widgets import (
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_WARN,
    LineChart,
    LogPanel,
    StatRow,
    ToolTip,
    _open_in_file_manager,
    _scrollable_table,
)

STATE_COLORS = {
    "Running": COLOR_OK,
    "Starting": COLOR_WARN,
    "Stopping": COLOR_WARN,
    "Paused": COLOR_WARN,
    "Finished": COLOR_MUTED,
    "Error": COLOR_ERROR,
    "running": COLOR_OK,
    "finished": COLOR_MUTED,
    "failed": COLOR_ERROR,
}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class Page(ttk.Frame):
    title = ""
    subtitle = ""

    def __init__(self, parent: tk.Misc, app: Any) -> None:
        super().__init__(parent, padding=14)
        self.app = app
        self.adapter = app.adapter
        self._built = False
        # A poll can touch a large run directory or a process registry. Keep
        # at most one request per page operation in flight: piling identical
        # reads into the shared three-worker pool makes a slow disk look like
        # a frozen GUI and can render stale telemetry long after it mattered.
        # A second request is retained as one latest-only follow-up instead
        # of being lost: a refresh requested while the initial app-start poll
        # is still reading an empty run root must still observe a run that
        # appears before that poll comes back.
        self._polls_in_flight: set[str] = set()
        self._pending_polls: dict[
            str, tuple[Callable[[], Any], Callable[[Any, BaseException | None], None]]
        ] = {}

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

    def submit_poll(
        self,
        operation: str,
        fn: Callable[[], Any],
        callback: Callable[[Any, BaseException | None], None],
    ) -> None:
        """Submit one periodic read, coalescing overlap to one fresh retry.

        This is deliberately for idempotent refreshes only, never for a user
        command such as Start, Stop or Force stop. A request that arrives
        while its operation is live replaces one retained follow-up request;
        it never expands the executor queue.  The Tk callback releases the
        guard before handing its result to the page, so errors cannot wedge a
        later refresh and work requested during a slow read is not discarded.
        """
        if operation in self._polls_in_flight:
            self._pending_polls[operation] = (fn, callback)
            return
        self._start_poll(operation, fn, callback)

    def _start_poll(
        self,
        operation: str,
        fn: Callable[[], Any],
        callback: Callable[[Any, BaseException | None], None],
    ) -> None:
        self._polls_in_flight.add(operation)

        def complete(result: Any, error: BaseException | None) -> None:
            self._polls_in_flight.discard(operation)
            try:
                callback(result, error)
            finally:
                pending = self._pending_polls.pop(operation, None)
                if pending is not None:
                    self._start_poll(operation, *pending)

        if not self.app.background.submit(fn, complete):
            self._polls_in_flight.discard(operation)

    def reset_polls(self) -> None:
        """Forget work tied to a background runner that has been replaced."""
        self._polls_in_flight.clear()
        self._pending_polls.clear()

    def report_error(self, context: str, error: BaseException) -> None:
        self.app.set_status(f"{context}: {error}", error=True)


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Live overview of the most recent training run - real, persisted/live data only."

    STAT_LABELS = (
        "state",
        "run id",
        "progress",
        "fps",
        "elapsed / eta",
        "envs / workers",
        "agents",
        "device",
        "reward",
    )

    def build(self) -> None:
        self.warning_banner = ttk.Label(
            self, text="", style="Warning.TLabel", wraplength=900, justify="left"
        )
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
        ttk.Button(actions, text="View in Runs / Checkpoints", command=self._open_in_runs).pack(
            side="left", padx=(8, 0)
        )

        self._last_run_dir: str | None = None

    def refresh(self) -> None:
        self.submit_poll("dashboard", self.adapter.dashboard_snapshot, self._on_snapshot)
        self.submit_poll("agents-summary", self.adapter.agents.views, self._on_agents_summary)

    def _on_agents_summary(
        self, views: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or views is None:
            return
        counts: dict[str, int] = {}
        for view in views:
            lifecycle = str(view.get("lifecycle", ""))
            counts[lifecycle] = counts.get(lifecycle, 0) + 1
        if not counts:
            text, color = "none launched", None
        else:
            text = ", ".join(f"{count} {name.lower()}" for name, count in sorted(counts.items()))
            color = COLOR_ERROR if counts.get("FAILED") else None
        self.stats.update_values({"agents": (text, color)})

    def _on_snapshot(self, snapshot: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or snapshot is None:
            self.report_error("Dashboard refresh failed", error or RuntimeError("unknown error"))
            return
        view = vm.dashboard_view(snapshot)
        previous_run_dir = self._last_run_dir
        self._last_run_dir = view.get("run_dir")
        if self._last_run_dir != previous_run_dir:
            # Never temporarily chart a prior run while the next telemetry
            # read is still in flight for the newly discovered run.
            for chart in (self.reward_chart, self.fps_chart, self.kl_chart):
                chart.set_points([])
        self.stats.update_values(
            {
                "state": (view["state"] or "no runs yet", STATE_COLORS.get(view["state"] or "")),
                "run id": (view["run_id"] or "n/a", None),
                "progress": (
                    f"{vm.format_fraction_as_percent((view['progress_percent'] or 0) / 100.0)}"
                    f" ({vm.format_number(view['timesteps'])}/{vm.format_number(view['target_timesteps'])})"
                    if view["progress_percent"] is not None
                    else "n/a",
                    None,
                ),
                "fps": (vm.format_number(view["fps"], 1), None),
                "elapsed / eta": (
                    f"{vm.format_duration(view['elapsed_seconds'])} / {vm.format_duration(view['eta_seconds'])}",
                    None,
                ),
                "envs / workers": (
                    f"{vm.format_number(view['environment_count'])} / {vm.format_number(view['env_workers'])}",
                    None,
                ),
                "device": (view["device"] or "n/a", None),
                "reward": (vm.format_number(view["reward"], 3), None),
            }
        )
        if view["stale"]:
            self.warning_banner.configure(text="\n".join(view["warnings"]), style="Error.TLabel")
        elif view["warnings"] or view["problems"]:
            self.warning_banner.configure(
                text="\n".join(view["warnings"] + view["problems"]), style="Warning.TLabel"
            )
        else:
            self.warning_banner.configure(text="")

        checkpoint_lines = [
            f"current: {view['current_checkpoint'] or 'n/a'}",
            f"final: {view['final_checkpoint'] or 'n/a'}",
        ]
        if view["best_checkpoint"]:
            best = view["best_checkpoint"]
            checkpoint_lines.append(
                "best evaluation: reward="
                + vm.format_number(best.get("mean_episode_reward"), 3)
                + f", win rate={vm.format_fraction_as_percent(best.get('win_rate'))}"
            )
        else:
            checkpoint_lines.append("best evaluation: n/a")
        if view["ppo_diagnostics"]:
            diag = view["ppo_diagnostics"]
            checkpoint_lines.append(
                "PPO diagnostics: approx_kl="
                + vm.format_number(diag.get("approx_kl"), 4)
                + f", clip_fraction={vm.format_number(diag.get('clip_fraction'), 3)}"
                + f", explained_variance={vm.format_number(diag.get('explained_variance'), 3)}"
                + f", entropy={vm.format_number(diag.get('entropy'), 3)}"
            )
        self.checkpoints_label.configure(text="\n".join(checkpoint_lines))

        run_dir = self._last_run_dir
        if run_dir:
            self.submit_poll(
                "telemetry",
                lambda: self.adapter.telemetry_series(run_dir),
                lambda series, error: self._on_telemetry(run_dir, series, error),
            )
        else:
            for chart in (self.reward_chart, self.fps_chart, self.kl_chart):
                chart.set_points([])

    def _on_telemetry(
        self, run_dir: str, series: dict[str, Any] | None, error: BaseException | None
    ) -> None:
        if (
            run_dir != self._last_run_dir
            or error is not None
            or series is None
            or not series.get("available")
        ):
            return
        data = series.get("series", {})
        self.reward_chart.set_points(data.get("mean_episode_reward", []))
        self.fps_chart.set_points(data.get("steps_per_second", []))
        self.kl_chart.set_points(data.get("approx_kl", []))

    def _open_run_folder(self) -> None:
        if self._last_run_dir:
            _open_in_file_manager(Path(self._last_run_dir))
        else:
            messagebox.showinfo(
                "No run yet", "No training run has been found under the output root."
            )

    def _open_in_runs(self) -> None:
        self.app.show_page("Runs / Checkpoints")
        page = self.app.pages["Runs / Checkpoints"]
        if self._last_run_dir:
            page.select_run(self._last_run_dir)


class AgentsPage(Page):
    """Launch configuration + agent lifecycle in one operational page.

    The launch form is the single source of the training configuration
    (the Benchmarks page applies its recommendation here); every launched
    agent - training, benchmark or evaluation - is listed with its
    lifecycle state, its Environment -> Worker topology and its live
    backend metrics, with the full action set (Pause/Resume where the
    backend supports it, Stop, Restart, Force stop) and a bounded log.
    """

    title = "Agents"
    subtitle = "Launch, operate and restart headless agents - real backend states only."

    COLUMNS = (
        ("name", "Agent", 150),
        ("kind", "Type", 80),
        ("lifecycle", "Lifecycle", 100),
        ("pid", "PID", 70),
        ("environment_count", "Envs", 55),
        ("env_workers", "Workers", 65),
        ("device", "Device", 60),
        ("timesteps", "Steps", 90),
        ("progress", "Progress", 80),
        ("steps_per_second", "Steps/s", 80),
        ("mean_episode_reward", "Reward", 80),
        ("started", "Started", 130),
        ("error", "Error", 200),
    )

    def build(self) -> None:
        # ---- Launch configuration (the former Training page) ----------
        form_frame = ttk.LabelFrame(self, text="Launch configuration", padding=10)
        form_frame.pack(fill="x")
        self.field_vars: dict[str, tk.StringVar] = {}
        # Real measured defaults: the hardware wizard's device choice and -
        # when the automatic benchmark has been applied - its winning
        # topology. Both are single small JSON reads of persisted state.
        profile = None
        recommendation = None
        try:
            profile = self.adapter.hardware_profile()
            recommendation = self.adapter.recommended_configuration()
        except OSError:
            pass
        defaults = vm.training_values_from_profile(profile)
        if recommendation and recommendation.get("applied_utc"):
            for field, key in (
                ("environment_count", "environment_count"),
                ("env_workers", "env_workers"),
                ("device", "device"),
                ("inference_device", "inference_device"),
            ):
                value = recommendation.get(key)
                if value is not None and str(value):
                    defaults[field] = str(value)
        groups = vm.training_field_groups()
        basic_frame = ttk.Frame(form_frame)
        basic_frame.pack(fill="x")
        self._build_fields(basic_frame, groups["basic"], defaults)
        self._advanced_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form_frame,
            text="Show advanced options",
            variable=self._advanced_visible,
            command=self._toggle_advanced,
        ).pack(anchor="w", pady=(8, 0))
        self.advanced_frame = ttk.Frame(form_frame)
        self._build_fields(self.advanced_frame, groups["advanced"], defaults)

        launch_bar = ttk.Frame(self)
        launch_bar.pack(fill="x", pady=(8, 0))
        self.launch_status_label = ttk.Label(launch_bar, text="", justify="left", wraplength=760)
        self.launch_status_label.pack(side="left", fill="x", expand=True)
        self.launch_button = ttk.Button(
            launch_bar, text="Launch agent", command=self._launch, style="Primary.TButton"
        )
        self.launch_button.pack(side="right")

        # ---- Agent table + actions ------------------------------------
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.tree = _scrollable_table(table_frame, self.COLUMNS)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        # Lifecycle at a glance: failures red, live work green, terminal
        # states muted - same palette the rest of the GUI uses.
        self.tree.tag_configure("lifecycle-failed", foreground=COLOR_ERROR)
        self.tree.tag_configure("lifecycle-running", foreground=COLOR_OK)
        self.tree.tag_configure("lifecycle-attention", foreground=COLOR_WARN)
        self.tree.tag_configure("lifecycle-done", foreground=COLOR_MUTED)

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(6, 0))
        self.pause_button = ttk.Button(actions, text="Pause", command=self._pause, state="disabled")
        self.pause_button.pack(side="left")
        self._pause_tooltip = ToolTip(
            self.pause_button, "Pause the training agent at its next safe boundary"
        )
        self.resume_button = ttk.Button(
            actions, text="Resume", command=self._resume, state="disabled"
        )
        self.resume_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.restart_button = ttk.Button(
            actions, text="Restart", command=self._restart, state="disabled"
        )
        self.restart_button.pack(side="left", padx=(8, 0))
        self.force_stop_button = ttk.Button(
            actions,
            text="Force stop",
            command=self._force_stop,
            state="disabled",
            style="Danger.TButton",
        )
        self.force_stop_button.pack(side="left", padx=(16, 0))
        self.remove_button = ttk.Button(
            actions, text="Remove", command=self._remove, state="disabled"
        )
        self.remove_button.pack(side="left", padx=(8, 0))
        self.stop_all_button = ttk.Button(
            actions, text="Stop all", command=self._stop_all, state="disabled"
        )
        self.stop_all_button.pack(side="right")
        self.clear_button = ttk.Button(actions, text="Clear exited", command=self._clear)
        self.clear_button.pack(side="right", padx=(8, 0))

        # ---- Topology + log -------------------------------------------
        bottom = ttk.Panedwindow(self, orient="horizontal")
        bottom.pack(fill="both", expand=True, pady=(8, 0))
        topology_frame = ttk.LabelFrame(
            bottom, text="Environment → Worker topology (selected agent)", padding=8
        )
        bottom.add(topology_frame, weight=1)
        self.topology_label = ttk.Label(topology_frame, text="no agent selected", justify="left")
        self.topology_label.pack(anchor="nw")
        log_frame = ttk.LabelFrame(bottom, text="Agent log", padding=8)
        bottom.add(log_frame, weight=2)
        self.log_panel = LogPanel(log_frame)
        self.log_panel.pack(fill="both", expand=True)

        self._row_to_agent: dict[str, str] = {}
        self._selected_agent_id: str | None = None
        self._last_views: list[dict[str, Any]] = []
        self._last_slot_values: dict[str, str] | None = None
        self._compatibility: dict[str, Any] | None = None
        # The process id alone is not sufficient: A -> B -> A can happen
        # while A's first disk read is still in flight. The generation keeps
        # that old A result from filling a freshly reset A log view.
        self._log_selection_generation = 0
        self._log_in_flight: tuple[str, int] | None = None
        self._launching = False

    def _build_fields(
        self, parent: tk.Misc, specs: list[vm.TrainingFieldSpec], defaults: dict[str, str]
    ) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=2)
        for row_index, spec in enumerate(specs):
            ttk.Label(parent, text=spec.label, width=26).grid(
                row=row_index, column=0, sticky="w", pady=2
            )
            var = tk.StringVar(value=defaults.get(spec.name, ""))
            self.field_vars[spec.name] = var
            if spec.kind == "choice" and spec.choices:
                widget: tk.Widget = ttk.Combobox(
                    parent, textvariable=var, values=spec.choices, state="readonly", width=28
                )
            elif spec.kind == "bool":
                widget = ttk.Checkbutton(parent, variable=var, onvalue="true", offvalue="false")
            elif spec.name in ("bc_checkpoint", "godot_executable"):
                widget = ttk.Frame(parent)
                ttk.Entry(widget, textvariable=var, width=30).pack(side="left")
                ttk.Button(
                    widget,
                    text="Browse",
                    width=8,
                    command=lambda v=var: self._browse_file(v),  # type: ignore[misc]
                ).pack(side="left", padx=(4, 0))
            else:
                widget = ttk.Entry(parent, textvariable=var, width=30)
            widget.grid(row=row_index, column=1, sticky="ew", pady=4, padx=(8, 16))
            if spec.help:
                ttk.Label(parent, text=spec.help, foreground=COLOR_MUTED, wraplength=360).grid(
                    row=row_index, column=2, sticky="ew"
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

    # -- launch slot -----------------------------------------------------

    def current_values(self) -> dict[str, str]:
        return {name: var.get() for name, var in self.field_vars.items()}

    def apply_launch_values(self, values: dict[str, str]) -> list[str]:
        """Apply external launch values (e.g. a benchmark recommendation).

        Returns the list of field names that were applied; unknown field
        names are ignored so a saved recommendation from an older version
        cannot break the form.
        """
        applied = []
        for name, value in values.items():
            if name in self.field_vars and value is not None:
                self.field_vars[name].set(str(value))
                applied.append(name)
        return applied

    def refresh(self) -> None:
        self.submit_poll("agents", self.adapter.agents.views, self._on_agents)
        values = self.current_values()
        if values != self._last_slot_values or self._compatibility is None:
            self._last_slot_values = dict(values)
            self._update_launch_slot(values)
        process_id = self._selected_agent_process_id()
        if process_id:
            self._request_log(process_id)

    def _update_launch_slot(self, values: dict[str, str]) -> None:
        slot = vm.launch_slot_view(values, self._compatibility)
        self._launch_slot = slot
        if slot["state"] == "AVAILABLE":
            summary = slot["summary"]
            topology = "+".join(
                str(row["environments"])
                for row in vm.topology_rows(summary["environment_count"], summary["env_workers"])
            )
            text = (
                f"AVAILABLE — {summary['environment_count']} environments / "
                f"{summary['env_workers']} workers ({topology}) on {summary['device']}, "
                f"{vm.format_number(summary['total_training_steps'])} steps"
            )
            for warning in slot["warnings"]:
                text += f"\nwarning: {warning}"
            self.launch_status_label.configure(
                text=text,
                foreground=COLOR_WARN if slot["warnings"] else COLOR_OK,
            )
            self.launch_button.configure(state="disabled" if self._launching else "normal")
        else:
            self.launch_status_label.configure(
                text="INVALID — " + "; ".join(slot["errors"]), foreground=COLOR_ERROR
            )
            self.launch_button.configure(state="disabled")
        # Runtime compatibility (worker topology vs. this host, CUDA
        # availability, a resolvable Godot executable) is a separate,
        # slower check: it probes the runtime. The form's explicit Godot
        # executable is forwarded so the verdict matches what a launch
        # would actually run.
        if slot["state"] == "AVAILABLE":
            summary = slot["summary"]
            godot_override = (values.get("godot_executable") or "").strip() or None
            self.submit_poll(
                "launch-compatibility",
                lambda: self.adapter.validate_runtime_configuration(
                    summary["environment_count"],
                    summary["env_workers"],
                    summary["device"],
                    godot_executable=godot_override,
                ),
                self._on_compatibility,
            )

    def _on_compatibility(
        self, validation: dict[str, Any] | None, error: BaseException | None
    ) -> None:
        if error is not None or validation is None:
            return
        previous = self._compatibility
        self._compatibility = validation
        if (
            previous is None
            or previous.get("errors") != validation.get("errors")
            or previous.get("warnings") != validation.get("warnings")
        ):
            self._update_launch_slot(self.current_values())

    def _launch(self) -> None:
        slot = getattr(self, "_launch_slot", None)
        if not slot or slot["state"] != "AVAILABLE":
            return
        try:
            config = vm.parse_training_form(self.current_values())
        except ValueError as exc:
            messagebox.showerror("Invalid launch configuration", str(exc))
            return
        self._launching = True
        self.launch_button.configure(state="disabled")
        self.app.background.submit(
            lambda: self.adapter.agents.launch_training(config),
            self._on_launched,
        )

    def _on_launched(self, view: dict[str, Any] | None, error: BaseException | None) -> None:
        self._launching = False
        slot = getattr(self, "_launch_slot", None)
        self.launch_button.configure(
            state="normal" if slot and slot["state"] == "AVAILABLE" else "disabled"
        )
        if error is not None or view is None:
            messagebox.showerror("Agent could not start", str(error))
            return
        if view.get("lifecycle") == "FAILED":
            messagebox.showerror("Agent failed to start", str(view.get("error") or "unknown error"))
            return
        self.app.set_status(f"Agent launched ({view.get('name', 'agent')})")

    # -- agent table -----------------------------------------------------

    def _on_agents(self, views: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or views is None:
            self.report_error("Agents refresh failed", error or RuntimeError("unknown"))
            return
        self._last_views = views
        rows = vm.agent_table_rows(views)
        selected = self._selected_agent_id
        selected_still_present = False
        self.tree.delete(*self.tree.get_children())
        self._row_to_agent.clear()
        lifecycle_tags = {
            "FAILED": "lifecycle-failed",
            "RUNNING": "lifecycle-running",
            "LAUNCHING": "lifecycle-attention",
            "PAUSED": "lifecycle-attention",
            "STOPPING": "lifecycle-attention",
            "RESTARTING": "lifecycle-attention",
            "FINISHED": "lifecycle-done",
            "STOPPED": "lifecycle-done",
        }
        for row in rows:
            progress = vm.agent_progress_percent(row)
            tag = lifecycle_tags.get(str(row["lifecycle"] or ""))
            item_id = self.tree.insert(
                "",
                "end",
                tags=(tag,) if tag else (),
                values=(
                    row["name"],
                    row["kind"] or "n/a",
                    row["lifecycle"] or "n/a",
                    row["pid"] if row["pid"] is not None else "n/a",
                    vm.format_number(row["environment_count"]),
                    vm.format_number(row["env_workers"]),
                    row["device"] or "n/a",
                    vm.format_number(row["timesteps"]),
                    vm.format_fraction_as_percent(progress / 100.0)
                    if progress is not None
                    else "n/a",
                    vm.format_number(row["steps_per_second"], 1),
                    vm.format_number(row["mean_episode_reward"], 3),
                    vm.format_timestamp(row["started_at"]),
                    row["error"] or "",
                ),
            )
            self._row_to_agent[item_id] = row["agent_id"]
            if row["agent_id"] == selected:
                selected_still_present = True
                self.tree.selection_set(item_id)
        if selected is not None and not selected_still_present:
            self._clear_selection()
        self.stop_all_button.configure(
            state="normal"
            if any(
                str(view.get("lifecycle")) in ("LAUNCHING", "RUNNING", "PAUSED") for view in views
            )
            else "disabled"
        )
        self._update_topology(rows)
        if self._selected_agent_id is not None:
            self._refresh_action_buttons()

    def _update_topology(self, rows: list[dict[str, Any]]) -> None:
        agent_id = self._selected_agent_id
        row = next((item for item in rows if item["agent_id"] == agent_id), None)
        if row is None:
            self.topology_label.configure(text="no agent selected")
            return
        environment_count = row["environment_count"]
        env_workers = row["env_workers"]
        shards = vm.topology_rows(environment_count, env_workers)
        if not shards:
            self.topology_label.configure(
                text=f"{row['name']}: no topology published by this agent kind"
            )
            return
        lines = [
            f"{row['name']} — {environment_count} environments across {env_workers} worker(s):"
        ]
        for shard in shards:
            lines.append(
                f"  worker {shard['worker']}: environments "
                f"{shard['first_environment']}-{shard['last_environment']} "
                f"({shard['environments']} envs)"
            )
        self.topology_label.configure(text="\n".join(lines))

    def _on_select(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            self._clear_selection()
            return
        agent_id = self._row_to_agent.get(selection[0])
        if agent_id is None:
            self._clear_selection()
            return
        if agent_id != self._selected_agent_id:
            self._selected_agent_id = agent_id
            self._log_selection_generation += 1
            self.log_panel.reset_cursor()
        self._refresh_action_buttons()

    def _refresh_action_buttons(self) -> None:
        agent_id = self._selected_agent_id
        if agent_id is None:
            for button in (
                self.pause_button,
                self.resume_button,
                self.stop_button,
                self.restart_button,
                self.force_stop_button,
                self.remove_button,
            ):
                button.configure(state="disabled")
            return
        views = {str(view.get("agent_id")): view for view in self._last_views}
        view = views.get(agent_id)
        if view is None:
            return
        availability = vm.agent_action_availability(
            str(view.get("lifecycle", "")), str(view.get("kind", ""))
        )
        for name, button in (
            ("pause", self.pause_button),
            ("resume", self.resume_button),
            ("stop", self.stop_button),
            ("restart", self.restart_button),
            ("force_stop", self.force_stop_button),
            ("remove", self.remove_button),
        ):
            button.configure(state="normal" if availability.get(name) else "disabled")
        # The tooltip states why pause is impossible for backends without
        # the cooperative protocol instead of leaving a silent dead button.
        reason = availability.get("pause_unsupported_reason")
        self._pause_tooltip.text = (
            str(reason) if reason else "Pause the training agent at its next safe boundary"
        )

    def _clear_selection(self) -> None:
        self._selected_agent_id = None
        self._log_selection_generation += 1
        self.log_panel.reset_cursor()
        self._refresh_action_buttons()
        self.topology_label.configure(text="no agent selected")

    def _selected_agent_process_id(self) -> str | None:
        agent_id = self._selected_agent_id
        if agent_id is None:
            return None
        for view in self._last_views:
            if str(view.get("agent_id")) == agent_id:
                return view.get("process_id")
        return None

    def _request_log(self, process_id: str) -> None:
        """Fetch one selected agent's log without duplicate or stale updates."""
        generation = self._log_selection_generation
        token = (process_id, generation)
        if token == self._log_in_flight:
            return
        self._log_in_flight = token
        stdout_after = self.log_panel.stdout_after
        stderr_after = self.log_panel.stderr_after
        self.app.background.submit(
            lambda: self.adapter.process_log(process_id, stdout_after, stderr_after),
            lambda log, error: self._on_log(process_id, generation, log, error),
        )

    def _on_log(
        self,
        process_id: str,
        generation: int,
        log: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        token = (process_id, generation)
        if token == self._log_in_flight:
            self._log_in_flight = None
        if (
            process_id != self._selected_agent_process_id()
            or generation != self._log_selection_generation
            or error is not None
            or log is None
        ):
            return
        self.log_panel.apply_log(log)

    # -- lifecycle actions -------------------------------------------------

    def _agent_action(self, label: str, action: Callable[[str], dict[str, Any]]) -> None:
        agent_id = self._selected_agent_id
        if not agent_id:
            return

        def _run() -> dict[str, Any]:
            return action(agent_id)

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None:
                self.app.set_status(f"{label} failed: {error}", error=True)
            elif result and not result.get("ok"):
                self.app.set_status(f"{label}: {result.get('error')}", error=True)
            else:
                self.app.set_status(f"{label} requested")

        self.app.background.submit(_run, _done)

    def _pause(self) -> None:
        self._agent_action("Pause", self.adapter.agents.pause)

    def _resume(self) -> None:
        self._agent_action("Resume", self.adapter.agents.resume)

    def _stop(self) -> None:
        self._agent_action("Stop", self.adapter.agents.stop)

    def _restart(self) -> None:
        self._agent_action("Restart", self.adapter.agents.restart)

    def _force_stop(self) -> None:
        if not messagebox.askyesno(
            "Force stop",
            "Skip cooperative shutdown and kill this process now? "
            "The final checkpoint will NOT be saved.",
        ):
            return
        self._agent_action("Force stop", self.adapter.agents.force_stop)

    def _remove(self) -> None:
        self._agent_action("Remove", self.adapter.agents.remove)

    def _stop_all(self) -> None:
        if not messagebox.askyesno(
            "Stop all agents", "Request a safe stop for every running agent?"
        ):
            return

        def _run() -> dict[str, Any]:
            return {"stopped": self.adapter.agents.stop_all()}

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None:
                self.app.set_status(f"Stop all failed: {error}", error=True)
                return
            stopped = result or {}
            failures = [item for item in stopped.get("stopped", []) if not item.get("ok")]
            if failures:
                self.app.set_status(
                    f"Stop all: {len(failures)} agent(s) could not be stopped", error=True
                )
            else:
                self.app.set_status(
                    f"Stop all requested for {len(stopped.get('stopped', []))} agent(s)"
                )

        self.app.background.submit(_run, _done)

    def _clear(self) -> None:
        def _run() -> dict[str, Any]:
            return {"removed": self.adapter.agents.clear_finished()}

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None:
                self.app.set_status(f"Clear failed: {error}", error=True)
            else:
                cleared = result or {}
                self.app.set_status(f"Cleared {cleared.get('removed', 0)} exited agent(s)")

        self.app.background.submit(_run, _done)


class BenchmarkPage(Page):
    """The one-button automatic benchmark: measure, pick best, apply it.

    There is deliberately nothing to configure here. Start runs the staged
    pipeline (runtime discovery, env/worker screening, device comparison,
    real PPO validation slices) with the project's host-scaled defaults,
    the recommendation is chosen by the pipeline's own criteria (validated
    throughput, stability over an unstable peak) and the winning
    configuration is persisted and applied to the launch configuration
    automatically. Every number shown was measured; failures stay visible.
    """

    title = "Benchmarks"
    subtitle = (
        "Fully automatic - one click measures this machine, picks the best stable "
        "configuration and applies it to every new agent launch. No parameters, no estimates."
    )

    RESULT_COLUMNS = (
        ("stage", "Stage", 90),
        ("status", "Status", 80),
        ("environments", "Envs", 55),
        ("workers", "Workers", 65),
        ("device", "Device", 60),
        ("steps", "Steps", 90),
        ("steps_per_second", "Steps/s", 90),
        ("p50_ms", "p50 ms", 70),
        ("p95_ms", "p95 ms", 70),
        ("jitter", "p95/p50", 70),
        ("startup_seconds", "Startup s", 80),
        ("error", "Error", 240),
    )

    def build(self) -> None:
        intro = ttk.LabelFrame(self, text="Automatic benchmark", padding=10)
        intro.pack(fill="x")
        ttk.Label(
            intro,
            text=(
                "Start measures the real runtime end to end: it screens a host-scaled "
                "grid of environment/worker topologies through the actual bridge, compares "
                "devices where more than one exists, validates the best candidates with "
                "short real PPO training slices, then picks the fastest stable "
                "configuration and applies it automatically. Results and the winning "
                "configuration are persisted and reused across restarts."
            ),
            wraplength=980,
            justify="left",
            foreground=COLOR_MUTED,
        ).pack(anchor="w")

        run_bar = ttk.Frame(intro)
        run_bar.pack(fill="x", pady=(10, 0))
        self.run_button = ttk.Button(
            run_bar, text="Start benchmark", command=self._start, style="Primary.TButton"
        )
        self.run_button.pack(side="left")
        ToolTip(self.run_button, "Run the complete automatic benchmark workflow")
        self.cancel_button = ttk.Button(
            run_bar, text="Cancel", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.progress_label = ttk.Label(
            run_bar, text="idle", foreground=COLOR_MUTED, wraplength=620, justify="left"
        )
        self.progress_label.pack(side="left", padx=(14, 0))
        self.phase_label = ttk.Label(intro, text="", foreground=COLOR_MUTED, justify="left")
        self.phase_label.pack(anchor="w", pady=(8, 0))

        best = ttk.LabelFrame(
            self, text="Best configuration (selected and applied automatically)", padding=10
        )
        best.pack(fill="x", pady=(10, 0))
        self.recommendation_label = ttk.Label(best, text="n/a", justify="left")
        self.recommendation_label.pack(anchor="nw")
        self.applied_label = ttk.Label(best, text="", justify="left", foreground=COLOR_MUTED)
        self.applied_label.pack(anchor="w", pady=(4, 0))

        results_frame = ttk.LabelFrame(
            self, text="Measurements (every configuration actually tested)", padding=8
        )
        results_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.tree = _scrollable_table(results_frame, self.RESULT_COLUMNS)

        self._cancel_event: threading.Event | None = None
        self._latest_progress: dict[str, Any] | None = None
        self._progress_lock = threading.Lock()
        self._latest_report: dict[str, Any] | None = None
        self._applied: bool | None = None
        self._failure_text: str | None = None
        self._running = False

    # -- workflow ----------------------------------------------------------

    def refresh(self) -> None:
        self.submit_poll(
            "pipeline-history", self.adapter.benchmark_pipeline_history, self._on_history
        )
        self.submit_poll(
            "recommendation", self.adapter.recommended_configuration, self._on_recommendation
        )
        self._refresh_workflow_labels()
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.run_button.configure(state="disabled" if self._running else "normal")
        self.cancel_button.configure(state="normal" if self._running else "disabled")

    def _refresh_workflow_labels(self) -> None:
        with self._progress_lock:
            progress = dict(self._latest_progress) if self._latest_progress else None
        view = vm.benchmark_workflow_view(
            running=self._running,
            event=progress,
            report=self._latest_report,
            applied=self._applied,
        )
        self.phase_label.configure(text=view["phase_line"])
        self.progress_label.configure(
            text=self._failure_text or view["detail"],
            foreground=COLOR_ERROR if self._failure_text else COLOR_MUTED,
        )

    def _start(self) -> None:
        """Run the complete workflow - no form, no parameters to validate."""
        if self._running:
            return
        self._running = True
        self._latest_report = None
        self._applied = None
        self._failure_text = None
        self._cancel_event = threading.Event()
        with self._progress_lock:
            self._latest_progress = None
        self._update_buttons()
        self.progress_label.configure(text="starting...", foreground=COLOR_MUTED)
        self.app.set_status("Benchmark started - measuring this machine")

        def on_progress(event: dict[str, Any]) -> None:
            with self._progress_lock:
                self._latest_progress = dict(event)

        cancel_event = self._cancel_event

        def _run() -> dict[str, Any]:
            return self.adapter.run_benchmark_pipeline(
                cancel=cancel_event.is_set if cancel_event else None,
                on_progress=on_progress,
            )

        self.app.background.submit(_run, self._on_finished)

    def _cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.progress_label.configure(text="cancelling - the in-flight measurement finishes")

    def _on_finished(self, report: dict[str, Any] | None, error: BaseException | None) -> None:
        self._running = False
        self._update_buttons()
        if error is not None or report is None:
            self._applied = False
            self._failure_text = f"failed: {error}"
            self.app.set_status(f"Benchmark failed: {error}", error=True)
            self._refresh_workflow_labels()
            return
        self._latest_report = report
        self._render_report(report)
        recommendation = report.get("recommendation")
        if recommendation:
            self._render_recommendation(recommendation)
            self._apply_automatically()
        else:
            self._applied = False
            reason = report.get("recommendation_reason") or report.get("status") or "unknown"
            self._render_recommendation(None, reason=str(reason))
            self.app.set_status(
                f"Benchmark finished without a usable configuration: {reason}",
                error=report.get("status") != "completed",
            )
        self._refresh_workflow_labels()

    def _apply_automatically(self) -> None:
        """Final workflow step: activate the persisted winning configuration."""

        def _run() -> dict[str, Any] | None:
            return self.adapter.apply_recommended_configuration()

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not result:
                self._applied = False
                self.applied_label.configure(
                    text=f"could not apply automatically: {error or 'no persisted recommendation'}",
                    foreground=COLOR_ERROR,
                )
                self.app.set_status(f"Benchmark: applying the result failed: {error}", error=True)
            else:
                self._applied = True
                self._push_to_launch_form(result)
                self.applied_label.configure(
                    text=(
                        "applied automatically - the launch configuration (Agents page) "
                        "now uses this topology"
                    ),
                    foreground=COLOR_OK,
                )
                self.app.set_status("Benchmark finished - best configuration applied")
            self._refresh_workflow_labels()

        self.app.background.submit(_run, _done)

    def _push_to_launch_form(self, recommendation: dict[str, Any]) -> None:
        """Mirror the applied configuration into an already-built Agents page.

        A not-yet-built Agents page needs nothing here: its build() reads
        the persisted applied recommendation itself.
        """
        agents_page = self.app.pages.get("Agents")
        if agents_page is None or not getattr(agents_page, "_built", False):
            return
        agents_page.apply_launch_values(
            {
                "environment_count": str(recommendation.get("environment_count", "")),
                "env_workers": str(recommendation.get("env_workers", "")),
                "device": recommendation.get("device") or "auto",
                "inference_device": recommendation.get("inference_device") or "auto",
            }
        )

    # -- results/history ---------------------------------------------------

    def _on_history(
        self, history: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or history is None:
            self.report_error("Benchmark history refresh failed", error or RuntimeError("unknown"))
            return
        if self._latest_report is None and not self._running and history:
            self._render_report(history[0].get("report"))

    def _render_report(self, report: dict[str, Any] | None) -> None:
        rows = vm.benchmark_pipeline_rows(report)
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert(
                "",
                "end",
                values=(
                    row["stage"] or "n/a",
                    row["status"] or "ok",
                    vm.format_number(row["environments"]),
                    vm.format_number(row["workers"]),
                    row["device"] or "-",
                    vm.format_number(row["steps"]),
                    vm.format_number(row["steps_per_second"], 1),
                    vm.format_number(row["p50_ms"], 2),
                    vm.format_number(row["p95_ms"], 2),
                    vm.format_number(row["jitter"], 2),
                    vm.format_number(row["startup_seconds"], 2),
                    row["error"] or "",
                ),
                tags=("failed",) if (row["status"] or "ok") not in ("ok", "measured") else (),
            )
        self.tree.tag_configure("failed", foreground=COLOR_ERROR)

    # -- best configuration --------------------------------------------------

    def _on_recommendation(
        self, recommendation: dict[str, Any] | None, error: BaseException | None
    ) -> None:
        if error is not None:
            return
        if not self._running and not self._latest_report:
            self._render_recommendation(recommendation)

    def _render_recommendation(
        self, recommendation: dict[str, Any] | None, reason: str | None = None
    ) -> None:
        view = vm.benchmark_recommendation_view(recommendation)
        if not view.get("available"):
            self.recommendation_label.configure(
                text=(reason or str(view.get("reason")))
                + "\nPress Start - the benchmark measures this machine and applies the result."
            )
            self.applied_label.configure(text="")
            return
        lines = [
            view["summary"],
            f"basis: {view['basis']}",
            f"measured: {view['created_utc'] or 'n/a'}",
        ]
        lines.extend(f"- {line}" for line in view["rationale"])
        lines.extend(f"warning: {warning}" for warning in view["warnings"])
        self.recommendation_label.configure(text="\n".join(lines))
        if view.get("applied_utc") and self._applied is None:
            self.applied_label.configure(
                text=f"active since {view['applied_utc']} (persisted)", foreground=COLOR_OK
            )


class EvaluationPage(Page):
    title = "Evaluations"
    subtitle = "Evaluates a frozen checkpoint with the existing evaluator; action-head diagnostics included."

    CHECKPOINT_COLUMNS = (
        ("run_id", "Run", 140),
        ("kind", "Kind", 80),
        ("path", "Path", 320),
        ("modified_utc", "Modified", 160),
    )
    EVAL_COLUMNS = (
        ("path", "Path", 260),
        ("timesteps", "Timesteps", 90),
        ("episodes", "Episodes", 80),
        ("win_rate", "Win rate", 80),
        ("loss_rate", "Loss rate", 80),
        ("mean_episode_reward", "Reward", 80),
    )

    def build(self) -> None:
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left, weight=1)
        ttk.Label(left, text="Checkpoints", style="Section.TLabel").pack(anchor="w")
        self.checkpoint_tree = _scrollable_table(left, self.CHECKPOINT_COLUMNS)

        form = ttk.LabelFrame(left, text="Run evaluation on the selected checkpoint", padding=10)
        form.pack(fill="x", pady=(8, 0))
        self.episodes_var = tk.StringVar(value="20")
        self.env_count_var = tk.StringVar(value="1")
        self.device_var = tk.StringVar(value="auto")
        for row, (label, var, kind) in enumerate(
            (
                ("Episodes", self.episodes_var, "entry"),
                ("Environment count", self.env_count_var, "entry"),
                ("Device", self.device_var, "choice"),
            )
        ):
            ttk.Label(form, text=label, width=16).grid(row=row, column=0, sticky="w", pady=2)
            if kind == "choice":
                ttk.Combobox(
                    form,
                    textvariable=var,
                    values=("auto", "cpu", "cuda"),
                    state="readonly",
                    width=16,
                ).grid(row=row, column=1, sticky="w")
            else:
                ttk.Entry(form, textvariable=var, width=18).grid(row=row, column=1, sticky="w")
        self.run_button = ttk.Button(
            form, text="Start evaluation", command=self._start, style="Primary.TButton"
        )
        self.run_button.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.selected_checkpoint_label = ttk.Label(
            left, text="selected checkpoint: none", foreground=COLOR_MUTED, wraplength=380
        )
        self.selected_checkpoint_label.pack(anchor="w")
        self.state_label = ttk.Label(left, text="", foreground=COLOR_MUTED)
        self.state_label.pack(anchor="w")

        right = ttk.Frame(paned, padding=(8, 0, 0, 0))
        paned.add(right, weight=1)
        ttk.Label(
            right,
            text="Evaluation results (select multiple rows to compare)",
            style="Section.TLabel",
        ).pack(anchor="w")
        self.eval_tree = _scrollable_table(right, self.EVAL_COLUMNS, expand=False)
        self.eval_tree.configure(selectmode="extended")
        self.eval_tree.bind("<<TreeviewSelect>>", self._on_eval_select)

        detail_frame = ttk.LabelFrame(right, text="Structured result / comparison", padding=8)
        detail_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.detail_text = tk.Text(
            detail_frame,
            wrap="word",
            state="disabled",
            height=18,
            font=("Consolas", 9),
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            selectbackground="#164e63",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
        )
        self.detail_text.pack(fill="both", expand=True)

        self._checkpoint_paths: dict[str, str] = {}
        self._eval_paths: dict[str, str] = {}
        self._selected_evaluation_paths: tuple[str, ...] = ()
        self._evaluation_detail_generation = 0
        self.selected_checkpoint: str | None = None
        self.checkpoint_tree.bind("<<TreeviewSelect>>", self._on_checkpoint_select)
        self.process_id: str | None = None

    def select_checkpoint(self, path: str) -> None:
        self.selected_checkpoint = path
        self.selected_checkpoint_label.configure(text=f"selected checkpoint: {path}")

    def refresh(self) -> None:
        self.submit_poll("checkpoint-list", self.adapter.discover_checkpoints, self._on_checkpoints)
        self.submit_poll("evaluation-list", self.adapter.discover_evaluations, self._on_evaluations)
        process_id = self.process_id
        if process_id:
            self.submit_poll(
                "evaluation-status",
                lambda: self.adapter.process_status(process_id),
                lambda status, error: self._on_process_status(process_id, status, error),
            )

    def _on_checkpoints(
        self, entries: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or entries is None:
            self.report_error("Checkpoint list refresh failed", error or RuntimeError("unknown"))
            return
        self.checkpoint_tree.delete(*self.checkpoint_tree.get_children())
        self._checkpoint_paths.clear()
        for entry in entries:
            item_id = self.checkpoint_tree.insert(
                "",
                "end",
                values=(
                    entry["run_id"],
                    entry["kind"],
                    entry["path"],
                    entry.get("modified_utc") or "n/a",
                ),
            )
            self._checkpoint_paths[item_id] = entry["path"]

    def _on_checkpoint_select(self, _event: object) -> None:
        selection = self.checkpoint_tree.selection()
        if selection:
            self.selected_checkpoint = self._checkpoint_paths.get(selection[0])

    def _start(self) -> None:
        if not self.selected_checkpoint:
            messagebox.showwarning(
                "Checkpoint required", "Select a checkpoint from the list on the left."
            )
            return
        try:
            episodes = int(self.episodes_var.get())
            environment_count = int(self.env_count_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid evaluation configuration",
                "Episodes and environment count must be integers.",
            )
            return
        checkpoint = self.selected_checkpoint
        device = self.device_var.get()

        def _launch() -> dict[str, Any]:
            return self.adapter.start_evaluation(
                checkpoint, episodes=episodes, environment_count=environment_count, device=device
            )

        self.run_button.configure(state="disabled")
        self.app.background.submit(_launch, self._on_started)

    def _on_started(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        self.run_button.configure(state="normal")
        if error is not None or result is None:
            messagebox.showerror("Evaluation could not start", str(error))
            return
        self.process_id = result["process_id"]
        self.app.set_status(f"Evaluation started ({self.process_id[:8]})")

    def _on_process_status(
        self,
        process_id: str,
        status: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        if process_id != self.process_id or error is not None or status is None:
            return
        self.state_label.configure(
            text=f"evaluation process: {status.get('state')}",
            foreground=STATE_COLORS.get(str(status.get("state", "")), COLOR_MUTED),
        )
        if status.get("state") in ("finished", "failed"):
            self.process_id = None

    def _on_evaluations(
        self, entries: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or entries is None:
            self.report_error("Evaluation list refresh failed", error or RuntimeError("unknown"))
            return
        selected = set(self._selected_evaluation_paths)
        restored_paths: list[str] = []
        self.eval_tree.delete(*self.eval_tree.get_children())
        self._eval_paths.clear()
        for entry in entries:
            path = str(entry["path"])
            item_id = self.eval_tree.insert(
                "",
                "end",
                values=(
                    path,
                    vm.format_number(entry.get("timesteps")),
                    vm.format_number(entry.get("episodes")),
                    vm.format_fraction_as_percent(entry.get("win_rate")),
                    vm.format_fraction_as_percent(entry.get("loss_rate")),
                    vm.format_number(entry.get("mean_episode_reward"), 3),
                ),
            )
            self._eval_paths[item_id] = path
            if path in selected:
                self.eval_tree.selection_add(item_id)
                restored_paths.append(path)
        restored = tuple(restored_paths)
        if restored != self._selected_evaluation_paths:
            self._selected_evaluation_paths = restored
            self._evaluation_detail_generation += 1
            if not restored:
                self._clear_evaluation_detail()

    def _on_eval_select(self, _event: object) -> None:
        selection = self.eval_tree.selection()
        paths = tuple(
            self._eval_paths[item_id] for item_id in selection if item_id in self._eval_paths
        )
        if paths == self._selected_evaluation_paths:
            return
        self._selected_evaluation_paths = paths
        self._evaluation_detail_generation += 1
        generation = self._evaluation_detail_generation
        if not paths:
            self._clear_evaluation_detail()
            return
        self.app.background.submit(
            lambda: [self.adapter.evaluation_detail(path) for path in paths],
            lambda details, error: self._on_details(paths, generation, details, error),
        )

    def _clear_evaluation_detail(self) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.configure(state="disabled")

    def _on_details(
        self,
        paths: tuple[str, ...],
        generation: int,
        details: list[dict[str, Any]] | None,
        error: BaseException | None,
    ) -> None:
        # A slow comparison from an earlier multi-selection must never
        # overwrite the report for the selection the operator currently sees.
        if (
            paths != self._selected_evaluation_paths
            or generation != self._evaluation_detail_generation
        ):
            return
        if error is not None or details is None:
            self.report_error("Evaluation detail failed", error or RuntimeError("unknown"))
            return
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        if len(details) == 1:
            self.detail_text.insert(
                "end", _render_evaluation_detail(vm.evaluation_view(details[0]))
            )
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

    COLUMNS = (
        ("run_id", "Run", 160),
        ("state", "State", 90),
        ("progress_percent", "Progress", 80),
        ("device", "Device", 70),
        ("environment_count", "Envs", 55),
        ("env_workers", "Workers", 65),
        ("checkpoints", "Checkpoints", 90),
        ("reward", "Reward", 80),
        ("win_rate", "Win rate", 80),
        ("modified_utc", "Modified", 160),
    )

    def build(self) -> None:
        paned = ttk.Panedwindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)
        top = ttk.Frame(paned)
        paned.add(top, weight=1)
        self.tree = _scrollable_table(top, self.COLUMNS)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        bottom = ttk.LabelFrame(paned, text="Run detail", padding=8)
        paned.add(bottom, weight=1)
        self.detail_text = tk.Text(
            bottom,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            selectbackground="#164e63",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
        )
        self.detail_text.pack(fill="both", expand=True)
        actions = ttk.Frame(bottom)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Open run folder", command=self._open_folder).pack(side="left")
        ttk.Button(actions, text="Evaluate latest checkpoint", command=self._evaluate).pack(
            side="left", padx=(8, 0)
        )

        self._row_to_dir: dict[str, str] = {}
        self._selected_run_dir: str | None = None
        self._run_detail_generation = 0
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
        self.submit_poll("run-list", self.adapter.list_runs, self._on_runs)

    def _on_runs(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or result is None:
            self.report_error("Run list refresh failed", error or RuntimeError("unknown"))
            return
        rows = vm.runs_table_rows(result)
        selected = self._selected_run_dir
        selected_still_present = False
        self.tree.delete(*self.tree.get_children())
        self._row_to_dir.clear()
        for row in rows:
            item_id = self.tree.insert(
                "",
                "end",
                values=(
                    row["run_id"],
                    row["state"] or "n/a",
                    vm.format_fraction_as_percent((row["progress_percent"] or 0) / 100.0)
                    if row["progress_percent"] is not None
                    else "n/a",
                    row["device"] or "n/a",
                    vm.format_number(row["environment_count"]),
                    vm.format_number(row["env_workers"]),
                    vm.format_number(row["checkpoints"]),
                    vm.format_number(row["reward"], 3),
                    vm.format_fraction_as_percent(row["win_rate"]),
                    row["modified_utc"] or "n/a",
                ),
            )
            self._row_to_dir[item_id] = row["run_dir"]
            if row["run_dir"] == selected:
                selected_still_present = True
                self.tree.selection_set(item_id)
        if selected is not None and not selected_still_present:
            self._clear_run_selection()
        if self._pending_run_selection is not None and self._select_existing_row(
            self._pending_run_selection
        ):
            self._pending_run_selection = None

    def _on_select(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            self._clear_run_selection()
            return
        run_dir = self._row_to_dir.get(selection[0])
        if run_dir is None:
            self._clear_run_selection()
            return
        if run_dir == self._selected_run_dir:
            return
        self._selected_run_dir = run_dir
        self._run_detail_generation += 1
        generation = self._run_detail_generation
        self.app.background.submit(
            lambda: self.adapter.inspect_run(run_dir),
            lambda report, error: self._on_detail(run_dir, generation, report, error),
        )

    def _clear_run_selection(self) -> None:
        self._selected_run_dir = None
        self._run_detail_generation += 1
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.configure(state="disabled")

    def _on_detail(
        self,
        run_dir: str,
        generation: int,
        report: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        if (
            run_dir != self._selected_run_dir
            or generation != self._run_detail_generation
            or error is not None
            or report is None
        ):
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

        chart_frame = ttk.LabelFrame(
            self,
            text="Host resource usage (sampled each refresh, bounded to the last 300 samples)",
            padding=10,
        )
        chart_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.cpu_chart = LineChart(chart_frame, "CPU percent (this process)")
        self.cpu_chart.pack(fill="both", expand=True, pady=(0, 4))
        self.rss_chart = LineChart(chart_frame, "process RSS (MB)")
        self.rss_chart.pack(fill="both", expand=True)

        from collections import deque

        self._cpu_series: deque[tuple[float, float]] = deque(maxlen=300)
        self._rss_series: deque[tuple[float, float]] = deque(maxlen=300)
        self._sample_index = 0.0

    def refresh(self) -> None:
        self.submit_poll("system-status", self.adapter.system_status, self._on_status)

    def _on_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            self.report_error("System status refresh failed", error or RuntimeError("unknown"))
            return
        godot_text = "not found"
        if status.get("godot_available"):
            version = status.get("godot_version")
            godot_text = f"available ({version})" if version else "available (version unknown)"
        self.stats.update_values(
            {
                "cpu": (
                    vm.format_fraction_as_percent((status.get("cpu_percent") or 0) / 100.0),
                    None,
                ),
                "process memory": (
                    vm.format_bytes((status.get("process_rss_mb") or 0) * 1024 * 1024)
                    if status.get("process_rss_mb") is not None
                    else "n/a",
                    None,
                ),
                "python": (status.get("python_version", "n/a"), None),
                "godot": (godot_text, COLOR_OK if status.get("godot_available") else COLOR_WARN),
                "torch": (
                    "available" if status.get("torch_available") else "not installed",
                    COLOR_OK if status.get("torch_available") else COLOR_MUTED,
                ),
                "cuda": (
                    "available" if status.get("cuda_available") else "not available",
                    COLOR_OK if status.get("cuda_available") else COLOR_MUTED,
                ),
            }
        )
        deps = status.get("dependencies", {})
        self.deps_label.configure(
            text="   ".join(
                f"{name}: {'yes' if available else 'no'}" for name, available in deps.items()
            )
        )
        self._sample_index += 1.0
        if status.get("cpu_percent") is not None:
            self._cpu_series.append((self._sample_index, float(status["cpu_percent"])))
        if status.get("process_rss_mb") is not None:
            self._rss_series.append((self._sample_index, float(status["process_rss_mb"])))
        self.cpu_chart.set_points(list(self._cpu_series))
        self.rss_chart.set_points(list(self._rss_series))


class SettingsPage(Page):
    title = "Settings"
    subtitle = "Project/output roots and the machine-local Godot executable every launch resolves."

    def build(self) -> None:
        roots = ttk.LabelFrame(self, text="Directories", padding=10)
        roots.pack(fill="x")
        self.project_root_label = ttk.Label(roots, text="")
        self.project_root_label.pack(anchor="w", pady=2)
        self.output_root_label = ttk.Label(roots, text="")
        self.output_root_label.pack(anchor="w", pady=2)
        ttk.Button(roots, text="Change output root...", command=self._change_output_root).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            roots,
            text="Changing the output root points this session's Runs/Checkpoints/Evaluations/"
            "Benchmarks pages at a different directory; it does not move existing runs.",
            foreground=COLOR_MUTED,
            wraplength=700,
        ).pack(anchor="w", pady=(4, 0))

        # ---- Godot executable (the #1 reason launches fail) -------------
        godot = ttk.LabelFrame(self, text="Godot executable", padding=10)
        godot.pack(fill="x", pady=(12, 0))
        ttk.Label(
            godot,
            text="Training, benchmarks and evaluations all resolve the engine through this "
            "remembered setting when none is given explicitly (resolution order: explicit "
            "path, GODOT_PATH/GODOT_EXECUTABLE, this setting, PATH). Verify & save probes "
            "the executable before persisting it, so a saved value was actually seen "
            "working.",
            foreground=COLOR_MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w")
        entry_row = ttk.Frame(godot)
        entry_row.pack(fill="x", pady=(8, 0))
        self.godot_var = tk.StringVar(value=self.adapter.godot_executable_setting() or "")
        ttk.Entry(entry_row, textvariable=self.godot_var, width=60).pack(side="left")
        ttk.Button(entry_row, text="Browse", width=8, command=self._browse_godot).pack(
            side="left", padx=(4, 0)
        )
        self.godot_save_button = ttk.Button(
            entry_row, text="Verify && save", command=self._save_godot, style="Primary.TButton"
        )
        self.godot_save_button.pack(side="left", padx=(8, 0))
        self.godot_status_label = ttk.Label(godot, text="", justify="left", wraplength=700)
        self.godot_status_label.pack(anchor="w", pady=(6, 0))

    def refresh(self) -> None:
        self.project_root_label.configure(text=f"project root: {self.adapter.project_root}")
        self.output_root_label.configure(text=f"output root: {self.adapter.output_root}")
        self.submit_poll("godot-status", self.adapter.system_status, self._on_system_status)

    def _on_system_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            return
        if status.get("godot_available"):
            resolved = status.get("godot_resolved_executable") or "?"
            version = status.get("godot_version") or "version unknown"
            self.godot_status_label.configure(
                text=f"currently resolves to: {resolved} ({version})", foreground=COLOR_OK
            )
        else:
            self.godot_status_label.configure(
                text="no usable Godot executable found - launches will fail until one is "
                "configured here, put on PATH or set via GODOT_PATH",
                foreground=COLOR_ERROR,
            )

    def _browse_godot(self) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.godot_var.set(path)

    def _save_godot(self) -> None:
        candidate = self.godot_var.get().strip()
        if not candidate:
            self.godot_status_label.configure(
                text="enter or browse to a Godot executable first", foreground=COLOR_WARN
            )
            return
        self.godot_save_button.configure(state="disabled")
        self.godot_status_label.configure(text="verifying...", foreground=COLOR_MUTED)

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            self.godot_save_button.configure(state="normal")
            if error is not None or not result:
                self.godot_status_label.configure(
                    text=f"verification failed: {error}", foreground=COLOR_ERROR
                )
                return
            if result.get("ok"):
                saved = (
                    "remembered in .sandboxai/settings.json"
                    if result.get("settings_path")
                    else "verified (settings file not writable here)"
                )
                version = result.get("version") or "version unknown"
                self.godot_status_label.configure(
                    text=f"OK: {result.get('resolved')} ({version}) - {saved}",
                    foreground=COLOR_OK,
                )
                self.app.set_status("Godot executable verified and remembered")
            else:
                self.godot_status_label.configure(
                    text=str(result.get("error")), foreground=COLOR_ERROR
                )

        self.app.background.submit(
            lambda: self.adapter.configure_godot_executable(candidate), _done
        )

    def _change_output_root(self) -> None:
        directory = filedialog.askdirectory(initialdir=str(self.adapter.output_root))
        if directory:
            self.app.set_output_root(directory)
            self.refresh()


PAGE_CLASSES: tuple[type[Page], ...] = (
    DashboardPage,
    AgentsPage,
    BenchmarkPage,
    EvaluationPage,
    RunsPage,
    SystemPage,
    SettingsPage,
)

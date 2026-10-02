"""Desktop Control Center pages.

Each page translates UI intent into adapter calls; formatting and validation
remain in the independently tested viewmodel.
"""

from __future__ import annotations

import contextlib
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from . import control_center_viewmodel as vm
from .control_center_widgets import (
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_WARN,
    LineChart,
    LogPanel,
    PhaseStepper,
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
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text=self.title, style="PageTitle.TLabel").pack(anchor="w")
        if self.subtitle:
            ttk.Label(header, text=self.subtitle, style="PageSubtitle.TLabel").pack(
                anchor="w", pady=(2, 6)
            )
        tk.Frame(header, height=1, background=COLOR_BORDER, borderwidth=0).pack(
            fill="x", pady=(2, 0)
        )

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
        # ---- Clean 1-2-3 Workflow & Live Roblox TTK Testing Bridge ------
        workflow_bar = ttk.LabelFrame(
            self,
            text="Quick workflow  ·  1. Roblox TTK Testing   2. Hardware Benchmark   3. Launch Agent",
            padding=12,
        )
        workflow_bar.pack(fill="x", pady=(0, 12))
        wf_top = ttk.Frame(workflow_bar, style="Surface.TFrame")
        wf_top.pack(fill="x")
        self.roblox_bridge_label = ttk.Label(
            wf_top,
            text="Roblox Bridge: probing local Roblox Player & TTK Testing logs...",
            style="Leader.TLabel",
        )
        self.roblox_bridge_label.pack(side="left", fill="x", expand=True)

        wf_buttons = ttk.Frame(workflow_bar, style="Surface.TFrame")
        wf_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(
            wf_buttons,
            text="1. Launch Roblox TTK Testing",
            command=self._launch_roblox_ttk,
            style="Primary.TButton",
        ).pack(side="left")
        ttk.Button(
            wf_buttons,
            text="Focus Window",
            command=self._focus_roblox_window,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            wf_buttons,
            text="Capture Screenshot",
            command=self._capture_roblox_window,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            wf_buttons,
            text="TTK Calibration",
            command=lambda: self.app.show_page("Settings"),
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            wf_buttons,
            text="Ubuntu CPU Turbo",
            command=self._enable_ubuntu_cpu_turbo,
        ).pack(side="left", padx=(14, 0))
        ttk.Button(
            wf_buttons,
            text="2. Run Benchmark",
            command=lambda: self.app.show_page("Benchmarks"),
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            wf_buttons,
            text="3. Deploy Agent",
            command=lambda: self.app.show_page("Agents"),
        ).pack(side="left", padx=(6, 0))

        self.warning_banner = ttk.Label(
            self, text="", style="Warning.TLabel", wraplength=900, justify="left"
        )
        self.stats = StatRow(self, self.STAT_LABELS)
        self.stats.pack(fill="x")
        self.warning_banner.pack(fill="x", pady=(8, 0))

        checkpoints_frame = ttk.LabelFrame(
            self, text="Checkpoints, PPO diagnostics & convergence", padding=12
        )
        checkpoints_frame.pack(fill="x", pady=(12, 0))
        self.checkpoints_label = ttk.Label(checkpoints_frame, text="n/a", justify="left")
        self.checkpoints_label.pack(anchor="w")
        self.convergence_label = ttk.Label(
            checkpoints_frame, text="", justify="left", foreground=COLOR_MUTED
        )
        self.convergence_label.pack(anchor="w", pady=(4, 0))

        charts_frame = ttk.LabelFrame(self, text="Live telemetry (bounded history)", padding=12)
        charts_frame.pack(fill="both", expand=True, pady=(12, 0))
        charts_grid = ttk.Frame(charts_frame)
        charts_grid.pack(fill="both", expand=True)
        self.reward_chart = LineChart(charts_grid, "Mean episode reward vs. timesteps")
        self.fps_chart = LineChart(charts_grid, "Steps/second vs. timesteps")
        self.kl_chart = LineChart(charts_grid, "PPO approx. KL vs. timesteps")
        for index, chart in enumerate((self.reward_chart, self.fps_chart, self.kl_chart)):
            chart.grid(row=index // 2, column=index % 2, sticky="nsew", padx=5, pady=5)
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
        ttk.Button(actions, text="Resume in Agents", command=self._resume_in_agents).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="Evaluate best checkpoint", command=self._evaluate_latest_best).pack(
            side="left", padx=(8, 0)
        )

        self._last_run_dir: str | None = None

    def refresh(self) -> None:
        self.submit_poll("dashboard", self.adapter.dashboard_snapshot, self._on_snapshot)
        self.submit_poll("agents-summary", self.adapter.agents.views, self._on_agents_summary)
        if hasattr(self.adapter, "ttk_testing_status"):
            self.submit_poll("roblox-bridge", self.adapter.ttk_testing_status, self._on_roblox_status)

    def _on_roblox_status(
        self, status: dict[str, Any] | None, error: BaseException | None
    ) -> None:
        if error is not None or status is None:
            return
        tview = vm.ttk_testing_view(status)
        summary = (
            f"{tview['status_badge']}   ·   Window: {tview['window_text']}   ·   "
            f"Place: {tview['place_text']}   ·   "
            f"Calibration: {tview['calibration_progress_text']}"
        )
        self.roblox_bridge_label.configure(
            text=summary,
            foreground=COLOR_OK if tview["connected"] else (COLOR_WARN if tview["roblox_running"] else COLOR_TEXT),
        )

    def _launch_roblox_ttk(self) -> None:
        if not hasattr(self.adapter, "launch_roblox_ttk_testing"):
            return

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not result or not result.get("ok"):
                self.app.set_status(f"Roblox launch failed: {error or (result or {}).get('error')}", error=True)
            else:
                self.app.set_status(str(result.get("message") or "Launched Roblox TTK Testing"))
                self.refresh()

        self.app.background.submit(self.adapter.launch_roblox_ttk_testing, _done)

    def _focus_roblox_window(self) -> None:
        if not hasattr(self.adapter, "focus_roblox_window"):
            return

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not result or not result.get("ok"):
                self.app.set_status(
                    f"Focus Roblox window: {error or (result or {}).get('message')}", error=True
                )
            else:
                self.app.set_status(str(result.get("message") or "Roblox window focused"))

        self.app.background.submit(self.adapter.focus_roblox_window, _done)

    def _enable_ubuntu_cpu_turbo(self) -> None:
        if not hasattr(self.adapter, "enable_ubuntu_cpu_turbo"):
            return

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not result:
                self.app.set_status(f"CPU Turbo failed: {error}", error=True)
                return
            uview = vm.ubuntu_cpu_turbo_view(result)
            self.app.set_status(f"Activated {uview['badge']} — {uview['summary']}")

        self.app.background.submit(self.adapter.enable_ubuntu_cpu_turbo, _done)

    def _capture_roblox_window(self) -> None:
        if not hasattr(self.adapter, "capture_roblox_screenshot"):
            return

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not result or not result.get("ok"):
                self.app.set_status(
                    f"Screenshot failed: {error or (result or {}).get('error')}", error=True
                )
            else:
                self.app.set_status(f"Captured Roblox screenshot: {result.get('path')}")

        self.app.background.submit(self.adapter.capture_roblox_screenshot, _done)

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
            health = vm.ppo_health_view(diag)
            checkpoint_lines.append(
                f"PPO diagnostics [{health['status']}]: approx_kl="
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
        reward_pts = data.get("mean_episode_reward", [])
        self.reward_chart.set_points(reward_pts)
        self.fps_chart.set_points(data.get("steps_per_second", []))
        self.kl_chart.set_points(data.get("approx_kl", []))
        conv = vm.training_convergence_view(reward_pts)
        conv_color = (
            COLOR_OK
            if conv["state"] == "IMPROVING"
            else (
                COLOR_WARN
                if conv["state"] == "PLATEAU"
                else (COLOR_ERROR if conv["state"] == "REGRESSING" else COLOR_MUTED)
            )
        )
        self.convergence_label.configure(
            text=f"Convergence radar: [{conv['badge']}] — {conv['recommendation']}",
            foreground=conv_color,
        )

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

    def _resume_in_agents(self) -> None:
        if not self._last_run_dir:
            messagebox.showinfo("No run yet", "No training run found to resume.")
            return
        latest = Path(self._last_run_dir) / "checkpoints" / "latest.zip"
        best = Path(self._last_run_dir) / "checkpoints" / "best.zip"
        ckpt = latest if latest.is_file() else (best if best.is_file() else None)
        self.app.show_page("Agents")
        agents = self.app.pages.get("Agents")
        if agents is not None and ckpt is not None and hasattr(agents, "resume_checkpoint_var"):
            agents.resume_checkpoint_var.set(str(ckpt))
            self.app.set_status(f"Selected checkpoint for resume: {ckpt}")

    def _evaluate_latest_best(self) -> None:
        if not self._last_run_dir:
            messagebox.showinfo("No run yet", "No training run found to evaluate.")
            return
        best = Path(self._last_run_dir) / "checkpoints" / "best.zip"
        latest = Path(self._last_run_dir) / "checkpoints" / "latest.zip"
        target = best if best.is_file() else (latest if latest.is_file() else None)
        if target is None:
            messagebox.showinfo(
                "No checkpoint yet", "This run has neither best.zip nor latest.zip yet."
            )
            return
        self.app.show_page("Evaluations")
        self.app.pages["Evaluations"].select_checkpoint(str(target))


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
        # ---- Launch configuration (streamlined 5-parameter deployment deck) ---
        form_frame = ttk.LabelFrame(self, text="Launch configuration", padding=12)
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

        # Keep StringVars for all training fields so programmatic overrides
        # (such as inference_device from a hybrid benchmark recommendation)
        # remain preserved even though the GUI exposes only the 5 core controls.
        self._measured_sps: float | None = None
        if recommendation and isinstance(
            recommendation.get("expected_steps_per_second"), (int, float)
        ):
            self._measured_sps = float(recommendation["expected_steps_per_second"])
        for spec in vm.TRAINING_FIELDS:
            self.field_vars[spec.name] = tk.StringVar(value=defaults.get(spec.name, ""))

        basic_frame = ttk.Frame(form_frame, style="Surface.TFrame")
        basic_frame.pack(fill="x")
        self._build_fields(basic_frame, vm.launch_field_specs(), defaults)

        presets_bar = ttk.Frame(form_frame, style="Surface.TFrame")
        presets_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(presets_bar, text="Presets:", style="FieldTitle.TLabel").pack(
            side="left", padx=(0, 8)
        )
        for label, steps_val in (
            ("25k Smoke", "25000"),
            ("100k Standard", "100000"),
            ("500k Deep", "500000"),
        ):
            ttk.Button(
                presets_bar,
                text=label,
                command=lambda s=steps_val: self.field_vars["total_training_steps"].set(s),  # type: ignore[misc]
            ).pack(side="left", padx=(0, 6))
        ttk.Button(
            presets_bar,
            text="Sync Optimal Benchmark",
            command=self._sync_optimal_benchmark,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            presets_bar,
            text="Ubuntu CPU Turbo",
            command=self._apply_ubuntu_cpu_turbo,
        ).pack(side="left", padx=(6, 0))

        resume_bar = ttk.Frame(form_frame, style="Surface.TFrame")
        resume_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(resume_bar, text="Resume checkpoint (optional):", style="FieldTitle.TLabel").pack(
            side="left", padx=(0, 8)
        )
        self.resume_checkpoint_var = tk.StringVar(value="")
        self.resume_combo = ttk.Combobox(
            resume_bar,
            textvariable=self.resume_checkpoint_var,
            values=("",),
            width=64,
        )
        self.resume_combo.pack(side="left")
        ttk.Button(
            resume_bar,
            text="Clear (Fresh Run)",
            command=lambda: self.resume_checkpoint_var.set(""),
        ).pack(side="left", padx=(6, 0))

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
        for col_index, spec in enumerate(specs):
            parent.columnconfigure(col_index, weight=1)
            cell = ttk.Frame(parent, style="Surface.TFrame", padding=(0, 2, 14, 2))
            cell.grid(row=0, column=col_index, sticky="nsew")
            ttk.Label(cell, text=spec.label, style="FieldTitle.TLabel").pack(anchor="w")
            var = self.field_vars.get(spec.name)
            if var is None:
                var = tk.StringVar(value=defaults.get(spec.name, ""))
                self.field_vars[spec.name] = var
            else:
                var.set(defaults.get(spec.name, ""))
            if spec.kind == "choice" and spec.choices:
                widget: tk.Widget = ttk.Combobox(
                    cell, textvariable=var, values=spec.choices, state="readonly", width=18
                )
            elif spec.kind == "bool":
                widget = ttk.Checkbutton(cell, variable=var, onvalue="true", offvalue="false")
            else:
                widget = ttk.Entry(cell, textvariable=var, width=18)
            widget.pack(fill="x", pady=(4, 2))
            if spec.help:
                ToolTip(widget, spec.help)
                ttk.Label(
                    cell, text=spec.help, style="FieldHelp.TLabel", wraplength=210, justify="left"
                ).pack(anchor="w")

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

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

    def _sync_optimal_benchmark(self) -> None:
        """Apply the persisted benchmark recommendation directly to the launch form."""
        try:
            recommendation = self.adapter.recommended_configuration()
        except OSError:
            recommendation = None
        if not recommendation or not isinstance(recommendation.get("environment_count"), int):
            self.app.set_status("No benchmark recommendation persisted yet", error=True)
            return
        if isinstance(recommendation.get("expected_steps_per_second"), (int, float)):
            self._measured_sps = float(recommendation["expected_steps_per_second"])
        self.apply_launch_values(
            {
                "environment_count": str(recommendation["environment_count"]),
                "env_workers": str(recommendation.get("env_workers", 1)),
                "device": str(recommendation.get("device") or "auto"),
                "inference_device": str(recommendation.get("inference_device") or "auto"),
            }
        )
        self.app.set_status("Synced optimal benchmark topology to launch form")

    def _apply_ubuntu_cpu_turbo(self) -> None:
        """Activate Ubuntu CPU Turbo mode and populate optimal CPU shard topology."""
        if not hasattr(self.adapter, "enable_ubuntu_cpu_turbo"):
            return
        try:
            profile = self.adapter.enable_ubuntu_cpu_turbo()
        except Exception as exc:
            self.app.set_status(f"Ubuntu CPU Turbo failed: {exc}", error=True)
            return
        uview = vm.ubuntu_cpu_turbo_view(profile)
        self.apply_launch_values(
            {
                "environment_count": str(uview["recommended_envs"]),
                "env_workers": str(uview["recommended_workers"]),
                "device": "cpu",
                "inference_device": "cpu",
            }
        )
        self.app.set_status(
            f"Ubuntu CPU Turbo active: {uview['recommended_envs']} Envs x {uview['recommended_workers']} Workers on CPU (OMP/MKL=1)"
        )

    def refresh(self) -> None:
        self.submit_poll("agents", self.adapter.agents.views, self._on_agents)
        if hasattr(self.adapter, "discover_checkpoints"):
            self.submit_poll(
                "agents-checkpoints", self.adapter.discover_checkpoints, self._on_resume_checkpoints
            )
        values = self.current_values()
        if values != self._last_slot_values or self._compatibility is None:
            self._last_slot_values = dict(values)
            self._update_launch_slot(values)
        process_id = self._selected_agent_process_id()
        if process_id:
            self._request_log(process_id)

    def _on_resume_checkpoints(
        self, entries: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or entries is None:
            return
        paths = [""] + [str(e.get("path")) for e in entries if e.get("path")]
        self.resume_combo.configure(values=tuple(paths))

    def _update_launch_slot(self, values: dict[str, str]) -> None:
        slot = vm.launch_slot_view(values, self._compatibility)
        self._launch_slot = slot
        if slot["state"] == "AVAILABLE":
            summary = slot["summary"]
            topology = "+".join(
                str(row["environments"])
                for row in vm.topology_rows(summary["environment_count"], summary["env_workers"])
            )
            eta = vm.estimate_training_duration(
                summary["total_training_steps"], getattr(self, "_measured_sps", None)
            )
            eta_suffix = f" (est. ~{eta} @ {vm.format_number(self._measured_sps, 1)} steps/s)" if eta else ""
            text = (
                f"AVAILABLE — {summary['environment_count']} environments / "
                f"{summary['env_workers']} workers ({topology}) on {summary['device']}, "
                f"{vm.format_number(summary['total_training_steps'])} steps{eta_suffix}"
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
        resume_ckpt = (
            self.resume_checkpoint_var.get().strip()
            if hasattr(self, "resume_checkpoint_var")
            else ""
        )
        self._launching = True
        self.launch_button.configure(state="disabled")
        self.app.background.submit(
            (
                (lambda: self.adapter.agents.launch_training(config, checkpoint=resume_ckpt))
                if resume_ckpt
                else (lambda: self.adapter.agents.launch_training(config))
            ),
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
        max_envs = max((int(s["environments"]) for s in shards), default=1)
        lines = [
            f"{row['name']} — {environment_count} environments across {env_workers} worker(s):"
        ]
        for shard in shards:
            bar = vm.format_ascii_bar(float(shard["environments"]) / max(max_envs, 1), width=8)
            lines.append(
                f"  worker {shard['worker']}: {bar} environments "
                f"{shard['first_environment']}-{shard['last_environment']} "
                f"({shard['environments']} envs)"
            )
        sps = row.get("steps_per_second")
        steps = row.get("timesteps")
        target = row.get("target_timesteps")
        if (
            isinstance(sps, (int, float))
            and float(sps) > 0.0
            and isinstance(steps, (int, float))
            and isinstance(target, (int, float))
            and float(target) > float(steps)
        ):
            eta = vm.estimate_training_duration(float(target) - float(steps), sps)
            if eta:
                lines.append(f"  live ETA: ~{eta} remaining @ {vm.format_number(sps, 1)} steps/s")
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
    """The one-button automatic benchmark with live hardware & bridge telemetry.

    There is deliberately nothing to configure here. Start runs the staged
    pipeline (runtime discovery, env/worker screening, device comparison,
    real PPO validation slices) with the project's host-scaled defaults,
    streams live step/FPS/latency/resource telemetry while every candidate
    is running, chooses the fastest stable configuration, and persists and
    applies it to the launch configuration automatically.
    """

    title = "Benchmarks"
    subtitle = (
        "Autonomous real-time calibration — streams live FPS, step counters, latency "
        "& stability, then applies the fastest stable topology automatically."
    )

    RESULT_COLUMNS = (
        ("stage", "Stage", 85),
        ("status", "Status", 75),
        ("environments", "Envs", 50),
        ("workers", "Workers", 60),
        ("device", "Device", 55),
        ("steps", "Steps", 80),
        ("steps_per_second", "Steps/s", 85),
        ("speedup", "Speedup", 70),
        ("p50_ms", "p50 ms", 65),
        ("p95_ms", "p95 ms", 65),
        ("jitter", "p95/p50", 65),
        ("startup_seconds", "Startup s", 75),
        ("bottleneck", "Regime", 95),
        ("error", "Error", 200),
    )

    LIVE_CARD_NAMES = (
        "stage / progress",
        "active config",
        "live fps (steps/s)",
        "peak fps",
        "live steps",
        "latency (p50 / p95)",
        "stability (jitter)",
        "elapsed / host",
    )

    def build(self) -> None:
        intro = ttk.LabelFrame(self, text="Automatic benchmark & live telemetry", padding=10)
        intro.pack(fill="x")
        ttk.Label(
            intro,
            text=(
                "Start measures the real runtime end to end: it screens a host-scaled "
                "grid of environment/worker topologies through the actual bridge, compares "
                "devices where more than one exists, validates the best candidates with "
                "short real PPO training slices, then picks the fastest stable "
                "configuration and applies it automatically."
            ),
            wraplength=980,
            justify="left",
            foreground=COLOR_MUTED,
        ).pack(anchor="w")

        run_bar = ttk.Frame(intro, style="Surface.TFrame")
        run_bar.pack(fill="x", pady=(8, 0))
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
            run_bar, text="idle", foreground=COLOR_MUTED, wraplength=680, justify="left"
        )
        self.progress_label.pack(side="left", padx=(14, 0))

        self.phase_stepper = PhaseStepper(intro)
        self.phase_stepper.pack(fill="x", pady=(8, 2))
        self.phase_label = ttk.Label(intro, text="", foreground=COLOR_MUTED, justify="left")
        self.phase_label.pack(anchor="w", pady=(2, 0))

        # ---- 8 Live Real-Time Telemetry Cards -------------------------
        self.live_cards = StatRow(self, self.LIVE_CARD_NAMES, max_columns=4)
        self.live_cards.pack(fill="x", pady=(8, 0))

        # ---- Best Configuration & Live Leader Banner ------------------
        best = ttk.LabelFrame(
            self, text="Best configuration (selected and applied automatically)", padding=10
        )
        best.pack(fill="x", pady=(8, 0))
        self.leader_banner = ttk.Label(
            best, text="LEADING SO FAR: awaiting benchmark telemetry", style="Leader.TLabel"
        )
        self.leader_banner.pack(fill="x", pady=(0, 6))
        self.recommendation_label = ttk.Label(best, text="n/a", justify="left")
        self.recommendation_label.pack(anchor="nw")
        self.applied_label = ttk.Label(best, text="", justify="left", foreground=COLOR_MUTED)
        self.applied_label.pack(anchor="w", pady=(4, 0))

        # ---- Split Live Results Table + Throughput Chart --------------
        bottom = ttk.Panedwindow(self, orient="horizontal")
        bottom.pack(fill="both", expand=True, pady=(8, 0))

        results_frame = ttk.LabelFrame(
            bottom, text="Measurements (live stream of every tested configuration)", padding=8
        )
        bottom.add(results_frame, weight=3)
        self.tree = _scrollable_table(results_frame, self.RESULT_COLUMNS)
        self.tree.tag_configure("failed", foreground=COLOR_ERROR)
        self.tree.tag_configure("leader", foreground=COLOR_OK)

        chart_frame = ttk.LabelFrame(
            bottom, text="Throughput scaling (steps/s across configurations)", padding=8
        )
        bottom.add(chart_frame, weight=2)
        self.throughput_chart = LineChart(
            chart_frame, "Measured throughput (steps/s)", color=COLOR_ACCENT
        )
        self.throughput_chart.pack(fill="both", expand=True)

        self._cancel_event: threading.Event | None = None
        self._latest_progress: dict[str, Any] | None = None
        self._progress_lock = threading.Lock()
        self._latest_report: dict[str, Any] | None = None
        self._applied: bool | None = None
        self._failure_text: str | None = None
        self._running = False
        self._rendered_row_count = -1

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
        live_view = vm.benchmark_live_telemetry_view(
            running=self._running,
            event=progress,
            report=self._latest_report,
        )
        self.phase_stepper.set_phases(view["phases"], fraction=live_view["progress_fraction"])
        self._update_live_telemetry_cards(live_view)
        if self._running and progress is not None:
            completed_rows = progress.get("completed_rows")
            if isinstance(completed_rows, list) and len(completed_rows) != self._rendered_row_count:
                self._rendered_row_count = len(completed_rows)
                self._render_report(None, live_rows=completed_rows)

    def _update_live_telemetry_cards(self, live_view: dict[str, Any]) -> None:
        stage_text = live_view["stage_label"]
        if live_view["index"] and live_view["total"]:
            stage_text = f"{stage_text} ({live_view['index']}/{live_view['total']})"
        fps_text = (
            f"{vm.format_number(live_view['live_fps'], 1)} ({live_view['live_phase']})"
            if live_view["live_fps"] is not None and live_view["live_phase"]
            else vm.format_number(live_view["live_fps"], 1)
        )
        peak_text = vm.format_number(live_view["peak_fps"], 1)
        if live_view.get("peak_speedup") is not None:
            peak_text += f" ({vm.format_number(live_view['peak_speedup'], 2)}x)"
        if live_view["live_steps"] is not None:
            steps_text = vm.format_number(live_view["live_steps"])
            if live_view["steps_per_env"] is not None:
                steps_text += f" ({vm.format_number(live_view['steps_per_env'])}/env)"
        else:
            steps_text = "n/a"
        latency_text = (
            f"{vm.format_number(live_view['p50_ms'], 2)} / "
            f"{vm.format_number(live_view['p95_ms'], 2)} ms"
            if live_view["p50_ms"] is not None or live_view["p95_ms"] is not None
            else "n/a"
        )
        jitter = live_view["jitter"]
        jitter_text = f"{vm.format_number(jitter, 2)}x" if jitter is not None else "n/a"
        jitter_color = (
            COLOR_WARN
            if isinstance(jitter, (int, float)) and jitter > 4.0
            else (COLOR_OK if isinstance(jitter, (int, float)) else None)
        )
        elapsed = live_view["elapsed_seconds"]
        elapsed_str = f"{vm.format_number(elapsed, 1)}s" if elapsed is not None else "0.0s"
        if live_view["cpu_percent"] is not None:
            elapsed_str += f" | CPU {vm.format_number(live_view['cpu_percent'], 0)}%"
        self.live_cards.update_values(
            {
                "stage / progress": (
                    stage_text,
                    COLOR_ACCENT if self._running else None,
                ),
                "active config": (live_view["active_config"], None),
                "live fps (steps/s)": (
                    fps_text,
                    COLOR_OK if live_view["live_fps"] is not None else None,
                ),
                "peak fps": (
                    peak_text,
                    COLOR_ACCENT if live_view["peak_fps"] is not None else None,
                ),
                "live steps": (steps_text, None),
                "latency (p50 / p95)": (latency_text, None),
                "stability (jitter)": (jitter_text, jitter_color),
                "elapsed / host": (elapsed_str, None),
            }
        )
        self.leader_banner.configure(
            text=f"Leading configuration: {live_view['leader_summary']}"
        )
        chart_points = list(live_view["chart_points"])
        if (
            self._running
            and live_view["live_fps"] is not None
            and isinstance(live_view["live_fps"], (int, float))
        ):
            next_idx = len(chart_points) + 1
            chart_points.append((next_idx, float(live_view["live_fps"])))
        self.throughput_chart.set_points(chart_points)

    def _start(self) -> None:
        """Run the complete workflow - no form, no parameters to validate."""
        if self._running:
            return
        self._running = True
        self._latest_report = None
        self._applied = None
        self._failure_text = None
        self._rendered_row_count = -1
        self._cancel_event = threading.Event()
        with self._progress_lock:
            self._latest_progress = None
        self.tree.delete(*self.tree.get_children())
        self.throughput_chart.set_points([])
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
            report = history[0].get("report")
            self._latest_report = report
            self._render_report(report)
            self._refresh_workflow_labels()

    def _render_report(
        self,
        report: dict[str, Any] | None,
        *,
        live_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        rows = vm.benchmark_pipeline_rows(report, live_rows=live_rows)
        self.tree.delete(*self.tree.get_children())
        best_fps = max(
            (
                float(r["steps_per_second"])
                for r in rows
                if (r.get("status") or "ok") in ("ok", "measured")
                and isinstance(r.get("steps_per_second"), (int, float))
            ),
            default=None,
        )
        for row in rows:
            status = row["status"] or "ok"
            fps = row.get("steps_per_second")
            if status not in ("ok", "measured"):
                tags: tuple[str, ...] = ("failed",)
            elif best_fps is not None and isinstance(fps, (int, float)) and float(fps) >= best_fps:
                tags = ("leader",)
            else:
                tags = ()
            speedup = row.get("speedup")
            speedup_str = f"{vm.format_number(speedup, 2)}x" if speedup is not None else "n/a"
            self.tree.insert(
                "",
                "end",
                values=(
                    row["stage"] or "n/a",
                    status,
                    vm.format_number(row["environments"]),
                    vm.format_number(row["workers"]),
                    row["device"] or "-",
                    vm.format_number(row["steps"]),
                    vm.format_number(fps, 1),
                    speedup_str,
                    vm.format_number(row["p50_ms"], 2),
                    vm.format_number(row["p95_ms"], 2),
                    vm.format_number(row["jitter"], 2),
                    vm.format_number(row["startup_seconds"], 2),
                    row.get("bottleneck") or "-",
                    row["error"] or "",
                ),
                tags=tags,
            )

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
        self.eval_tree.tag_configure("best-eval", foreground=COLOR_OK)
        self._eval_paths.clear()
        best_reward = max(
            (
                float(e["mean_episode_reward"])
                for e in entries
                if isinstance(e.get("mean_episode_reward"), (int, float))
            ),
            default=None,
        )
        for entry in entries:
            path = str(entry["path"])
            reward_val = entry.get("mean_episode_reward")
            is_best = (
                best_reward is not None
                and isinstance(reward_val, (int, float))
                and float(reward_val) >= best_reward
            )
            item_id = self.eval_tree.insert(
                "",
                "end",
                tags=("best-eval",) if is_best else (),
                values=(
                    path,
                    vm.format_number(entry.get("timesteps")),
                    vm.format_number(entry.get("episodes")),
                    vm.format_fraction_as_percent(entry.get("win_rate")),
                    vm.format_fraction_as_percent(entry.get("loss_rate")),
                    vm.format_number(reward_val, 3),
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
            tactical = vm.tactical_combat_profile_view(details[0])
            self.detail_text.insert(
                "end", _render_evaluation_detail(vm.evaluation_view(details[0]), tactical)
            )
        else:
            rows = vm.evaluation_comparison_rows(details)
            self.detail_text.insert("end", _render_evaluation_comparison(rows))
        self.detail_text.configure(state="disabled")


def _render_evaluation_detail(
    view: dict[str, Any], tactical: dict[str, Any] | None = None
) -> str:
    if not view.get("available"):
        return f"Evaluation summary unavailable: {view.get('error')}"
    win_bar = vm.format_ascii_bar(view["outcomes"]["win_rate"], 10)
    acc_bar = vm.format_ascii_bar(view["accuracy"]["mean_accuracy"], 10)
    shoot_rate = view["action_head_diagnostics"]["policy_shoot_request_rate"]
    discharge_rate = view["action_head_diagnostics"]["discharge_rate"]
    shoot_bar = vm.format_ascii_bar(shoot_rate, 10)
    discharge_bar = vm.format_ascii_bar(discharge_rate, 10)
    lines = [
        f"path: {view['path']}",
        f"episodes: {vm.format_number(view['episodes'])}   timesteps: {vm.format_number(view['timesteps'])}",
    ]
    if tactical and tactical.get("available"):
        lines.extend(
            [
                "",
                f"Tactical Combat Lab // Archetype: {tactical['archetype']}",
                f"  K/D ratio: {tactical['kd_ratio']}   damage trade: {tactical['damage_trade']}"
                f"   lethality: {tactical['lethality']}   survival: {tactical['survival_rate']}",
                f"  assessment: {tactical['archetype_summary']}",
            ]
        )
    lines.extend(
        [
            "",
            "Outcomes",
            f"  win rate: {win_bar} {vm.format_fraction_as_percent(view['outcomes']['win_rate'])}"
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
            f"  accuracy: {acc_bar} {vm.format_fraction_as_percent(view['accuracy']['mean_accuracy'])}"
            f"   shots fired: {vm.format_number(view['accuracy']['mean_shots_fired'], 1)}"
            f"   shots hit: {vm.format_number(view['accuracy']['mean_shots_hit'], 1)}",
            "",
            "Action-head diagnostics (zero-shot / policy discharge behavior)",
            f"  shoot request rate: {shoot_bar} {vm.format_fraction_as_percent(shoot_rate)}",
            f"  discharge rate:     {discharge_bar} {vm.format_fraction_as_percent(discharge_rate)}",
            f"  action-pipeline localization: {view['action_head_diagnostics']['localization'] or 'n/a'}",
        ]
    )
    return "\n".join(lines)


def _render_evaluation_comparison(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No comparable evaluation summaries in this selection."
    best_reward = max(
        (
            float(r["reward"])
            for r in rows
            if isinstance(r.get("reward"), (int, float))
        ),
        default=None,
    )
    base_reward = next(
        (float(r["reward"]) for r in rows if isinstance(r.get("reward"), (int, float))),
        None,
    )
    header = (
        f"{'path':36} {'reward':>8} {'Δrew':>8} {'win%':>7} {'loss%':>7} {'acc%':>7} {'discharge%':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        rew = row.get("reward")
        is_best = (
            best_reward is not None
            and isinstance(rew, (int, float))
            and float(rew) >= best_reward
        )
        prefix = "★ " if is_best else "  "
        delta_str = (
            f"{float(rew) - base_reward:+.2f}"
            if (isinstance(rew, (int, float)) and base_reward is not None)
            else "n/a"
        )
        lines.append(
            f"{prefix + Path(row['path']).name:36} {vm.format_number(rew, 2):>8} "
            f"{delta_str:>8} {vm.format_fraction_as_percent(row['win_rate']):>7} "
            f"{vm.format_fraction_as_percent(row['loss_rate']):>7} "
            f"{vm.format_fraction_as_percent(row['accuracy']):>7} "
            f"{vm.format_fraction_as_percent(row['discharge_rate']):>11}"
        )
    return "\n".join(lines)


class RunsPage(Page):
    title = "Runs / Checkpoints"
    subtitle = "Read-only inventory from run_inspection.py — inspect details, reward curves & evaluate checkpoints."

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
        self.tree.tag_configure("run-running", foreground=COLOR_OK)
        self.tree.tag_configure("run-failed", foreground=COLOR_ERROR)
        self.tree.tag_configure("run-warn", foreground=COLOR_WARN)

        bottom = ttk.Frame(paned)
        paned.add(bottom, weight=1)
        bottom_split = ttk.Panedwindow(bottom, orient="horizontal")
        bottom_split.pack(fill="both", expand=True)

        detail_frame = ttk.LabelFrame(bottom_split, text="Run detail & diagnostics", padding=8)
        bottom_split.add(detail_frame, weight=3)
        self.detail_text = tk.Text(
            detail_frame,
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

        chart_frame = ttk.LabelFrame(
            bottom_split, text="Selected run telemetry (reward vs. timesteps)", padding=8
        )
        bottom_split.add(chart_frame, weight=2)
        self.run_reward_chart = LineChart(
            chart_frame, "Mean episode reward", color=COLOR_OK
        )
        self.run_reward_chart.pack(fill="both", expand=True, pady=(0, 4))
        self.run_fps_chart = LineChart(
            chart_frame, "Throughput (steps/s)", color=COLOR_ACCENT
        )
        self.run_fps_chart.pack(fill="both", expand=True)

        actions = ttk.Frame(bottom)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Open run folder", command=self._open_folder).pack(side="left")
        ttk.Button(actions, text="Evaluate latest checkpoint", command=self._evaluate).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="Evaluate best checkpoint", command=self._evaluate_best).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(
            actions, text="Clone topology to Agents", command=self._clone_to_agents
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="Resume checkpoint in Agents",
            command=self._resume_run_in_agents,
            style="Primary.TButton",
        ).pack(side="left", padx=(8, 0))

        self._row_to_dir: dict[str, str] = {}
        self._selected_run_dir: str | None = None
        self._selected_run_report: dict[str, Any] | None = None
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
            state_str = str(row["state"] or "").lower()
            if state_str == "running":
                tags: tuple[str, ...] = ("run-running",)
            elif state_str in ("failed", "error"):
                tags = ("run-failed",)
            elif state_str in ("starting", "paused", "stopping"):
                tags = ("run-warn",)
            else:
                tags = ()
            item_id = self.tree.insert(
                "",
                "end",
                tags=tags,
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
        self._selected_run_report = None
        self._run_detail_generation += 1
        generation = self._run_detail_generation
        self.app.background.submit(
            lambda: self.adapter.inspect_run(run_dir),
            lambda report, error: self._on_detail(run_dir, generation, report, error),
        )
        if hasattr(self.adapter, "telemetry_series"):
            self.app.background.submit(
                lambda: self.adapter.telemetry_series(run_dir),
                lambda series, error: self._on_run_telemetry(run_dir, generation, series, error),
            )

    def _clear_run_selection(self) -> None:
        self._selected_run_dir = None
        self._selected_run_report = None
        self._run_detail_generation += 1
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.configure(state="disabled")
        self.run_reward_chart.set_points([])
        self.run_fps_chart.set_points([])

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
        self._selected_run_report = report
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", _render_run_detail(report))
        self.detail_text.configure(state="disabled")

    def _on_run_telemetry(
        self,
        run_dir: str,
        generation: int,
        series: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        if (
            run_dir != self._selected_run_dir
            or generation != self._run_detail_generation
            or error is not None
            or series is None
            or not series.get("available")
        ):
            return
        data = series.get("series") or {}
        self.run_reward_chart.set_points(data.get("mean_episode_reward", []))
        self.run_fps_chart.set_points(data.get("steps_per_second", []))

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

    def _evaluate_best(self) -> None:
        if not self._selected_run_dir:
            return
        best = Path(self._selected_run_dir) / "checkpoints" / "best.zip"
        target = best if best.is_file() else (Path(self._selected_run_dir) / "checkpoints" / "latest.zip")
        if not target.is_file():
            messagebox.showinfo(
                "No checkpoint yet", "This run has neither best.zip nor latest.zip yet."
            )
            return
        self.app.show_page("Evaluations")
        self.app.pages["Evaluations"].select_checkpoint(str(target))

    def _clone_to_agents(self) -> None:
        report = self._selected_run_report
        if not report:
            return
        config = report.get("config") or {}
        agents_page = self.app.pages.get("Agents")
        if agents_page is None:
            return
        self.app.show_page("Agents")
        agents_page.apply_launch_values(
            {
                "environment_count": str(config.get("environment_count", 1)),
                "env_workers": str(config.get("env_workers", 1)),
                "total_training_steps": str(config.get("total_training_steps", 100000)),
                "device": str(config.get("device", "auto")),
            }
        )
        self.app.set_status(f"Cloned topology from {report.get('run_id', 'run')} to Agents")

    def _resume_run_in_agents(self) -> None:
        if not self._selected_run_dir:
            return
        latest = Path(self._selected_run_dir) / "checkpoints" / "latest.zip"
        best = Path(self._selected_run_dir) / "checkpoints" / "best.zip"
        target = latest if latest.is_file() else (best if best.is_file() else None)
        if target is None:
            messagebox.showinfo(
                "No checkpoint yet", "This run has neither latest.zip nor best.zip yet."
            )
            return
        self._clone_to_agents()
        agents_page = self.app.pages.get("Agents")
        if agents_page is not None and hasattr(agents_page, "resume_checkpoint_var"):
            agents_page.resume_checkpoint_var.set(str(target))
            self.app.set_status(f"Ready to resume from {target}")


def _render_run_detail(report: dict[str, Any]) -> str:
    manifest = report.get("manifest") or {}
    config = report.get("config") or {}
    status = report.get("status") or {}
    checkpoints = report.get("checkpoints") or {}
    evaluation = report.get("evaluation") or {}
    warnings = report.get("warnings") or []
    problems = report.get("problems") or []
    progress = status.get("progress_percent")
    prog_bar = (
        f"{vm.format_ascii_bar(float(progress) / 100.0, 12)} {vm.format_fraction_as_percent(float(progress) / 100.0)}"
        if isinstance(progress, (int, float))
        else "n/a"
    )
    lines = [
        f"run_id: {report.get('run_id')}    experiment: {report.get('experiment_id') or 'n/a'}",
        f"state: {status.get('state')}    progress: {prog_bar}",
        f"created: {manifest.get('created_utc', 'n/a')}",
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
    ppo_diag = (report.get("summary") or {}).get("ppo_diagnostics")
    if ppo_diag:
        health = vm.ppo_health_view(ppo_diag)
        lines.append(health["summary"])
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

    STAT_LABELS = (
        "cpu",
        "process memory",
        "optimal topology",
        "python",
        "godot",
        "torch",
        "cuda",
    )

    def build(self) -> None:
        self.stats = StatRow(self, self.STAT_LABELS, max_columns=4)
        self.stats.pack(fill="x")
        deps_frame = ttk.LabelFrame(self, text="Optional dependencies & runtime capabilities", padding=10)
        deps_frame.pack(fill="x", pady=(12, 0))
        self.deps_label = ttk.Label(deps_frame, text="n/a", justify="left")
        self.deps_label.pack(anchor="w")

        chart_frame = ttk.LabelFrame(
            self,
            text="Host resource usage (sampled each refresh, bounded to the last 300 samples)",
            padding=10,
        )
        chart_frame.pack(fill="both", expand=True, pady=(12, 0))
        charts_grid = ttk.Frame(chart_frame)
        charts_grid.pack(fill="both", expand=True)
        self.cpu_chart = LineChart(
            charts_grid, "CPU percent (this process)", color=COLOR_ACCENT
        )
        self.cpu_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.rss_chart = LineChart(
            charts_grid, "Process RSS (MB)", color=COLOR_OK
        )
        self.rss_chart.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        charts_grid.columnconfigure(0, weight=1)
        charts_grid.columnconfigure(1, weight=1)
        charts_grid.rowconfigure(0, weight=1)

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
        cpu_pct = status.get("cpu_percent")
        cpu_color = (
            COLOR_ERROR
            if isinstance(cpu_pct, (int, float)) and cpu_pct >= 90.0
            else (
                COLOR_WARN
                if isinstance(cpu_pct, (int, float)) and cpu_pct >= 75.0
                else COLOR_OK
            )
        )
        rec_text = "uncalibrated"
        rec_color = COLOR_MUTED
        if hasattr(self.adapter, "recommended_configuration"):
            try:
                rec = self.adapter.recommended_configuration()
                if rec and isinstance(rec.get("environment_count"), int):
                    rec_text = (
                        f"{rec['environment_count']}e / {rec.get('env_workers', 1)}w "
                        f"({rec.get('device', 'cpu')})"
                    )
                    rec_color = COLOR_OK
            except OSError:
                pass
        self.stats.update_values(
            {
                "cpu": (
                    vm.format_fraction_as_percent((cpu_pct or 0) / 100.0)
                    if cpu_pct is not None
                    else "n/a",
                    cpu_color if cpu_pct is not None else None,
                ),
                "process memory": (
                    vm.format_bytes((status.get("process_rss_mb") or 0) * 1024 * 1024)
                    if status.get("process_rss_mb") is not None
                    else "n/a",
                    None,
                ),
                "optimal topology": (rec_text, rec_color),
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
    subtitle = "Project/output roots, machine-local Godot executable & persisted calibration status."

    def build(self) -> None:
        top_grid = ttk.Frame(self)
        top_grid.pack(fill="x")
        top_grid.columnconfigure(0, weight=1)
        top_grid.columnconfigure(1, weight=1)

        roots = ttk.LabelFrame(top_grid, text="Directories", padding=12)
        roots.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.project_root_label = ttk.Label(roots, text="")
        self.project_root_label.pack(anchor="w", pady=2)
        self.output_root_label = ttk.Label(roots, text="")
        self.output_root_label.pack(anchor="w", pady=2)
        dir_buttons = ttk.Frame(roots, style="Surface.TFrame")
        dir_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(
            dir_buttons, text="Change output root...", command=self._change_output_root
        ).pack(side="left")
        ttk.Button(
            dir_buttons,
            text="Open project folder",
            command=lambda: _open_in_file_manager(Path(self.adapter.project_root)),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            dir_buttons,
            text="Open output folder",
            command=lambda: _open_in_file_manager(Path(self.adapter.output_root)),
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            roots,
            text="Changing the output root points Runs, Evaluations & Benchmarks at a different directory.",
            foreground=COLOR_MUTED,
            wraplength=520,
        ).pack(anchor="w", pady=(6, 0))

        # ---- Godot executable (the #1 reason launches fail) -------------
        godot = ttk.LabelFrame(top_grid, text="Godot executable", padding=12)
        godot.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(
            godot,
            text="Training, benchmarks and evaluations resolve the Godot binary through this "
            "remembered setting when none is given explicitly.",
            foreground=COLOR_MUTED,
            wraplength=520,
            justify="left",
        ).pack(anchor="w")
        entry_row = ttk.Frame(godot, style="Surface.TFrame")
        entry_row.pack(fill="x", pady=(8, 0))
        self.godot_var = tk.StringVar(value=self.adapter.godot_executable_setting() or "")
        ttk.Entry(entry_row, textvariable=self.godot_var, width=42).pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Browse", width=8, command=self._browse_godot).pack(
            side="left", padx=(6, 0)
        )
        self.godot_save_button = ttk.Button(
            entry_row, text="Verify & save", command=self._save_godot, style="Primary.TButton"
        )
        self.godot_save_button.pack(side="left", padx=(8, 0))
        self.godot_status_label = ttk.Label(godot, text="", justify="left", wraplength=520)
        self.godot_status_label.pack(anchor="w", pady=(6, 0))

        # ---- Machine calibration & Ubuntu CPU Performance Turbo ---------
        calib = ttk.LabelFrame(
            self,
            text="Machine calibration & Ubuntu CPU Performance Turbo",
            padding=12,
        )
        calib.pack(fill="x", pady=(10, 0))
        self.calibration_label = ttk.Label(
            calib, text="Checking persisted calibration...", justify="left", wraplength=920
        )
        self.calibration_label.pack(anchor="w")
        self.cpu_turbo_label = ttk.Label(
            calib, text="Ubuntu CPU Turbo: probing host topology...", justify="left", wraplength=920
        )
        self.cpu_turbo_label.pack(anchor="w", pady=(4, 0))
        calib_buttons = ttk.Frame(calib, style="Surface.TFrame")
        calib_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(
            calib_buttons,
            text="Open Benchmarks calibration",
            command=lambda: self.app.show_page("Benchmarks"),
        ).pack(side="left")
        ttk.Button(
            calib_buttons,
            text="Enable Ubuntu CPU Turbo (OMP/MKL=1 + Optimal Shards)",
            command=self._activate_ubuntu_cpu_turbo,
            style="Primary.TButton",
        ).pack(side="left", padx=(8, 0))

        # ---- Roblox TTK Testing Live Bridge & Calibration ---------------
        self._build_roblox_ttk_section()

    def _build_roblox_ttk_section(self) -> None:
        roblox_box = ttk.LabelFrame(
            self,
            text="Roblox TTK Testing [MAP VOTING]  ·  Sable Digital (PlaceId 120189115846709  ·  Universe 10090256806)",
            padding=12,
        )
        roblox_box.pack(fill="both", expand=True, pady=(10, 0))
        shortcut_row = ttk.Frame(roblox_box, style="Surface.TFrame")
        shortcut_row.pack(fill="x")
        ttk.Label(shortcut_row, text="Roblox shortcut / exe:", width=20).pack(side="left")
        self.roblox_shortcut_var = tk.StringVar(
            value=r"C:\Users\jonas\OneDrive\Desktop\Roblox Player.lnk"
        )
        ttk.Entry(shortcut_row, textvariable=self.roblox_shortcut_var, width=42).pack(side="left")
        ttk.Button(
            shortcut_row,
            text="Launch Shortcut",
            command=lambda: self._launch_roblox(direct_place=False),
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            shortcut_row,
            text="Join TTK Testing",
            command=lambda: self._launch_roblox(direct_place=True),
            style="Primary.TButton",
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            shortcut_row,
            text="Focus Window",
            command=self._focus_roblox_window,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            shortcut_row,
            text="Capture Screenshot",
            command=self._capture_ttk_screenshot,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            shortcut_row,
            text="Open Captures",
            command=self._open_captures_folder,
        ).pack(side="left", padx=(6, 0))

        self.roblox_live_label = ttk.Label(
            roblox_box, text="Probing Roblox Player...", foreground=COLOR_ACCENT, justify="left"
        )
        self.roblox_live_label.pack(anchor="w", pady=(8, 6))

        # ---- Interactive TTK & DPS Calculator + 1-Click Presets ---------
        calc_row = ttk.Frame(roblox_box, style="Surface.TFrame")
        calc_row.pack(fill="x", pady=(2, 8))
        ttk.Label(calc_row, text="TTK / DPS Calculator:", style="FieldTitle.TLabel").pack(side="left")
        ttk.Label(calc_row, text="DMG:").pack(side="left", padx=(10, 4))
        self.ttk_dmg_var = tk.StringVar(value="34")
        ttk.Entry(calc_row, textvariable=self.ttk_dmg_var, width=6).pack(side="left")
        ttk.Label(calc_row, text="RPM:").pack(side="left", padx=(10, 4))
        self.ttk_rpm_var = tk.StringVar(value="750")
        ttk.Entry(calc_row, textvariable=self.ttk_rpm_var, width=7).pack(side="left")
        ttk.Label(calc_row, text="HP:").pack(side="left", padx=(10, 4))
        self.ttk_hp_var = tk.StringVar(value="100")
        ttk.Entry(calc_row, textvariable=self.ttk_hp_var, width=6).pack(side="left")
        ttk.Button(
            calc_row,
            text="Calculate",
            command=self._recalc_ttk_lab,
        ).pack(side="left", padx=(10, 8))
        self.ttk_calc_result_label = ttk.Label(
            calc_row,
            text="3 STK  ·  160.0 ms TTK  ·  425.0 Burst DPS  ·  Instant-Lethal CQB (<170 ms)",
            foreground=COLOR_OK,
        )
        self.ttk_calc_result_label.pack(side="left", padx=(4, 10))
        ttk.Label(calc_row, text="Presets:", style="FieldTitle.TLabel").pack(
            side="left", padx=(8, 4)
        )
        for preset_id, btn_label in (
            ("sable_cqb_carbine", "Sable CQB (160ms)"),
            ("tactical_rifle_ffa", "8P FFA Rifle (265ms)"),
            ("precision_marksman", "Marksman (286ms)"),
        ):
            ttk.Button(
                calc_row,
                text=btn_label,
                command=lambda pid=preset_id: self._apply_ttk_preset(pid),  # type: ignore[misc]
            ).pack(side="left", padx=(4, 0))

        ttk_columns = (
            ("mechanic", "Mechanic", 180),
            ("status", "Status", 145),
            ("measured_value", "Measured / Calibrated Value", 190),
            ("rule", "Implementation Rule", 360),
            ("source", "Evidence Source", 200),
        )
        self.ttk_tree = _scrollable_table(roblox_box, ttk_columns)
        self.ttk_tree.tag_configure("ttk-verified", foreground=COLOR_OK)
        self.ttk_tree.tag_configure("ttk-pending", foreground=COLOR_WARN)
        self.ttk_tree.tag_configure("ttk-excluded", foreground=COLOR_MUTED)
        self.ttk_tree.bind("<<TreeviewSelect>>", self._on_select_ttk_mechanic)

        edit_row = ttk.Frame(roblox_box, style="Surface.TFrame")
        edit_row.pack(fill="x", pady=(6, 0))
        self._selected_mechanic: str | None = None
        self.mechanic_label = ttk.Label(
            edit_row, text="Mechanic: (select row)", width=26, style="FieldTitle.TLabel"
        )
        self.mechanic_label.pack(side="left")
        ttk.Label(edit_row, text="Measured value:").pack(side="left", padx=(6, 4))
        self.mechanic_value_var = tk.StringVar(value="")
        ttk.Entry(edit_row, textvariable=self.mechanic_value_var, width=28).pack(side="left")
        ttk.Label(edit_row, text="Notes:").pack(side="left", padx=(8, 4))
        self.mechanic_notes_var = tk.StringVar(value="")
        ttk.Entry(edit_row, textvariable=self.mechanic_notes_var, width=28).pack(side="left")
        ttk.Button(
            edit_row,
            text="Save calibration",
            command=self._save_ttk_mechanic,
            style="Primary.TButton",
        ).pack(side="left", padx=(8, 0))
        self._ttk_row_map: dict[str, dict[str, Any]] = {}

    def refresh(self) -> None:
        self.project_root_label.configure(text=f"project root: {self.adapter.project_root}")
        self.output_root_label.configure(text=f"output root: {self.adapter.output_root}")
        self._refresh_calibration_summary()
        self.submit_poll("godot-status", self.adapter.system_status, self._on_system_status)
        if hasattr(self.adapter, "ttk_testing_status"):
            shortcut = self.roblox_shortcut_var.get().strip() or None
            self.submit_poll(
                "settings-ttk-status",
                lambda: self.adapter.ttk_testing_status(shortcut),
                self._on_ttk_status,
            )

    def _on_ttk_status(
        self, status: dict[str, Any] | None, error: BaseException | None
    ) -> None:
        if error is not None or status is None:
            return
        tview = vm.ttk_testing_view(status)
        self.roblox_live_label.configure(
            text=(
                f"{tview['status_badge']}   ·   Launcher: {tview['launcher_text']}   ·   "
                f"Window: {tview['window_text']}   ·   Place: {tview['place_text']}   ·   "
                f"Progress: {tview['calibration_progress_text']}"
            ),
            foreground=COLOR_OK if tview["connected"] else (COLOR_WARN if tview["roblox_running"] else COLOR_MUTED),
        )
        selected_mech = self._selected_mechanic
        self.ttk_tree.delete(*self.ttk_tree.get_children())
        self._ttk_row_map.clear()
        for row in tview["rows"]:
            st = row["status"]
            tag = (
                "ttk-verified"
                if "VERIFIED" in st or st == "CALIBRATED"
                else ("ttk-pending" if "NEEDS" in st else "ttk-excluded")
            )
            item_id = self.ttk_tree.insert(
                "",
                "end",
                tags=(tag,),
                values=(
                    row["mechanic"],
                    row["status"],
                    row["measured_value"],
                    row["rule"],
                    row["source"],
                ),
            )
            self._ttk_row_map[item_id] = row
            if row["mechanic"] == selected_mech:
                self.ttk_tree.selection_set(item_id)

    def _focus_roblox_window(self) -> None:
        if not hasattr(self.adapter, "focus_roblox_window"):
            return

        def _done(res: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not res or not res.get("ok"):
                self.app.set_status(
                    f"Focus Roblox window: {error or (res or {}).get('message')}", error=True
                )
            else:
                self.app.set_status(str(res.get("message") or "Roblox window focused"))
                self.refresh()

        self.app.background.submit(self.adapter.focus_roblox_window, _done)

    def _open_captures_folder(self) -> None:
        captures_dir = Path(self.adapter.project_root) / ".sandboxai" / "ttk_captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
        _open_in_file_manager(captures_dir)

    def _recalc_ttk_lab(self) -> None:
        try:
            dmg = float(self.ttk_dmg_var.get())
            rpm = float(self.ttk_rpm_var.get())
            hp = float(self.ttk_hp_var.get())
        except ValueError:
            self.app.set_status("DMG, RPM and HP must be numeric", error=True)
            return
        from .ttk_testing import calculate_ttk_metrics

        metrics = calculate_ttk_metrics(damage=dmg, rpm=rpm, target_hp=hp)
        summary = (
            f"{metrics['shots_to_kill']} STK ({metrics['headshots_to_kill']} HS)   ·   "
            f"{metrics['ttk_ms']:.1f} ms TTK   ·   "
            f"{metrics['burst_dps']:.1f} Burst DPS ({metrics['sustained_dps']:.1f} Sust.)   ·   "
            f"{metrics['pace_label']}"
        )
        self.ttk_calc_result_label.configure(text=summary, foreground=COLOR_OK)
        if self._selected_mechanic == "weapon_damage_and_rpm_ttk_curve":
            self.mechanic_value_var.set(summary)

    def _apply_ttk_preset(self, preset_id: str) -> None:
        if not hasattr(self.adapter, "apply_ttk_preset"):
            return

        def _done(res: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not res or not res.get("ok"):
                self.app.set_status(f"Preset failed: {error}", error=True)
                return
            metrics = res.get("metrics") or {}
            if metrics:
                self.ttk_dmg_var.set(str(metrics.get("damage", 34)))
                self.ttk_rpm_var.set(str(metrics.get("rpm", 750)))
                self._recalc_ttk_lab()
            self.app.set_status(f"Applied TTK preset: {res.get('label')}")
            self.refresh()

        self.app.background.submit(lambda: self.adapter.apply_ttk_preset(preset_id), _done)

    def _activate_ubuntu_cpu_turbo(self) -> None:
        if not hasattr(self.adapter, "enable_ubuntu_cpu_turbo"):
            return

        def _done(res: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not res:
                self.app.set_status(f"Ubuntu CPU Turbo failed: {error}", error=True)
                return
            uview = vm.ubuntu_cpu_turbo_view(res)
            self.cpu_turbo_label.configure(
                text=f"Ubuntu CPU Turbo: {uview['badge']} — {uview['summary']}",
                foreground=COLOR_OK,
            )
            self.app.set_status(f"Activated {uview['badge']}")

        self.app.background.submit(self.adapter.enable_ubuntu_cpu_turbo, _done)

    def _on_select_ttk_mechanic(self, _event: object) -> None:
        sel = self.ttk_tree.selection()
        if not sel:
            return
        row = self._ttk_row_map.get(sel[0])
        if not row:
            return
        self._selected_mechanic = str(row["mechanic"])
        self.mechanic_label.configure(text=f"Mechanic: {self._selected_mechanic}")
        val = str(row.get("measured_value") or "")
        self.mechanic_value_var.set("" if val == "—" else val)
        self.mechanic_notes_var.set(str(row.get("notes") or ""))

    def _save_ttk_mechanic(self) -> None:
        if not self._selected_mechanic or not hasattr(self.adapter, "save_ttk_calibration"):
            return
        mech = self._selected_mechanic
        val = self.mechanic_value_var.get()
        notes = self.mechanic_notes_var.get()

        def _done(res: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or res is None:
                self.app.set_status(f"Failed to save TTK calibration: {error}", error=True)
            else:
                self.app.set_status(f"Saved TTK calibration for '{mech}'")
                self.refresh()

        self.app.background.submit(
            lambda: self.adapter.save_ttk_calibration(mech, val, notes=notes),
            _done,
        )

    def _launch_roblox(self, *, direct_place: bool) -> None:
        if not hasattr(self.adapter, "launch_roblox_ttk_testing"):
            return
        shortcut = self.roblox_shortcut_var.get().strip() or None

        def _done(res: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not res or not res.get("ok"):
                self.app.set_status(
                    f"Roblox launch failed: {error or (res or {}).get('error')}", error=True
                )
            else:
                self.app.set_status(str(res.get("message") or "Launched Roblox"))
                self.refresh()

        self.app.background.submit(
            lambda: self.adapter.launch_roblox_ttk_testing(shortcut, direct_place=direct_place),
            _done,
        )

    def _capture_ttk_screenshot(self) -> None:
        if not hasattr(self.adapter, "capture_roblox_screenshot"):
            return

        def _done(res: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not res or not res.get("ok"):
                self.app.set_status(
                    f"Screenshot failed: {error or (res or {}).get('error')}", error=True
                )
            else:
                self.app.set_status(f"Saved screenshot: {res.get('path')}")
                if self._selected_mechanic:
                    self.mechanic_notes_var.set(f"screenshot: {res.get('path')}")

        self.app.background.submit(self.adapter.capture_roblox_screenshot, _done)

    def _refresh_calibration_summary(self) -> None:
        rec = None
        if hasattr(self.adapter, "recommended_configuration"):
            with contextlib.suppress(OSError):
                rec = self.adapter.recommended_configuration()
        rec_view = vm.benchmark_recommendation_view(rec)
        if rec_view.get("available"):
            applied_str = (
                f"applied ({rec_view['applied_utc']})"
                if rec_view.get("applied_utc")
                else "not yet applied"
            )
            self.calibration_label.configure(
                text=f"Optimal topology: {rec_view['summary']} — {applied_str}",
                foreground=COLOR_OK,
            )
        else:
            self.calibration_label.configure(
                text="No benchmark recommendation persisted yet — run the automatic benchmark on the Benchmarks page.",
                foreground=COLOR_MUTED,
            )
        if hasattr(self.adapter, "ubuntu_cpu_status") and hasattr(self, "cpu_turbo_label"):
            with contextlib.suppress(Exception):
                uview = vm.ubuntu_cpu_turbo_view(self.adapter.ubuntu_cpu_status())
                self.cpu_turbo_label.configure(
                    text=f"Ubuntu CPU Turbo: {uview['badge']} — {uview['summary']}",
                    foreground=COLOR_OK if uview["anti_thrash_active"] else COLOR_ACCENT,
                )

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

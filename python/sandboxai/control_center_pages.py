"""Desktop Control Center pages.

Each page translates UI intent into adapter calls; formatting and validation
remain in the independently tested viewmodel.
"""

from __future__ import annotations

import contextlib
import threading
import tkinter as tk
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from . import control_center_viewmodel as vm
from .adapter import SandboxAIAdapter
from .control_center_layout import (
    LayoutState,
    WidgetSpec,
    move,
    normalize,
    reset_page,
    set_span,
    set_visible,
)
from .control_center_theme import (
    ACCENT_PRESETS,
    DENSITIES,
    LAYOUT_MODES,
    MOTION_LEVELS,
    THEME_NAMES,
    THEMES,
    Theme,
    normalize_accent,
)
from .control_center_ui import (
    LayoutBoard,
    ScrollArea,
    SegmentedControl,
    StatusDot,
)
from .control_center_widgets import (
    LineChart,
    LogPanel,
    PhaseStepper,
    StatRow,
    ToolTip,
    _open_in_file_manager,
    _scrollable_table,
    refresh_table_empty,
)


def state_colors(theme: Theme) -> dict[str, str]:
    """Lifecycle -> colour map for the current theme (built per theme change)."""
    return {
        "Running": theme.ok,
        "Starting": theme.warn,
        "Stopping": theme.warn,
        "Paused": theme.warn,
        "Finished": theme.text_muted,
        "Error": theme.error,
        "running": theme.ok,
        "finished": theme.text_muted,
        "failed": theme.error,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class Page(ttk.Frame):
    """Base class for one navigable page.

    A page owns its adapter calls, its widgets and its polling gate. It never
    hard-codes a colour: every value comes from ``self.palette`` so a theme
    switch is a repaint rather than a rewrite, and pages that colour table
    tags/canvas text implement :meth:`on_theme` to re-apply them.
    """

    title = ""
    subtitle = ""
    #: Movable cards this page exposes to the layout studio.
    widgets: tuple[WidgetSpec, ...] = ()
    #: Columns the layout board packs the cards into.
    board_columns = 3

    def __init__(self, parent: tk.Misc, app: Any) -> None:
        super().__init__(parent, padding=app.px(18, minimum=10))
        self.app = app
        # Cached ThemeBus for helper widgets that discover it by walking up
        # from themselves (see ToolTip) - a page constructs dozens of small
        # widgets and forgetting `bus=` used to leave them on the default
        # Corz palette forever.
        self._cc_bus = app.bus
        # Typed deliberately: ``app`` is an untyped shell, but the adapter is
        # a real class, so every ``self.adapter.<call>`` on the pages is
        # checked against the adapter's public API. That is what catches a
        # renamed adapter method on a machine where the Tk suite cannot run.
        self.adapter: SandboxAIAdapter = app.adapter
        self._built = False
        self._board: LayoutBoard | None = None
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
        # Treeview tag colours are the one colour that is not carried by a
        # ttk style, so a page has to re-apply them itself on a theme switch.
        # Recording the (tree, tag, role) triples here means a page cannot
        # forget: ``on_theme`` replays the list.
        self._tag_roles: list[tuple[Any, str, str]] = []

    @property
    def palette(self) -> Theme:
        """The live theme (alias kept short because pages use it constantly)."""
        return self.app.palette

    def tag_style(self, tree: Any, tag: str, role: str) -> None:
        """Colour a Treeview tag from the live palette and keep it themed.

        Idempotent: a page may call this from a poll handler (the Evaluations
        table does, once per tick), and recording the same triple twice would
        grow ``_tag_roles`` without bound - a slow leak that a theme switch
        then replays entry by entry onto the same tree.
        """
        record = (tree, tag, role)
        if record not in self._tag_roles:
            self._tag_roles.append(record)
        self._apply_tag_style(tree, tag, role)

    def _apply_tag_style(self, tree: Any, tag: str, role: str) -> None:
        with contextlib.suppress(tk.TclError):
            tree.tag_configure(tag, foreground=self.palette.color(role))

    def show(self) -> None:
        if not self._built:
            # A rebuild (density change, shell change) destroys the trees that
            # the tag-colour records point at. Dropping the records with them
            # keeps ``on_theme`` from replaying colours onto dead widgets, and
            # keeps the list from growing once per rebuild.
            self._tag_roles.clear()
            self._heading()
            self.build()
            self._built = True
        self.refresh()

    def _heading(self) -> None:
        """The page header: title, status pill, subtitle and a hint slot.

        The subtitle sits on its own wrapped line instead of trailing the
        title on the same row: a long one used to be clipped as soon as the
        window was narrow, which is exactly when an operator needs to read
        what a page is for.
        """
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, self.app.px(14, minimum=8)))
        self._heading_frame = header
        row = ttk.Frame(header)
        row.pack(fill="x")
        ttk.Label(row, text=self.title, style="PageTitle.TLabel").pack(side="left")
        self._heading_pill = ttk.Label(row, text="", style="Pill.TLabel")
        self._heading_pill.pack(side="right")
        if self.subtitle:
            ttk.Label(
                header,
                text=self.subtitle,
                style="PageSubtitle.TLabel",
                justify="left",
                wraplength=self.app.px(1180, minimum=420),
            ).pack(anchor="w", fill="x", pady=(self.app.px(4, minimum=2), 0))
        tk.Frame(header, height=1, background=self.palette.border, borderwidth=0).pack(
            fill="x", pady=(self.app.px(10, minimum=6), 0)
        )
        # The hint is created here (so it always packs *below* the heading)
        # and stays hidden until a page has something to explain - an empty
        # page reads like a broken one, and a page that says "no runs yet,
        # start one here" does not.
        self._hint = ttk.Label(
            self,
            text="",
            style="Hint.TLabel",
            justify="left",
            anchor="w",
            wraplength=self.app.px(1180, minimum=420),
        )

    def set_hint(self, text: str) -> None:
        """Explain an empty page in one line (``""`` hides the hint)."""
        hint = getattr(self, "_hint", None)
        if hint is None:
            return
        with contextlib.suppress(tk.TclError):
            hint.configure(text=text)
            if not text:
                hint.pack_forget()
                return
            if hint.winfo_manager():
                return
            body = next(
                (
                    child
                    for child in self.winfo_children()
                    if child is not getattr(self, "_heading_frame", None)
                ),
                None,
            )
            if body is not None:
                hint.pack(
                    fill="x",
                    before=body,
                    pady=(0, self.app.px(10, minimum=5)),
                )

    def set_heading_pill(self, text: str, kind: str = "") -> None:
        """Update the small status pill next to the page title."""
        style = {"ok": "PillOk.TLabel", "warn": "PillWarn.TLabel", "error": "PillError.TLabel"}.get(
            kind, "Pill.TLabel"
        )
        pill = getattr(self, "_heading_pill", None)
        if pill is None:
            return
        with contextlib.suppress(tk.TclError):
            pill.configure(text=text, style=style)

    def board(self, parent: tk.Misc, specs: tuple[WidgetSpec, ...] | None = None) -> LayoutBoard:
        """Create this page's layout board (cards the operator can rearrange).

        The board is attached *here*, not by the caller. A page used to build
        and fill its board and then never hand it to a geometry manager, so
        the page rendered its heading and stopped: the cards were laid out
        inside an unmapped frame, which is why Dashboard/Training/Benchmarks
        looked empty under every heading.
        """
        board = LayoutBoard(
            parent,
            self.app.layout_bus,
            page_key=self.title,
            specs=specs or self.widgets,
            columns=self.board_columns,
            gap=self.app.px(12, minimum=6),
            min_column_width=self.app.px(320, minimum=260),
        )
        self._board = board
        board.pack(fill="both", expand=True)
        return board

    def card(
        self,
        parent: tk.Misc,
        title: str,
        subtitle: str = "",
        *,
        accent: bool = True,
        nested: bool = False,
    ) -> Any:
        """A themed rounded card whose ``.body`` frame hosts content.

        ``nested=True`` is for a card that lives *inside* another card: its
        rounded corners have to be painted in the surface it actually sits
        on (the parent card), not in the page background, or every corner
        shows a small square of a different colour.
        """
        from .control_center_ui import RoundedPanel

        return RoundedPanel(
            parent,
            self.app.bus,
            radius=self.app.prefs.radius,
            padding=self.app.px(14, minimum=10),
            title=title,
            subtitle=subtitle,
            accent=accent,
            background_role="card" if nested else "bg",
        )

    def on_theme(self, theme: Theme) -> None:
        """Re-apply colours that live outside ttk styles (canvas/tags/etc.)."""
        for tree, tag, role in getattr(self, "_tag_roles", ()):
            self._apply_tag_style(tree, tag, role)

    def rebuild_after_restyle(self) -> None:
        """Rebuild the widgets after a density change so paddings/fonts fit.

        Destroying every child is what makes a new density fit, but it used
        to wipe what the operator was looking at: the log they were
        reading, the selected row, the page's scroll offset. The transient
        view state is captured first and applied to the fresh widgets
        afterwards, so a density change feels like a restyle and not like a
        reload.
        """
        if not self._built:
            return
        view_state = self._capture_view_state()
        for child in self.winfo_children():
            child.destroy()
        self._built = False
        self.show()
        self._restore_view_state(view_state)

    def _capture_view_state(self) -> dict[str, Any]:
        """Everything a rebuild should put back, keyed by attribute name."""
        logs: dict[str, dict[str, Any]] = {}
        selections: dict[str, tuple[str, ...]] = {}
        scroll: list[float] = []
        for name, value in vars(self).items():
            if isinstance(value, LogPanel):
                with contextlib.suppress(tk.TclError):
                    logs[name] = value.snapshot_view()
            elif isinstance(value, ttk.Treeview):
                with contextlib.suppress(tk.TclError):
                    selections[name] = tuple(value.selection())
        for area in self._scroll_areas():
            with contextlib.suppress(tk.TclError):
                scroll.append(float(area.canvas.yview()[0]))
        return {
            "logs": logs,
            "selections": selections,
            "scroll": scroll,
            "focus": self._focused_attribute(),
        }

    def _focused_attribute(self) -> str:
        """Attribute name of the focused widget, when it is one of ours.

        After a rebuild the operator should keep typing where they were
        (the log, a table, a form field), so the keyboard focus is restored
        to the same *element*, not to whatever happens to be first.
        """
        try:
            focused = self.focus_get()
        except (tk.TclError, KeyError):  # a destroyed widget has no focus
            return ""
        if focused is None:
            return ""
        for name, value in vars(self).items():
            if value is focused:
                return name
        return ""

    def _restore_view_state(self, state: dict[str, Any]) -> None:
        focus_name = str(state.get("focus") or "")
        if focus_name:
            widget = getattr(self, focus_name, None)
            if widget is not None:
                with contextlib.suppress(tk.TclError):
                    widget.focus_set()
        for name, snapshot in state["logs"].items():
            panel = getattr(self, name, None)
            if isinstance(panel, LogPanel):
                panel.restore_view(snapshot)
        for name, selection in state["selections"].items():
            tree = getattr(self, name, None)
            if not selection or not isinstance(tree, ttk.Treeview):
                continue
            present = [item for item in selection if tree.exists(item)]
            if present:
                tree.selection_set(present)
        areas = self._scroll_areas()
        for index, fraction in enumerate(state["scroll"]):
            if fraction <= 0.0 or index >= len(areas):
                continue
            # The scroll region only exists after the next layout pass, so
            # the offset is applied on idle rather than right now.
            self._restore_scroll_offset(areas[index], fraction)

    def _restore_scroll_offset(self, area: ScrollArea, fraction: float) -> None:
        """Apply a saved scroll fraction once the rebuilt page has laid out."""
        self.after(0, self._apply_scroll_offset, area, fraction)  # type: ignore[arg-type]

    @staticmethod
    def _apply_scroll_offset(area: ScrollArea, fraction: float) -> None:
        with contextlib.suppress(tk.TclError):
            area.canvas.yview_moveto(fraction)

    def _scroll_areas(self) -> list[ScrollArea]:
        """Every ScrollArea in this page, in build order (stable per page)."""
        found: list[ScrollArea] = []
        pending = list(self.winfo_children())
        while pending:
            widget = pending.pop(0)
            if isinstance(widget, ScrollArea):
                found.append(widget)
            pending.extend(widget.winfo_children())
        return found

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


class ActiveRunStrip(ttk.Frame):
    """Compact list of live processes with a Stop button, shown on the Dashboard.

    The former Agents page was the only place a benchmark/evaluation process
    could be stopped from. With lifecycle actions moved to the Training page
    (training-only), this strip keeps every *other* running process reachable
    without reintroducing a second process table.
    """

    LIVE = ("LAUNCHING", "RUNNING", "PAUSED", "STOPPING", "RESTARTING")

    def __init__(self, parent: tk.Misc, app: Any) -> None:
        super().__init__(parent)
        self.app = app
        self.adapter = app.adapter
        self._rows: dict[str, tk.Widget] = {}
        self.empty_label = ttk.Label(
            self, text="No process running right now.", style="FieldHelp.TLabel"
        )
        self.empty_label.pack(anchor="w")
        self._views: list[dict[str, Any]] = []

    def update_views(self, views: list[dict[str, Any]]) -> None:
        """Re-render the strip from the adapter's agent views."""
        self._views = views
        live = [
            view
            for view in views
            if str(view.get("lifecycle")) in self.LIVE and view.get("kind") != "training"
        ]
        for child in self._rows.values():
            child.destroy()
        self._rows.clear()
        if not live:
            self.empty_label.pack(anchor="w")
            return
        self.empty_label.pack_forget()
        for view in live:
            self._rows[str(view.get("agent_id"))] = self._build_row(view)

    def _build_row(self, view: dict[str, Any]) -> tk.Widget:
        row = ttk.Frame(self, style="CardInner.TFrame")
        row.pack(fill="x", pady=2)
        dot = StatusDot(row, self.app.bus, size=self.app.px(9, minimum=7), background_role="card")
        dot.pack(side="left", padx=(0, self.app.px(6, minimum=3)))
        lifecycle = str(view.get("lifecycle", ""))
        dot.set_state(
            self.app.color("ok") if lifecycle == "RUNNING" else self.app.color("warn"),
            pulse=lifecycle in ("RUNNING", "LAUNCHING"),
            motion=self.app.motion,
        )
        label = ttk.Label(
            row,
            text=f"{view.get('name') or view.get('kind')}  ·  {lifecycle.lower()}",
            style="CardLabel.TLabel",
        )
        label.pack(side="left")
        agent_id = str(view.get("agent_id") or "")
        kind = str(view.get("kind") or "")
        ttk.Button(
            row,
            text="Stop",
            style="Ghost.TButton",
            command=lambda: self._stop(agent_id),
        ).pack(side="right")
        ttk.Button(
            row,
            text="Details",
            style="Ghost.TButton",
            command=lambda: self._open_page(kind),
        ).pack(side="right", padx=(0, self.app.px(6, minimum=3)))
        return row

    def _stop(self, agent_id: str) -> None:
        def done(_result: Any, error: BaseException | None) -> None:
            if error is not None:
                self.app.set_status(f"Stop failed: {error}", error=True)
            else:
                self.app.set_status("Stop requested — the process ends at its next safe boundary")

        self.app.background.submit(lambda: self.adapter.agents.stop(agent_id), done)

    def _open_page(self, kind: str) -> None:
        page = {"benchmark": "Benchmarks", "evaluation": "Evaluations"}.get(kind, "Training")
        if page in self.app.pages:
            self.app.show_page(page)


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Live state of the newest run, the active processes and this machine."

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

    widgets = (
        WidgetSpec(
            "kpis",
            "KPI row",
            "Run state, progress, throughput, resources.",
            default_span=3,
            max_span=3,
            removable=False,
        ),
        WidgetSpec("workflow", "Workflow", "Roblox/TTK bridge plus the one-click run chain."),
        WidgetSpec(
            "run_insight", "Run insight", "Checkpoints, PPO health and convergence.", default_span=2
        ),
        WidgetSpec(
            "charts",
            "Telemetry charts",
            "Reward, throughput and PPO KL history.",
            default_span=3,
            max_span=3,
        ),
        WidgetSpec("active_runs", "Active processes", "Everything running besides this window."),
        WidgetSpec(
            "quick_actions", "Quick actions", "Open folders, resume, evaluate.", default_span=2
        ),
    )

    def build(self) -> None:
        area = ScrollArea(
            self,
            self.app.bus,
            style="Content.TFrame",
            scale_px=self.app.px,
        )
        area.pack(fill="both", expand=True)
        # The warning banner is packed before the board on purpose: both share
        # ``area.body``, and ``Page.board()`` attaches the board immediately,
        # so a banner built afterwards would sit *under* every card instead of
        # above them where a run warning belongs.
        self._warning_banner = ttk.Label(
            area.body,
            text="",
            style="Warning.TLabel",
            wraplength=self.app.px(1100, minimum=600),
            justify="left",
        )
        self._warning_banner.pack(fill="x", pady=(self.app.px(6, minimum=3), 0))
        self.warning_banner = self._warning_banner
        board = self.board(area.body)
        board.add("kpis", self._build_kpis)
        board.add("workflow", self._build_workflow)
        board.add("run_insight", self._build_run_insight)
        board.add("charts", self._build_charts)
        board.add("active_runs", self._build_active_runs)
        board.add("quick_actions", self._build_quick_actions)
        board.rebuild()
        self._last_run_dir: str | None = None

    # -- cards ------------------------------------------------------------

    def _build_kpis(self, parent: tk.Misc) -> tk.Widget:
        self.stats = StatRow(
            parent,
            self.STAT_LABELS,
            max_columns=5,
            bus=self.app.bus,
            motion=self.app.motion,
        )
        self.stats.pack(fill="x")
        return self.stats

    def _build_workflow(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Workflow", "1. TTK bridge   ·   2. benchmark   ·   3. training")
        self.roblox_bridge_label = ttk.Label(
            card.body,
            text="Probing Roblox Player & TTK Testing…",
            style="FieldTitle.TLabel",
            wraplength=280,
        )
        self.roblox_bridge_label.pack(anchor="w", pady=(0, self.app.px(8, minimum=4)))
        buttons = ttk.Frame(card.body, style="CardInner.TFrame")
        buttons.pack(fill="x")
        for label, command, style in (
            ("Launch TTK Testing", self._launch_roblox_ttk, "Primary.TButton"),
            ("Focus window", self._focus_roblox_window, "Ghost.TButton"),
            ("Screenshot", self._capture_roblox_window, "Ghost.TButton"),
            ("CPU turbo", self._enable_ubuntu_cpu_turbo, "Ghost.TButton"),
            ("Run benchmark", lambda: self.app.show_page("Benchmarks"), "TButton"),
            ("Open training", lambda: self.app.show_page("Training"), "TButton"),
        ):
            ttk.Button(buttons, text=label, command=command, style=style).pack(
                fill="x", pady=self.app.px(2, minimum=1)
            )
        return card

    def _build_run_insight(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Checkpoints, PPO diagnostics & convergence")
        self.checkpoints_label = ttk.Label(
            card.body, text="n/a", justify="left", style="CardLabel.TLabel"
        )
        self.checkpoints_label.pack(anchor="w")
        self.convergence_label = ttk.Label(
            card.body, text="", justify="left", style="FieldHelp.TLabel"
        )
        self.convergence_label.pack(anchor="w", pady=(self.app.px(6, minimum=3), 0))
        return card

    def _build_charts(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Live telemetry", "Bounded history, nothing estimated")
        grid = ttk.Frame(card.body, style="CardInner.TFrame")
        grid.pack(fill="both", expand=True)
        self.reward_chart = LineChart(
            grid, "Mean episode reward vs. timesteps", bus=self.app.bus, color=self.app.color("ok")
        )
        self.fps_chart = LineChart(
            grid, "Steps/second vs. timesteps", bus=self.app.bus, color=self.app.color("accent")
        )
        self.kl_chart = LineChart(
            grid, "PPO approx. KL vs. timesteps", bus=self.app.bus, color=self.app.color("warn")
        )
        for index, chart in enumerate((self.reward_chart, self.fps_chart, self.kl_chart)):
            chart.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=self.app.px(5, minimum=2),
                pady=self.app.px(5, minimum=2),
            )
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)
        return card

    def _build_active_runs(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Active processes", "Benchmarks & evaluations")
        self.active_runs = ActiveRunStrip(card.body, self.app)
        self.active_runs.pack(fill="x")
        return card

    def _build_quick_actions(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Quick actions")
        for label, command in (
            ("Open run folder", self._open_run_folder),
            ("Open in Runs / Checkpoints", self._open_in_runs),
            ("Resume latest in Training", self._resume_in_training),
            ("Evaluate best checkpoint", self._evaluate_latest_best),
        ):
            ttk.Button(card.body, text=label, command=command, style="Ghost.TButton").pack(
                fill="x", pady=self.app.px(2, minimum=1)
            )
        return card

    # -- polling ----------------------------------------------------------

    def refresh(self) -> None:
        self.submit_poll("dashboard", self.adapter.dashboard_snapshot, self._on_snapshot)
        self.submit_poll("agents-summary", self.adapter.agents.views, self._on_agents_summary)
        if hasattr(self.adapter, "ttk_testing_status"):
            self.submit_poll(
                "roblox-bridge", self.adapter.ttk_testing_status, self._on_roblox_status
            )

    def _on_roblox_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
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
            foreground=self.app.color("ok")
            if tview["connected"]
            else (self.app.color("warn") if tview["roblox_running"] else self.app.color("text")),
        )

    def _launch_roblox_ttk(self) -> None:
        if not hasattr(self.adapter, "launch_roblox_ttk_testing"):
            return

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            if error is not None or not result or not result.get("ok"):
                self.app.set_status(
                    f"Roblox launch failed: {error or (result or {}).get('error')}", error=True
                )
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
            color = self.app.color("error") if counts.get("FAILED") else None
        if getattr(self, "stats", None) is not None:
            self.stats.update_values({"agents": (text, color)})
        if getattr(self, "active_runs", None) is not None:
            self.active_runs.update_views(views)

    def _on_snapshot(self, snapshot: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or snapshot is None:
            self.report_error("Dashboard refresh failed", error or RuntimeError("unknown error"))
            return
        view = vm.dashboard_view(snapshot)
        self.set_hint(
            ""
            if view["run_id"]
            else (
                "No runs yet. The workflow card starts the TTK bridge, a benchmark "
                "and a training run - in that order."
            )
        )
        previous_run_dir = self._last_run_dir
        self._last_run_dir = view.get("run_dir")
        if self._last_run_dir != previous_run_dir:
            # Never temporarily chart a prior run while the next telemetry
            # read is still in flight for the newly discovered run.
            for chart in (self.reward_chart, self.fps_chart, self.kl_chart):
                chart.set_points([])
        state = view["state"] or ""
        state_color = state_colors(self.palette).get(state)
        progress_text = (
            f"{vm.format_fraction_as_percent((view['progress_percent'] or 0) / 100.0)}"
            f" ({vm.format_number(view['timesteps'])}/{vm.format_number(view['target_timesteps'])})"
            if view["progress_percent"] is not None
            else "n/a"
        )
        self.stats.update_values(
            {
                "state": (view["state"] or "no runs yet", state_color),
                "run id": (view["run_id"] or "n/a", None),
                "progress": (progress_text, None),
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
            },
            numeric={
                "fps": (
                    float(view["fps"]) if isinstance(view["fps"], (int, float)) else 0.0,
                    vm.format_number(view["fps"], 1),
                    lambda value: vm.format_number(value, 1),
                )
            }
            if isinstance(view["fps"], (int, float))
            else None,
        )
        self.set_heading_pill(
            f"{view['state'] or 'no run'}"
            + (
                f"  ·  {vm.format_fraction_as_percent((view['progress_percent'] or 0) / 100.0)}"
                if view["progress_percent"] is not None
                else ""
            ),
            "error"
            if view["stale"]
            else ("warn" if (view["warnings"] or view["problems"]) else "ok"),
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
        conv_kind = {"IMPROVING": "ok", "PLATEAU": "warn", "REGRESSING": "error"}.get(
            conv["state"], ""
        )
        self.convergence_label.configure(
            text=f"Convergence radar: [{conv['badge']}] — {conv['recommendation']}",
            foreground=self.app.color(conv_kind) if conv_kind else self.palette.text_dim,
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

    def _resume_in_training(self) -> None:
        if not self._last_run_dir:
            messagebox.showinfo("No run yet", "No training run found to resume.")
            return
        latest = Path(self._last_run_dir) / "checkpoints" / "latest.zip"
        best = Path(self._last_run_dir) / "checkpoints" / "best.zip"
        ckpt = latest if latest.is_file() else (best if best.is_file() else None)
        self.app.show_page("Training")
        page = self.app.pages.get("Training")
        if page is not None and ckpt is not None and hasattr(page, "resume_checkpoint_var"):
            page.resume_checkpoint_var.set(str(ckpt))
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


class TrainingPage(Page):
    """Training launch deck + the lifecycle of every training agent.

    This page owns the training configuration (the Benchmarks page applies
    its recommendation here) and the *training* half of the process registry:
    one row per launched training agent with its Environment -> Worker
    topology, live backend metrics, the full action set (Pause/Resume, Stop,
    Restart from the newest checkpoint, Force stop) and a bounded log.
    Benchmarks and evaluations keep their own pages; any other live process
    stays reachable through the Dashboard's active-process strip.

    The budget is selected with a segmented control: **Steps** ends the run at
    an exact step count, **Time** lets the window request the cooperative stop
    once the published elapsed time reaches the budget, after which the
    trainer saves its final checkpoint at the next safe boundary.
    """

    title = "Training"
    subtitle = "Launch, budget, operate and restart headless training runs."

    COLUMNS = (
        ("name", "Run", 150),
        ("lifecycle", "Lifecycle", 100),
        ("pid", "PID", 70),
        ("environment_count", "Envs", 55),
        ("env_workers", "Workers", 65),
        ("device", "Device", 60),
        ("timesteps", "Steps", 90),
        ("progress", "Progress", 80),
        ("steps_per_second", "Steps/s", 80),
        ("mean_episode_reward", "Reward", 80),
        ("budget", "Budget", 90),
        ("started", "Started", 130),
        ("error", "Error", 200),
    )

    widgets = (
        WidgetSpec(
            "launch",
            "Launch deck",
            "Environments, workers, device, budget (Steps or Time) and the verdict.",
            default_span=3,
            max_span=3,
            removable=False,
        ),
        WidgetSpec(
            "runs",
            "Training runs",
            "Every training agent of this session with its lifecycle actions.",
            default_span=3,
            max_span=3,
            removable=False,
        ),
        WidgetSpec(
            "topology", "Worker topology", "Environment -> Worker shard plan of the selection."
        ),
        WidgetSpec("log", "Run log", "Bounded stdout/stderr of the selected run.", default_span=2),
    )

    def build(self) -> None:
        area = ScrollArea(self, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        board = self.board(area.body)
        board.add("launch", self._build_launch_card)
        board.add("runs", self._build_runs_card)
        board.add("topology", self._build_topology_card)
        board.add("log", self._build_log_card)
        board.rebuild()

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
        # Time budget bookkeeping: which agent this window must stop, and
        # whether the cooperative stop has already been sent.
        self._budget_agent_id: str | None = None
        self._budget_stop_sent = False
        self._budget_label_by_agent: dict[str, str] = {}

    # -- cards ------------------------------------------------------------

    def _build_launch_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent, "Launch deck", "Measured defaults, validated against the real trainer"
        )
        form_frame = card.body
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
        # remain preserved even though the GUI exposes only the core controls.
        self._measured_sps: float | None = None
        if recommendation and isinstance(
            recommendation.get("expected_steps_per_second"), (int, float)
        ):
            self._measured_sps = float(recommendation["expected_steps_per_second"])
        for spec in vm.TRAINING_FIELDS:
            self.field_vars[spec.name] = tk.StringVar(value=defaults.get(spec.name, ""))

        basic_frame = ttk.Frame(form_frame, style="CardInner.TFrame")
        basic_frame.pack(fill="x")
        self._build_fields(basic_frame, vm.launch_field_specs(), defaults)

        # ---- Budget: Steps | Time -------------------------------------
        budget_row = ttk.Frame(form_frame, style="CardInner.TFrame")
        budget_row.pack(fill="x", pady=(self.app.px(10, minimum=6), 0))
        ttk.Label(budget_row, text="Budget", style="FieldTitle.TLabel").pack(
            side="left", padx=(0, 8)
        )
        self.budget_mode = SegmentedControl(
            budget_row,
            self.app.bus,
            vm.BUDGET_MODES,
            value="steps",
            on_change=self._on_budget_mode,
            motion=self.app.motion,
            height=self.app.px(30, minimum=24),
            width=self.app.px(180, minimum=140),
        )
        self.budget_mode.pack(side="left")
        ttk.Label(budget_row, text="Time budget (minutes)", style="FieldHelp.TLabel").pack(
            side="left", padx=(self.app.px(14, minimum=8), 6)
        )
        self.budget_minutes_var = tk.StringVar(value="45")
        self.budget_entry = ttk.Entry(
            budget_row, textvariable=self.budget_minutes_var, width=7, state="disabled"
        )
        self.budget_entry.pack(side="left")
        self.budget_hint = ttk.Label(
            budget_row, text="", style="FieldHelp.TLabel", wraplength=self.app.px(420, minimum=240)
        )
        self.budget_hint.pack(side="left", padx=(self.app.px(12, minimum=6), 0))
        self.budget_minutes_var.trace_add("write", lambda *_args: self._on_budget_values_changed())

        presets_bar = ttk.Frame(form_frame, style="CardInner.TFrame")
        presets_bar.pack(fill="x", pady=(self.app.px(10, minimum=6), 0))
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
                style="Ghost.TButton",
                command=lambda s=steps_val: self._apply_step_preset(s),  # type: ignore[misc]
            ).pack(side="left", padx=(0, 6))
        ttk.Button(
            presets_bar,
            text="Sync benchmark topology",
            style="Ghost.TButton",
            command=self._sync_optimal_benchmark,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            presets_bar,
            text="Ubuntu CPU turbo",
            style="Ghost.TButton",
            command=self._apply_ubuntu_cpu_turbo,
        ).pack(side="left", padx=(6, 0))

        resume_bar = ttk.Frame(form_frame, style="CardInner.TFrame")
        resume_bar.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        ttk.Label(resume_bar, text="Resume checkpoint (optional):", style="FieldTitle.TLabel").pack(
            side="left", padx=(0, 8)
        )
        self.resume_checkpoint_var = tk.StringVar(value="")
        self.resume_combo = ttk.Combobox(
            resume_bar,
            textvariable=self.resume_checkpoint_var,
            values=("",),
            width=52,
        )
        self.resume_combo.pack(side="left")
        ttk.Button(
            resume_bar,
            text="Clear (fresh run)",
            style="Ghost.TButton",
            command=lambda: self.resume_checkpoint_var.set(""),
        ).pack(side="left", padx=(6, 0))

        launch_bar = ttk.Frame(form_frame, style="CardInner.TFrame")
        launch_bar.pack(fill="x", pady=(self.app.px(10, minimum=6), 0))
        self.launch_status_label = ttk.Label(
            launch_bar,
            text="",
            justify="left",
            wraplength=self.app.px(760, minimum=420),
            style="CardLabel.TLabel",
        )
        self.launch_status_label.pack(side="left", fill="x", expand=True)
        self.launch_button = ttk.Button(
            launch_bar, text="Launch training", command=self._launch, style="Primary.TButton"
        )
        self.launch_button.pack(side="right")
        return card

    def _build_runs_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Training runs", "Real backend states only")
        table_frame = ttk.Frame(card.body, style="CardInner.TFrame")
        table_frame.pack(fill="both", expand=True)
        self.tree = _scrollable_table(
            table_frame,
            self.COLUMNS,
            bus=self.app.bus,
            empty_text=(
                "No training agents yet.\nSet the topology above and press Launch training."
            ),
        )
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self._apply_lifecycle_tags()
        actions = ttk.Frame(card.body, style="CardInner.TFrame")
        actions.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        self.pause_button = ttk.Button(actions, text="Pause", command=self._pause, state="disabled")
        self.pause_button.pack(side="left")
        self._pause_tooltip = ToolTip(
            self.pause_button,
            "Pause the training agent at its next safe boundary",
            bus=self.app.bus,
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
        self.clear_button = ttk.Button(
            actions, text="Clear exited", style="Ghost.TButton", command=self._clear
        )
        self.clear_button.pack(side="right", padx=(8, 0))
        return card

    def _apply_lifecycle_tags(self) -> None:
        for tag, role in (
            ("lifecycle-failed", "error"),
            ("lifecycle-running", "ok"),
            ("lifecycle-attention", "warn"),
            ("lifecycle-done", "text_muted"),
        ):
            self.tag_style(self.tree, tag, role)

    def _build_topology_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Environment → Worker topology")
        self.topology_label = ttk.Label(
            card.body,
            text="no run selected",
            justify="left",
            style="CardLabel.TLabel",
            wraplength=self.app.px(320, minimum=200),
        )
        self.topology_label.pack(anchor="nw")
        return card

    def _build_log_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Run log")
        self.log_panel = LogPanel(card.body, bus=self.app.bus, motion=self.app.motion)
        self.log_panel.pack(fill="both", expand=True)
        return card

    # -- budget -----------------------------------------------------------

    def _on_budget_mode(self, mode: str) -> None:
        time_mode = mode == "time"
        with contextlib.suppress(tk.TclError):
            self.budget_entry.configure(state="normal" if time_mode else "disabled")
        self._last_slot_values = None
        if time_mode:
            self.budget_hint.configure(
                text=(
                    "The trainer stops itself at the next step boundary after this budget "
                    "and still saves the final checkpoint; the window only steps in if a run "
                    "can no longer reach a safe boundary."
                )
            )
        else:
            self.budget_hint.configure(
                text="The trainer ends exactly at the configured step count."
            )
        self._update_launch_slot(self.current_values())

    def _on_budget_values_changed(self) -> None:
        self._last_slot_values = None

    def _apply_step_preset(self, steps: str) -> None:
        if self.budget_mode.get() != "steps":
            self.budget_mode.set("steps")
            self._on_budget_mode("steps")
        self.field_vars["total_training_steps"].set(steps)
        self._update_launch_slot(self.current_values())

    def current_budget(self) -> dict[str, Any]:
        """Validated budget view for the current selector state."""
        steps_var = self.field_vars.get("total_training_steps")
        return vm.budget_view(
            self.budget_mode.get(),
            steps_raw=steps_var.get() if steps_var is not None else "",
            minutes_raw=self.budget_minutes_var.get(),
            steps_per_second=getattr(self, "_measured_sps", None),
            checkpoint_frequency=None,
        )

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
                ToolTip(widget, spec.help, bus=self.app.bus)
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
        self._check_time_budget()
        process_id = self._selected_agent_process_id()
        if process_id:
            self._request_log(process_id)

    # -- time budget ------------------------------------------------------

    def _budget_seconds(self) -> float | None:
        """The configured Time budget in seconds, or ``None`` in Steps mode."""
        if self.budget_mode.get() != "time":
            return None
        try:
            minutes = float(self.budget_minutes_var.get().strip())
        except (TypeError, ValueError):
            return None
        return minutes * 60.0 if minutes > 0 else None

    def _check_time_budget(self) -> None:
        """Watchdog for a run that cannot reach its own budget stop.

        The trainer enforces ``max_train_minutes`` itself and writes the
        final checkpoint there, so this is the fallback path: only if the
        run is still alive well past the budget (a wedged engine, a bridge
        that never returns) does the window request a cooperative stop of
        its own. The elapsed value is the trainer's own published number,
        not a client-side clock, so a paused process does not burn budget.
        """
        seconds = self._budget_seconds()
        if seconds is None or self._budget_agent_id is None or self._budget_stop_sent:
            return
        grace = max(60.0, seconds * 0.1)
        view = next(
            (item for item in self._last_views if item.get("agent_id") == self._budget_agent_id),
            None,
        )
        if view is None or str(view.get("lifecycle")) not in ("RUNNING", "PAUSED"):
            return
        elapsed = view.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or float(elapsed) < seconds + grace:
            return
        self._budget_stop_sent = True
        agent_id = self._budget_agent_id
        self.app.notify(
            f"Time budget passed ({vm.format_duration(elapsed)}) and the run did not stop on "
            "its own — requesting the stop now",
            kind="warn",
            timeout_ms=6000,
        )

        def done(_result: Any, error: BaseException | None) -> None:
            if error is not None:
                self.app.set_status(f"Time-budget stop failed: {error}", error=True)

        self.app.background.submit(lambda: self.adapter.agents.stop(agent_id), done)

    def _on_resume_checkpoints(
        self, entries: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or entries is None:
            return
        paths = [""] + [str(e.get("path")) for e in entries if e.get("path")]
        self.resume_combo.configure(values=tuple(paths))

    def _update_launch_slot(self, values: dict[str, str]) -> None:
        slot = vm.launch_slot_view(values, self._compatibility)
        budget = self.current_budget()
        if slot["state"] == "AVAILABLE" and budget["errors"]:
            slot = dict(slot)
            slot["state"] = "INVALID"
            slot["errors"] = list(budget["errors"])
        elif slot["state"] == "AVAILABLE":
            slot = dict(slot)
            slot["warnings"] = list(slot["warnings"]) + list(budget["warnings"])
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
            eta_suffix = (
                f" (est. ~{eta} @ {vm.format_number(self._measured_sps, 1)} steps/s)" if eta else ""
            )
            budget_hint = budget["budget_line"]
            estimate = budget["estimate"] or eta_suffix.strip(" ()")
            if estimate:
                budget_hint += f"   ·   {estimate}"
            text = (
                f"AVAILABLE — {summary['environment_count']} environments / "
                f"{summary['env_workers']} workers ({topology}) on {summary['device']}\n"
                f"Budget: {budget_hint}   ·   step cap {vm.format_number(summary['total_training_steps'])}"
            )
            for warning in slot["warnings"]:
                text += f"\nwarning: {warning}"
            self.launch_status_label.configure(
                text=text,
                foreground=self.palette.warn if slot["warnings"] else self.palette.ok,
            )
            self.launch_button.configure(state="disabled" if self._launching else "normal")
        else:
            self.launch_status_label.configure(
                text="INVALID — " + "; ".join(slot["errors"]), foreground=self.palette.error
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
        values = self.current_values()
        # Time is a trainer-enforced budget, not just a window-side request:
        # the launch config carries the minutes so the run stops itself at
        # the next safe boundary even if this window is closed meanwhile.
        budget_seconds = self._budget_seconds()
        if budget_seconds is not None:
            values["max_train_minutes"] = f"{budget_seconds / 60.0:g}"
        try:
            config = vm.parse_training_form(values)
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
            messagebox.showerror("Training run could not start", str(error))
            return
        if view.get("lifecycle") == "FAILED":
            messagebox.showerror(
                "Training run failed to start", str(view.get("error") or "unknown error")
            )
            return
        self._budget_stop_sent = False
        self._budget_agent_id = (
            str(view.get("agent_id")) if self._budget_seconds() is not None else None
        )
        label = self.current_budget()["budget_line"]
        if self._budget_agent_id is not None and view.get("agent_id"):
            self._budget_label_by_agent[str(view.get("agent_id"))] = label
        self.app.set_status(
            f"Training run launched ({view.get('name', 'run')}) · {label}", toast=True
        )

    # -- agent table -----------------------------------------------------

    def _on_agents(self, views: list[dict[str, Any]] | None, error: BaseException | None) -> None:
        if error is not None or views is None:
            self.report_error("Training refresh failed", error or RuntimeError("unknown"))
            return
        training_views = [view for view in views if str(view.get("kind")) == "training"]
        self._last_views = training_views
        views = training_views
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
                    self._budget_label_by_agent.get(str(row["agent_id"]), ""),
                    vm.format_timestamp(row["started_at"]),
                    row["error"] or "",
                ),
            )
            self._row_to_agent[item_id] = row["agent_id"]
            if row["agent_id"] == selected:
                selected_still_present = True
                self.tree.selection_set(item_id)
        refresh_table_empty(self.tree)
        self.set_hint(
            ""
            if rows
            else (
                "No training agents yet. Check the topology and budget above, "
                "then press Launch training."
            )
        )
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
    """The automatic benchmark with live hardware & bridge telemetry.

    Auto runs the staged pipeline (runtime discovery, env/worker screening,
    device comparison, real PPO validation slices) with the project's
    host-scaled defaults; Push widens the host-scaled ladder towards the
    saturation limit; Custom takes explicit candidate lists. Every candidate
    streams live step/FPS/latency/resource telemetry, the fastest stable
    configuration is chosen, and it is persisted and applied to the launch
    configuration automatically.
    """

    title = "Benchmarks"
    widgets = (
        WidgetSpec(
            "plan",
            "Benchmark plan",
            "Auto / Push / Custom, candidate ladders and the start controls.",
            default_span=3,
            max_span=3,
            removable=False,
        ),
        WidgetSpec(
            "live",
            "Live telemetry",
            "Eight live counters plus the leading configuration.",
            default_span=3,
            max_span=3,
        ),
        WidgetSpec(
            "results",
            "Measurements",
            "Live measurement table and the throughput scaling chart.",
            default_span=3,
            max_span=3,
        ),
    )
    subtitle = (
        "Auto, Push or Custom sweeps — measures this machine, then applies the fastest "
        "stable topology automatically."
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
        area = ScrollArea(self, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        board = self.board(area.body)
        board.add("plan", self._build_plan_card)
        board.add("live", self._build_live_card)
        board.add("results", self._build_results_card)
        board.rebuild()

        self._cancel_event: threading.Event | None = None
        self._latest_progress: dict[str, Any] | None = None
        self._progress_lock = threading.Lock()
        self._latest_report: dict[str, Any] | None = None
        self._applied: bool | None = None
        self._failure_text: str | None = None
        self._running = False
        self._rendered_row_count = -1

    # -- cards -------------------------------------------------------------

    def _build_plan_card(self, parent: tk.Misc) -> tk.Widget:
        intro_card = self.card(
            parent, "Automatic benchmark", "Plan, run and apply the winning topology"
        )
        intro = ttk.Frame(intro_card.body, style="CardInner.TFrame")
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
            foreground=self.palette.text_dim,
        ).pack(anchor="w")

        mode_row = ttk.Frame(intro, style="Surface.TFrame")
        mode_row.pack(fill="x", pady=(self.app.px(10, minimum=6), 0))
        ttk.Label(mode_row, text="Mode", style="FieldTitle.TLabel").pack(side="left", padx=(0, 8))
        self.mode_control = SegmentedControl(
            mode_row,
            self.app.bus,
            vm.BENCHMARK_MODES,
            value="auto",
            on_change=self._on_mode_changed,
            motion=self.app.motion,
            height=self.app.px(30, minimum=24),
            width=self.app.px(240, minimum=180),
        )
        self.mode_control.pack(side="left")
        self.mode_plan_label = ttk.Label(
            mode_row,
            text="",
            style="FieldHelp.TLabel",
            wraplength=self.app.px(560, minimum=280),
            justify="left",
        )
        self.mode_plan_label.pack(side="left", padx=(self.app.px(14, minimum=8), 0))

        # Custom mode: the same candidate lists the CLI accepts. Auto and
        # Push derive their ladders from this host and keep these disabled.
        custom_row = ttk.Frame(intro, style="Surface.TFrame")
        custom_row.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        self.custom_env_var = tk.StringVar(value="16,32,64,128")
        self.custom_worker_var = tk.StringVar(value="4,8,16")
        self.custom_steps_var = tk.StringVar(value="")
        self.custom_minutes_var = tk.StringVar(value="5")
        self._custom_fields: list[tk.Widget] = []
        for label, var, width in (
            ("Environments", self.custom_env_var, 22),
            ("Workers", self.custom_worker_var, 12),
            ("Steps / config", self.custom_steps_var, 10),
            ("or minutes", self.custom_minutes_var, 8),
        ):
            ttk.Label(custom_row, text=label, style="FieldHelp.TLabel").pack(
                side="left", padx=(0, 4)
            )
            entry = ttk.Entry(custom_row, textvariable=var, width=width, state="disabled")
            entry.pack(side="left", padx=(0, self.app.px(12, minimum=6)))
            self._custom_fields.append(entry)

        run_bar = ttk.Frame(intro, style="Surface.TFrame")
        run_bar.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        self.run_button = ttk.Button(
            run_bar, text="Start benchmark", command=self._start, style="Primary.TButton"
        )
        self.run_button.pack(side="left")
        ToolTip(
            self.run_button,
            "Run the complete automatic benchmark workflow",
            bus=self.app.bus,
        )
        self.cancel_button = ttk.Button(
            run_bar, text="Cancel", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.progress_label = ttk.Label(
            run_bar, text="idle", foreground=self.palette.text_dim, wraplength=680, justify="left"
        )
        self.progress_label.pack(side="left", padx=(14, 0))

        self.phase_stepper = PhaseStepper(intro, bus=self.app.bus)
        self.phase_stepper.pack(fill="x", pady=(8, 2))
        self.phase_label = ttk.Label(
            intro, text="", foreground=self.palette.text_dim, justify="left"
        )
        self.phase_label.pack(anchor="w", pady=(2, 0))

        # Re-plan while typing: a malformed list must disable Start right
        # away instead of failing only after the click.
        for var in (
            self.custom_env_var,
            self.custom_worker_var,
            self.custom_steps_var,
            self.custom_minutes_var,
        ):
            var.trace_add("write", lambda *_args: self._update_mode_plan())

        return intro_card

    def _build_live_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent,
            "Live telemetry",
            "Stage, throughput and the current leader while a run is active",
        )
        self.live_cards = StatRow(card.body, self.LIVE_CARD_NAMES, max_columns=4, bus=self.app.bus)
        self.live_cards.pack(fill="x")

        best_card = self.card(
            card.body,
            "Best configuration",
            "Selected and applied automatically when a run finishes",
            nested=True,
        )
        best_card.pack(fill="x", pady=(self.app.px(10, minimum=5), 0))
        self.leader_banner = ttk.Label(
            best_card.body,
            text="LEADING SO FAR: awaiting benchmark telemetry",
            style="Leader.TLabel",
        )
        self.leader_banner.pack(fill="x", pady=(0, 6))
        self.recommendation_label = ttk.Label(
            best_card.body, text="n/a", justify="left", style="CardLabel.TLabel"
        )
        self.recommendation_label.pack(anchor="nw")
        self.applied_label = ttk.Label(
            best_card.body, text="", justify="left", style="FieldHelp.TLabel"
        )
        self.applied_label.pack(anchor="w", pady=(4, 0))

        return card

    def _build_results_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Measurements & scaling", "Every tested configuration, live")
        # A ttk.Panedwindow would be the obvious way to split this card, but a
        # panedwindow re-arranges its panes whenever a child reports a new
        # requested size - and a table that fits its columns to the width it
        # was given reports exactly that. The two of them together kept the
        # benchmark page resizing itself forever (the desktop suite timed out
        # inside the resulting event storm). A plain grid gives the same 3:2
        # split, decides once, and cannot re-arrange anything.
        bottom = ttk.Frame(card.body, style="CardInner.TFrame")
        bottom.pack(fill="both", expand=True)

        results_card = self.card(
            bottom,
            "Measurements",
            "Live stream of every tested configuration",
            nested=True,
        )
        results_card.grid(row=0, column=0, sticky="nsew")
        self.tree = _scrollable_table(
            results_card.body,
            self.RESULT_COLUMNS,
            bus=self.app.bus,
            empty_text=("No measurements yet.\nPress Start benchmark to measure this machine."),
        )
        self.tag_style(self.tree, "failed", "error")
        self.tag_style(self.tree, "leader", "ok")

        chart_card = self.card(
            bottom,
            "Throughput scaling",
            "Steps per second across configurations",
            nested=True,
        )
        chart_card.grid(row=0, column=1, sticky="nsew", padx=(self.app.px(10, minimum=5), 0))
        self.throughput_chart = LineChart(
            chart_card.body,
            "Measured throughput (steps/s)",
            color=self.palette.accent,
            bus=self.app.bus,
        )
        self.throughput_chart.pack(fill="both", expand=True)
        bottom.columnconfigure(0, weight=3, uniform="results")
        bottom.columnconfigure(1, weight=2, uniform="results")
        bottom.rowconfigure(0, weight=1)
        return card

    # -- workflow ----------------------------------------------------------

    def refresh(self) -> None:
        self._update_mode_plan()
        self.submit_poll(
            "pipeline-history", self.adapter.benchmark_pipeline_history, self._on_history
        )
        self.submit_poll(
            "recommendation", self.adapter.recommended_configuration, self._on_recommendation
        )
        self._refresh_workflow_labels()
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Start follows both the run state and the plan's validity."""
        invalid = bool(self.current_plan()["errors"])
        self.run_button.configure(state="disabled" if (self._running or invalid) else "normal")
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
            foreground=self.palette.error if self._failure_text else self.palette.text_dim,
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
            self.palette.warn
            if isinstance(jitter, (int, float)) and jitter > 4.0
            else (self.palette.ok if isinstance(jitter, (int, float)) else None)
        )
        elapsed = live_view["elapsed_seconds"]
        elapsed_str = f"{vm.format_number(elapsed, 1)}s" if elapsed is not None else "0.0s"
        if live_view["cpu_percent"] is not None:
            elapsed_str += f" | CPU {vm.format_number(live_view['cpu_percent'], 0)}%"
        self.live_cards.update_values(
            {
                "stage / progress": (
                    stage_text,
                    self.palette.accent if self._running else None,
                ),
                "active config": (live_view["active_config"], None),
                "live fps (steps/s)": (
                    fps_text,
                    self.palette.ok if live_view["live_fps"] is not None else None,
                ),
                "peak fps": (
                    peak_text,
                    self.palette.accent if live_view["peak_fps"] is not None else None,
                ),
                "live steps": (steps_text, None),
                "latency (p50 / p95)": (latency_text, None),
                "stability (jitter)": (jitter_text, jitter_color),
                "elapsed / host": (elapsed_str, None),
            }
        )
        self.leader_banner.configure(text=f"Leading configuration: {live_view['leader_summary']}")
        chart_points = list(live_view["chart_points"])
        if (
            self._running
            and live_view["live_fps"] is not None
            and isinstance(live_view["live_fps"], (int, float))
        ):
            next_idx = len(chart_points) + 1
            chart_points.append((next_idx, float(live_view["live_fps"])))
        self.throughput_chart.set_points(chart_points)

    def _on_mode_changed(self, mode: str) -> None:
        """Enable the custom fields only in Custom mode and re-plan."""
        custom = mode == "custom"
        for field in getattr(self, "_custom_fields", []):
            with contextlib.suppress(tk.TclError):
                field.configure(state="normal" if custom else "disabled")
        self._update_mode_plan()

    def current_plan(self) -> dict[str, Any]:
        """The planned sweep for the selected mode (a plan, not a measurement)."""
        import os

        return vm.benchmark_mode_view(
            self.mode_control.get(),
            environment_text=self.custom_env_var.get(),
            worker_text=self.custom_worker_var.get(),
            steps_raw=self.custom_steps_var.get(),
            minutes_raw=self.custom_minutes_var.get(),
            cpu_count=os.cpu_count(),
        )

    def _update_mode_plan(self) -> None:
        plan = self.current_plan()
        if plan["errors"]:
            self.mode_plan_label.configure(
                text="; ".join(plan["errors"]), foreground=self.palette.error
            )
            self.run_button.configure(state="disabled")
            return
        text = plan["summary"]
        if plan["budget_mode"] == "steps" and plan["steps"]:
            text += f"   ·   {vm.format_number(plan['steps'])} steps per configuration"
        elif plan["minutes"]:
            text += f"   ·   {vm.format_number(plan['minutes'], 1)} min total budget"
        note = plan.get("environments_note")
        if note:
            text += f"\n{note}"
        for warning in plan["warnings"]:
            text += f"\nwarning: {warning}"
        self.mode_plan_label.configure(
            text=text, foreground=self.palette.warn if plan["warnings"] else self.palette.text_dim
        )
        if not self._running:
            self.run_button.configure(state="normal")

    def _start(self) -> None:
        """Run the staged workflow for the selected mode."""
        if self._running:
            return
        plan = self.current_plan()
        if plan["errors"]:
            messagebox.showerror("Invalid benchmark plan", "\n".join(plan["errors"]))
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
        self.progress_label.configure(text="starting...", foreground=self.palette.text_dim)
        self.app.set_status(f"Benchmark started ({plan['mode']}): {plan['summary']}", toast=True)

        def on_progress(event: dict[str, Any]) -> None:
            with self._progress_lock:
                self._latest_progress = dict(event)

        cancel_event = self._cancel_event

        budget_mode = str(plan["budget_mode"])
        steps = int(plan["steps"]) if isinstance(plan["steps"], (int, float)) else None
        minutes = float(plan["minutes"]) if isinstance(plan["minutes"], (int, float)) else None

        def _run() -> dict[str, Any]:
            return self.adapter.run_benchmark_pipeline(
                budget_mode=budget_mode,
                steps=steps,
                minutes=minutes,
                environment_counts=list(plan["environments"]) or None,
                worker_counts=list(plan["workers"]) or None,
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
                    foreground=self.palette.error,
                )
                self.app.set_status(f"Benchmark: applying the result failed: {error}", error=True)
            else:
                self._applied = True
                self._push_to_launch_form(result)
                self.applied_label.configure(
                    text=(
                        "applied automatically - the launch deck on the Training page "
                        "now uses this topology"
                    ),
                    foreground=self.palette.ok,
                )
                self.app.set_status("Benchmark finished - best configuration applied")
            self._refresh_workflow_labels()

        self.app.background.submit(_run, _done)

    def _push_to_launch_form(self, recommendation: dict[str, Any]) -> None:
        """Mirror the applied configuration into an already-built Training page.

        A not-yet-built Training page needs nothing here: its build() reads
        the persisted applied recommendation itself.
        """
        agents_page = self.app.pages.get("Training")
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
        self.set_hint(
            ""
            if (history or self._running or self._latest_report)
            else (
                "No benchmark recorded yet. Start measures this machine end to end, "
                "picks the fastest stable topology and applies it to the launch deck."
            )
        )
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
        refresh_table_empty(self.tree)

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
                text=f"active since {view['applied_utc']} (persisted)", foreground=self.palette.ok
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
        # Two columns of cards over a plain grid. A panedwindow here would
        # re-arrange both halves whenever a table (or a chart) reports a new
        # requested size, which is a feedback loop with content that sizes
        # itself - the layout has exactly two sensible columns and does not
        # need a movable sash to express that.
        split = ttk.Frame(self)
        split.pack(fill="both", expand=True)
        split.columnconfigure(0, weight=1, uniform="eval")
        split.columnconfigure(1, weight=1, uniform="eval")
        split.rowconfigure(0, weight=1)

        left_area = ScrollArea(split, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        left_area.grid(row=0, column=0, sticky="nsew")
        left = left_area.body

        checkpoint_card = self.card(left, "Checkpoints", "Pick the checkpoint to evaluate")
        checkpoint_card.pack(fill="both", expand=True)
        self.checkpoint_tree = _scrollable_table(
            checkpoint_card.body,
            self.CHECKPOINT_COLUMNS,
            bus=self.app.bus,
            expand=False,
            empty_text="No checkpoints yet.\nTrain a run, then evaluate its checkpoints here.",
        )

        form = self.card(left, "Run evaluation on the selected checkpoint")
        form.pack(fill="x", pady=(self.app.px(10, minimum=5), 0))
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
            ttk.Label(form.body, text=label, style="FieldTitle.TLabel").grid(
                row=row, column=0, sticky="w", pady=2
            )
            if kind == "choice":
                ttk.Combobox(
                    form.body,
                    textvariable=var,
                    values=("auto", "cpu", "cuda"),
                    state="readonly",
                    width=16,
                ).grid(row=row, column=1, sticky="w", padx=(self.app.px(8, minimum=4), 0))
            else:
                ttk.Entry(form.body, textvariable=var, width=18).grid(
                    row=row, column=1, sticky="w", padx=(self.app.px(8, minimum=4), 0)
                )
        self.run_button = ttk.Button(
            form.body, text="Start evaluation", command=self._start, style="Primary.TButton"
        )
        self.run_button.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.selected_checkpoint_label = ttk.Label(
            left,
            text="selected checkpoint: none",
            style="FieldHelp.TLabel",
            wraplength=self.app.px(420, minimum=260),
        )
        self.selected_checkpoint_label.pack(anchor="w", pady=(self.app.px(6, minimum=3), 0))
        self.state_label = ttk.Label(left, text="", style="FieldHelp.TLabel")
        self.state_label.pack(anchor="w")

        right_area = ScrollArea(split, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        right_area.grid(row=0, column=1, sticky="nsew", padx=(self.app.px(12, minimum=6), 0))
        right = right_area.body

        results_card = self.card(right, "Evaluation results", "Select multiple rows to compare")
        results_card.pack(fill="both", expand=True)
        self.eval_tree = _scrollable_table(
            results_card.body,
            self.EVAL_COLUMNS,
            bus=self.app.bus,
            expand=False,
            empty_text="No evaluations recorded yet.\nPick a checkpoint on the left and press Start evaluation.",
        )
        self.eval_tree.configure(selectmode="extended")
        self.eval_tree.bind("<<TreeviewSelect>>", self._on_eval_select)

        detail_card = self.card(right, "Structured result / comparison")
        detail_card.pack(fill="both", expand=True, pady=(self.app.px(10, minimum=5), 0))
        self.detail_text = tk.Text(
            detail_card.body,
            wrap="word",
            state="disabled",
            height=18,
            font=self.app.bus.font("mono"),
            background=self.palette.panel,
            foreground=self.palette.text,
            insertbackground=self.palette.text,
            selectbackground=self.palette.accent_soft,
            relief="flat",
            borderwidth=0,
            padx=self.app.px(10, minimum=6),
            pady=self.app.px(10, minimum=6),
        )
        self.detail_text.pack(fill="both", expand=True)

        self._checkpoint_paths: dict[str, str] = {}
        self._checkpoint_signature: tuple[tuple[Any, ...], ...] | None = None
        self._eval_paths: dict[str, str] = {}
        self._evaluation_signature: tuple[tuple[Any, ...], ...] | None = None
        self._selected_evaluation_paths: tuple[str, ...] = ()
        self._evaluation_detail_generation = 0
        self.selected_checkpoint: str | None = None
        self.checkpoint_tree.bind("<<TreeviewSelect>>", self._on_checkpoint_select)
        self.process_id: str | None = None

    #: The keys both inventories render; also their table identity.
    CHECKPOINT_KEYS = ("run_id", "kind", "path", "modified_utc")
    EVALUATION_KEYS = (
        "path",
        "timesteps",
        "episodes",
        "mean_episode_reward",
        "win_rate",
        "loss_rate",
    )

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
        # Same rule as the runs table: an unchanged list is left alone, which
        # is the only way the operator's row selection survives a poll tick
        # (deleting the rows clears the highlight, even when the very same
        # rows are re-inserted).
        signature = vm.table_signature(entries, self.CHECKPOINT_KEYS)
        if signature == self._checkpoint_signature:
            return
        self._checkpoint_signature = signature
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
        refresh_table_empty(self.checkpoint_tree)
        self.set_hint(
            ""
            if (entries or self.selected_checkpoint)
            else (
                "Nothing to evaluate yet. Training writes checkpoints into its run "
                "directory; they appear here as soon as one exists."
            )
        )

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
            foreground=state_colors(self.palette).get(
                str(status.get("state", "")), self.palette.text_dim
            ),
        )
        if status.get("state") in ("finished", "failed"):
            self.process_id = None

    def _on_evaluations(
        self, entries: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or entries is None:
            self.report_error("Evaluation list refresh failed", error or RuntimeError("unknown"))
            return
        signature = vm.table_signature(entries, self.EVALUATION_KEYS)
        if signature == self._evaluation_signature:
            return
        self._evaluation_signature = signature
        selected = set(self._selected_evaluation_paths)
        restored_paths: list[str] = []
        self.eval_tree.delete(*self.eval_tree.get_children())
        self.tag_style(self.eval_tree, "best-eval", "ok")
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
        refresh_table_empty(self.eval_tree)
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


def _render_evaluation_detail(view: dict[str, Any], tactical: dict[str, Any] | None = None) -> str:
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
        (float(r["reward"]) for r in rows if isinstance(r.get("reward"), (int, float))),
        default=None,
    )
    base_reward = next(
        (float(r["reward"]) for r in rows if isinstance(r.get("reward"), (int, float))),
        None,
    )
    header = f"{'path':36} {'reward':>8} {'Δrew':>8} {'win%':>7} {'loss%':>7} {'acc%':>7} {'discharge%':>11}"
    lines = [header, "-" * len(header)]
    for row in rows:
        rew = row.get("reward")
        is_best = (
            best_reward is not None and isinstance(rew, (int, float)) and float(rew) >= best_reward
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
        # One scroll context per page: the inventory on top, the detail and
        # the telemetry under it, all inside a single ScrollArea. The two
        # panedwindows this used to be built from re-arranged their panes on
        # every requested-size change of a child (a table fits its columns to
        # the width it gets, a chart repaints on Configure), which is a
        # feedback loop - and their native sashes were the "window is mostly
        # chrome" complaint the single scroll area replaced.
        area = ScrollArea(self, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        paned = area.body

        top = self.card(paned, "Runs & checkpoints", "Real directories on disk, newest first")
        top.pack(fill="x")
        self.tree = _scrollable_table(
            top.body,
            self.COLUMNS,
            bus=self.app.bus,
            empty_text=(
                "No runs found under the output root yet.\n"
                "Launch a training run and it appears here while it trains."
            ),
        )
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tag_style(self.tree, "run-running", "ok")
        self.tag_style(self.tree, "run-failed", "error")
        self.tag_style(self.tree, "run-warn", "warn")

        bottom = ttk.Frame(paned)
        bottom.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        bottom_split = ttk.Frame(bottom)
        bottom_split.pack(fill="x")
        bottom_split.columnconfigure(0, weight=3, uniform="rundetail")
        bottom_split.columnconfigure(1, weight=2, uniform="rundetail")
        bottom_split.rowconfigure(0, weight=1)

        detail_card = self.card(bottom_split, "Run detail & diagnostics")
        detail_card.grid(row=0, column=0, sticky="nsew")
        self.detail_text = tk.Text(
            detail_card.body,
            wrap="word",
            state="disabled",
            font=self.app.bus.font("mono"),
            background=self.palette.panel,
            foreground=self.palette.text,
            insertbackground=self.palette.text,
            selectbackground=self.palette.accent_soft,
            relief="flat",
            borderwidth=0,
            padx=self.app.px(10, minimum=6),
            pady=self.app.px(10, minimum=6),
        )
        self.detail_text.pack(fill="both", expand=True)

        chart_card = self.card(
            bottom_split, "Selected run telemetry", "Reward and throughput against timesteps"
        )
        chart_card.grid(row=0, column=1, sticky="nsew", padx=(self.app.px(12, minimum=6), 0))
        self.run_reward_chart = LineChart(
            chart_card.body, "Mean episode reward", color=self.palette.ok, bus=self.app.bus
        )
        self.run_reward_chart.pack(fill="both", expand=True, pady=(0, 4))
        self.run_fps_chart = LineChart(
            chart_card.body, "Throughput (steps/s)", color=self.palette.accent, bus=self.app.bus
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
            actions, text="Clone topology to Training", command=self._clone_to_training
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="Resume checkpoint in Training",
            command=self._resume_run_in_training,
            style="Primary.TButton",
        ).pack(side="left", padx=(8, 0))

        self._row_to_dir: dict[str, str] = {}
        self._selected_run_dir: str | None = None
        self._selected_run_report: dict[str, Any] | None = None
        self._run_detail_generation = 0
        self._pending_run_selection: str | None = None
        self._rows_signature: tuple[tuple[Any, ...], ...] | None = None

    #: The row keys whose values are rendered; also the identity of the table.
    COLUMN_KEYS = (
        "run_id",
        "state",
        "progress_percent",
        "device",
        "environment_count",
        "env_workers",
        "checkpoints",
        "reward",
        "win_rate",
        "modified_utc",
    )

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
        # Rebuilding the table costs a delete plus one insert per run, and it
        # also drops the selection (and the scroll position) the operator is
        # looking at. A run list that did not change is left alone; the
        # pending-selection path below still runs, because navigating here
        # while a scan is in flight must work even when the table is current.
        signature = vm.table_signature(rows, self.COLUMN_KEYS)
        if signature == self._rows_signature and self._row_to_dir:
            if self._pending_run_selection is not None and self._select_existing_row(
                self._pending_run_selection
            ):
                self._pending_run_selection = None
            return
        self._rows_signature = signature
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
        refresh_table_empty(self.tree)
        self.set_hint(
            ""
            if rows
            else (
                "No run directories under the output root yet - a training run "
                "creates one and shows up here while it trains."
            )
        )
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
        target = (
            best
            if best.is_file()
            else (Path(self._selected_run_dir) / "checkpoints" / "latest.zip")
        )
        if not target.is_file():
            messagebox.showinfo(
                "No checkpoint yet", "This run has neither best.zip nor latest.zip yet."
            )
            return
        self.app.show_page("Evaluations")
        self.app.pages["Evaluations"].select_checkpoint(str(target))

    def _clone_to_training(self) -> None:
        report = self._selected_run_report
        if not report:
            return
        config = report.get("config") or {}
        agents_page = self.app.pages.get("Training")
        if agents_page is None:
            return
        self.app.show_page("Training")
        agents_page.apply_launch_values(
            {
                "environment_count": str(config.get("environment_count", 1)),
                "env_workers": str(config.get("env_workers", 1)),
                "total_training_steps": str(config.get("total_training_steps", 100000)),
                "device": str(config.get("device", "auto")),
            }
        )
        self.app.set_status(f"Cloned topology from {report.get('run_id', 'run')} to Training")

    def _resume_run_in_training(self) -> None:
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
        self._clone_to_training()
        agents_page = self.app.pages.get("Training")
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
        area = ScrollArea(self, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        host = area.body

        self.stats = StatRow(host, self.STAT_LABELS, max_columns=4, bus=self.app.bus)
        self.stats.pack(fill="x")

        deps_card = self.card(host, "Optional dependencies & runtime capabilities")
        deps_card.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        self.deps_label = ttk.Label(
            deps_card.body, text="n/a", justify="left", style="CardLabel.TLabel"
        )
        self.deps_label.pack(anchor="w")

        chart_card = self.card(
            host,
            "Host resource usage",
            "Sampled each refresh — the last 300 samples are kept",
        )
        chart_card.pack(fill="both", expand=True, pady=(self.app.px(12, minimum=6), 0))
        charts_grid = ttk.Frame(chart_card.body, style="CardInner.TFrame")
        charts_grid.pack(fill="both", expand=True)
        self.cpu_chart = LineChart(
            charts_grid,
            "CPU percent (this process)",
            color=self.palette.accent,
            bus=self.app.bus,
        )
        self.cpu_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.rss_chart = LineChart(
            charts_grid, "Process RSS (MB)", color=self.palette.ok, bus=self.app.bus
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
            self.palette.error
            if isinstance(cpu_pct, (int, float)) and cpu_pct >= 90.0
            else (
                self.palette.warn
                if isinstance(cpu_pct, (int, float)) and cpu_pct >= 75.0
                else self.palette.ok
            )
        )
        rec_text = "uncalibrated"
        rec_color = self.palette.text_dim
        if hasattr(self.adapter, "recommended_configuration"):
            try:
                rec = self.adapter.recommended_configuration()
                if rec and isinstance(rec.get("environment_count"), int):
                    rec_text = (
                        f"{rec['environment_count']}e / {rec.get('env_workers', 1)}w "
                        f"({rec.get('device', 'cpu')})"
                    )
                    rec_color = self.palette.ok
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
                "godot": (
                    godot_text,
                    self.palette.ok if status.get("godot_available") else self.palette.warn,
                ),
                "torch": (
                    "available" if status.get("torch_available") else "not installed",
                    self.palette.ok if status.get("torch_available") else self.palette.text_dim,
                ),
                "cuda": (
                    "available" if status.get("cuda_available") else "not available",
                    self.palette.ok if status.get("cuda_available") else self.palette.text_dim,
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
    subtitle = "Appearance, movable layouts, presets, roots, Godot executable and calibration."

    def build(self) -> None:
        area = ScrollArea(self, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        host = area.body

        self._build_appearance_section(host)
        self._build_layout_studio(host)
        self._build_preset_section(host)

        top_grid = ttk.Frame(host)
        top_grid.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        top_grid.columnconfigure(0, weight=1, uniform="settings")
        top_grid.columnconfigure(1, weight=1, uniform="settings")

        roots_card = self.card(top_grid, "Directories", "Where the project and its runs live")
        roots_card.grid(row=0, column=0, sticky="nsew", padx=(0, self.app.px(6, minimum=3)))
        roots = roots_card.body
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
            foreground=self.palette.text_dim,
            wraplength=520,
        ).pack(anchor="w", pady=(6, 0))

        # ---- Godot executable (the #1 reason launches fail) -------------
        godot_card = self.card(top_grid, "Godot executable", "The #1 reason a launch fails")
        godot_card.grid(row=0, column=1, sticky="nsew", padx=(self.app.px(6, minimum=3), 0))
        godot = godot_card.body
        ttk.Label(
            godot,
            text="Training, benchmarks and evaluations resolve the Godot binary through this "
            "remembered setting when none is given explicitly.",
            foreground=self.palette.text_dim,
            wraplength=520,
            justify="left",
        ).pack(anchor="w")
        entry_row = ttk.Frame(godot, style="Surface.TFrame")
        entry_row.pack(fill="x", pady=(8, 0))
        self.godot_var = tk.StringVar(value=self.adapter.godot_executable_setting() or "")
        ttk.Entry(entry_row, textvariable=self.godot_var, width=42).pack(
            side="left", fill="x", expand=True
        )
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
        calib_card = self.card(
            host,
            "Machine calibration & Ubuntu CPU Performance Turbo",
            "Read-only summary; the calibration itself runs on the Benchmarks page",
        )
        calib_card.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        calib = calib_card.body
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
        self._build_roblox_ttk_section(host)

    # -- appearance -------------------------------------------------------

    def _build_appearance_section(self, parent: tk.Misc) -> None:
        card = self.card(parent, "Appearance", "Applies live — every page repaints immediately")
        card.pack(fill="x")
        row = ttk.Frame(card.body, style="CardInner.TFrame")
        row.pack(fill="x")
        self.theme_var = tk.StringVar(value=self.app.palette.label)
        self.layout_var = tk.StringVar(value=LAYOUT_MODES[self.app.prefs.layout])
        self.density_var = tk.StringVar(value=self.app.bus.density.label)
        self.motion_var = tk.StringVar(value=MOTION_LEVELS[self.app.motion.level])
        fields = (
            (
                "Theme",
                self.theme_var,
                tuple(THEMES[name].label for name in THEME_NAMES),
                self._on_theme_selected,
            ),
            (
                "Shell layout",
                self.layout_var,
                tuple(LAYOUT_MODES.values()),
                self._on_layout_selected,
            ),
            (
                "Density",
                self.density_var,
                tuple(d.label for d in DENSITIES.values()),
                self._on_density_selected,
            ),
            ("Motion", self.motion_var, tuple(MOTION_LEVELS.values()), self._on_motion_selected),
        )
        for label, variable, values, handler in fields:
            cell = ttk.Frame(row, style="CardInner.TFrame")
            cell.pack(side="left", padx=(0, self.app.px(14, minimum=8)))
            ttk.Label(cell, text=label, style="FieldTitle.TLabel").pack(anchor="w")
            picker = ttk.Combobox(
                cell, textvariable=variable, values=values, state="readonly", width=16
            )
            picker.pack(anchor="w", pady=(self.app.px(4, minimum=2), 0))
            picker.bind("<<ComboboxSelected>>", handler)

        self.radius_var = tk.IntVar(value=self.app.prefs.radius)
        radius_cell = ttk.Frame(row, style="CardInner.TFrame")
        radius_cell.pack(side="left", padx=(0, self.app.px(10, minimum=6)))
        ttk.Label(radius_cell, text="Corner radius", style="FieldTitle.TLabel").pack(anchor="w")
        scale = tk.Scale(
            radius_cell,
            from_=0,
            to=20,
            orient="horizontal",
            variable=self.radius_var,
            command=lambda _value: self._on_radius_changed(),
            background=self.palette.card,
            foreground=self.palette.text,
            troughcolor=self.palette.panel,
            highlightthickness=0,
            showvalue=True,
            length=self.app.px(150, minimum=110),
        )
        scale.pack(anchor="w")

        self._build_accent_row(card.body)

        toggles = ttk.Frame(card.body, style="CardInner.TFrame")
        toggles.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        self.glow_var = tk.BooleanVar(value=self.app.prefs.show_glow)
        self.grid_var = tk.BooleanVar(value=self.app.prefs.show_grid)
        ttk.Checkbutton(
            toggles,
            text="Accent glow",
            variable=self.glow_var,
            command=self._on_effect_toggled,
        ).pack(side="left")
        ttk.Checkbutton(
            toggles,
            text="Grid backdrop",
            variable=self.grid_var,
            command=self._on_effect_toggled,
        ).pack(side="left", padx=(self.app.px(12, minimum=6), 0))
        ttk.Button(
            toggles,
            text="Reset appearance",
            style="Ghost.TButton",
            command=self._reset_appearance,
        ).pack(side="right")

    def _build_accent_row(self, parent: tk.Misc) -> None:
        """Accent colour: quick swatches plus a hex field for anything else."""
        row = ttk.Frame(parent, style="CardInner.TFrame")
        row.pack(fill="x", pady=(self.app.px(10, minimum=5), 0))
        ttk.Label(row, text="Accent", style="FieldTitle.TLabel").pack(side="left", padx=(0, 8))
        swatches = ttk.Frame(row, style="CardInner.TFrame")
        swatches.pack(side="left")
        self._accent_swatches: list[tk.Widget] = []
        for color, label in ACCENT_PRESETS:
            button = tk.Button(
                swatches,
                text="",
                width=2,
                height=1,
                background=color,
                activebackground=color,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                cursor="hand2",
                command=partial(self._pick_accent, color),
            )
            button.pack(side="left", padx=(0, 3))
            ToolTip(button, f"{label} ({color})", bus=self.app.bus)
            self._accent_swatches.append(button)
        self.accent_var = tk.StringVar(value=self.app.prefs.accent or self.app.palette.accent)
        entry = ttk.Entry(row, textvariable=self.accent_var, width=10)
        entry.pack(side="left", padx=(self.app.px(10, minimum=6), 4))
        entry.bind("<Return>", lambda _event: self._apply_accent())
        ttk.Button(row, text="Apply", command=self._apply_accent).pack(side="left")
        ttk.Button(
            row,
            text="Theme accent",
            style="Ghost.TButton",
            command=self._clear_accent,
        ).pack(side="left", padx=(6, 0))
        self.accent_hint = ttk.Label(row, text="", style="FieldHelp.TLabel")
        self.accent_hint.pack(side="left", padx=(self.app.px(10, minimum=6), 0))
        self._refresh_accent_hint()

    def _refresh_accent_hint(self) -> None:
        hint = getattr(self, "accent_hint", None)
        if hint is None:
            return
        current = self.app.prefs.accent
        hint.configure(
            text=(
                f"Custom accent {current} (every theme wears it)"
                if current
                else "Using the theme's own accent"
            )
        )

    def _pick_accent(self, color: str) -> None:
        if self.app.set_accent(color):
            self.accent_var.set(color)
            self._refresh_accent_hint()

    def _apply_accent(self) -> None:
        typed = self.accent_var.get().strip()
        if typed and not normalize_accent(typed):
            self.app.notify(f"'{typed}' is not a colour like #4F7CFF", kind="error")
            return
        if self.app.set_accent(typed):
            self._refresh_accent_hint()

    def _clear_accent(self) -> None:
        self.app.set_accent("")
        self.accent_var.set(self.app.palette.accent)
        self._refresh_accent_hint()

    def _on_theme_selected(self, _event: object = None) -> None:
        labels = {THEMES[name].label: name for name in THEME_NAMES}
        picked = labels.get(str(self.theme_var.get()))
        if picked:
            self.app.set_theme(picked)
            if not self.app.prefs.accent:
                self.accent_var.set(self.app.palette.accent)
            self._refresh_accent_hint()

    def _on_layout_selected(self, _event: object = None) -> None:
        modes = {label: key for key, label in LAYOUT_MODES.items()}
        picked = modes.get(str(self.layout_var.get()))
        if picked:
            self.app.set_layout_mode(picked)

    def _on_density_selected(self, _event: object = None) -> None:
        names = {density.label: name for name, density in DENSITIES.items()}
        picked = names.get(str(self.density_var.get()))
        if picked:
            self.app.set_density(picked)

    def _on_motion_selected(self, _event: object = None) -> None:
        levels = {label: key for key, label in MOTION_LEVELS.items()}
        picked = levels.get(str(self.motion_var.get()))
        if picked:
            self.app.set_motion(picked)

    def _on_radius_changed(self) -> None:
        self.app.prefs.radius = int(self.radius_var.get())
        self.app.save_preferences()

    def _on_effect_toggled(self) -> None:
        self.app.prefs.show_glow = bool(self.glow_var.get())
        self.app.prefs.show_grid = bool(self.grid_var.get())
        self.app.save_preferences()
        self.app.notify("Visual effects updated", kind="info", timeout_ms=1800)

    def _reset_appearance(self) -> None:
        self.app.set_accent("")
        self.app.set_theme("corz")
        self.app.set_density("comfort")
        self.app.set_motion("normal")
        self.theme_var.set(THEMES["corz"].label)
        self.density_var.set(DENSITIES["comfort"].label)
        self.motion_var.set(MOTION_LEVELS["normal"])
        self.app.notify("Appearance reset to defaults", kind="ok")

    # -- layout studio ----------------------------------------------------

    def _build_layout_studio(self, parent: tk.Misc) -> None:
        card = self.card(
            parent, "Layout studio", "Move, resize or hide cards, then save the result as a preset"
        )
        card.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        controls = ttk.Frame(card.body, style="CardInner.TFrame")
        controls.pack(fill="x")
        pages = [title for title in PAGE_WIDGETS]
        self.studio_page_var = tk.StringVar(value=pages[0] if pages else "")
        ttk.Label(controls, text="Page", style="FieldTitle.TLabel").pack(side="left", padx=(0, 6))
        self.studio_page_combo = ttk.Combobox(
            controls,
            textvariable=self.studio_page_var,
            values=tuple(pages),
            state="readonly",
            width=18,
        )
        self.studio_page_combo.pack(side="left")
        self.studio_page_combo.bind("<<ComboboxSelected>>", lambda _e: self._render_studio_rows())
        ttk.Button(
            controls,
            text="Reset this page",
            style="Ghost.TButton",
            command=self._reset_studio_page,
        ).pack(side="right")
        ttk.Button(
            controls,
            text="Reset everything",
            style="Ghost.TButton",
            command=self._reset_studio_all,
        ).pack(side="right", padx=(0, self.app.px(6, minimum=3)))
        ttk.Button(
            controls,
            text="Move up / down, span and visibility apply immediately",
            style="Ghost.TButton",
            state="disabled",
        )
        self.studio_container = ttk.Frame(card.body, style="CardInner.TFrame")
        self.studio_container.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        self._studio_rows: list[tk.Widget] = []
        self._render_studio_rows()

    def _render_studio_rows(self) -> None:
        for row in self._studio_rows:
            row.destroy()
        self._studio_rows.clear()
        page_title = self.studio_page_var.get()
        specs = PAGE_WIDGETS.get(page_title, ())
        state = self.app.layout_bus.state
        placements = {placement.widget_id: placement for placement in state.placements(page_title)}
        for spec in specs:
            placement = placements.get(spec.widget_id)
            row = ttk.Frame(self.studio_container, style="CardInner.TFrame")
            row.pack(fill="x", pady=self.app.px(2, minimum=1))
            label = spec.title
            if placement is not None and not placement.visible:
                label += "   (hidden)"
            ttk.Label(row, text=label, style="CardLabel.TLabel").pack(
                side="left", fill="x", expand=True
            )
            ttk.Button(
                row,
                text="Up",
                style="Ghost.TButton",
                command=lambda widget_id=spec.widget_id: self._move_studio_widget(widget_id, -1),  # type: ignore[misc]
            ).pack(side="left", padx=(0, 3))
            ttk.Button(
                row,
                text="Down",
                style="Ghost.TButton",
                command=lambda widget_id=spec.widget_id: self._move_studio_widget(widget_id, 1),  # type: ignore[misc]
            ).pack(side="left", padx=(0, 3))
            for span in range(spec.min_span, spec.max_span + 1):
                style = (
                    "Primary.TButton" if placement and placement.span == span else "Ghost.TButton"
                )
                ttk.Button(
                    row,
                    text=f"{span}×",
                    style=style,
                    width=3,
                    command=lambda widget_id=spec.widget_id, value=span: (  # type: ignore[misc]
                        self._span_studio_widget(widget_id, value)
                    ),
                ).pack(side="left", padx=(0, 2))
            if spec.removable:
                visible_var = tk.BooleanVar(value=bool(placement.visible) if placement else True)
                ttk.Checkbutton(
                    row,
                    text="Show",
                    variable=visible_var,
                    command=lambda widget_id=spec.widget_id, var=visible_var: (  # type: ignore[misc]
                        self._toggle_studio_widget(widget_id, bool(var.get()))
                    ),
                ).pack(side="left", padx=(self.app.px(10, minimum=5), 0))
            self._studio_rows.append(row)

    def _publish_layout(self, state: Any) -> None:
        self.app.set_layout_state(state)
        self._render_studio_rows()

    def _move_studio_widget(self, widget_id: str, delta: int) -> None:
        state = move(self.app.layout_bus.state, self.studio_page_var.get(), widget_id, delta)
        self._publish_layout(state)

    def _span_studio_widget(self, widget_id: str, span: int) -> None:
        state = set_span(self.app.layout_bus.state, self.studio_page_var.get(), widget_id, span)
        self._publish_layout(state)

    def _toggle_studio_widget(self, widget_id: str, visible: bool) -> None:
        state = set_visible(
            self.app.layout_bus.state, self.studio_page_var.get(), widget_id, visible
        )
        self._publish_layout(state)

    def _reset_studio_page(self) -> None:
        state = reset_page(self.app.layout_bus.state, self.studio_page_var.get(), PAGE_WIDGETS)
        self._publish_layout(state)
        self.app.notify(f"Layout reset: {self.studio_page_var.get()}", kind="info")

    def _reset_studio_all(self) -> None:
        state = normalize(LayoutState(pages={}), PAGE_WIDGETS)
        self._publish_layout(state)
        self.app.notify("All layouts reset to defaults", kind="ok")

    # -- presets ----------------------------------------------------------

    def _build_preset_section(self, parent: tk.Misc) -> None:
        card = self.card(
            parent, "Presets", "Theme, shell layout, density, motion and every card position"
        )
        card.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        row = ttk.Frame(card.body, style="CardInner.TFrame")
        row.pack(fill="x")
        self.preset_name_var = tk.StringVar(value="")
        ttk.Label(row, text="Name", style="FieldTitle.TLabel").pack(side="left", padx=(0, 6))
        ttk.Entry(row, textvariable=self.preset_name_var, width=22).pack(side="left")
        ttk.Button(
            row,
            text="Save preset",
            style="Primary.TButton",
            command=self._save_preset,
        ).pack(side="left", padx=(self.app.px(8, minimum=4), 0))
        self.preset_choice_var = tk.StringVar(value="")
        self.preset_combo = ttk.Combobox(
            row, textvariable=self.preset_choice_var, values=(), state="readonly", width=24
        )
        self.preset_combo.pack(side="left", padx=(self.app.px(14, minimum=8), 0))
        ttk.Button(row, text="Apply", command=self._apply_preset).pack(side="left", padx=(6, 0))
        ttk.Button(
            row,
            text="Rename to name",
            command=self._rename_preset,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Delete", style="Danger.TButton", command=self._delete_preset).pack(
            side="left", padx=(6, 0)
        )
        self.preset_status_label = ttk.Label(
            card.body, text="", style="FieldHelp.TLabel", wraplength=self.app.px(880, minimum=420)
        )
        self.preset_status_label.pack(anchor="w", pady=(self.app.px(8, minimum=4), 0))

        transfer = ttk.Frame(card.body, style="CardInner.TFrame")
        transfer.pack(fill="x", pady=(self.app.px(8, minimum=4), 0))
        ttk.Label(
            transfer,
            text="Share or back up a preset as a file",
            style="FieldHelp.TLabel",
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            transfer, text="Export…", style="Ghost.TButton", command=self._export_preset
        ).pack(side="left")
        ttk.Button(
            transfer, text="Import…", style="Ghost.TButton", command=self._import_preset
        ).pack(side="left", padx=(6, 0))
        self._refresh_preset_list()

    def _export_preset(self) -> None:
        """Write the selected preset to a file the operator chooses."""
        selected = self.preset_choice_var.get().strip()
        if not selected:
            self.app.notify("Select a preset to export", kind="warn")
            return
        target = filedialog.asksaveasfilename(
            parent=self,
            title="Export preset",
            defaultextension=".json",
            initialfile=f"{selected}.json",
            filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")],
        )
        if target:
            self.app.export_layout_preset(selected, target)

    def _import_preset(self) -> None:
        """Add a preset from a file, asking before a name is replaced."""
        source = filedialog.askopenfilename(
            parent=self,
            title="Import preset",
            filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")],
        )
        if not source:
            return
        name = self.app.preset_name_for_import(source)
        if name and name in self.app.list_layout_presets():
            if not messagebox.askyesno(
                "Preset already exists",
                f"A preset named '{name}' already exists.\nReplace it with the imported one?",
            ):
                return
            self.app.import_layout_preset(source, overwrite=True)
        else:
            self.app.import_layout_preset(source)
        self._refresh_preset_list()

    def _refresh_preset_list(self) -> None:
        names = self.app.list_layout_presets()
        with contextlib.suppress(tk.TclError):
            self.preset_combo.configure(values=tuple(names))
            if names and not self.preset_choice_var.get():
                self.preset_choice_var.set(names[0])
        active = self.app.prefs.active_preset
        self.preset_status_label.configure(
            text=(
                f"Active preset: {active}"
                if active
                else "No preset active — the current arrangement is kept automatically."
            )
        )

    def _save_preset(self) -> None:
        name = self.preset_name_var.get().strip()
        if not name:
            self.app.notify("Give the preset a name first", kind="warn")
            return
        try:
            saved = self.app.save_layout_preset(name)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            self.app.notify(f"Preset could not be saved: {exc}", kind="error")
            return
        self.preset_name_var.set("")
        self._refresh_preset_list()
        self.app.notify(f"Preset saved: {saved}", kind="ok")

    def _apply_preset(self) -> None:
        name = self.preset_choice_var.get()
        if not name:
            self.app.notify("No preset selected", kind="warn")
            return
        if not self.app.apply_layout_preset(name):
            return
        self._refresh_preset_list()
        self._render_studio_rows()
        self.theme_var.set(self.app.palette.label)
        self.density_var.set(self.app.bus.density.label)
        self.motion_var.set(MOTION_LEVELS[self.app.motion.level])
        self.layout_var.set(LAYOUT_MODES[self.app.prefs.layout])

    def _rename_preset(self) -> None:
        """Rename the selected preset to the name in the entry (both required)."""
        selected = self.preset_choice_var.get()
        new_name = self.preset_name_var.get().strip()
        if not selected:
            self.app.notify("Select a preset to rename", kind="warn")
            return
        if not new_name:
            self.app.notify("Type the new name first", kind="warn")
            return
        if not self.app.rename_layout_preset(selected, new_name):
            return
        self.preset_name_var.set("")
        self.preset_choice_var.set("")
        self._refresh_preset_list()

    def _delete_preset(self) -> None:
        name = self.preset_choice_var.get()
        if name and self.app.delete_layout_preset(name):
            self.preset_choice_var.set("")
            self._refresh_preset_list()

    def _build_roblox_ttk_section(self, parent: tk.Misc) -> None:
        roblox_card = self.card(
            parent,
            "Roblox TTK Testing [MAP VOTING]",
            "Sable Digital  ·  PlaceId 120189115846709  ·  Universe 10090256806",
        )
        roblox_card.pack(fill="x", pady=(self.app.px(12, minimum=6), 0))
        roblox_box = roblox_card.body
        shortcut_row = ttk.Frame(roblox_box, style="Surface.TFrame")
        shortcut_row.pack(fill="x")
        ttk.Label(shortcut_row, text="Roblox shortcut / exe:").pack(side="left")
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
            roblox_box,
            text="Probing Roblox Player...",
            foreground=self.palette.accent,
            justify="left",
        )
        self.roblox_live_label.pack(anchor="w", pady=(8, 6))

        # ---- Interactive TTK & DPS Calculator + 1-Click Presets ---------
        calc_row = ttk.Frame(roblox_box, style="Surface.TFrame")
        calc_row.pack(fill="x", pady=(2, 8))
        ttk.Label(calc_row, text="TTK / DPS Calculator:", style="FieldTitle.TLabel").pack(
            side="left"
        )
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
            foreground=self.palette.ok,
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
        self.ttk_tree = _scrollable_table(
            roblox_box,
            ttk_columns,
            expand=False,
            bus=self.app.bus,
            empty_text="No mechanics manifest recorded yet.",
        )
        self.tag_style(self.ttk_tree, "ttk-verified", "ok")
        self.tag_style(self.ttk_tree, "ttk-pending", "warn")
        self.tag_style(self.ttk_tree, "ttk-excluded", "text_dim")
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

    def _on_ttk_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            return
        tview = vm.ttk_testing_view(status)
        self.roblox_live_label.configure(
            text=(
                f"{tview['status_badge']}   ·   Launcher: {tview['launcher_text']}   ·   "
                f"Window: {tview['window_text']}   ·   Place: {tview['place_text']}   ·   "
                f"Progress: {tview['calibration_progress_text']}"
            ),
            foreground=self.palette.ok
            if tview["connected"]
            else (self.palette.warn if tview["roblox_running"] else self.palette.text_dim),
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
        self.ttk_calc_result_label.configure(text=summary, foreground=self.palette.ok)
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
                foreground=self.palette.ok,
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
        self.mechanic_value_var.set(str(row.get("measured_value") or ""))
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
                foreground=self.palette.ok,
            )
        else:
            self.calibration_label.configure(
                text="No benchmark recommendation persisted yet — run the automatic benchmark on the Benchmarks page.",
                foreground=self.palette.text_dim,
            )
        if hasattr(self.adapter, "ubuntu_cpu_status") and hasattr(self, "cpu_turbo_label"):
            with contextlib.suppress(Exception):
                uview = vm.ubuntu_cpu_turbo_view(self.adapter.ubuntu_cpu_status())
                self.cpu_turbo_label.configure(
                    text=f"Ubuntu CPU Turbo: {uview['badge']} — {uview['summary']}",
                    foreground=self.palette.ok
                    if uview["anti_thrash_active"]
                    else self.palette.accent,
                )

    def _on_system_status(self, status: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or status is None:
            return
        if status.get("godot_available"):
            resolved = status.get("godot_resolved_executable") or "?"
            version = status.get("godot_version") or "version unknown"
            self.godot_status_label.configure(
                text=f"currently resolves to: {resolved} ({version})", foreground=self.palette.ok
            )
        else:
            self.godot_status_label.configure(
                text="no usable Godot executable found - launches will fail until one is "
                "configured here, put on PATH or set via GODOT_PATH",
                foreground=self.palette.error,
            )

    def _browse_godot(self) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.godot_var.set(path)

    def _save_godot(self) -> None:
        candidate = self.godot_var.get().strip()
        if not candidate:
            self.godot_status_label.configure(
                text="enter or browse to a Godot executable first", foreground=self.palette.warn
            )
            return
        self.godot_save_button.configure(state="disabled")
        self.godot_status_label.configure(text="verifying...", foreground=self.palette.text_dim)

        def _done(result: dict[str, Any] | None, error: BaseException | None) -> None:
            self.godot_save_button.configure(state="normal")
            if error is not None or not result:
                self.godot_status_label.configure(
                    text=f"verification failed: {error}", foreground=self.palette.error
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
                    foreground=self.palette.ok,
                )
                self.app.set_status("Godot executable verified and remembered")
            else:
                self.godot_status_label.configure(
                    text=str(result.get("error")), foreground=self.palette.error
                )

        self.app.background.submit(
            lambda: self.adapter.configure_godot_executable(candidate), _done
        )

    def _change_output_root(self) -> None:
        directory = filedialog.askdirectory(initialdir=str(self.adapter.output_root))
        if directory:
            self.app.set_output_root(directory)
            self.refresh()


class StatsPage(Page):
    """What the trained policy receives, decoded from a real recording.

    This page answers the operator's actual question - "where are the
    enemies, what objects are around the agent, and why did the policy do
    that?" - with the *same* numbers the network gets. Two honest sources:

    * ``contract.OBSERVATION_SPEC`` is the contract itself (all 106 fields,
      their meaning and their normalisation), so the table is complete even
      before a single replay exists;
    * a recorded replay is the only thing that can show real values. Only
      ``replay_detail=detailed`` stores the observation vector per tick, and
      a light replay is reported as such instead of being rendered with
      zeros that would look like data.

    The TTK Testing evidence manifest sits next to it, because the
    calibration boundary ("what is verified about the real game, what still
    needs a manual measurement") is part of the same picture.
    """

    title = "Stats"
    subtitle = (
        "Every value the policy receives: contacts, objects, hearing, memory, raw vector, action."
    )

    REPLAY_COLUMNS = (
        ("name", "Replay", 200),
        ("run", "Run", 140),
        ("detail", "Detail", 70),
        ("ticks", "Ticks", 60),
        ("seed", "Seed", 70),
        ("map", "Map", 120),
        ("scenario", "Scenario", 120),
        ("curriculum", "Level", 55),
        ("enemies", "Enemies", 65),
        ("observations", "Observations", 95),
        ("size", "Size", 80),
    )

    CONTACT_COLUMNS = (
        ("slot", "Contact", 110),
        ("state", "State", 150),
        ("relative_position", "Relative position", 150),
        ("distance", "Distance", 75),
        ("bearing", "Bearing", 75),
        ("elevation", "Elevation", 75),
        ("health", "Health", 70),
        ("visible", "Visible", 60),
        ("in_fov", "In FOV", 60),
        ("los", "LOS", 55),
        ("info_age", "Age", 60),
        ("confidence", "Confidence", 85),
        ("source", "Source", 70),
    )

    WORLD_COLUMNS = (
        ("field", "Object / memory", 190),
        ("distance", "Value", 80),
        ("bearing", "Bearing", 80),
        ("note", "What it is", 260),
    )

    AUDIO_COLUMNS = (
        ("field", "Hearing", 190),
        ("value", "Value", 120),
        ("note", "What it is", 260),
    )

    ACTION_COLUMNS = (
        ("component", "Action component", 130),
        ("value", "Value", 60),
        ("meaning", "Meaning", 420),
    )

    VECTOR_COLUMNS = (
        ("index", "#", 60),
        ("field", "Field", 260),
        ("value", "Value this tick", 150),
        ("meaning", "Meaning", 380),
        ("normalization", "Normalisation", 260),
    )

    EVIDENCE_COLUMNS = (
        ("mechanic", "Mechanic", 220),
        ("status", "Evidence status", 150),
        ("measured_value", "Measured / notes", 240),
        ("rule", "Allowed project behaviour", 420),
        ("source", "Source", 200),
    )

    #: How many poll ticks between two automatic replay-list rescans. The
    #: list only changes when a run writes a replay, so polling it every
    #: tick would re-read every replay header for nothing.
    RESCAN_EVERY = 20

    widgets = (
        WidgetSpec(
            "source",
            "Replay source",
            "Which recording the page decodes: run, seed, map, level.",
            default_span=3,
            max_span=3,
            removable=False,
        ),
        WidgetSpec(
            "summary",
            "Policy input at a glance",
            "Tick navigation plus the headline values of one observation.",
            default_span=3,
            max_span=3,
        ),
        WidgetSpec(
            "vector",
            "Observation vector (raw)",
            "All 106 fields, grouped, exactly as the contract defines them.",
            default_span=3,
            max_span=3,
        ),
        WidgetSpec(
            "contacts",
            "Contacts / enemies",
            "The three tracked enemies: position, distance, bearing, perception.",
            default_span=2,
        ),
        WidgetSpec(
            "world",
            "Objects, cover and memory",
            "World geometry, corpses, remembered cover/danger and exploration.",
        ),
        WidgetSpec(
            "hearing",
            "Hearing",
            "What the policy perceives through sound.",
        ),
        WidgetSpec(
            "action",
            "Action the policy took",
            "The six action components emitted for the selected tick.",
            default_span=2,
        ),
        WidgetSpec(
            "evidence",
            "TTK Testing evidence",
            "Verified mechanics versus measurements that are still required.",
            default_span=3,
            max_span=3,
        ),
    )

    def build(self) -> None:
        area = ScrollArea(self, self.app.bus, style="Content.TFrame", scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        board = self.board(area.body)
        board.add("source", self._build_source_card)
        board.add("summary", self._build_summary_card)
        board.add("vector", self._build_vector_card)
        board.add("contacts", self._build_contacts_card)
        board.add("world", self._build_world_card)
        board.add("hearing", self._build_hearing_card)
        board.add("action", self._build_action_card)
        board.add("evidence", self._build_evidence_card)
        board.rebuild()

        self._replays: list[dict[str, Any]] = []
        self._replay: dict[str, Any] | None = None
        self._replay_signature: tuple[Any, ...] | None = None
        self._evidence_loaded = False
        self._tick = 0
        self._poll_count = 0
        self._scan_requested = True
        # The contract tables are complete before any recording exists: the
        # field list is part of the contract, only the values need a replay.
        # An empty page used to imply "the policy receives nothing".
        self._fill_vector(None)
        self._fill_contacts(None)
        self._fill_world(None)
        self._fill_audio(None)
        self._fill_action(None)

    # -- cards -------------------------------------------------------------

    def _build_source_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent,
            "Replay source",
            "Replays live in <output root>/<run>/replays; only detailed ones store the vector",
        )
        row = ttk.Frame(card.body, style="CardInner.TFrame")
        row.pack(fill="x")
        self.replay_info_label = ttk.Label(
            row,
            text="Scanning for recorded replays…",
            style="CardLabel.TLabel",
            justify="left",
            wraplength=self.app.px(900, minimum=420),
        )
        self.replay_info_label.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Rescan", style="Ghost.TButton", command=self._rescan_replays).pack(
            side="right"
        )
        self.contract_label = ttk.Label(
            card.body,
            text="",
            style="CardLabel.TLabel",
            justify="left",
            wraplength=self.app.px(900, minimum=420),
        )
        self.contract_label.pack(anchor="w", pady=(0, self.app.px(6, minimum=2)))
        self.replay_tree = _scrollable_table(
            card.body,
            self.REPLAY_COLUMNS,
            expand=False,
            bus=self.app.bus,
            empty_text=("No recordings yet.\nA run writes them when replay recording is enabled."),
        )
        self.replay_tree.bind("<<TreeviewSelect>>", self._on_replay_selected)
        self.tag_style(self.replay_tree, "detailed", "ok")
        self.tag_style(self.replay_tree, "light", "text_dim")
        return card

    def _build_summary_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Policy input at a glance", "One recorded tick at a time")
        controls = ttk.Frame(card.body, style="CardInner.TFrame")
        controls.pack(fill="x")
        ttk.Button(
            controls, text="◀ Prev", style="Ghost.TButton", command=lambda: self._step_tick(-1)
        ).pack(side="left")
        ttk.Button(
            controls, text="Next ▶", style="Ghost.TButton", command=lambda: self._step_tick(1)
        ).pack(side="left", padx=(4, 8))
        self.tick_var = tk.StringVar(value="0")
        ttk.Label(controls, text="Tick").pack(side="left")
        tick_entry = ttk.Entry(controls, textvariable=self.tick_var, width=8)
        tick_entry.pack(side="left", padx=(4, 4))
        tick_entry.bind("<Return>", lambda _e: self._jump_to_tick())
        ttk.Button(controls, text="Go", style="Ghost.TButton", command=self._jump_to_tick).pack(
            side="left"
        )
        self.tick_label = ttk.Label(controls, text="no replay selected", style="CardLabel.TLabel")
        self.tick_label.pack(side="left", padx=(12, 0))
        self.summary_label = ttk.Label(
            card.body,
            text="Select a detailed replay to decode what the policy received.",
            style="CardLabel.TLabel",
            justify="left",
            wraplength=self.app.px(900, minimum=420),
        )
        self.summary_label.pack(anchor="w", pady=(self.app.px(8, minimum=4), 0))
        self.event_label = ttk.Label(
            card.body,
            text="",
            style="PageSubtitle.TLabel",
            justify="left",
            wraplength=self.app.px(900, minimum=420),
        )
        self.event_label.pack(anchor="w", pady=(self.app.px(4, minimum=2), 0))
        return card

    def _build_vector_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent,
            "Observation vector (raw)",
            "Grouped by contract section; the value column is the recorded tick",
        )
        self.vector_tree = _scrollable_table(
            card.body,
            self.VECTOR_COLUMNS,
            bus=self.app.bus,
            empty_text="Select a detailed replay to see the values the policy received.",
        )
        self.tag_style(self.vector_tree, "section", "accent")
        return card

    def _build_contacts_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent,
            "Contacts / enemies",
            "Relative position of each tracked enemy — never a world coordinate",
        )
        self.contact_tree = _scrollable_table(
            card.body,
            self.CONTACT_COLUMNS,
            expand=False,
            bus=self.app.bus,
            empty_text="Select a detailed replay to see the tracked contacts.",
        )
        return card

    def _build_world_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent,
            "Objects, cover and memory",
            "Geometry the agent can see or remembers",
        )
        self.world_tree = _scrollable_table(
            card.body,
            self.WORLD_COLUMNS,
            expand=False,
            bus=self.app.bus,
            empty_text="Select a detailed replay to see visible objects and memory.",
        )
        return card

    def _build_hearing_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Hearing", "Directional perception, not ground truth")
        self.audio_tree = _scrollable_table(
            card.body,
            self.AUDIO_COLUMNS,
            expand=False,
            bus=self.app.bus,
            empty_text="Select a detailed replay to see what was heard.",
        )
        return card

    def _build_action_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(parent, "Action the policy took", "One row per action component")
        self.action_tree = _scrollable_table(
            card.body,
            self.ACTION_COLUMNS,
            expand=False,
            bus=self.app.bus,
            empty_text="Select a detailed replay to see the recorded action.",
        )
        return card

    def _build_evidence_card(self, parent: tk.Misc) -> tk.Widget:
        card = self.card(
            parent,
            "TTK Testing evidence",
            "Verified / calibration-required / excluded, from sandboxai.ttk_testing",
        )
        self.evidence_label = ttk.Label(
            card.body, text="Loading evidence manifest…", style="CardLabel.TLabel", justify="left"
        )
        self.evidence_label.pack(anchor="w", pady=(0, self.app.px(6, minimum=3)))
        self.evidence_tree = _scrollable_table(
            card.body,
            self.EVIDENCE_COLUMNS,
            expand=False,
            bus=self.app.bus,
            empty_text="No TTK Testing evidence manifest available yet.",
        )
        # Registered once: ``tag_style`` records the (tree, tag, role) triple so
        # a theme switch replays it, and re-registering per row would grow that
        # list every poll.
        self.tag_style(self.evidence_tree, "ttk-verified", "ok")
        self.tag_style(self.evidence_tree, "ttk-pending", "warn")
        self.tag_style(self.evidence_tree, "ttk-excluded", "text_dim")
        return card

    # -- polling -----------------------------------------------------------

    def refresh(self) -> None:
        self._poll_count += 1
        # The replay list is re-read on a timer, not on every 600 ms tick: it
        # only changes when a run writes a recording, and this scan is the
        # only disk work the page does. The scan keeps running while a replay
        # is selected - otherwise a replay written during the session would
        # never appear without pressing Rescan - and an unchanged listing is
        # left alone in ``_on_replays``, so the operator's selection and
        # scroll position survive it.
        if self._scan_requested or self._poll_count % self.RESCAN_EVERY == 1:
            self._scan_requested = False
            self.submit_poll("replays", self.adapter.list_replays, self._on_replays)
        if not self._evidence_loaded:
            self.submit_poll("ttk-evidence", self.adapter.ttk_evidence, self._on_evidence)

    def _rescan_replays(self) -> None:
        self._scan_requested = True
        self.submit_poll("replays", self.adapter.list_replays, self._on_replays)

    def _on_replays(
        self, replays: list[dict[str, Any]] | None, error: BaseException | None
    ) -> None:
        if error is not None or replays is None:
            self.report_error("Replay scan failed", error or RuntimeError("unknown error"))
            return
        signature = tuple(
            (entry.get("path"), entry.get("size_bytes"), entry.get("modified")) for entry in replays
        )
        self._replays = list(replays)
        if signature == self._replay_signature and self._replay is not None:
            # Nothing on disk changed since the last scan. Rebuilding the
            # table would drop and re-add every row for no reason, and with it
            # the operator's selection and scroll position.
            return
        self._replay_signature = signature
        previous = str((self._replay or {}).get("path") or "")
        rows = vm.replay_table_rows(self._replays)
        selected = ""
        self.replay_tree.delete(*self.replay_tree.get_children())
        for index, row in enumerate(rows):
            item = self.replay_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    row["name"],
                    row["run"],
                    row["detail"],
                    row["ticks"],
                    row["seed"],
                    row["map"],
                    row["scenario"],
                    row["curriculum"],
                    row["enemies"],
                    row["observations"],
                    row["size"],
                ),
            )
            self.replay_tree.item(
                item, tags=("detailed" if row["observations"] == "yes" else "light",)
            )
            if str(self._replays[index].get("path")) == previous:
                selected = item
        if not rows:
            # A scan that finds nothing must also clear what the last scan
            # decoded: the values on screen belong to files that are gone, and
            # leaving them up would present stale evidence as current.
            self._clear_decode()
            self.replay_info_label.configure(
                text=(
                    "No replay found under "
                    f"{self.adapter.output_root}. Train a run with replay_detail=detailed "
                    "(Control Center → Training → Advanced, or --replay-detail detailed) "
                    "and this page will decode it."
                )
            )
            return
        detailed = sum(1 for row in rows if row["observations"] == "yes")
        self.replay_info_label.configure(
            text=(
                f"{len(rows)} replay(s) found — {detailed} store the observation vector. "
                "Only a detailed replay can show what the policy received."
            )
        )
        if selected:
            self.replay_tree.selection_set(selected)
        self._select_row(selected or "0")

    def _select_row(self, item: str) -> None:
        try:
            index = int(item)
        except ValueError:
            return
        if not 0 <= index < len(self._replays):
            return
        self.replay_tree.selection_set(item)
        self._load_replay(self._replays[index], 0)

    def _on_replay_selected(self, _event: object = None) -> None:
        selection = self.replay_tree.selection()
        if not selection:
            return
        try:
            index = int(selection[0])
        except ValueError:
            return
        if not 0 <= index < len(self._replays):
            return
        self._load_replay(self._replays[index], 0)

    def _load_replay(self, replay: dict[str, Any], tick: int) -> None:
        path = str(replay.get("path") or "")
        if not path:
            return
        self.submit_poll(
            "replay-tick",
            partial(self.adapter.replay_stats, path, tick),
            self._on_replay_stats,
        )

    def _on_replay_stats(self, result: dict[str, Any] | None, error: BaseException | None) -> None:
        if error is not None or not result:
            self.report_error(
                "Replay decode failed", error or RuntimeError("no replay data returned")
            )
            # Clear first, then say why: a failed decode must not leave the
            # previous replay's values (or its contract warning) on screen.
            self._clear_decode()
            self.summary_label.configure(
                text="This replay could not be decoded. Rescan after the run finishes writing it."
            )
            return
        self._replay = result
        self._tick = int(result.get("tick_index") or 0)
        self.tick_var.set(str(self._tick))
        view = vm.stats_source_view(result)
        self.replay_info_label.configure(text=f"{view['headline']}\n{view['detail']}")
        contract = vm.replay_contract_view(result)
        # A theme style, never a raw colour: the label must repaint with the
        # rest of the window when the operator switches theme.
        self.contract_label.configure(
            text=contract["text"],
            style="CardLabel.TLabel" if contract["matches"] else "Warning.TLabel",
        )
        summary = vm.observation_summary_view(result.get("observation"))
        self.summary_label.configure(text=summary["summary"])
        count = int(result.get("tick_count") or 0)
        self.tick_label.configure(
            text=f"tick {self._tick + 1} / {count}" if count else "this replay has no ticks"
        )
        parts: list[str] = []
        events = result.get("events") or []
        if events:
            parts.append(
                "Events at this tick: " + ", ".join(str(event.get("kind")) for event in events)
            )
        reward = result.get("reward")
        if isinstance(reward, (int, float)):
            parts.append(f"reward {vm.format_number(reward, 3)}")
        self.event_label.configure(text="   ·   ".join(parts))
        self._fill_vector(result.get("observation"))
        self._fill_contacts(result.get("observation"))
        self._fill_world(result.get("observation"))
        self._fill_audio(result.get("observation"))
        self._fill_action(result.get("action"))

    def _clear_decode(self) -> None:
        """Show the contract without a recording again.

        Used when a scan finds no replay and when a decode fails: the page
        must not keep presenting values (or a contract warning) from a file
        that is no longer there. Everything returns to the same
        "no recording yet" state the page starts in, with the contract table
        still complete - the contract does not need a replay, only the values
        do.
        """
        self._replay = None
        self._tick = 0
        self.tick_var.set("0")
        self.tick_label.configure(text="no replay selected")
        self.summary_label.configure(
            text="Select a detailed replay to decode what the policy received."
        )
        self.event_label.configure(text="")
        self.contract_label.configure(text="", style="CardLabel.TLabel")
        self._fill_vector(None)
        self._fill_contacts(None)
        self._fill_world(None)
        self._fill_audio(None)
        self._fill_action(None)

    def _step_tick(self, delta: int) -> None:
        if self._replay is None:
            return
        count = int(self._replay.get("tick_count") or 0)
        if count <= 0:
            return
        target = max(0, min(count - 1, self._tick + delta))
        if target == self._tick:
            return
        path = str(self._replay.get("path") or "")
        self.submit_poll(
            "replay-tick",
            partial(self.adapter.replay_stats, path, target),
            self._on_replay_stats,
        )

    def _jump_to_tick(self) -> None:
        if self._replay is None:
            return
        count = max(1, int(self._replay.get("tick_count") or 1))
        try:
            wanted = int(self.tick_var.get().strip())
        except ValueError:
            self.tick_var.set(str(self._tick))
            return
        target = max(0, min(count - 1, wanted))
        if target == self._tick:
            return
        path = str(self._replay.get("path") or "")
        self.submit_poll(
            "replay-tick",
            partial(self.adapter.replay_stats, path, target),
            self._on_replay_stats,
        )

    def _on_evidence(self, summary: dict[str, Any] | None, error: BaseException | None) -> None:
        # One result is enough: the manifest is a static local table, so it
        # either loaded or the page says it could not. Polling forever would
        # just repeat a failure every 600 ms.
        self._evidence_loaded = True
        if error is not None or summary is None:
            self.evidence_label.configure(text="Evidence manifest unavailable.")
            return
        view = vm.ttk_evidence_view(summary)
        self.evidence_label.configure(text=f"{view['headline']}   ·   checked {view['checked_on']}")
        self.evidence_tree.delete(*self.evidence_tree.get_children())
        for row in vm.ttk_evidence_rows(summary):
            item = self.evidence_tree.insert(
                "",
                "end",
                values=(
                    row["mechanic"],
                    row["status"],
                    row["measured_value"],
                    row["rule"],
                    row["source"],
                ),
            )
            tag = {
                "verified": "ttk-verified",
                "calibration required": "ttk-pending",
                "excluded": "ttk-excluded",
            }.get(str(row["status"]))
            if tag:
                self.evidence_tree.item(item, tags=(tag,))

    # -- tables ------------------------------------------------------------

    def _fill_vector(self, observation: Any) -> None:
        self.vector_tree.delete(*self.vector_tree.get_children())
        rows = vm.observation_field_rows(observation)
        for section, fields in vm.observation_section_rows(rows):
            parent = self.vector_tree.insert("", "end", text=section, open=True, tags=("section",))
            for row in fields:
                self.vector_tree.insert(
                    parent,
                    "end",
                    values=(
                        row["index"],
                        row["field"],
                        row["value"],
                        row["meaning"],
                        row["normalization"],
                    ),
                )

    def _fill_table(self, tree: Any, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", "end", values=tuple(row.get(column, "") for column in columns))

    def _fill_contacts(self, observation: Any) -> None:
        self._fill_table(
            self.contact_tree,
            vm.contact_rows(observation),
            tuple(key for key, _title, _width in self.CONTACT_COLUMNS),
        )

    def _fill_world(self, observation: Any) -> None:
        self._fill_table(
            self.world_tree,
            vm.world_rows(observation),
            tuple(key for key, _title, _width in self.WORLD_COLUMNS),
        )

    def _fill_audio(self, observation: Any) -> None:
        self._fill_table(
            self.audio_tree,
            vm.audio_rows(observation),
            tuple(key for key, _title, _width in self.AUDIO_COLUMNS),
        )

    def _fill_action(self, action: Any) -> None:
        self._fill_table(
            self.action_tree,
            vm.action_rows(action),
            tuple(key for key, _title, _width in self.ACTION_COLUMNS),
        )


PAGE_CLASSES: tuple[type[Page], ...] = (
    DashboardPage,
    TrainingPage,
    BenchmarkPage,
    EvaluationPage,
    RunsPage,
    StatsPage,
    SystemPage,
    SettingsPage,
)

#: Every page's movable cards, in the shape the layout studio and the
#: per-page layout boards consume. Pages without an entry keep their fixed
#: composition.
PAGE_WIDGETS: dict[str, tuple[WidgetSpec, ...]] = {
    page_class.title: page_class.widgets for page_class in PAGE_CLASSES if page_class.widgets
}

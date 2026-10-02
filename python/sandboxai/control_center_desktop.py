"""Tk desktop Control Center backed exclusively by :mod:`sandboxai.adapter`.

Architecture (kept intentionally unchanged from the first version):

    Desktop Control Center (this file)
        -> SandboxAIAdapter (python/sandboxai/adapter.py)
            -> existing CLI/training/evaluation/benchmark infrastructure
                -> headless Godot workers

This is the Control Center: a normal movable/resizable desktop application
that operates the *headless* training stack. There is no rendering, no
environment visualisation and no human-play mode — the window only reads
what the backends actually measured (agents, workers, environments, steps,
throughput, rewards/losses, resources, checkpoints, logs) and issues
lifecycle commands through the cooperative control protocol.

This module owns the *shell*: window chrome, navigation, DPI scaling, theme
and layout switching, toasts and polling cadence. Pages live in
:mod:`sandboxai.control_center_pages`, presentation primitives in
:mod:`sandboxai.control_center_ui`, tokens in
:mod:`sandboxai.control_center_theme`, and the movable-card model in
:mod:`sandboxai.control_center_layout`. No page invents a metric: a value
the adapter did not return is shown as "n/a", never estimated.

Tk is a small, supported local desktop dependency that ships with CPython on
Windows; no additional GUI toolkit is required. All adapter calls that touch
disk or a subprocess run on a background thread pool
(:class:`BackgroundRunner`) and hand their result back to the Tk thread
through a queue, so a slow poll (a big run directory, a stuck process list)
never freezes the window.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import (  # messagebox re-export keeps the public test/embedding seam stable
    messagebox,
    ttk,
)

from .adapter import SandboxAIAdapter
from .control_center_layout import (
    LayoutState,
    PresetError,
    PresetStore,
    deserialize,
    normalize,
    safe_preset_name,
)
from .control_center_pages import PAGE_CLASSES, PAGE_WIDGETS, Page
from .control_center_theme import (
    DENSITIES,
    LAYOUT_MODES,
    MOTION_LEVELS,
    THEME_NAMES,
    THEMES,
    FontSpec,
    PreferencesStore,
    Theme,
    UiPreferences,
    UiScale,
    apply_ttk_styles,
    enable_dpi_awareness,
    get_theme,
    prefs_as_dict,
)
from .control_center_ui import (
    LayoutBus,
    MotionController,
    StatusDot,
    ThemeBus,
    ToastHost,
)
from .control_center_widgets import BackgroundRunner, ToolTip

__all__ = ["ControlCenter", "PAGE_CLASSES", "PAGE_WIDGETS", "main", "messagebox"]


class ControlCenter(tk.Tk):
    """The operator window: navigation, theming and the page host."""

    POLL_MS = 600

    def __init__(self, adapter: SandboxAIAdapter | None = None) -> None:
        # DPI awareness must be enabled before the window exists, or Windows
        # scales the already-laid-out bitmap and every measurement in the UI
        # is a lie.
        enable_dpi_awareness()
        super().__init__()
        self.title("SandboxAI Studio")
        self.adapter = adapter or SandboxAIAdapter()
        self.scale = UiScale.from_root(self)
        self.prefs_store = PreferencesStore(self.adapter.project_root)
        self.prefs: UiPreferences = self.prefs_store.load()
        self.bus = ThemeBus(
            get_theme(self.prefs.theme),
            scale=self.scale,
            density=DENSITIES.get(self.prefs.density, DENSITIES["comfort"]),
        )
        self.motion = MotionController(self, self.prefs.motion)
        self.preset_store = PresetStore(self.adapter.project_root)
        self.layout_bus = LayoutBus(self._load_layout())
        self.background = BackgroundRunner(self)

        self._apply_window_geometry()
        self.style = apply_ttk_styles(
            self, self.bus.theme, density=self.bus.density, scale=self.scale
        )

        self.pages: dict[str, Page] = {}
        self._current: Page | None = None
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._nav_indicator: tk.Frame | None = None
        self._layouter: tk.Misc | None = None
        self.toasts = ToastHost(self, self.bus, self.motion, width=self.bus.px(380, minimum=280))
        self._build_shell()
        self.bind("<Control-k>", lambda _event: self.open_command_palette())
        self.bind("<Control-K>", lambda _event: self.open_command_palette())
        self.after(self.POLL_MS, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Preferences, layout, theme
    # ------------------------------------------------------------------

    @property
    def palette(self) -> Theme:
        """The live theme; pages read every colour from here."""
        return self.bus.theme

    def color(self, role: str) -> str:
        return self.bus.theme.color(role)

    def font(self, role: str, *, bold: bool = False, mono: bool = False) -> FontSpec:
        return self.bus.font(role, bold=bold, mono=mono)

    def px(self, value: int, *, minimum: int = 0) -> int:
        return self.bus.px(value, minimum=minimum)

    def notify(self, message: str, *, kind: str = "info", timeout_ms: int = 4200) -> None:
        """Show a transient toast without touching the persistent status pill."""
        self.toasts.show(message, kind=kind, timeout_ms=timeout_ms)

    def _load_layout(self) -> LayoutState:
        base = normalize(LayoutState(pages={}), PAGE_WIDGETS)
        if not self.prefs.active_preset:
            return base
        loaded = self.preset_store.load_layout(self.prefs.active_preset, PAGE_WIDGETS)
        return loaded if loaded is not None else base

    def _apply_window_geometry(self) -> None:
        self.geometry(self.prefs.geometry or "1720x1000")
        self.minsize(self.px(1280, minimum=1000), self.px(800, minimum=640))
        if self.prefs.show_grid:
            with contextlib.suppress(tk.TclError):
                self.option_add("*Canvas.background", self.bus.theme.bg)

    def save_preferences(self) -> None:
        """Persist theme/layout/density/motion, window size and last page."""
        with contextlib.suppress(tk.TclError):
            self.prefs.geometry = self.winfo_geometry()
        self.prefs.theme = self.bus.theme.name
        self.prefs.density = self.bus.density.name
        self.prefs.motion = self.motion.level
        self.prefs.last_page = self._current.title if self._current else self.prefs.last_page
        self.prefs_store.save(self.prefs)

    def preferences_snapshot(self) -> dict[str, object]:
        """Preferences as a plain dict (diagnostics and tests)."""
        return prefs_as_dict(self.prefs)

    def set_theme(self, name: str, *, persist: bool = True) -> None:
        """Switch the palette and repaint the whole window."""
        theme = get_theme(name)
        self.prefs.theme = theme.name
        self.bus.set_theme(
            theme, scale=self.scale, density=DENSITIES.get(self.prefs.density, DENSITIES["comfort"])
        )
        self.style = apply_ttk_styles(self, theme, density=self.bus.density, scale=self.scale)
        self._restyle_shell()
        for page in self.pages.values():
            page.on_theme(theme)
        if persist:
            self.save_preferences()
        self.notify(f"Theme: {theme.label}", kind="info", timeout_ms=2200)

    def set_density(self, name: str, *, persist: bool = True) -> None:
        """Switch spacing/row height and rebuild the built pages."""
        density = DENSITIES.get(name, DENSITIES["comfort"])
        self.prefs.density = density.name
        self.bus.set_theme(self.bus.theme, scale=self.scale, density=density)
        self.style = apply_ttk_styles(self, self.bus.theme, density=density, scale=self.scale)
        self._restyle_shell()
        for page in list(self.pages.values()):
            page.rebuild_after_restyle()
        if persist:
            self.save_preferences()
        self.notify(f"Density: {density.label}", kind="info", timeout_ms=2200)

    def set_motion(self, name: str, *, persist: bool = True) -> None:
        """Switch the animation level (``off`` disables every tween)."""
        level = name if name in MOTION_LEVELS else "normal"
        self.prefs.motion = level
        self.motion.set_level(level)
        if persist:
            self.save_preferences()
        self.notify(f"Motion: {MOTION_LEVELS[level]}", kind="info", timeout_ms=2000)

    def set_layout_mode(self, name: str, *, persist: bool = True) -> None:
        """Switch between the rail, topbar and command-board shells.

        Rebuilding the shell recreates every page, which drops live page
        state (a selected run, a scrolled log). It therefore only happens
        when the mode actually changes: re-picking the current layout is a
        no-op instead of an expensive, state-destroying one.
        """
        mode = name if name in LAYOUT_MODES else "rail"
        changed = mode != self.prefs.layout
        self.prefs.layout = mode
        if changed:
            self._rebuild_shell()
        if persist:
            self.save_preferences()
        if changed:
            self.notify(f"Layout: {LAYOUT_MODES[mode]}", kind="info", timeout_ms=2200)

    def set_layout_state(self, state: LayoutState, *, persist: bool = True) -> None:
        """Publish a new movable-card layout to every page board."""
        self.layout_bus.set_state(normalize(state, PAGE_WIDGETS))
        if persist:
            self.save_preferences()

    def save_layout_preset(self, name: str, *, note: str = "") -> str:
        """Persist the current layout + appearance as a named preset."""
        clean = self.preset_store.save(
            name,
            self.layout_bus.state,
            appearance={
                "theme": self.bus.theme.name,
                "layout": self.prefs.layout,
                "density": self.bus.density.name,
                "motion": self.motion.level,
                "radius": self.prefs.radius,
                "show_glow": self.prefs.show_glow,
                "show_grid": self.prefs.show_grid,
            },
            note=note,
        )
        self.prefs.active_preset = clean
        self.save_preferences()
        self.notify(f"Preset saved: {clean}", kind="ok")
        return clean

    def apply_layout_preset(self, name: str) -> bool:
        """Load a preset's layout and appearance; returns whether it applied."""
        document = self.preset_store.load(name)
        if document is None:
            self.notify(self.preset_store.error or f"Preset '{name}' is unreadable", kind="error")
            return False
        layout = document.get("layout")
        state = deserialize(layout, PAGE_WIDGETS) if isinstance(layout, dict) else None
        if state is None:
            return False
        previous_mode = self.prefs.layout
        appearance = document.get("appearance")
        if isinstance(appearance, dict):
            theme = appearance.get("theme")
            if isinstance(theme, str) and theme in THEMES:
                self.bus.set_theme(get_theme(theme), scale=self.scale, density=self.bus.density)
                self.prefs.theme = theme
                self.style = apply_ttk_styles(
                    self, self.bus.theme, density=self.bus.density, scale=self.scale
                )
                self._restyle_shell()
            density = appearance.get("density")
            if isinstance(density, str) and density in DENSITIES:
                self.prefs.density = density
                self.bus.set_theme(self.bus.theme, scale=self.scale, density=DENSITIES[density])
                self.style = apply_ttk_styles(
                    self, self.bus.theme, density=DENSITIES[density], scale=self.scale
                )
            mode = appearance.get("layout")
            if isinstance(mode, str) and mode in LAYOUT_MODES:
                self.prefs.layout = mode
            motion = appearance.get("motion")
            if isinstance(motion, str) and motion in MOTION_LEVELS:
                self.motion.set_level(motion)
                self.prefs.motion = motion
        self.prefs.active_preset = safe_preset_name(name)
        mode_changed = self.prefs.layout != previous_mode
        self.layout_bus.set_state(state)
        if mode_changed:
            # Only a different shell needs new widgets; a preset that keeps
            # the current shell must not throw away live page state.
            self._rebuild_shell()
        self.save_preferences()
        self.notify(f"Preset applied: {self.prefs.active_preset}", kind="ok")
        return True

    def rename_layout_preset(self, old: str, new: str) -> bool:
        """Rename a saved preset, keeping the active marker on it."""
        try:
            renamed = self.preset_store.rename(old, new)
        except PresetError as exc:
            self.notify(f"Preset could not be renamed: {exc}", kind="error")
            return False
        if self.prefs.active_preset == old:
            self.prefs.active_preset = renamed
            self.save_preferences()
        self.notify(f"Preset renamed: {old} → {renamed}", kind="info")
        return True

    def delete_layout_preset(self, name: str) -> bool:
        """Delete a preset, clearing the active marker when it was the one."""
        removed = self.preset_store.delete(name)
        if removed:
            with contextlib.suppress(PresetError):
                if self.prefs.active_preset == safe_preset_name(name):
                    self.prefs.active_preset = ""
                    self.save_preferences()
            self.notify(f"Preset deleted: {name}", kind="warn")
        return removed

    def list_layout_presets(self) -> list[str]:
        return self.preset_store.list_presets()

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def _build_shell(self) -> None:
        self.configure(background=self.bus.theme.bg)
        self.outer = ttk.Frame(self, style="Shell.TFrame")
        self.outer.pack(fill="both", expand=True)
        self._header = self._build_header(self.outer)
        self._nav_host: ttk.Frame | None = None
        self._content_host: ttk.Frame | None = None
        self._build_body()
        self.show_page(self.prefs.last_page if self.prefs.last_page in self.pages else "Dashboard")

    def _rebuild_shell(self) -> None:
        """Recreate the body for a different shell layout.

        Tk cannot reparent a widget, so the pages are rebuilt from their
        classes. That is deliberate: a shell change is a rare, explicit
        action, and rebuilding is the only way to guarantee that no widget
        is left in a destroyed parent (which would show up as a frozen or
        half-drawn window).
        """
        current_title = self._current.title if self._current is not None else None
        for page in list(self.pages.values()):
            with contextlib.suppress(tk.TclError):
                page.destroy()
        self.pages.clear()
        self._current = None
        if self._nav_host is not None:
            with contextlib.suppress(tk.TclError):
                self._nav_host.destroy()
        if self._content_host is not None:
            with contextlib.suppress(tk.TclError):
                self._content_host.destroy()
        self._nav_buttons.clear()
        self._build_body()
        target = current_title if current_title in self.pages else "Dashboard"
        self.show_page(target)

    def _build_header(self, parent: tk.Misc) -> ttk.Frame:
        theme = self.bus.theme
        header = ttk.Frame(parent, style="Header.TFrame", padding=(self.px(22), self.px(11)))
        header.pack(fill="x")
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left")
        self.brand_title = ttk.Label(brand, text="SandboxAI", style="BrandTitle.TLabel")
        self.brand_title.pack(side="left")
        self.brand_sub = ttk.Label(
            brand, text="   Training & Calibration Studio", style="BrandEyebrow.TLabel"
        )
        self.brand_sub.pack(side="left", pady=(self.px(4), 0))

        # Quick switches live in the header so switching a theme/layout is
        # always one click, never a trip to Settings.
        self.quick = ttk.Frame(header, style="Header.TFrame")
        self.quick.pack(side="right", padx=(self.px(12), 0))
        self.theme_picker = ttk.Combobox(
            self.quick,
            values=tuple(THEMES[name].label for name in THEME_NAMES),
            state="readonly",
            width=self.px(14, minimum=10),
        )
        self.theme_picker.set(theme.label)
        self.theme_picker.pack(side="right", padx=(self.px(6), 0))
        self.theme_picker.bind("<<ComboboxSelected>>", self._on_theme_picked)
        self.layout_picker = ttk.Combobox(
            self.quick,
            values=tuple(LAYOUT_MODES.values()),
            state="readonly",
            width=self.px(14, minimum=10),
        )
        self.layout_picker.set(LAYOUT_MODES[self.prefs.layout])
        self.layout_picker.pack(side="right", padx=(self.px(6), 0))
        self.layout_picker.bind("<<ComboboxSelected>>", self._on_layout_picked)

        status = ttk.Frame(header, style="Header.TFrame")
        status.pack(side="right", padx=(self.px(12), 0))
        self.status_dot = StatusDot(status, self.bus, size=self.px(10, minimum=8))
        self.status_dot.pack(side="left", padx=(0, self.px(6, minimum=4)))
        self.status_label = ttk.Label(status, text="Ready", style="OperatorStatus.TLabel")
        self.status_label.pack(side="left")
        self.telemetry_badge = ttk.Label(
            status,
            text="Bridge v3  ·  84-Obs  ·  6-Head",
            style="Pill.TLabel",
        )
        self.telemetry_badge.pack(side="left", padx=(self.px(8, minimum=4), 0))
        self.preset_label = ttk.Label(status, text="", style="Pill.TLabel")
        self.preset_label.pack(side="left", padx=(self.px(8, minimum=4), 0))
        self._refresh_preset_label()
        tk.Frame(header, height=1, background=theme.border, borderwidth=0).pack(
            side="bottom", fill="x"
        )
        return header

    def _restyle_shell(self) -> None:
        """Repaint everything that caches a colour outside the ttk styles."""
        theme = self.bus.theme
        self.configure(background=theme.bg)
        self.theme_picker.set(theme.label)
        self.layout_picker.set(LAYOUT_MODES[self.prefs.layout])
        self._refresh_preset_label()
        if self._nav_indicator is not None:
            with contextlib.suppress(tk.TclError):
                self._nav_indicator.configure(background=theme.accent)

    def _refresh_preset_label(self) -> None:
        text = f"Preset: {self.prefs.active_preset}" if self.prefs.active_preset else "No preset"
        with contextlib.suppress(tk.TclError):
            self.preset_label.configure(text=text)

    def _on_theme_picked(self, _event: object = None) -> None:
        labels = {THEMES[name].label: name for name in THEME_NAMES}
        picked = labels.get(str(self.theme_picker.get()))
        if picked is not None:
            self.set_theme(picked)

    def _on_layout_picked(self, _event: object = None) -> None:
        modes = {label: key for key, label in LAYOUT_MODES.items()}
        picked = modes.get(str(self.layout_picker.get()))
        if picked is not None:
            self.set_layout_mode(picked)

    def _build_body(self) -> None:
        mode = self.prefs.layout
        body = ttk.Frame(self.outer, style="Shell.TFrame")
        body.pack(fill="both", expand=True)
        if mode == "topbar":
            self._build_topbar(body)
            return
        nav = ttk.Frame(
            body,
            style="Nav.TFrame",
            width=self.px(238, minimum=180),
            padding=(self.px(12), self.px(16)),
        )
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        nav.bind("<Enter>", lambda _e: self._set_nav_hover(True), add="+")
        nav.bind("<Leave>", lambda _e: self._set_nav_hover(False), add="+")
        tk.Frame(body, width=1, background=self.bus.theme.border, borderwidth=0).pack(
            side="left", fill="y"
        )
        self._nav_host = nav
        self._nav_indicator = tk.Frame(nav, height=2, background=self.bus.theme.accent)
        self._build_nav_items(nav, vertical=True)
        content = ttk.Frame(body, style="Content.TFrame", padding=(self.px(24), self.px(18)))
        content.pack(side="left", fill="both", expand=True)
        self._content_host = content
        self._layouter = content
        self._build_pages(content)

    def _build_topbar(self, body: ttk.Frame) -> None:
        bar = ttk.Frame(body, style="Nav.TFrame", padding=(self.px(14), self.px(8)))
        bar.pack(fill="x")
        self._nav_host = bar
        self._build_nav_items(bar, vertical=False)
        tk.Frame(body, height=1, background=self.bus.theme.border, borderwidth=0).pack(fill="x")
        content = ttk.Frame(body, style="Content.TFrame", padding=(self.px(24), self.px(18)))
        content.pack(fill="both", expand=True)
        self._content_host = content
        self._layouter = content
        self._build_pages(content)

    def _build_nav_items(self, host: tk.Misc, *, vertical: bool) -> None:
        groups = {0: "OVERVIEW", 1: "WORKFLOWS", 4: "ARTIFACTS", 5: "SYSTEM"}
        for index, page_class in enumerate(PAGE_CLASSES):
            if vertical and index in groups:
                if index:
                    ttk.Separator(host).pack(fill="x", pady=(self.px(14), self.px(8)))
                ttk.Label(
                    host,
                    text=groups[index],
                    style="NavGroup.TLabel",
                ).pack(fill="x", padx=self.px(10), pady=(0, self.px(4)))
            button = ttk.Button(
                host,
                text=page_class.title,
                style="Nav.TButton",
                command=lambda name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )
            if vertical:
                button.pack(fill="x", pady=1)
            else:
                button.pack(side="left", padx=(0, self.px(4, minimum=2)))
            ToolTip(button, f"Open {page_class.title}   ·   Ctrl+{index + 1}")
            self._nav_buttons[page_class.title] = button
            self.bind(
                f"<Control-Key-{index + 1}>",
                lambda _evt, name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )
        if vertical:
            ttk.Separator(host).pack(fill="x", pady=self.px(14))
            ttk.Label(
                host,
                text=(
                    "Ctrl+K  command palette\n"
                    f"Ctrl+1 .. Ctrl+{len(PAGE_CLASSES)}  pages\n"
                    "Headless Godot Bridge v3"
                ),
                style="NavFootnote.TLabel",
                justify="left",
            ).pack(anchor="w", padx=self.px(10))

    def _set_nav_hover(self, hovering: bool) -> None:
        # Subtle affordance: the shell surface lifts slightly under the pointer,
        # which is what makes the navigation feel alive without any motion.
        target = self.bus.theme.card if hovering else self.bus.theme.shell
        if self._nav_host is None:
            return
        with contextlib.suppress(tk.TclError):
            self._nav_host.configure(style="Shell.TFrame")
            self._tween_shell_background(target)

    def _tween_shell_background(self, target: str) -> None:
        del target  # ttk styles cannot be tweened per-frame; the pill carries the motion.
        return

    def _build_pages(self, host: tk.Misc) -> None:
        for page_class in PAGE_CLASSES:
            if page_class.title in self.pages:
                continue
            self.pages[page_class.title] = page_class(host, self)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_page(self, name: str) -> None:
        if name not in self.pages:
            raise KeyError(f"unknown page: {name}")
        if self._current is not None and self._current.title == name:
            return
        if self._current is not None:
            self._current.pack_forget()
        page = self.pages[name]
        page.pack(fill="both", expand=True)
        self._current = page
        for title, button in self._nav_buttons.items():
            selected = title == name
            button.configure(style="NavSelected.TButton" if selected else "Nav.TButton")
        self._move_nav_indicator(name)
        page.show()
        self.prefs.last_page = name
        self.set_status(f"{name}  ·  ready")

    def _move_nav_indicator(self, name: str) -> None:
        button = self._nav_buttons.get(name)
        indicator = self._nav_indicator
        if indicator is None or button is None or not indicator.winfo_exists():
            return
        try:
            y = button.winfo_y()
            height = button.winfo_height()
        except tk.TclError:  # pragma: no cover - layout race during rebuild
            return
        target = y + max(1, height - 2)
        start = int(indicator.place_info().get("y", target) or target)
        animate = abs(target - start) > 4

        def place_at(value: float) -> None:
            with contextlib.suppress(tk.TclError):
                indicator.place(
                    x=self.px(10),
                    y=int(value),
                    width=self.px(3, minimum=2),
                    height=self.px(18, minimum=10),
                )
                indicator.lift()

        if animate:
            self.motion.tween(140, lambda progress: place_at(start + (target - start) * progress))
        else:
            place_at(target)

    # ------------------------------------------------------------------
    # Status, pages and lifecycle
    # ------------------------------------------------------------------

    def set_status(self, message: str, error: bool = False, *, toast: bool | None = None) -> None:
        """Update the persistent status pill, and toast when it matters."""
        self.status_label.configure(
            text=message,
            foreground=self.bus.theme.error if error else self.bus.theme.ok,
        )
        self.status_dot.set_state(
            self.bus.theme.error if error else self.bus.theme.ok,
            pulse=False,
            motion=self.motion,
        )
        if toast is True or (toast is None and error):
            self.notify(message, kind="error" if error else "info")

    def set_busy(self, busy: bool, message: str = "Working…") -> None:
        """Show a pulsing activity dot while a long operation runs."""
        self.status_label.configure(
            text=message if busy else "Ready",
            foreground=self.bus.theme.warn if busy else self.bus.theme.ok,
        )
        self.status_dot.set_state(
            self.bus.theme.warn if busy else self.bus.theme.ok, pulse=busy, motion=self.motion
        )

    def set_output_root(self, output_root: str) -> None:
        self.background.close()
        self.adapter = SandboxAIAdapter(
            project_root=self.adapter.project_root, output_root=output_root
        )
        self.background = BackgroundRunner(self)
        for page in self.pages.values():
            # close() deliberately discards queued delivery callbacks. A page
            # must therefore not retain a coalesced-poll guard that belonged
            # to the retired runner, or a later visit would never refresh.
            page.reset_polls()
            page.adapter = self.adapter

    def open_command_palette(self) -> None:
        """A keyboard-driven page switcher (Ctrl+K)."""
        window = tk.Toplevel(self)
        window.title("Command palette")
        window.configure(background=self.bus.theme.card)
        window.transient(self)
        width, height = self.px(520, minimum=380), self.px(320, minimum=240)
        window.geometry(f"{width}x{height}+{self.winfo_rootx() + 120}+{self.winfo_rooty() + 90}")
        entry = ttk.Entry(window, font=self.font("body"))
        entry.pack(fill="x", padx=self.px(14), pady=self.px(12))
        listing = tk.Listbox(
            window,
            background=self.bus.theme.panel,
            foreground=self.bus.theme.text,
            selectbackground=self.bus.theme.accent,
            selectforeground=self.bus.theme.on_accent,
            highlightthickness=0,
            borderwidth=0,
            font=self.font("body"),
        )
        listing.pack(fill="both", expand=True, padx=self.px(14), pady=(0, self.px(14)))

        def refresh(_event: object = None) -> None:
            query = entry.get().strip().lower()
            listing.delete(0, "end")
            for title in self.pages:
                if query in title.lower():
                    listing.insert("end", title)
            if listing.size():
                listing.selection_clear(0, "end")
                listing.selection_set(0)

        def choose(_event: object = None) -> None:
            selection = listing.curselection()
            if not selection:
                return
            self.show_page(str(listing.get(selection[0])))
            window.destroy()

        entry.bind("<KeyRelease>", refresh)
        entry.bind("<Return>", choose)
        entry.bind("<Escape>", lambda _e: window.destroy())
        listing.bind("<Double-Button-1>", choose)
        listing.bind("<Return>", choose)
        refresh()
        entry.focus_set()

    def _tick(self) -> None:
        if self._current is not None:
            try:
                self._current.refresh()
            except Exception as exc:  # noqa: BLE001 - a page bug must not stop polling entirely
                self.set_status(
                    f"internal error refreshing {self._current.title}: {exc}", error=True
                )
        self.after(self.POLL_MS, self._tick)

    def _on_close(self) -> None:
        self.save_preferences()
        self.motion.stop_all()
        self.background.close()
        self.adapter.close()
        self.destroy()


def main(project_root: str | None = None, output_root: str = "training") -> int:
    app = ControlCenter(
        adapter=SandboxAIAdapter(project_root=project_root, output_root=output_root)
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

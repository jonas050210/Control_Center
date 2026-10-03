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
from collections.abc import Callable, Iterable
from pathlib import Path
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
    fit_window_geometry,
    normalize,
    safe_preset_name,
)
from .control_center_pages import PAGE_CLASSES, PAGE_WIDGETS, Page
from .control_center_theme import (
    DENSITIES,
    MOTION_LEVELS,
    THEMES,
    FontSpec,
    PreferencesStore,
    Theme,
    UiPreferences,
    UiScale,
    apply_ttk_styles,
    enable_dpi_awareness,
    get_theme,
    normalize_accent,
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

__all__ = ["ControlCenter", "NAV_WIDTH_PX", "PAGE_CLASSES", "PAGE_WIDGETS", "main", "messagebox"]

#: The rail's width as ``(preferred, minimum)``. It is fixed: a collapsible
#: rail has to be narrower than its own page titles, and the window has room
#: for the titles. One constant for the build and every later rescale keeps
#: the rail from twitching the first time the window is resized.
NAV_WIDTH_PX: tuple[int, int] = (238, 180)


def _palette_matches(pages: Iterable[str], query: str) -> list[str]:
    """The page titles a palette query keeps, in page order."""
    needle = query.strip().lower()
    return [title for title in pages if needle in title.lower()]


def _palette_highlight(listing: tk.Listbox, rows: list[str], index: int = 0) -> None:
    """Replace the palette rows and highlight one of them.

    An out-of-range index (the filter shrank the list) falls back to the
    first row, and an empty result simply has nothing highlighted.
    """
    listing.delete(0, "end")
    for row in rows:
        listing.insert("end", row)
    if not listing.size():
        return
    index = max(0, min(index, listing.size() - 1))
    listing.selection_clear(0, "end")
    listing.selection_set(index)
    listing.activate(index)
    listing.see(index)


def _palette_move(listing: tk.Listbox, direction: int) -> str:
    """Move the highlight one row, clamped to the list.

    Returns ``"break"`` so Tk does not also move the listbox cursor itself.
    """
    count = listing.size()
    if not count:
        return "break"
    current = listing.curselection()
    index = max(0, min(count - 1, (current[0] if current else 0) + direction))
    listing.selection_clear(0, "end")
    listing.selection_set(index)
    listing.activate(index)
    listing.see(index)
    return "break"


def _palette_choice(listing: tk.Listbox) -> str:
    """The highlighted page title, or an empty string when nothing is."""
    selection = listing.curselection()
    return str(listing.get(selection[0])) if selection else ""


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
        # Normalize Tk's internal point-to-pixel font scaling to 1.0 after
        # measuring the display DPI in UiScale.from_root(self), so UiScale
        # font sizes map 1:1 to pixels and are never double-scaled on
        # Windows 125%/150% DPI displays.
        with contextlib.suppress(Exception):
            self.tk.call("tk", "scaling", 1.0)
        self.prefs_store = PreferencesStore(self.adapter.project_root)
        self.prefs: UiPreferences = self.prefs_store.load()
        self.bus = ThemeBus(
            get_theme(self.prefs.theme, self.prefs.accent),
            scale=self.scale,
            density=DENSITIES.get(self.prefs.density, DENSITIES["comfort"]),
        )
        self.motion = MotionController(self, self.prefs.motion)
        self.preset_store = PresetStore(self.adapter.project_root)
        self.layout_bus = LayoutBus(self._load_layout())
        self.background = BackgroundRunner(self)
        self._viewport_resize_job: str | None = None
        self._last_viewport_size: tuple[int, int] = (0, 0)
        self._compact_header: bool = False

        self._apply_window_geometry()
        self.style = apply_ttk_styles(
            self, self.bus.theme, density=self.bus.density, scale=self.scale
        )

        self.pages: dict[str, Page] = {}
        self._current: Page | None = None
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._nav_group_widgets: list[tk.Widget] = []
        self._nav_footnote: tk.Widget | None = None
        self._nav_indicator: tk.Frame | None = None
        self._layouter: tk.Misc | None = None
        self.toasts = ToastHost(self, self.bus, self.motion, width=self.bus.px(380, minimum=280))
        self._build_shell()
        self.bind("<Configure>", self._on_root_configure, add="+")
        self.bind("<Control-k>", lambda _event: self.open_command_palette())
        self.bind("<Control-K>", lambda _event: self.open_command_palette())
        self.bind("<F11>", lambda _event: self.toggle_zoom())
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

    def screen_size(self) -> tuple[int, int]:
        """The pixel size of the display this window opens on.

        A missing or nonsensical answer (a stub Tk in a headless harness, a
        display server that reports 0) falls back to a 1920x1080 desktop: the
        window must open somewhere sane rather than divide by zero.
        """

        def probe(attribute: str, fallback: int) -> int:
            with contextlib.suppress(tk.TclError, TypeError, ValueError):
                value = int(getattr(self, attribute)())
                if value > 1:
                    return value
            return fallback

        return probe("winfo_screenwidth", 1920), probe("winfo_screenheight", 1080)

    def _apply_window_geometry(self) -> None:
        screen_w, screen_h = self.screen_size()
        fitted = fit_window_geometry(
            self.prefs.geometry,
            screen_width=screen_w,
            screen_height=screen_h,
            preferred_width=min(screen_w - 40, self.px(1800, minimum=1200)),
            preferred_height=min(screen_h - 60, self.px(980, minimum=700)),
            min_width=min(screen_w - 40, 960),
            min_height=min(screen_h - 60, 600),
        )
        self.geometry(fitted)
        # Allow resizing well below 1920x1080 so smaller displays or split-screen
        # windows can shrink smoothly while UiScale scales elements down.
        self.minsize(min(820, max(640, screen_w - 80)), min(520, max(440, screen_h - 80)))
        with contextlib.suppress(ValueError, IndexError):
            size_part = fitted.split("+", 1)[0].split("-", 1)[0]
            w_str, h_str = size_part.lower().split("x", 1)
            self.scale = self.scale.with_viewport(int(w_str), int(h_str))
            self.bus.scale = self.scale
        if self.prefs.zoomed:
            # Restore a maximized window as maximized; `state("zoomed")` is
            # the Tk spelling on Windows and on X11 alike, and a window
            # manager that does not support it must not break startup.
            with contextlib.suppress(tk.TclError):
                self.state("zoomed")
        if self.prefs.show_grid:
            with contextlib.suppress(tk.TclError):
                self.option_add("*Canvas.background", self.bus.theme.bg)

    def save_preferences(self) -> None:
        """Persist theme/layout/density/motion, window size and last page."""
        with contextlib.suppress(tk.TclError):
            self.prefs.geometry = self.winfo_geometry()
            self.prefs.zoomed = self.state() == "zoomed"
        self.prefs.theme = self.bus.theme.name
        self.prefs.density = self.bus.density.name
        self.prefs.motion = self.motion.level
        self.prefs.last_page = self._current.title if self._current else self.prefs.last_page
        self.prefs_store.save(self.prefs)

    def fit_window_to_screen(self) -> None:
        """Pull the window back onto this screen and centre it.

        The recovery for a window that was closed on a bigger display (or
        maximized on one): unzoom, clamp the size to the screen, centre. It is
        what the header's *Fit window to screen* button calls, so an operator
        never has to find and delete ``preferences.json``.
        """
        with contextlib.suppress(tk.TclError):
            if self.state() == "zoomed":
                self.state("normal")
        screen_w, screen_h = self.screen_size()
        self.geometry(
            fit_window_geometry(
                None,
                screen_width=screen_w,
                screen_height=screen_h,
                preferred_width=self.px(1800, minimum=1400),
                preferred_height=self.px(980, minimum=820),
                min_width=self.px(1280, minimum=1000),
                min_height=self.px(800, minimum=640),
            )
        )
        self.prefs.zoomed = False
        self.notify("Window fitted to this screen", kind="ok", timeout_ms=1800)

    def toggle_zoom(self) -> None:
        """Maximize/restore the window (F11).

        ``state("zoomed")`` is Tk's spelling on Windows and on X11 alike; a
        window manager that does not support it must not break the action.
        """
        with contextlib.suppress(tk.TclError):
            if self.state() == "zoomed":
                self.state("normal")
                self.prefs.zoomed = False
            else:
                self.state("zoomed")
                self.prefs.zoomed = True

    def preferences_snapshot(self) -> dict[str, object]:
        """Preferences as a plain dict (diagnostics and tests)."""
        return prefs_as_dict(self.prefs)

    def set_theme(self, name: str, *, persist: bool = True) -> None:
        """Switch the palette and repaint the whole window."""
        theme = get_theme(name, self.prefs.accent)
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

    def set_accent(self, value: str, *, persist: bool = True) -> bool:
        """Override the accent colour of whichever theme is active.

        ``""`` restores the theme's own designed accent. An unusable value is
        refused with a message instead of being applied half-way, and the
        chosen colour is stored in the preferences so it survives a restart.
        """
        accent = normalize_accent(value)
        if value and not accent:
            self.notify(f"'{value}' is not a colour like #4F7CFF", kind="error")
            return False
        self.prefs.accent = accent
        self._reapply_appearance()
        if persist:
            self.save_preferences()
        self.notify(
            "Accent: theme default" if not accent else f"Accent: {accent}",
            kind="info",
            timeout_ms=2200,
        )
        return True

    def _reapply_appearance(self, *, rebuild: bool = False) -> None:
        """Re-apply the theme colours (and density) from the preferences.

        ``rebuild`` is for a density change: the paddings have to be laid
        out again, which recreates the page widgets (and restores what the
        operator was looking at - see ``Page.rebuild_after_restyle``).
        """
        theme = get_theme(self.prefs.theme, self.prefs.accent)
        density = DENSITIES.get(self.prefs.density, DENSITIES["comfort"])
        self.bus.set_theme(theme, scale=self.scale, density=density)
        self.style = apply_ttk_styles(self, theme, density=density, scale=self.scale)
        self._restyle_shell()
        for page in self.pages.values():
            if rebuild:
                page.rebuild_after_restyle()
            else:
                page.on_theme(theme)

    def set_density(self, name: str, *, persist: bool = True) -> None:
        """Switch spacing/row height and rebuild the built pages."""
        density = DENSITIES.get(name, DENSITIES["comfort"])
        self.prefs.density = density.name
        self._reapply_appearance(rebuild=True)
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
                "accent": self.prefs.accent,
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
        appearance = document.get("appearance")
        if isinstance(appearance, dict):
            before = (self.prefs.theme, self.prefs.accent, self.prefs.density)
            accent = appearance.get("accent")
            if isinstance(accent, str):
                # Old presets have no accent; they keep whatever is active.
                self.prefs.accent = normalize_accent(accent)
            theme = appearance.get("theme")
            if isinstance(theme, str) and theme in THEMES:
                self.prefs.theme = theme
            density = appearance.get("density")
            if isinstance(density, str) and density in DENSITIES:
                self.prefs.density = density
            if before != (self.prefs.theme, self.prefs.accent, self.prefs.density):
                # A different density needs the widgets laid out again, which
                # is the same path the Settings pickers use. Nothing changed
                # means nothing to repaint - a preset that only rearranges
                # cards must not restyle the window.
                self._reapply_appearance(rebuild=before[2] != self.prefs.density)
            # A preset written by an older build carries a shell choice; the
            # shell is fixed now, so it is read past rather than applied.
            motion = appearance.get("motion")
            if isinstance(motion, str) and motion in MOTION_LEVELS:
                self.motion.set_level(motion)
                self.prefs.motion = motion
        self.prefs.active_preset = safe_preset_name(name)
        self.layout_bus.set_state(state)
        # A preset's card arrangement applies to the shell that is on screen.
        # The shell itself is not switchable any more, so a preset that came
        # from an older build keeps its cards and drops its shell choice
        # instead of tearing down every page to rebuild an identical rail.
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

    def export_layout_preset(self, name: str, target: str | Path) -> bool:
        """Write one preset to ``target`` so it can be shared or backed up."""
        try:
            written = self.preset_store.export(name, Path(target))
        except (PresetError, OSError) as exc:
            self.notify(f"Preset could not be exported: {exc}", kind="error")
            return False
        self.notify(f"Preset exported to {written}", kind="ok")
        return True

    def preset_name_for_import(self, source: str | Path) -> str:
        """The preset name an import file would take (``""`` when unusable)."""
        return self.preset_store.import_target_name(source)

    def import_layout_preset(self, source: str | Path, *, overwrite: bool = False) -> str | None:
        """Add a preset from a file, keeping the local library intact.

        The name comes from the file. An existing preset of that name is
        refused unless ``overwrite`` is set, so importing a stranger's
        export cannot silently replace the operator's own arrangement.
        """
        try:
            imported = self.preset_store.import_preset(Path(source), overwrite=overwrite)
        except (PresetError, OSError) as exc:
            self.notify(f"Preset could not be imported: {exc}", kind="error")
            return None
        self.notify(f"Preset imported: {imported}", kind="ok")
        return imported

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

        status = ttk.Frame(header, style="Header.TFrame")
        status.pack(side="right", padx=(self.px(12), 0))
        self.status_dot = StatusDot(status, self.bus, size=self.px(10, minimum=8))
        self.status_dot.pack(side="left", padx=(0, self.px(6, minimum=4)))
        self.status_label = ttk.Label(status, text="Ready", style="OperatorStatus.TLabel")
        self.status_label.pack(side="left")
        self.telemetry_badge = ttk.Label(
            status,
            text="Bridge v3  ·  106-Obs  ·  6-Head",
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
        self._refresh_preset_label()
        if self._nav_indicator is not None:
            with contextlib.suppress(tk.TclError):
                self._nav_indicator.configure(background=theme.accent)

    def _refresh_preset_label(self) -> None:
        text = f"Preset: {self.prefs.active_preset}" if self.prefs.active_preset else "No preset"
        with contextlib.suppress(tk.TclError):
            self.preset_label.configure(text=text)

    def _build_body(self) -> None:
        body = ttk.Frame(self.outer, style="Shell.TFrame")
        body.pack(fill="both", expand=True)
        nav = ttk.Frame(
            body,
            style="Nav.TFrame",
            width=self.px(NAV_WIDTH_PX[0], minimum=NAV_WIDTH_PX[1]),
            padding=(self.px(12, minimum=6), self.px(16, minimum=8)),
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
        self._build_nav_items(nav)
        content = ttk.Frame(body, style="Content.TFrame", padding=(self.px(24), self.px(18)))
        content.pack(side="left", fill="both", expand=True)
        self._content_host = content
        self._layouter = content
        self._build_pages(content)

    def _build_nav_items(self, host: tk.Misc) -> None:
        """Fill the rail with one button per page, under its group heading.

        The rail is the only shell and it keeps its full width and its full
        page titles. A collapsed rail used to trade the titles for two-letter
        codes (``DB``, ``TR``, ``BM``) to win back ~160 px, which is a bad
        trade for a window with eight pages: it needs a shortcut sheet at the
        bottom to stay readable, and it makes every page look identical until
        the pointer is over it.
        """
        self._nav_group_widgets.clear()
        self._nav_footnote = None
        groups = {0: "OVERVIEW", 1: "WORKFLOWS", 4: "ARTIFACTS", 5: "SYSTEM"}
        for index, page_class in enumerate(PAGE_CLASSES):
            if index in groups:
                if index:
                    sep = ttk.Separator(host)
                    sep.pack(fill="x", pady=(self.px(14), self.px(8)))
                    self._nav_group_widgets.append(sep)
                grp_lbl = ttk.Label(
                    host,
                    text=groups[index],
                    style="NavGroup.TLabel",
                )
                grp_lbl.pack(fill="x", padx=self.px(10), pady=(0, self.px(4)))
                self._nav_group_widgets.append(grp_lbl)
            button = ttk.Button(
                host,
                text=page_class.title,
                style="Nav.TButton",
                command=lambda name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )
            button.pack(fill="x", pady=1)

            def on_nav_button_configure(
                _event: tk.Event[tk.Misc], name: str = page_class.title
            ) -> None:
                self._on_nav_button_configure(name)

            button.bind("<Configure>", on_nav_button_configure, add=True)
            ToolTip(button, f"Open {page_class.title}")
            self._nav_buttons[page_class.title] = button
            self.bind(
                f"<Control-Key-{index + 1}>",
                lambda _evt, name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )
        sep_bottom = ttk.Separator(host)
        sep_bottom.pack(fill="x", pady=self.px(14))
        self._nav_group_widgets.append(sep_bottom)
        self._nav_footnote = ttk.Label(
            host,
            text="Headless Godot Bridge v3",
            style="NavFootnote.TLabel",
            justify="left",
        )
        self._nav_footnote.pack(anchor="w", padx=self.px(10))

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

    def _on_nav_button_configure(self, name: str) -> None:
        if self._current is not None and self._current.title == name:
            self._move_nav_indicator(name, animate=False)

    def _on_root_configure(self, event: object = None) -> None:
        if getattr(event, "widget", None) is not self:
            return
        try:
            width = int(getattr(event, "width", 0) or self.winfo_width() or 0)
            height = int(getattr(event, "height", 0) or self.winfo_height() or 0)
        except (tk.TclError, TypeError, ValueError):
            return
        if width < 320 or height < 240:
            return
        if (width, height) == self._last_viewport_size:
            return
        self._last_viewport_size = (width, height)
        if self._viewport_resize_job is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._viewport_resize_job)
        with contextlib.suppress(tk.TclError):
            self._viewport_resize_job = self.after(65, self._apply_viewport_scale)

    def _apply_viewport_scale(self) -> None:
        self._viewport_resize_job = None
        width, height = self._last_viewport_size
        if width < 320 or height < 240:
            return
        compact = width < 1220
        if compact != self._compact_header:
            self._compact_header = compact
            with contextlib.suppress(tk.TclError):
                if compact:
                    self.telemetry_badge.pack_forget()
                    self.preset_label.pack_forget()
                    self.brand_sub.pack_forget()
                else:
                    self.brand_sub.pack(side="left", pady=(self.px(4), 0))
                    self.telemetry_badge.pack(side="left", padx=(self.px(8, minimum=4), 0))
                    self.preset_label.pack(side="left", padx=(self.px(8, minimum=4), 0))
        new_scale = self.scale.with_viewport(width, height)
        if new_scale != self.scale:
            self.scale = new_scale
            theme = self.bus.theme
            density = self.bus.density
            self.bus.set_theme(theme, scale=self.scale, density=density)
            self.style = apply_ttk_styles(self, theme, density=density, scale=self.scale)
            with contextlib.suppress(tk.TclError):
                if self._header is not None:
                    self._header.configure(
                        padding=(self.px(20, minimum=10), self.px(10, minimum=5))
                    )
                if self._nav_host is not None:
                    self._nav_host.configure(
                        width=self.px(NAV_WIDTH_PX[0], minimum=NAV_WIDTH_PX[1]),
                        padding=(self.px(12, minimum=6), self.px(16, minimum=8)),
                    )
                if self._content_host is not None:
                    self._content_host.configure(
                        padding=(self.px(22, minimum=10), self.px(16, minimum=8))
                    )
        if self._current is not None:
            self._move_nav_indicator(self._current.title, animate=False)

    def _move_nav_indicator(self, name: str, *, animate: bool = True) -> None:
        button = self._nav_buttons.get(name)
        indicator = self._nav_indicator
        if indicator is None or button is None or not indicator.winfo_exists():
            return
        try:
            height = int(button.winfo_height())
            if height <= 4 and self._nav_host is not None:
                with contextlib.suppress(tk.TclError):
                    self._nav_host.update_idletasks()
                height = int(button.winfo_height())
            y = int(button.winfo_y())
            x = int(button.winfo_x())
        except tk.TclError:  # pragma: no cover - layout race during rebuild
            return
        ind_w = self.px(3, minimum=2)
        ind_h = self.px(18, minimum=12)
        if height > 4:
            ind_h = min(ind_h, max(10, height - 8))
            target = y + max(0, (height - ind_h) // 2)
        else:
            target = y + max(1, height - 2)
        x_pos = max(2, x - self.px(6, minimum=4)) if x > 4 else self.px(4, minimum=2)
        start = int(indicator.place_info().get("y", target) or target)
        do_animate = animate and abs(target - start) > 4 and height > 4

        def place_at(value: float) -> None:
            with contextlib.suppress(tk.TclError):
                indicator.place(
                    x=x_pos,
                    y=int(value),
                    width=ind_w,
                    height=ind_h,
                    bordermode="outside",
                )
                indicator.lift()

        if do_animate:
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
        """A keyboard-driven page switcher (Ctrl+K).

        Ctrl+K binds on the root window, which Tk includes in the bindtags of
        every widget inside it, so the shortcut works from any field of the
        main window. The palette itself is a separate toplevel with its own
        bindtags, so it carries its own bindings: Ctrl+K closes it again
        (pressing it twice must not stack a second palette), Escape closes
        it, Up/Down move the highlight from the entry and from the list, and
        the page accelerators Ctrl+1..7 switch straight from it.
        """
        existing = getattr(self, "_palette_window", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            if existing.winfo_viewable():
                existing.focus_force()
            return
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
            """Filter the page list; keep the highlight while it is unchanged.

            ``<KeyRelease>`` fires for every key, arrow keys included, so a
            blind "select the first row" would undo the move the operator just
            made with Up/Down. Rebuilding the listbox is therefore skipped
            whenever the filtered titles are the same as the ones on screen.
            """
            rows = _palette_matches(self.pages, entry.get())
            if list(listing.get(0, "end")) != rows:
                current = listing.curselection()
                _palette_highlight(listing, rows, current[0] if current else 0)

        def move(direction: int) -> str:
            return _palette_move(listing, direction)

        def choose(_event: object = None) -> None:
            title = _palette_choice(listing)
            if title:
                self.show_page(title)
                window.destroy()

        def close(_event: object = None) -> None:
            window.destroy()

        def jump(name: str) -> None:
            self.show_page(name)
            window.destroy()

        def focus_entry(event: object = None) -> None:
            """Once the palette is really on screen, make it take the keys.

            ``focus_set`` alone can be ignored while the toplevel is still
            being mapped (and by a window manager that hands the focus to the
            parent window), which would leave the palette open but deaf.
            """
            if getattr(event, "widget", None) is window:
                entry.focus_force()

        # The arrows are bound on the two focusable children themselves, so
        # they work from the query field and after a click into the list. A
        # binding on the toplevel would fire *after* the Listbox class
        # binding, which already steps the cursor: one keypress, two rows.
        # Both handlers return "break" (see _palette_move), which suppresses
        # the class binding and keeps the step at exactly one row.
        entry.bind("<Down>", lambda _e: move(1))
        entry.bind("<Up>", lambda _e: move(-1))
        listing.bind("<Down>", lambda _e: move(1))
        listing.bind("<Up>", lambda _e: move(-1))
        entry.bind("<KeyRelease>", refresh)
        entry.bind("<Return>", choose)
        listing.bind("<Double-Button-1>", choose)
        listing.bind("<Return>", choose)
        window.bind("<Map>", focus_entry)
        window.bind("<Escape>", close)
        window.bind("<Control-Key-k>", close)

        def jump_to(title: str) -> Callable[[object], None]:
            """One handler per page: the lambda's default argument confused
            mypy's type inference, and a named factory reads better anyway."""

            def handler(_event: object = None) -> None:
                jump(title)

            return handler

        for index, page_class in enumerate(PAGE_CLASSES):
            window.bind(f"<Control-Key-{index + 1}>", jump_to(page_class.title))
        window.protocol("WM_DELETE_WINDOW", close)
        self._palette_window = window
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

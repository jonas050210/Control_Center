"""Reusable Tk primitives for the Control Center's restyled interface.

These are the pieces the pages compose instead of raw ttk widgets: an
overlay scrollbar that only appears when there is something to scroll, a
scroll area, rounded cards, an animated segmented control, toasts, a motion
controller and a layout board that arranges cards according to a saved
:mod:`sandboxai.control_center_layout` state.

Three rules shape this module:

1. **Animation never blocks and never leaves a wrong final state.** Every
   animation runs on the Tk thread through a single ``after`` ticker, and a
   disabled motion level applies the final value immediately instead of
   skipping it.
2. **Themed, not hard-coded.** Widgets subscribe to a :class:`ThemeBus` and
   repaint on a theme switch; no colour literal lives in a page.
3. **No invisible click targets.** An overlay scrollbar that has nothing to
   scroll is removed from the geometry manager (not merely drawn
   transparent), so it can never swallow a click or look like clutter.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import ttk
from typing import Any

from .control_center_layout import LayoutState, WidgetSpec
from .control_center_theme import DENSITIES, MOTION_SPEED, Density, Theme, UiScale

__all__ = [
    "AnimatedValue",
    "LayoutBoard",
    "LayoutBus",
    "MotionController",
    "RoundedPanel",
    "ScrollArea",
    "SegmentedControl",
    "SlimScrollbar",
    "StatusDot",
    "ThemeBus",
    "ToastHost",
    "attach_overlay_scrollbars",
    "ease_out_cubic",
    "lerp_color",
    "rounded_rect",
]

#: Overlay scrollbar thickness in px before DPI scaling.
SCROLLBAR_THICKNESS = 8
#: How long an overlay scrollbar stays visible after the last interaction.
SCROLLBAR_HIDE_MS = 900


# ---------------------------------------------------------------------------
# Colour and easing helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lerp_color(first: str, second: str, fraction: float) -> str:
    """Blend two ``#rrggbb`` colours; ``fraction`` 0 -> first, 1 -> second."""
    fraction = _clamp(fraction)
    left = first.lstrip("#")
    right = second.lstrip("#")
    channels = []
    for index in (0, 2, 4):
        start = int(left[index : index + 2], 16)
        end = int(right[index : index + 2], 16)
        channels.append(round(start + (end - start) * fraction))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def ease_out_cubic(t: float) -> float:
    """Fast-out, soft-land easing used for every slide/indicator motion."""
    t = _clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    """Symmetric easing for pulses and loops."""
    t = _clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def rounded_rect(
    canvas: tk.Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius: float,
    **kwargs: Any,
) -> int:
    """Draw a rounded rectangle as a smoothed polygon; returns its item id.

    Tk has no native rounded rectangle and ``create_arc``-based approaches
    need four items that then do not share a single fill/border update. A
    smoothed polygon is one item, repaints in one call and is what the
    cards use on every resize.
    """
    radius = max(0.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    points = [
        x0 + radius,
        y0,
        x1 - radius,
        y0,
        x1,
        y0,
        x1,
        y0 + radius,
        x1,
        y1 - radius,
        x1,
        y1,
        x1 - radius,
        y1,
        x0 + radius,
        y1,
        x0,
        y1,
        x0,
        y1 - radius,
        x0,
        y0 + radius,
        x0,
        y0,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)


class ThemeBus:
    """Holds the active theme/scale/density and notifies subscribers.

    One bus is passed down to every widget in the window, so a theme switch
    is a single ``set_theme`` call that repaints every canvas-drawn surface
    while ttk styles are re-applied once by the shell.
    """

    def __init__(
        self,
        theme: Theme,
        *,
        scale: UiScale | None = None,
        density: Density | None = None,
    ) -> None:
        self._theme = theme
        self.scale = scale or UiScale()
        self.density = density or DENSITIES["comfort"]
        self._listeners: list[Callable[[Theme], None]] = []

    @property
    def theme(self) -> Theme:
        return self._theme

    def font(self, role: str, *, bold: bool = False, mono: bool = False) -> tuple[Any, ...]:
        """Resolve a font role, honouring the active density preset."""
        family, size, *rest = self.scale.font(role, bold=bold, mono=mono)
        if self.density.font_delta:
            size = max(11, int(size) + self.density.font_delta)
        return (family, size, *rest)

    def px(self, value: float, *, minimum: int = 0) -> int:
        return self.scale.px(value, minimum=minimum)

    def subscribe(self, listener: Callable[[Theme], None]) -> Callable[[], None]:
        """Register a repaint callback; the return value unsubscribes it."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def set_theme(
        self,
        theme: Theme,
        *,
        scale: UiScale | None = None,
        density: Density | None = None,
    ) -> None:
        self._theme = theme
        if scale is not None:
            self.scale = scale
        if density is not None:
            self.density = density
        for listener in list(self._listeners):
            try:
                listener(theme)
            except tk.TclError:
                # A widget destroyed between the theme change and this call
                # must not stop the remaining widgets from repainting.
                with contextlib.suppress(ValueError):
                    self._listeners.remove(listener)


class LayoutBus:
    """Holds the active :class:`LayoutState` and notifies boards on change."""

    def __init__(self, state: LayoutState) -> None:
        self._state = state
        self._listeners: list[Callable[[LayoutState], None]] = []

    @property
    def state(self) -> LayoutState:
        return self._state

    def subscribe(self, listener: Callable[[LayoutState], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def set_state(self, state: LayoutState) -> None:
        self._state = state
        for listener in list(self._listeners):
            try:
                listener(state)
            except tk.TclError:
                with contextlib.suppress(ValueError):
                    self._listeners.remove(listener)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


@dataclass
class _Tween:
    duration_ms: int
    on_frame: Callable[[float], None]
    on_done: Callable[[], None] | None
    started_at: float | None = None


class MotionController:
    """One 16 ms ticker driving every animation, and nothing when idle.

    Motion levels are honest: ``off`` applies the end value immediately,
    ``reduced`` shortens durations, ``cinematic`` lengthens them. The ticker
    stops scheduling itself the moment the last animation finishes, which is
    what keeps a live GUI at ~0 % CPU between updates.
    """

    FRAME_MS = 16

    def __init__(self, root: tk.Misc, level: str = "normal") -> None:
        self._root = root
        self._level = level if level in MOTION_SPEED else "normal"
        self._tweens: dict[int, _Tween] = {}
        self._loops: dict[int, Callable[[float], None]] = {}
        self._next_handle = 1
        self._loop_started: float | None = None
        self._after_id: str | None = None

    @property
    def level(self) -> str:
        return self._level

    @property
    def enabled(self) -> bool:
        return MOTION_SPEED.get(self._level, 1.0) > 0.0

    @property
    def running(self) -> bool:
        """Whether the ticker currently has an outstanding ``after`` call."""
        return self._after_id is not None

    def set_level(self, level: str) -> None:
        self._level = level if level in MOTION_SPEED else "normal"

    def duration(self, milliseconds: int) -> int:
        """Scale a nominal duration for the current motion level."""
        speed = MOTION_SPEED.get(self._level, 1.0)
        if speed <= 0.0:
            return 0
        return max(1, int(round(milliseconds * speed)))

    def tween(
        self,
        milliseconds: int,
        on_frame: Callable[[float], None],
        on_done: Callable[[], None] | None = None,
    ) -> int:
        """Animate ``on_frame(0..1)``; returns a handle for :meth:`cancel`."""
        duration = self.duration(milliseconds)
        handle = self._next_handle
        self._next_handle += 1
        if duration <= 0:
            on_frame(1.0)
            if on_done is not None:
                on_done()
            return handle
        self._tweens[handle] = _Tween(duration, on_frame, on_done, None)
        self._schedule()
        return handle

    def loop(self, interval_ms: int, on_frame: Callable[[float], None]) -> int:
        """Register a repeating callback receiving a 0..1 triangle wave."""
        handle = self._next_handle
        self._next_handle += 1
        if not self.enabled:
            on_frame(0.0)
            return handle
        self._loops[handle] = on_frame
        if self._loop_started is None:
            import time

            self._loop_started = time.monotonic()
        self._schedule()
        return handle

    def cancel(self, handle: int) -> None:
        self._tweens.pop(handle, None)
        self._loops.pop(handle, None)

    def stop_all(self) -> None:
        self._tweens.clear()
        self._loops.clear()
        if self._after_id is not None:
            with contextlib.suppress(tk.TclError):
                self._root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule(self) -> None:
        if self._after_id is not None or not self._root.winfo_exists():
            return
        self._after_id = self._root.after(self.FRAME_MS, self._tick)

    def _tick(self) -> None:
        import time

        self._after_id = None
        now = time.monotonic() * 1000.0
        for handle, tween in list(self._tweens.items()):
            if tween.started_at is None:
                tween.started_at = now
            progress = (now - tween.started_at) / max(1, tween.duration_ms)
            if progress >= 1.0:
                self._tweens.pop(handle, None)
                tween.on_frame(1.0)
                if tween.on_done is not None:
                    tween.on_done()
            else:
                tween.on_frame(progress)
        if self._loops:
            started = self._loop_started or now
            elapsed = (now - started) / 1000.0
            wave = ease_in_out((elapsed % 2.0) / 2.0) * 2
            wave = wave if wave <= 1.0 else 2.0 - wave
            for callback in list(self._loops.values()):
                callback(wave)
        if self._tweens or self._loops:
            self._schedule()


# ---------------------------------------------------------------------------
# Overlay scrollbar + scroll area
# ---------------------------------------------------------------------------


class SlimScrollbar(tk.Canvas):
    """A thin scrollbar drawn as an overlay; hidden unless it is needed.

    ``set()``/``command`` follow the Tk scrollbar protocol, so it can be
    wired to a ``Treeview``, ``Text`` or ``Canvas`` without adapters. The
    widget removes itself from the geometry manager while the content fits,
    which is what stops the old "every table always has two bars" clutter.
    """

    #: The wheel over the bar already scrolls the widget it is wired to (see
    #: ``_on_wheel``), so a surrounding scroll area must stay put.
    _cc_wheel_owner = True

    def __init__(
        self,
        parent: tk.Misc,
        bus: ThemeBus,
        *,
        orient: str = "vertical",
        command: Callable[..., Any] | None = None,
        thickness: int = SCROLLBAR_THICKNESS,
        place: dict[str, Any] | None = None,
        autohide: bool = True,
    ) -> None:
        self._theme = bus.theme
        self._bus = bus
        self._cc_bus = self._bus
        self._orient = orient
        self._command = command
        self._thickness = thickness
        self._place: dict[str, Any] = dict(place or {})
        self._autohide = autohide
        self._first = 0.0
        self._last = 1.0
        self._hover = False
        self._dragging = False
        self._drag_offset = 0.0
        self._hide_after: str | None = None
        self._placed = False
        super().__init__(
            parent,
            width=thickness if orient == "vertical" else 1,
            height=1 if orient == "vertical" else thickness,
            highlightthickness=0,
            borderwidth=0,
            background=self._theme.bg,
            cursor="arrow",
            takefocus=False,
        )
        self._unsubscribe = bus.subscribe(self.apply_theme)
        self.bind("<Enter>", lambda _e: self._on_hover(True), add="+")
        self.bind("<Leave>", lambda _e: self._on_hover(False), add="+")
        self.bind("<Button-1>", self._on_press, add="+")
        self.bind("<B1-Motion>", self._on_drag, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        for sequence in ("<Button-4>", "<Button-5>"):
            self.bind(sequence, self._on_wheel, add="+")
        self.bind("<Destroy>", lambda _e: self._unsubscribe(), add="+")

    # -- protocol ---------------------------------------------------------

    def set(self, first: float, last: float) -> None:
        """Tk calls this with the visible fraction whenever the view moves."""
        try:
            self._first = _clamp(float(first))
            self._last = _clamp(float(last))
        except (TypeError, ValueError):
            return
        if self._overflow():
            if not self._placed and (self._hover or self._dragging):
                self._place_widget()
            self._redraw()
        elif self._placed and not self._dragging:
            self._remove()

    def reveal(self) -> None:
        """Show the bar briefly (used by the owning widget on hover/scroll)."""
        if not self._overflow():
            self._remove()
            return
        self._place_widget()
        self._redraw()
        self._schedule_hide()

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.configure(background=theme.bg)
            self._redraw()

    # -- geometry ---------------------------------------------------------

    def _place_widget(self) -> None:
        if self._placed:
            return
        with contextlib.suppress(tk.TclError):
            self.place(**self._place)
            self.lift(self)  # type: ignore[arg-type]
            self._placed = True

    def _remove(self) -> None:
        if not self._placed:
            return
        with contextlib.suppress(tk.TclError):
            self.place_forget()
        self._placed = False

    def _schedule_hide(self) -> None:
        if not self._autohide or self._dragging:
            return
        self._cancel_hide()
        with contextlib.suppress(tk.TclError):
            self._hide_after = self.after(SCROLLBAR_HIDE_MS, self._hide_now)

    def _cancel_hide(self) -> None:
        if self._hide_after is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._hide_after)
            self._hide_after = None

    def _hide_now(self) -> None:
        self._hide_after = None
        if not self._hover and not self._dragging:
            self._remove()

    def _on_hover(self, hovering: bool) -> None:
        self._hover = hovering
        if hovering:
            self.reveal()
        else:
            self._schedule_hide()

    # -- interaction ------------------------------------------------------

    def _overflow(self) -> bool:
        return (self._last - self._first) < 0.999

    def _track_length(self) -> float:
        return float(self.winfo_height() if self._orient == "vertical" else self.winfo_width())

    def _event_position(self, event: tk.Event) -> float:
        return float(event.y if self._orient == "vertical" else event.x)

    def _thumb_bounds(self) -> tuple[float, float]:
        length = self._track_length()
        return self._first * length, max(self._last * length, self._first * length + 24.0)

    def _scroll(self, *args: Any) -> None:
        if self._command is None:
            return
        try:
            self._command(*args)
        except tk.TclError:  # pragma: no cover - widget went away mid-drag
            return
        self.reveal()

    def _on_press(self, event: tk.Event) -> None:
        position = self._event_position(event)
        start, end = self._thumb_bounds()
        if start <= position <= end:
            self._dragging = True
            self._drag_offset = position - start
        else:
            self._scroll(
                "moveto",
                _clamp(position / max(1.0, self._track_length()) - (self._last - self._first) / 2),
            )
            self._dragging = True
            self._drag_offset = 0.0

    def _on_drag(self, event: tk.Event) -> None:
        if not self._dragging:
            return
        length = self._track_length()
        if length <= 0:
            return
        fraction = (self._event_position(event) - self._drag_offset) / length
        self._scroll("moveto", _clamp(fraction))

    def _on_release(self, _event: tk.Event) -> None:
        self._dragging = False
        self._schedule_hide()

    def _on_wheel(self, event: tk.Event) -> None:
        direction = -1 if getattr(event, "num", 4) == 4 else 1
        self._scroll("scroll", direction, "units")

    # -- painting ---------------------------------------------------------

    def _redraw(self) -> None:
        self.delete("thumb")
        length = self._track_length()
        if length <= 1 or not self._overflow():
            return
        thickness = float(self.winfo_width() if self._orient == "vertical" else self.winfo_height())
        inset = max(1.0, thickness * 0.15)
        start, end = self._thumb_bounds()
        track = self._theme.border
        thumb = (
            self._theme.text_muted if not (self._hover or self._dragging) else self._theme.text_dim
        )
        if self._orient == "vertical":
            rounded_rect(
                self,
                inset,
                inset,
                thickness - inset,
                length - inset,
                (thickness - 2 * inset) / 2,
                fill=track,
                outline=track,
                tags="thumb",
            )
            rounded_rect(
                self,
                inset,
                start + inset,
                thickness - inset,
                min(end, length) - inset,
                (thickness - 2 * inset) / 2,
                fill=thumb,
                outline=thumb,
                tags="thumb",
            )
        else:
            rounded_rect(
                self,
                inset,
                inset,
                length - inset,
                thickness - inset,
                (thickness - 2 * inset) / 2,
                fill=track,
                outline=track,
                tags="thumb",
            )
            rounded_rect(
                self,
                start + inset,
                inset,
                min(end, length) - inset,
                thickness - inset,
                (thickness - 2 * inset) / 2,
                fill=thumb,
                outline=thumb,
                tags="thumb",
            )


def attach_overlay_scrollbars(
    frame: tk.Misc,
    widget: Any,
    bus: ThemeBus,
    *,
    scale_px: Callable[[int], int] | None = None,
    horizontal: bool = True,
) -> tuple[SlimScrollbar, SlimScrollbar | None]:
    """Give a Treeview/Text overlay scrollbars that appear only when needed.

    The returned attributes are also stored on the widget as
    ``_vertical_scrollbar`` / ``_horizontal_scrollbar`` (the established
    presentation seam the desktop tests use), so pages never deal with
    geometry details.
    """
    px = scale_px or (lambda value: value)
    thickness = px(SCROLLBAR_THICKNESS)
    vertical = SlimScrollbar(
        frame,
        bus,
        orient="vertical",
        command=widget.yview,
        thickness=thickness,
        place={"relx": 1.0, "rely": 0.0, "relheight": 1.0, "anchor": "ne", "width": thickness},
    )
    horizontal_bar: SlimScrollbar | None = None
    if horizontal:
        horizontal_bar = SlimScrollbar(
            frame,
            bus,
            orient="horizontal",
            command=widget.xview,
            thickness=thickness,
            place={"relx": 0.0, "rely": 1.0, "relwidth": 1.0, "anchor": "sw", "height": thickness},
        )
        widget.configure(xscrollcommand=horizontal_bar.set)
    widget.configure(yscrollcommand=vertical.set)
    widget._vertical_scrollbar = vertical  # type: ignore[attr-defined]
    widget._horizontal_scrollbar = horizontal_bar  # type: ignore[attr-defined]

    def reveal(_event: object = None) -> None:
        vertical.reveal()
        if horizontal_bar is not None:
            horizontal_bar.reveal()

    for sequence in ("<Enter>", "<MouseWheel>", "<Button-4>", "<Button-5>", "<Configure>"):
        with contextlib.suppress(tk.TclError):
            widget.bind(sequence, reveal, add="+")
    return vertical, horizontal_bar


def _wheel_owner(widget: object) -> bool:
    """True when the wheel already belongs to ``widget`` itself.

    ``Treeview``, ``Text`` and ``Listbox`` scroll through Tk's class
    bindings, and :class:`SlimScrollbar` carries its own handler; a page
    scroll area that scrolled as well would move two viewports per click.
    """
    if getattr(widget, "_cc_wheel_owner", False):
        return True
    winfo_class = getattr(widget, "winfo_class", None)
    if not callable(winfo_class):
        return False
    with contextlib.suppress(tk.TclError):
        return winfo_class() in ("Treeview", "Text", "Listbox")
    return False


class ScrollArea(ttk.Frame):
    """A vertical scroll container for a page, with an overlay scrollbar.

    Pages used to stack ``Panedwindow`` panes, each with its own native
    scrollbar, until a window was mostly chrome. One scroll area per page
    keeps a single scroll context, and its scrollbar is invisible until the
    pointer is over the content or the wheel is used.

    The wheel is bound on the containing *toplevel*, which Tk includes in the
    bindtags of every descendant: a wheel event goes to the widget under the
    pointer, so without that binding the page only scrolled when the pointer
    happened to be over the canvas background - over a card's labels nothing
    moved and the scrollbar looked like the only way down. Each area's
    handler acts only when the pointer is inside *its* content and no inner
    widget owns the wheel.
    """

    def __init__(
        self,
        parent: tk.Misc,
        bus: ThemeBus,
        *,
        style: str = "Content.TFrame",
        background: str | None = None,
        scale_px: Callable[[int], int] | None = None,
    ) -> None:
        super().__init__(parent, style=style)
        self._bus = bus
        self._cc_bus = self._bus
        self._theme = bus.theme
        px = scale_px or (lambda value: value)
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            borderwidth=0,
            background=background or self._theme.bg,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas, style=style)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        thickness = px(SCROLLBAR_THICKNESS)
        self.scrollbar = SlimScrollbar(
            self,
            bus,
            orient="vertical",
            command=self.canvas.yview,
            thickness=thickness,
            place={"relx": 1.0, "rely": 0.0, "relheight": 1.0, "anchor": "ne", "width": thickness},
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        for sequence in ("<Enter>", "<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(sequence, self._on_interaction, add="+")
            self.canvas.bind(sequence, self._on_interaction, add="+")
        self._wheel_toplevel = toplevel = self.canvas.winfo_toplevel()
        self._wheel_bindings = [
            (sequence, toplevel.bind(sequence, self._on_page_wheel, add="+"))
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>")
        ]
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._unsubscribe = bus.subscribe(self.apply_theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.canvas.configure(background=theme.bg)

    def _on_body_configure(self, _event: object = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.scrollbar.set(*self.canvas.yview())

    def _on_canvas_configure(self, event: tk.Event) -> None:
        # Stretch the inner frame to the visible width so cards keep filling
        # the window instead of leaving a dead gutter on the right.
        self.canvas.itemconfigure(self._window, width=event.width)
        self.scrollbar.set(*self.canvas.yview())

    def _contains(self, widget: object) -> bool:
        """True while ``widget`` sits inside this area's scrollable body."""
        while widget is not None and widget is not self:
            widget = getattr(widget, "master", None)
        return widget is self

    def _wheel_goes_inner(self, widget: object) -> bool:
        """True when an inner widget of this area owns the wheel itself."""
        node = widget
        while node is not None and node is not self:
            if _wheel_owner(node):
                return True
            node = getattr(node, "master", None)
        return False

    def _on_page_wheel(self, event: tk.Event) -> None:
        """Toplevel wheel hook: scroll this area when the pointer is on it."""
        widget = getattr(event, "widget", None)
        if widget is self or widget is self.canvas:
            return  # bound directly, and Tk calls those bindings before this one
        if widget is None or not self._contains(widget) or self._wheel_goes_inner(widget):
            return
        self._on_interaction(event)

    def _on_destroy(self, event: object = None) -> None:
        """Drop the toplevel bindings so a rebuilt page leaves none behind."""
        if getattr(event, "widget", None) is not self:
            return
        for sequence, funcid in self._wheel_bindings:
            with contextlib.suppress(tk.TclError):
                self._wheel_toplevel.unbind(sequence, funcid)
        self._wheel_bindings = []

    def _on_interaction(self, event: tk.Event) -> None:
        self.scrollbar.reveal()
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = -3
        elif getattr(event, "num", None) == 5:
            delta = 3
        elif getattr(event, "delta", 0):
            delta = -1 * int(event.delta / 120) * 3
        if delta:
            self.canvas.yview_scroll(delta, "units")
            self.scrollbar.set(*self.canvas.yview())

    def scroll_to_top(self) -> None:
        self.canvas.yview_moveto(0.0)


# ---------------------------------------------------------------------------
# Cards, controls, toasts
# ---------------------------------------------------------------------------


class RoundedPanel(tk.Canvas):
    """A themed card with rounded corners, an optional header and a body frame.

    The card's visible surface (``card``/``card_hover``) is painted on the
    canvas, while a single child frame inset by the corner radius holds the
    content. Because the content never reaches the corners, the rounded
    silhouette stays intact, which a plain ``ttk.Frame`` cannot do.
    """

    def __init__(
        self,
        parent: tk.Misc,
        bus: ThemeBus,
        *,
        radius: int = 10,
        padding: int = 14,
        title: str = "",
        subtitle: str = "",
        accent: bool = False,
        background_role: str = "bg",
        body_style: str = "CardInner.TFrame",
    ) -> None:
        self._bus = bus
        self._cc_bus = self._bus
        self._theme = bus.theme
        self._radius = radius
        self._padding = padding
        self._title = title
        self._subtitle = subtitle
        self._accent = accent
        self._background_role = background_role
        self._outer = self._theme.color(background_role)
        super().__init__(
            parent,
            highlightthickness=0,
            borderwidth=0,
            background=self._outer,
            height=1,
        )
        self.body = ttk.Frame(self, style=body_style)
        self._body_window = self.create_window(
            padding, padding, window=self.body, anchor="nw", width=1, height=1
        )
        self._hover = False
        self.bind("<Configure>", lambda _e: self._redraw(), add="+")
        self.body.bind("<Configure>", lambda _e: self._on_body_configure(), add="+")
        self.bind("<Enter>", lambda _e: self._set_hover(True), add="+")
        self.bind("<Leave>", lambda _e: self._set_hover(False), add="+")
        self._unsubscribe = bus.subscribe(self.apply_theme)

    # -- theming ----------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self._outer = theme.color(self._background_role)
            self.configure(background=self._outer)
            self._redraw()

    def _set_hover(self, hovering: bool) -> None:
        if hovering == self._hover:
            return
        self._hover = hovering
        self._redraw()

    def _on_body_configure(self) -> None:
        header = self._header_height()
        self.configure(height=self.body.winfo_reqheight() + 2 * self._padding + header)

    def _header_height(self) -> int:
        return self._bus.px(34, minimum=26) if self._title else 0

    def _redraw(self) -> None:
        self.delete("surface")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        radius = self._radius
        fill = self._theme.card_hover if self._hover else self._theme.card
        border = self._theme.border_strong if self._hover else self._theme.border
        rounded_rect(
            self,
            0,
            0,
            width,
            height,
            radius,
            fill=fill,
            outline=border,
            width=1,
            tags="surface",
        )
        if self._accent:
            rounded_rect(
                self,
                self._padding,
                0,
                self._padding + self._bus.px(46, minimum=24),
                self._bus.px(3, minimum=2),
                self._bus.px(2, minimum=1),
                fill=self._theme.accent,
                outline=self._theme.accent,
                tags="surface",
            )
        if self._title:
            self.create_text(
                self._padding,
                self._padding,
                text=self._title,
                anchor="nw",
                fill=self._theme.text,
                font=self._bus.font("h2", bold=True),
                tags="surface",
            )
        if self._subtitle:
            self.create_text(
                self._padding,
                self._padding + self._bus.px(20, minimum=16),
                text=self._subtitle,
                anchor="nw",
                fill=self._theme.text_muted,
                font=self._bus.font("micro"),
                tags="surface",
            )
        self.tag_lower("surface")
        header = self._header_height()
        self.coords(self._body_window, self._padding, self._padding + header)
        self.itemconfigure(
            self._body_window,
            width=max(1, width - 2 * self._padding),
            height=max(1, height - 2 * self._padding - header),
        )


class SegmentedControl(tk.Canvas):
    """An animated segmented switch (``Steps | Time``, layout/theme picks).

    Clicking or using the arrow keys changes the selection; the highlight
    slides to the new segment instead of snapping, and the change callback
    only fires on a real change so a poll-driven ``set()`` cannot loop.
    """

    def __init__(
        self,
        parent: tk.Misc,
        bus: ThemeBus,
        options: tuple[tuple[str, str], ...],
        *,
        value: str = "",
        on_change: Callable[[str], None] | None = None,
        motion: MotionController | None = None,
        height: int = 30,
        width: int = 0,
    ) -> None:
        self._bus = bus
        self._cc_bus = self._bus
        self._theme = bus.theme
        self._segments = tuple(options)
        self._value = value or (self._segments[0][0] if self._segments else "")
        self._on_change = on_change
        self._motion = motion
        self._positions: list[tuple[float, float]] = []
        self._indicator = 0.0
        super().__init__(
            parent,
            height=height,
            width=width,
            highlightthickness=0,
            borderwidth=0,
            background=self._theme.panel,
            takefocus=True,
        )
        self.bind("<Configure>", lambda _e: self._redraw(force=True), add="+")
        self.bind("<Button-1>", self._on_click, add="+")
        self.bind("<Left>", lambda _e: self._step(-1), add="+")
        self.bind("<Right>", lambda _e: self._step(1), add="+")
        self.bind("<FocusIn>", lambda _e: self._redraw(), add="+")
        self.bind("<FocusOut>", lambda _e: self._redraw(), add="+")
        self._unsubscribe = bus.subscribe(self.apply_theme)

    # -- state ------------------------------------------------------------

    def get(self) -> str:
        return self._value

    def set(self, value: str, *, animate: bool = False) -> None:
        if value == self._value:
            return
        self._value = value
        self._animate_indicator()

    def choices(self) -> tuple[str, ...]:
        """The visible labels, in order (``("Steps", "Time")``)."""
        return tuple(label for _key, label in self._segments)

    def labels(self) -> tuple[str, ...]:
        """Alias for :meth:`choices`, kept for call sites that think in labels."""
        return self.choices()

    def _index(self) -> int:
        for index, (key, _label) in enumerate(self._segments):
            if key == self._value:
                return index
        return 0

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.configure(background=theme.panel)
            self._redraw(force=True)

    # -- interaction ------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        for index, (start, end) in enumerate(self._positions):
            if start <= event.x <= end:
                key = self._segments[index][0]
                if key != self._value:
                    self._value = key
                    self._animate_indicator()
                    if self._on_change is not None:
                        self._on_change(key)
                return

    def _step(self, direction: int) -> None:
        if not self._segments:
            return
        index = max(0, min(len(self._segments) - 1, self._index() + direction))
        key = self._segments[index][0]
        if key != self._value:
            self._value = key
            self._animate_indicator()
            if self._on_change is not None:
                self._on_change(key)

    def _animate_indicator(self) -> None:
        target = float(self._index())
        if self._motion is None or not self._motion.enabled:
            self._indicator = target
            self._redraw(force=True)
            return
        start = self._indicator
        self._motion.tween(
            150,
            lambda progress: self._set_indicator(
                start + (target - start) * ease_out_cubic(progress)
            ),
        )

    def _set_indicator(self, value: float) -> None:
        self._indicator = value
        self._redraw()

    # -- painting ---------------------------------------------------------

    def _redraw(self, force: bool = False) -> None:
        if not force and not self.winfo_exists():
            return
        self.delete("all")
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        theme = self._theme
        background = theme.panel
        self.configure(background=background)
        rounded_rect(
            self, 1, 1, width - 1, height - 1, height / 2, fill=theme.card, outline=theme.border
        )
        count = max(1, len(self._segments))
        segment = (width - 2) / count
        self._positions = [
            (1 + index * segment, 1 + (index + 1) * segment) for index in range(count)
        ]
        indicator_start = 1 + self._indicator * segment
        rounded_rect(
            self,
            indicator_start + 2,
            3,
            indicator_start + segment - 2,
            height - 3,
            (height - 6) / 2,
            fill=theme.accent,
            outline=theme.accent,
        )
        for index, (_key, label) in enumerate(self._segments):
            selected = index == self._index()
            self.create_text(
                1 + index * segment + segment / 2,
                height / 2,
                text=label,
                fill=theme.on_accent if selected else theme.text_dim,
                font=self._bus.font("small", bold=selected),
            )


class StatusDot(tk.Canvas):
    """A small status dot that breathes while a run is live."""

    def __init__(
        self, parent: tk.Misc, bus: ThemeBus, *, size: int = 10, background_role: str = "card"
    ) -> None:
        self._theme = bus.theme
        self._background_role = background_role
        self._size = size
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            borderwidth=0,
            background=self._theme.color(background_role),
        )
        self._color = self._theme.text_muted
        self._handle: int | None = None
        self._motion: MotionController | None = None
        self._unsubscribe = bus.subscribe(self.apply_theme)
        self._draw(1.0)

    def set_state(
        self, color: str, *, pulse: bool = False, motion: MotionController | None = None
    ) -> None:
        self._color = color
        if pulse and motion is not None and self._handle is None:
            self._motion = motion
            self._handle = motion.loop(1800, lambda wave: self._draw(0.45 + 0.55 * wave))
        elif not pulse and self._handle is not None and self._motion is not None:
            self._motion.cancel(self._handle)
            self._handle = None
            self._draw(1.0)
        else:
            self._draw(1.0)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.configure(background=theme.color(self._background_role))
            self._draw(1.0)

    def _draw(self, intensity: float) -> None:
        self.delete("all")
        size = float(self._size)
        color = lerp_color(self._theme.card, self._color, _clamp(intensity))
        self.create_oval(1, 1, size - 1, size - 1, fill=color, outline=color)


class ToastHost:
    """Slide-in notifications anchored to the bottom-right of the window."""

    def __init__(
        self,
        parent: tk.Misc,
        bus: ThemeBus,
        motion: MotionController,
        *,
        width: int = 380,
        margin: int = 22,
    ) -> None:
        self._bus = bus
        self._cc_bus = self._bus
        self._theme = bus.theme
        self._motion = motion
        self._width = width
        self._margin = margin
        self.frame = tk.Frame(parent, background=self._theme.bg)
        self.frame.place(relx=1.0, rely=1.0, anchor="se", x=-margin, y=-margin)
        self._toasts: list[tk.Frame] = []
        self._after_ids: dict[tk.Frame, str] = {}
        self._offset = margin
        self._unsubscribe = bus.subscribe(self.apply_theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.frame.configure(background=theme.bg)

    def show(self, message: str, *, kind: str = "info", timeout_ms: int = 4200) -> None:
        """Queue one toast; older toasts beyond four are dropped."""
        if not message:
            return
        theme = self._theme
        accent = {
            "info": theme.accent,
            "ok": theme.ok,
            "warn": theme.warn,
            "error": theme.error,
        }.get(kind, theme.accent)
        toast = tk.Frame(self.frame, background=theme.card, highlightthickness=0)
        bar = tk.Frame(toast, background=accent, width=4)
        bar.pack(side="left", fill="y")
        label = tk.Label(
            toast,
            text=message,
            justify="left",
            wraplength=self._width - 60,
            background=theme.card,
            foreground=theme.text,
            padx=12,
            pady=10,
            anchor="w",
        )
        label.pack(side="left", fill="both", expand=True)
        close = tk.Label(
            toast,
            text="✕",
            background=theme.card,
            foreground=theme.text_muted,
            padx=10,
            cursor="hand2",
        )
        close.pack(side="right", fill="y")

        def dismiss(_event: object = None, target: tk.Frame = toast) -> None:
            self._dismiss(target)

        def forget(_event: object = None, target: tk.Frame = toast) -> None:
            self._after_ids.pop(target, None)

        def slide(progress: float) -> None:
            self._slide(progress)

        close.bind("<Button-1>", dismiss)
        toast.pack(fill="x", pady=(6, 0))
        toast.bind("<Destroy>", forget, add="+")
        self._toasts.append(toast)
        while len(self._toasts) > 4:
            self._dismiss(self._toasts[0])
        self._motion.tween(200, slide)
        handle = toast.after(max(600, int(timeout_ms)), lambda: self._dismiss(toast))
        self._after_ids[toast] = handle

    def _slide(self, progress: float) -> None:
        with contextlib.suppress(tk.TclError):
            offset = int(self._offset * (1.0 - ease_out_cubic(progress)))
            self.frame.place_configure(y=-(self._margin + offset))

    def _dismiss(self, toast: tk.Frame) -> None:
        handle = self._after_ids.pop(toast, None)
        if handle is not None:
            with contextlib.suppress(tk.TclError):
                toast.after_cancel(handle)
        with contextlib.suppress(ValueError):
            self._toasts.remove(toast)
        with contextlib.suppress(tk.TclError):
            toast.destroy()


class AnimatedValue:
    """Animates a KPI label: numeric count-up, otherwise a colour settle."""

    def __init__(self, label: tk.Label, motion: MotionController) -> None:
        self._label = label
        self._motion = motion
        self._numeric: float | None = None

    def set(
        self,
        text: str,
        color: str | None = None,
        *,
        value: float | None = None,
        formatter: Callable[[float], str] | None = None,
        animate: bool = True,
    ) -> None:
        target = color or self._label.cget("foreground")
        if not animate or not self._motion.enabled:
            self._label.configure(text=text, foreground=target)
            self._numeric = value
            return
        if value is not None and self._numeric is not None and formatter is not None:
            start = self._numeric

            def count_up(progress: float) -> None:
                self._label.configure(
                    text=formatter(start + (value - start) * ease_out_cubic(progress))
                )

            def settle_number() -> None:
                self._label.configure(text=formatter(value))

            self._motion.tween(240, count_up, on_done=settle_number)
            self._label.configure(foreground=target)
        else:
            current = str(self._label.cget("foreground"))

            def frame(progress: float) -> None:
                eased = ease_out_cubic(progress)
                self._label.configure(foreground=lerp_color(current, target, eased))

            def settle_color() -> None:
                self._label.configure(foreground=target)

            self._motion.tween(220, frame, on_done=settle_color)
            self._label.configure(text=text)
        self._numeric = value


# ---------------------------------------------------------------------------
# Layout board
# ---------------------------------------------------------------------------


class LayoutBoard(ttk.Frame):
    """Arranges a page's cards according to the saved layout state.

    Cards are built once (their factories run on first use so an unvisited
    page costs nothing), then only re-gridded when the layout changes. A
    card the operator hid is forgotten rather than destroyed, so showing it
    again does not lose live state such as a selected table row.
    """

    def __init__(
        self,
        parent: tk.Misc,
        layout_bus: LayoutBus,
        *,
        page_key: str,
        specs: tuple[WidgetSpec, ...],
        columns: int = 3,
        gap: int = 12,
        min_column_width: int = 300,
    ) -> None:
        super().__init__(parent)
        self._layout_bus = layout_bus
        self._page_key = page_key
        self._specs = {spec.widget_id: spec for spec in specs}
        self._factories: dict[str, Callable[[tk.Misc], tk.Widget]] = {}
        self._widgets: dict[str, tk.Widget] = {}
        # ``_max_columns`` is the page's chosen grid (the design intent);
        # ``_columns`` is what actually fits right now. A narrow window
        # collapses the board, and resizing back must restore the page's own
        # column count - the hard-coded 3 used to cap a page that asked for 4.
        self._max_columns = max(1, columns)
        self._columns = self._max_columns
        self._gap = gap
        self._min_column_width = min_column_width
        for index in range(self._max_columns):
            self.columnconfigure(index, weight=1, uniform="board")
        self._unsubscribe = layout_bus.subscribe(lambda _state: self.rebuild())
        self.bind("<Configure>", lambda _e: self._on_resize(), add="+")

    def add(self, widget_id: str, factory: Callable[[tk.Misc], tk.Widget]) -> None:
        """Register a card builder; unregistered ids are ignored by rebuild."""
        if widget_id not in self._specs:
            raise KeyError(f"{widget_id!r} is not a registered widget of {self._page_key!r}")
        self._factories[widget_id] = factory

    def widget(self, widget_id: str) -> tk.Widget | None:
        """The built card, or ``None`` while it has not been requested yet."""
        return self._widgets.get(widget_id)

    def spec(self, widget_id: str) -> WidgetSpec | None:
        return self._specs.get(widget_id)

    def rebuild(self) -> None:
        # Every registered card is built, even while it is hidden. Hiding a
        # card and then refreshing the page must not change page behaviour:
        # the owning page keeps referencing ``self.log_panel`` and friends,
        # so a lazily skipped factory would turn "card hidden" into an
        # AttributeError on the next poll. Hidden cards are simply not
        # gridded, which keeps their live state (selected rows, log scroll
        # position) for when the operator shows them again.
        for widget_id in self._specs:
            self._ensure_widget(widget_id)
        placements = self._layout_bus.state.visible(self._page_key)
        for widget in self._widgets.values():
            with contextlib.suppress(tk.TclError):
                widget.grid_forget()
        row = 0
        column = 0
        for placement in placements:
            card = self._widgets.get(placement.widget_id)
            if card is None:
                continue
            span = max(1, min(self._columns, placement.span))
            if column + span > self._columns:
                row += 1
                column = 0
            card.grid(
                row=row,
                column=column,
                columnspan=span,
                sticky="nsew",
                padx=(0 if column == 0 else self._gap, 0),
                pady=(0 if row == 0 else self._gap, 0),
            )
            column += span
            if column >= self._columns:
                row += 1
                column = 0

    def _ensure_widget(self, widget_id: str) -> tk.Widget | None:
        existing = self._widgets.get(widget_id)
        if existing is not None:
            return existing
        factory = self._factories.get(widget_id)
        if factory is None:
            return None
        widget = factory(self)
        self._widgets[widget_id] = widget
        return widget

    def _on_resize(self) -> None:
        """Collapse to fewer columns on a narrow window instead of squeezing."""
        width = self.winfo_width()
        wanted = max(1, min(self._max_columns, max(1, width // self._min_column_width)))
        if wanted == self._columns:
            return
        self._columns = wanted
        for index in range(self._max_columns):
            self.columnconfigure(index, weight=1 if index < wanted else 0, uniform="board")
        self.rebuild()

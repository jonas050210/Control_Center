"""Reusable Tk infrastructure and presentation widgets for the desktop Control Center.

Kept separate from page orchestration so widgets, background execution and the
visual token palette can evolve without growing the application shell module.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from . import control_center_viewmodel as vm
from .contract import AGENT_VIEW_ASPECT
from .control_center_layout import fit_table_column_widths
from .control_center_theme import (
    FONT_FAMILY,
    MONO_FONT_FAMILY,
    Theme,
    get_theme,
)
from .control_center_ui import (
    SCROLLBAR_THICKNESS,
    AnimatedValue,
    MotionController,
    SlimScrollbar,
    ThemeBus,
    attach_overlay_scrollbars,
    ease_out_cubic,
    lerp_color,
    rounded_rect,
)
from .ttk_vision import encode_png

# ---------------------------------------------------------------------------
# Background work: keeps every adapter call off the Tk event loop thread.
# ---------------------------------------------------------------------------

# Tk objects may only be finalised on the thread that owns the Tcl
# interpreter, and CPython's cyclic collector honours no such rule: it
# runs in whichever thread happens to cross an allocation threshold. A
# background worker walking a directory tree allocates heavily, so it is
# a prime candidate - and the widget trees this GUI discards (page
# switches, closed windows) sit in reference cycles, because widgets
# point at their parent and callbacks point back at their page. Collect
# such a cycle on a worker and tkinter.Variable.__del__ reaches into Tcl
# from the wrong thread. On Linux that is usually survivable; on Windows
# the process dies instantly with exception code 0x80000003, no
# traceback and no failing test - which is exactly how CI found this.
#
# So automatic collection is suspended for as long as a runner is alive
# and driven explicitly from _pump instead, which runs on the Tk thread.
# Nothing leaks: the collector still runs, just somewhere it is allowed
# to. Reference-counted garbage - the overwhelming majority - is freed
# immediately as always; only cycles wait for the next sweep.

_gc_lock = threading.Lock()
_gc_suspenders = 0
_gc_was_enabled = True


def _suspend_automatic_gc() -> None:
    global _gc_suspenders, _gc_was_enabled
    with _gc_lock:
        if _gc_suspenders == 0:
            _gc_was_enabled = gc.isenabled()
            gc.disable()
        _gc_suspenders += 1


def _resume_automatic_gc() -> None:
    """Restore the collector once the last runner has gone."""
    global _gc_suspenders
    with _gc_lock:
        if _gc_suspenders == 0:
            return
        _gc_suspenders -= 1
        if _gc_suspenders == 0 and _gc_was_enabled:
            gc.enable()


class BackgroundRunner:
    """Runs adapter calls off the Tk thread; delivers results back on it.

    Tk widgets may only be touched from the thread running ``mainloop()``.
    Every page submits its adapter/viewmodel calls here instead of calling
    them directly, so a page's ``refresh()`` always returns immediately and
    the window never freezes while training/benchmark/evaluation artifacts
    are being read from disk or a process list is being walked.
    """

    #: How long ``close`` will wait for work already in flight.
    CLOSE_TIMEOUT_S = 5.0

    def __init__(self, root: tk.Misc, workers: int = 3, poll_ms: int = 100) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="control-center-bg"
        )
        self._queue: queue.Queue[
            tuple[Callable[[Any, BaseException | None], None], Any, BaseException | None]
        ] = queue.Queue()
        self._closed = False
        self._poll_ms = poll_ms
        self._pending: set[Future[None]] = set()
        self._pending_lock = threading.Lock()
        # Collect every tenth pump rather than every one: a full sweep of a
        # live GUI heap costs a few milliseconds, which is fine once a
        # second and wasteful ten times a second.
        self._pumps_between_collections = max(1, 1000 // max(poll_ms, 1))
        self._pumps_since_collection = 0
        # Exactly one Tk `after` callback may be outstanding. Tests manually
        # drain `_pump()` without mainloop; scheduling unconditionally from
        # those direct calls used to grow a second, third, then unbounded
        # callback chain, most visibly as a Windows desktop-test timeout.
        self._pump_after_id: str | None = None
        _suspend_automatic_gc()
        # Tied to the object, not just to close(): a runner that is dropped
        # without being closed - a constructor that raised half-way, a test
        # that forgot - must still hand the collector back. Leaving
        # automatic collection off process-wide because one runner leaked
        # would slow everything after it to a crawl, which is a worse
        # failure than the one this guards against, and a silent one.
        # Calling a finalize object runs it at most once, so close() and
        # collection cannot both release the same suspension.
        self._gc_release = weakref.finalize(self, _resume_automatic_gc)
        self._pump()

    def submit(
        self, fn: Callable[[], Any], callback: Callable[[Any, BaseException | None], None]
    ) -> bool:
        """Queue background work and report whether the runner accepted it.

        A caller may need to release local in-flight state when shutdown has
        already started.  Returning the acceptance result keeps that state
        from getting stranded without making normal fire-and-forget callers
        inspect a value they do not need.
        """
        if self._closed:
            return False

        def _run() -> None:
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to the GUI, never crashes a bg thread
                self._queue.put((callback, None, exc))
            else:
                self._queue.put((callback, result, None))

        future = self._executor.submit(_run)
        with self._pending_lock:
            self._pending.add(future)
        future.add_done_callback(self._forget)
        return True

    def _forget(self, future: Future[None]) -> None:
        with self._pending_lock:
            self._pending.discard(future)

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
        self._collect_if_due()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        """Queue one future pump, including when tests call `_pump` directly."""
        if self._closed or self._pump_after_id is not None:
            return
        self._pump_after_id = self._root.after(self._poll_ms, self._scheduled_pump)

    def _scheduled_pump(self) -> None:
        # Only Tk invokes this wrapper. A direct `_pump()` must leave its
        # already-pending timer in place rather than multiplying callbacks.
        self._pump_after_id = None
        if not self._closed:
            self._pump()

    def _cancel_scheduled_pump(self) -> None:
        if self._pump_after_id is None:
            return
        cancel = getattr(self._root, "after_cancel", None)
        if callable(cancel):
            with contextlib.suppress(tk.TclError):
                cancel(self._pump_after_id)
        self._pump_after_id = None

    def _collect_if_due(self) -> None:
        """Run the cyclic collector here, on the thread that owns Tk."""
        self._pumps_since_collection += 1
        if self._pumps_since_collection < self._pumps_between_collections:
            return
        self._pumps_since_collection = 0
        gc.collect()

    def close(self) -> None:
        """Stop accepting work and wait for what is already running.

        ``wait=True`` is load-bearing, not politeness. Callers close the
        runner and then immediately tear down the window - ``_on_close``
        does ``close()``, ``adapter.close()``, ``destroy()`` in a row. A
        worker still inside an adapter call at that point holds the
        closure that submitted it, and those closures capture pages,
        which hold Tk widgets. Whichever thread drops the last reference
        runs the finaliser, so with ``wait=False`` that is the worker,
        and Tk objects get finalised off the thread that owns the Tcl
        interpreter. On Linux that usually gets away with it; on Windows
        it is a hard interpreter crash with no traceback (exception code
        0x80000003, observed in CI during an unrelated test).

        ``cancel_futures`` throws away everything that has not started,
        so the wait is bounded by the single in-flight call rather than
        by the whole queue - and by ``CLOSE_TIMEOUT_S`` on top of that,
        because a wedged adapter call must not be able to hang the
        window shut.
        """
        if self._closed:
            return
        self._closed = True
        self._cancel_scheduled_pump()
        with self._pending_lock:
            pending = set(self._pending)
        for future in pending:
            future.cancel()
        # Bounded, not open-ended. Waiting is what keeps a worker from
        # outliving the widgets it can reach, but an adapter call that
        # has wedged - a process listing that will not come back, a
        # network path that stopped answering - must not take the window
        # with it. Whatever is still running after this has already lost
        # the race with the GC suspension above, which is the real guard.
        concurrent.futures.wait(pending, timeout=self.CLOSE_TIMEOUT_S)
        self._executor.shutdown(wait=False, cancel_futures=True)
        # Results that arrived while shutting down. Nothing will deliver
        # them now, and each one holds a callback holding widgets; drain
        # them here so they are released on the Tk thread instead of
        # whenever the queue itself is collected.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # One last sweep on this thread, so the widget tree the caller is
        # about to destroy does not become a worker's problem later, then
        # hand the collector back.
        gc.collect()
        self._gc_release()


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------

# Shared desktop Control Center palette. These module constants are the
# *default* theme's colours: they keep standalone widget construction (and
# the older imports in tests) working, while every live widget reads
# ``ThemeBus.theme`` so a theme switch repaints it. New code should prefer
# ``app.palette`` / ``bus.theme`` over these constants.
_FONT_FAMILY = FONT_FAMILY
_MONO_FONT = MONO_FONT_FAMILY
_DEFAULT_THEME = get_theme("corz")
_DEFAULT_BUS = ThemeBus(_DEFAULT_THEME)
COLOR_BG = _DEFAULT_THEME.bg
COLOR_SURFACE = _DEFAULT_THEME.panel
COLOR_SURFACE_RAISED = _DEFAULT_THEME.card
COLOR_HOVER = _DEFAULT_THEME.card_hover
COLOR_TEXT = _DEFAULT_THEME.text
COLOR_ACCENT = _DEFAULT_THEME.accent
COLOR_ACCENT_SECONDARY = _DEFAULT_THEME.accent_second
COLOR_OK = _DEFAULT_THEME.ok
COLOR_WARN = _DEFAULT_THEME.warn
COLOR_ERROR = _DEFAULT_THEME.error
COLOR_MUTED = _DEFAULT_THEME.text_muted
COLOR_BORDER = _DEFAULT_THEME.border
COLOR_GRID = _DEFAULT_THEME.card_hover


def _bus_from_ancestors(widget: tk.Misc) -> ThemeBus | None:
    """The nearest ancestor's ThemeBus, for helpers built without one.

    Widgets that take a bus stamp it as ``_cc_bus``, and a page stamps the
    shell's bus on itself, so a small helper (a tooltip on a button, a chart
    inside a card) can be constructed without repeating ``bus=`` and still
    follow a live theme switch instead of the module's default palette.
    """
    node: Any = widget
    seen = 0
    while node is not None and seen < 64:
        bus = getattr(node, "_cc_bus", None)
        if isinstance(bus, ThemeBus):
            return bus
        node = getattr(node, "master", None)
        seen += 1
    return None


class ToolTip:
    """Small, delayed keyboard/mouse help bubble for otherwise terse controls."""

    def __init__(self, widget: tk.Widget, text: str, *, bus: ThemeBus | None = None) -> None:
        self.widget = widget
        self.text = text
        self.bus = bus or _bus_from_ancestors(widget) or _DEFAULT_BUS
        self._cc_bus = self.bus
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<FocusIn>", self._schedule, add="+")
        widget.bind("<FocusOut>", self._hide, add="+")
        widget.bind("<Destroy>", self._on_widget_destroy, add="+")

    def _schedule(self, _event: object = None) -> None:
        if not self.widget.winfo_exists():
            return
        self._cancel()
        self._after_id = self.widget.after(380, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window is not None or not self.text or not self.widget.winfo_exists():
            return
        self._after_id = None
        theme = self.bus.theme
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(
            f"+{self.widget.winfo_rootx() + 12}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 6}"
        )
        tk.Label(
            tip,
            text=self.text,
            justify="left",
            background=theme.card,
            foreground=theme.text,
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=6,
            font=self.bus.font("small"),
            wraplength=360,
        ).pack()
        self._window = tip

    def _hide(self, _event: object = None) -> None:
        self._cancel()
        if self._window is not None:
            with contextlib.suppress(tk.TclError):
                self._window.destroy()
            self._window = None

    def _on_widget_destroy(self, _event: object = None) -> None:
        # A delayed callback must never try to query a widget after its page
        # was replaced or the application was closed.
        self._hide()


class StatCard(ttk.Frame):
    """One KPI card: rounded surface, accent rail and an animated value.

    The value animates (count-up for numbers, a colour settle for text) so a
    polling refresh reads as live telemetry rather than a label flicker. A
    disabled motion level applies the final value immediately, which keeps
    tests and low-motion operators on exactly the same numbers.
    """

    def __init__(
        self,
        parent: tk.Misc,
        label: str,
        *,
        bus: ThemeBus | None = None,
        motion: MotionController | None = None,
    ) -> None:
        super().__init__(parent, style="Card.TFrame", padding=(0, 0))
        self._bus = bus or _DEFAULT_BUS
        self._cc_bus = self._bus
        self._theme = self._bus.theme
        self._motion = motion
        self._accent_bar = tk.Frame(self, height=2, background=self._theme.border, borderwidth=0)
        self._accent_bar.pack(fill="x", side="top")
        padding = self._bus.px(16, minimum=10)
        self._body = ttk.Frame(
            self, style="CardInner.TFrame", padding=(padding, self._bus.px(11, minimum=8))
        )
        self._body.pack(fill="both", expand=True)
        clean_label = label[:1].upper() + label[1:] if label else ""
        self._title_lbl = ttk.Label(self._body, text=clean_label, style="CardLabel.TLabel")
        self._title_lbl.pack(anchor="w")
        self._value = tk.Label(
            self._body,
            text="n/a",
            background=self._theme.card,
            foreground=self._theme.text,
            anchor="w",
            font=self._bus.font("h2", bold=True, mono=True),
        )
        self._value.pack(anchor="w", pady=(self._bus.px(4, minimum=2), 0), fill="x")
        self._animated = AnimatedValue(self._value, motion or MotionController(self, level="off"))
        self._current_color: str | None = None
        self._unsubscribe = self._bus.subscribe(self.apply_theme, owner=self)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            padding = self._bus.px(16, minimum=8)
            self._body.configure(padding=(padding, self._bus.px(11, minimum=6)))
            self._value.configure(
                background=theme.card,
                font=self._bus.font("h2", bold=True, mono=True),
            )
            self._accent_bar.configure(background=self._current_color or theme.border)

    def set(
        self,
        text: str,
        color: str | None = None,
        *,
        value: float | None = None,
        formatter: Callable[[float], str] | None = None,
    ) -> None:
        self._current_color = color
        self._animated.set(
            text,
            color or self._theme.text,
            value=value,
            formatter=formatter,
            animate=self._motion is not None,
        )
        self._accent_bar.configure(background=color or self._theme.border)


class StatRow(ttk.Frame):
    """A balanced grid of :class:`StatCard` modules built from an ordered label tuple."""

    def __init__(
        self,
        parent: tk.Misc,
        labels: tuple[str, ...],
        *,
        max_columns: int = 5,
        bus: ThemeBus | None = None,
        motion: MotionController | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus or _DEFAULT_BUS
        self._cc_bus = self._bus
        self._cards: dict[str, StatCard] = {}
        columns = min(max_columns, max(1, len(labels)))
        gap = self._bus.px(6, minimum=3)
        for index, label in enumerate(labels):
            card = StatCard(self, label, bus=self._bus, motion=motion)
            row, column = divmod(index, columns)
            card.grid(row=row, column=column, sticky="nsew", padx=gap, pady=gap)
            self.columnconfigure(column, weight=1, uniform="stats")
            self._cards[label] = card

    def column_count(self) -> int:
        """How many KPI columns the current grid holds (layout tests use this)."""
        return max(1, len(self._cards))

    def update_values(
        self,
        values: dict[str, tuple[str, str | None]],
        *,
        numeric: dict[str, tuple[float, str, Callable[[float], str]]] | None = None,
    ) -> None:
        """Update cards; ``numeric`` optionally supplies count-up values."""
        for label, (text, color) in values.items():
            card = self._cards.get(label)
            if card is None:
                continue
            if numeric and label in numeric:
                value, _fallback, formatter = numeric[label]
                card.set(text, color, value=value, formatter=formatter)
            else:
                card.set(text, color)


class PhaseStepper(tk.Canvas):
    """Clean horizontal workflow stepper with a smoothly animated track."""

    def __init__(
        self,
        parent: tk.Misc,
        height: int = 52,
        *,
        bus: ThemeBus | None = None,
        motion: MotionController | None = None,
    ) -> None:
        self._bus = bus or _DEFAULT_BUS
        self._cc_bus = self._bus
        self._theme = self._bus.theme
        self._motion = motion
        super().__init__(
            parent,
            height=height if self._motion is None else max(height, 42),
            background=self._theme.panel,
            highlightthickness=1,
            highlightbackground=self._theme.border,
        )
        self._phases: list[dict[str, str]] = []
        self._fraction: float = 0.0
        self._display_fraction: float = 0.0
        self._anim_after_id: str | None = None
        self._motion_handle: int | None = None
        #: Size of the last paint. A ``<Configure>`` that does not change the
        #: size must not rebuild every canvas item again - a resize drag
        #: delivers a stream of them, and repainting the same picture is pure
        #: event churn.
        self._painted_size: tuple[int, int] | None = None
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Destroy>", self._cancel_anim, add="+")
        self._unsubscribe = self._bus.subscribe(self.apply_theme, owner=self)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.configure(background=theme.panel, highlightbackground=theme.border)
            self._redraw(force=True)

    def _status_palette(self) -> dict[str, tuple[str, str, str]]:
        theme = self._theme
        return {
            "pending": (theme.card, theme.border, theme.text_muted),
            "active": (theme.card_active, theme.accent, theme.text),
            "done": (theme.card, theme.ok, theme.ok),
            "skipped": (theme.card, theme.warn, theme.warn),
            "failed": (theme.card, theme.error, theme.error),
        }

    def _cancel_anim(self, event: object = None) -> None:
        # `<Destroy>` fires for every child as well; only the stepper itself
        # owns these handles.
        if event is not None and getattr(event, "widget", None) is not self:
            return
        if self._motion_handle is not None and self._motion is not None:
            self._motion.cancel(self._motion_handle)
            self._motion_handle = None
        if self._anim_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._anim_after_id)
            self._anim_after_id = None

    def set_state(self, phases: list[dict[str, str]], fraction: float = 0.0) -> None:
        self._phases = list(phases)
        target = max(0.0, min(1.0, float(fraction)))
        self._fraction = target
        if target == 0.0 and self._display_fraction > 0.5:
            self._display_fraction = 0.0
        if self._motion is not None:
            start = self._display_fraction
            if self._motion_handle is not None:
                # One transition at a time: a second call (a new episode
                # boundary) replaces the one in flight instead of stacking.
                self._motion.cancel(self._motion_handle)
            self._motion_handle = self._motion.tween(
                320,
                lambda progress: self._set_display(
                    start + (target - start) * ease_out_cubic(progress)
                ),
            )
            return
        self._schedule_smooth_step()

    def _set_display(self, value: float) -> None:
        self._display_fraction = max(0.0, min(1.0, value))
        self._redraw(force=True)

    def set_phases(self, phases: list[dict[str, str]], fraction: float = 0.0) -> None:
        self.set_state(phases, fraction)

    def _schedule_smooth_step(self) -> None:
        diff = self._fraction - self._display_fraction
        if abs(diff) <= 0.004 or not self.winfo_exists():
            self._display_fraction = self._fraction
            self._cancel_anim()
            self._redraw(force=True)
            return
        self._display_fraction += diff * 0.28
        self._redraw(force=True)
        if self._anim_after_id is None:
            with contextlib.suppress(tk.TclError):
                self._anim_after_id = self.after(16, self._on_anim_tick)

    def _on_anim_tick(self) -> None:
        self._anim_after_id = None
        self._schedule_smooth_step()

    def _redraw(self, *, force: bool = False) -> None:
        size = (max(int(self.winfo_width()), 1), max(int(self.winfo_height()), 1))
        if not force and size == self._painted_size:
            return
        self._painted_size = size
        self.delete("all")
        theme = self._theme
        width, height = size
        if not self._phases:
            return
        palette = self._status_palette()
        count = len(self._phases)
        pad_x = self._bus.px(14, minimum=8)
        gap = self._bus.px(10, minimum=6)
        box_h = self._bus.px(28, minimum=22)
        box_y0 = self._bus.px(8, minimum=5)
        box_y1 = box_y0 + box_h
        slot_w = max(40.0, (width - 2 * pad_x - (count - 1) * gap) / count)
        radius = min(self._bus.px(8, minimum=4), box_h / 2)
        for idx, phase in enumerate(self._phases):
            status = phase.get("status", "pending")
            bg, border, fg = palette.get(status, palette["pending"])
            x0 = pad_x + idx * (slot_w + gap)
            x1 = x0 + slot_w
            if idx < count - 1:
                conn_color = theme.ok if status == "done" else theme.border
                self.create_line(
                    x1,
                    (box_y0 + box_y1) / 2,
                    x1 + gap,
                    (box_y0 + box_y1) / 2,
                    fill=conn_color,
                    width=1,
                )
            rounded_rect(
                self,
                x0,
                box_y0,
                x1,
                box_y1,
                radius,
                fill=bg,
                outline=border,
                width=1,
            )
            dot_color = border if status != "pending" else theme.text_muted
            self.create_oval(
                x0 + self._bus.px(11, minimum=7),
                (box_y0 + box_y1) / 2 - 3,
                x0 + self._bus.px(17, minimum=12),
                (box_y0 + box_y1) / 2 + 3,
                fill=dot_color,
                outline="",
            )
            label = str(phase.get("label", "")).capitalize()
            self.create_text(
                x0 + self._bus.px(24, minimum=18),
                (box_y0 + box_y1) / 2,
                anchor="w",
                text=f"{idx + 1}. {label}",
                fill=fg,
                font=self._bus.font("small", bold=status == "active"),
            )
        # Animated bottom progress track.
        bar_y0 = height - self._bus.px(7, minimum=4)
        bar_y1 = height - self._bus.px(3, minimum=2)
        rounded_rect(
            self,
            pad_x,
            bar_y0,
            width - pad_x,
            bar_y1,
            2,
            fill=theme.bg,
            outline="",
        )
        if self._display_fraction > 0.0:
            fill_w = (width - 2 * pad_x) * self._display_fraction
            bar_color = theme.ok if self._fraction >= 0.999 else theme.accent
            rounded_rect(
                self,
                pad_x,
                bar_y0,
                max(pad_x + fill_w, pad_x + 4),
                bar_y1,
                2,
                fill=bar_color,
                outline="",
            )


class LineChart(tk.Canvas):
    """Clean, minimal telemetry chart with smooth Y-domain interpolation and hover pill."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        height: int = 152,
        *,
        color: str | None = None,
        bus: ThemeBus | None = None,
    ) -> None:
        self._bus = bus or _DEFAULT_BUS
        self._cc_bus = self._bus
        self._theme = self._bus.theme
        self._base_height = height
        eff_height = (
            max(96, int(round(height * self._bus.scale.viewport_scale)))
            if self._bus.scale.viewport_scale < 1.0
            else height
        )
        super().__init__(
            parent,
            height=eff_height,
            background=self._theme.card,
            highlightthickness=1,
            highlightbackground=self._theme.border,
        )
        self._title = title
        self._color_override = color
        self._points: list[tuple[float, float]] = []
        self._points_sig: tuple[Any, ...] | None = None
        self._hover_x: int | None = None
        #: Size of the last paint; see :attr:`PhaseStepper._painted_size`.
        self._painted_size: tuple[int, int] | None = None
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self._unsubscribe = self._bus.subscribe(self.apply_theme, owner=self)

    @property
    def _color(self) -> str:
        return self._color_override or self._theme.accent

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            eff_height = (
                max(96, int(round(self._base_height * self._bus.scale.viewport_scale)))
                if self._bus.scale.viewport_scale < 1.0
                else self._base_height
            )
            self.configure(
                background=theme.card,
                highlightbackground=theme.border,
                height=eff_height,
            )
            self._redraw(force=True)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        sig = (
            len(points),
            points[0] if points else None,
            points[-1] if points else None,
            points[len(points) // 2] if points else None,
        )
        self._points = points
        if sig == self._points_sig and self._painted_size is not None:
            return
        self._points_sig = sig
        self._redraw(force=True)

    def point_count(self) -> int:
        """Number of plotted samples (used by the animation/layout tests)."""
        return len(self._points)

    def _on_motion(self, event: tk.Event[Any]) -> None:
        self._hover_x = int(getattr(event, "x", 0))
        if len(self._points) >= 2:
            self._redraw(force=True)

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        if self._hover_x is not None:
            self._hover_x = None
            self._redraw(force=True)

    def _draw_grid(
        self,
        width: int,
        height: int,
        pad_left: int,
        pad_right: int,
        pad_top: int,
        pad_bottom: int,
        y_min: float,
        y_max: float,
    ) -> None:
        plot_h = height - pad_top - pad_bottom
        for fraction in (0.0, 0.5, 1.0):
            gy = pad_top + fraction * plot_h
            self.create_line(pad_left, gy, width - pad_right, gy, fill=self._theme.border)
            value = y_max - fraction * (y_max - y_min)
            self.create_text(
                pad_left - self._bus.px(8, minimum=5),
                gy,
                anchor="e",
                text=vm.format_number(value, 2),
                fill=self._theme.text_muted,
                font=self._bus.font("micro", mono=True),
            )

    def _draw_hover(
        self,
        points: list[tuple[float, float]],
        coords: list[float],
        width: int,
        height: int,
        pad_left: int,
        pad_right: int,
        pad_top: int,
        pad_bottom: int,
    ) -> str | None:
        hover_x = self._hover_x
        if hover_x is None or not points:
            return None
        best_idx = min(range(len(points)), key=lambda i: abs(coords[2 * i] - hover_x))
        hx, hy = coords[2 * best_idx], coords[2 * best_idx + 1]
        px, py = points[best_idx]
        self.create_line(
            hx,
            pad_top,
            hx,
            height - pad_bottom,
            fill=self._theme.border_strong,
            dash=(3, 3),
        )
        self.create_oval(
            hx - 4,
            hy - 4,
            hx + 4,
            hy + 4,
            fill=self._color,
            outline=self._theme.text,
            width=1,
        )
        return f"Step {vm.format_number(px, 0)}: {vm.format_number(py, 3)}   \u00b7   "

    def _redraw(self, *, force: bool = False) -> None:
        size = (max(int(self.winfo_width()), 1), max(int(self.winfo_height()), 1))
        if not force and size == self._painted_size:
            return
        self._painted_size = size
        self.delete("all")
        theme = self._theme
        width, height = size
        pad_left = self._bus.px(60, minimum=42)
        pad_right = self._bus.px(14, minimum=8)
        pad_top = self._bus.px(30, minimum=22)
        pad_bottom = self._bus.px(18, minimum=12)
        clean_title = self._title[:1].upper() + self._title[1:] if self._title else ""
        self.create_text(
            self._bus.px(12, minimum=6),
            self._bus.px(8, minimum=4),
            anchor="nw",
            text=clean_title,
            font=self._bus.font("small", bold=True),
            fill=theme.text,
        )
        points = vm.downsample_series(
            self._points, max_points=max(width - pad_left - pad_right, 10)
        )
        if len(points) < 2:
            self.create_text(
                width / 2,
                height / 2 + 6,
                text="Waiting for telemetry\u2026",
                fill=theme.text_muted,
                font=self._bus.font("small"),
            )
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        raw_y_min, raw_y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = raw_y_min, raw_y_max
        if y_min == y_max:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        if x_min == x_max:
            x_min, x_max = x_min - 1.0, x_max + 1.0
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom
        self._draw_grid(width, height, pad_left, pad_right, pad_top, pad_bottom, y_min, y_max)

        coords: list[float] = []
        for x, y in points:
            cx = pad_left + (x - x_min) / (x_max - x_min) * plot_w
            cy = pad_top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h
            coords.extend((cx, cy))
        baseline_y = height - pad_bottom
        poly_coords = [coords[0], baseline_y, *coords, coords[-2], baseline_y]
        area_color = lerp_color(theme.card, self._color, 0.18)
        self.create_polygon(*poly_coords, fill=area_color, outline="")
        if len(points) >= 4:
            ema_coords: list[float] = []
            ema_val = points[0][1]
            alpha = 0.25
            for idx, (_px, py) in enumerate(points):
                ema_val = py if idx == 0 else (alpha * py + (1.0 - alpha) * ema_val)
                ex = coords[2 * idx]
                ey = pad_top + (1.0 - (ema_val - y_min) / (y_max - y_min)) * plot_h
                ema_coords.extend((ex, ey))
            ema_color = lerp_color(self._color, theme.text, 0.45)
            self.create_line(*ema_coords, fill=ema_color, width=1, dash=(4, 2), smooth=False)
        self.create_line(*coords, fill=self._color, width=2, smooth=False)
        peak_idx = max(range(len(ys)), key=lambda idx: ys[idx])
        peak_x, peak_y = coords[2 * peak_idx], coords[2 * peak_idx + 1]
        self.create_oval(
            peak_x - 4,
            peak_y - 4,
            peak_x + 4,
            peak_y + 4,
            fill="",
            outline=theme.ok,
            width=1,
        )
        last_x, last_y = coords[-2], coords[-1]
        self.create_oval(
            last_x - 3,
            last_y - 3,
            last_x + 3,
            last_y + 3,
            fill=self._color,
            outline=theme.card,
        )
        cursor_prefix = (
            self._draw_hover(
                points, coords, width, height, pad_left, pad_right, pad_top, pad_bottom
            )
            or ""
        )
        summary = (
            f"{cursor_prefix}Min {vm.format_number(raw_y_min, 2)}   \u00b7   "
            f"Max {vm.format_number(raw_y_max, 2)}   \u00b7   "
            f"Latest {vm.format_number(ys[-1], 3)}"
        )
        self.create_text(
            width - pad_right,
            self._bus.px(8, minimum=4),
            anchor="ne",
            text=summary,
            fill=theme.text if cursor_prefix else theme.text_muted,
            font=self._bus.font("micro", mono=True),
        )


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

    def __init__(
        self,
        parent: tk.Misc,
        max_lines: int = 4000,
        *,
        bus: ThemeBus | None = None,
        motion: MotionController | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus or _DEFAULT_BUS
        self._theme = self._bus.theme
        self._motion = motion
        self._max_lines = max_lines
        self._stdout_after = -1
        self._stderr_after = -1
        self._user_scrolled_up = False
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        self._autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Auto-scroll",
            variable=self._autoscroll,
            command=self._on_autoscroll_toggled,
        ).pack(side="left")
        # Wrapping is on by default: process output is read, not compared,
        # and wrap=word removes the horizontal scrollbar entirely. Turning it
        # off keeps long command lines/tracebacks intact and reveals the
        # overlay bar only while the reader is actually interacting.
        self._wrap = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Wrap lines",
            variable=self._wrap,
            command=self._on_wrap_toggled,
        ).pack(side="left", padx=(10, 0))
        ttk.Button(toolbar, text="Clear view", command=self.clear).pack(side="left", padx=(8, 0))
        self._truncated_label = ttk.Label(toolbar, text="", style="PillWarn.TLabel")
        self._truncated_label.pack(side="right")
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.text = tk.Text(
            text_frame,
            height=16,
            wrap="word",
            state="disabled",
            background=self._text_surface(),
            foreground=self._theme.text,
            insertbackground=self._theme.text,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=6,
            font=self._bus.font("small", mono=True),
        )
        self.text.pack(fill="both", expand=True)
        thickness = self._bus.px(SCROLLBAR_THICKNESS, minimum=6)
        self._yscroll = SlimScrollbar(
            text_frame,
            self._bus,
            orient="vertical",
            command=self._scroll_text_y,
            thickness=thickness,
            place={"relx": 1.0, "rely": 0.0, "relheight": 1.0, "anchor": "ne", "width": thickness},
        )
        self._xscroll = SlimScrollbar(
            text_frame,
            self._bus,
            orient="horizontal",
            command=self.text.xview,
            thickness=thickness,
            place={"relx": 0.0, "rely": 1.0, "relwidth": 1.0, "anchor": "sw", "height": thickness},
        )
        self.text.configure(yscrollcommand=self._on_text_scroll, xscrollcommand=self._xscroll.set)
        self.text.tag_configure("stderr", foreground=self._theme.error)
        self.text.tag_configure("meta", foreground=self._theme.text_muted)
        self.text.bind("<Enter>", lambda _e: self._reveal_scrollbars(), add="+")
        self._unsubscribe = self._bus.subscribe(self.apply_theme, owner=self)

    def _text_surface(self) -> str:
        """The log surface: inset, like every other field on a card.

        It used to be ``panel``, which on the dark themes is *darker* than the
        card the log sits in - the opposite of the inset reading a text area
        wants, and one more black rectangle on the Settings and Runs pages.
        """
        return self._theme.field_surface()[0]

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            self.text.configure(
                background=self._text_surface(),
                foreground=theme.text,
                insertbackground=theme.text,
                font=self._bus.font("small", mono=True),
            )
            self.text.tag_configure("stderr", foreground=theme.error)
            self.text.tag_configure("meta", foreground=theme.text_muted)

    def _reveal_scrollbars(self) -> None:
        self._yscroll.reveal()
        if not self._wrap.get():
            self._xscroll.reveal()

    def _on_wrap_toggled(self) -> None:
        wrapping = bool(self._wrap.get())
        self.text.configure(wrap="word" if wrapping else "none")
        if not wrapping:
            self._reveal_scrollbars()
        if self._autoscroll.get() and not self._user_scrolled_up:
            self.text.see("end")
        # Wheel input covers the common pointer path, while the wrapped
        # scrollbar command and navigation keys cover track dragging,
        # scrollbar arrows and keyboard readers. Previously only wheel
        # events paused follow-mode, so dragging the visible scrollbar up
        # could still snap a reader back to the newest log line on refresh.
        self.text.bind("<MouseWheel>", self._on_manual_scroll, add="+")
        self.text.bind("<Button-4>", self._on_manual_scroll, add="+")
        self.text.bind("<Button-5>", self._on_manual_scroll, add="+")
        for sequence in ("<Prior>", "<Next>", "<Home>", "<End>", "<Up>", "<Down>"):
            self.text.bind(sequence, self._on_manual_scroll, add="+")

    def _on_manual_scroll(self, _event: object = None) -> None:
        self._user_scrolled_up = True
        # Instance bindings run before Tk's class binding moves the viewport.
        # Re-check after that binding so a downward wheel/key action at the
        # newest line immediately resumes follow mode instead of leaving it
        # silently paused.
        with contextlib.suppress(tk.TclError):
            self.text.after_idle(self._sync_follow_state)

    def _scroll_text_y(self, *args: str) -> None:
        """Forward scrollbar input and preserve an operator's reading position."""
        self._user_scrolled_up = True
        self.text.yview(*args)
        self._sync_follow_state()
        self._yscroll.reveal()

    def _sync_follow_state(self) -> None:
        """Resume live follow only when the viewport is genuinely at the end."""
        try:
            _first, last = self.text.yview()
        except tk.TclError:
            return
        if last >= 0.999:
            self._user_scrolled_up = False

    def _on_text_scroll(self, first: float | str, last: float | str) -> None:
        """Synchronize the native scrollbar and restore follow-at-bottom.

        Tk may invoke a registered callback with Tcl strings while its type
        stubs describe numeric fractions, so both forms are accepted here.
        A manual wheel event pauses live-follow while an operator reads
        historical output; reaching the newest line makes the already-enabled
        Auto-scroll checkbox useful immediately.
        """
        try:
            first_fraction = float(first)
            last_fraction = float(last)
        except ValueError:  # pragma: no cover - Tk sends numeric fractions
            return
        self._yscroll.set(first_fraction, last_fraction)
        if last_fraction >= 0.999:
            self._user_scrolled_up = False

    def _on_autoscroll_toggled(self) -> None:
        if self._autoscroll.get():
            self._user_scrolled_up = False
            self.text.see("end")

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

    def snapshot_view(self) -> dict[str, Any]:
        """What to put back when the page rebuilt around this panel.

        A density change destroys and recreates every widget on the page,
        which used to blank the log and drop the reader's position. The
        page captures this before the children go away and hands it to the
        fresh panel, so the log survives the restyle.
        """
        return {
            "text": self.text.get("1.0", "end-1c"),
            "autoscroll": bool(self._autoscroll.get()),
            "wrap": bool(self._wrap.get()),
            "scrolled_up": bool(self._user_scrolled_up),
            "stdout_after": self._stdout_after,
            "stderr_after": self._stderr_after,
            "yview": float(self.text.yview()[0]),
        }

    def restore_view(self, snapshot: Mapping[str, Any]) -> None:
        """Re-apply a :meth:`snapshot_view` result to a freshly built panel."""
        content = str(snapshot.get("text") or "")
        self._autoscroll.set(bool(snapshot.get("autoscroll", True)))
        self._wrap.set(bool(snapshot.get("wrap", True)))
        self._on_wrap_toggled()
        self._user_scrolled_up = bool(snapshot.get("scrolled_up", False))
        # The cursors matter as much as the text: the next poll appends only
        # what is new, so restoring the text without them would duplicate it.
        self._stdout_after = int(snapshot.get("stdout_after", -1))
        self._stderr_after = int(snapshot.get("stderr_after", -1))
        if content:
            self.text.configure(state="normal")
            self.text.insert("1.0", content)
            self.text.configure(state="disabled")
            self.text.yview_moveto(float(snapshot.get("yview", 1.0)))
        if self._autoscroll.get() and not self._user_scrolled_up:
            self.text.see("end")

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


def _scrollable_table(
    parent: tk.Misc,
    columns: tuple[tuple[str, str, int], ...],
    *,
    expand: bool = True,
    bus: ThemeBus | None = None,
    empty_text: str = "",
    fit_columns: bool = False,
) -> ttk.Treeview:
    """Build a sortable table whose overlay bars appear only when needed.

    Dense inventories still have to keep every column reachable on a narrow
    window, but two permanently visible native scrollbars were the single
    loudest piece of chrome in the old UI. The table stretches its columns to
    *stretch* into spare width (Tk's own column stretching, no Python in the
    loop) and otherwise exposes an overlay scrollbar while the pointer is
    inside it.

    ``empty_text`` gives the table a quiet overlay line for its empty state
    ("No runs recorded yet - launch a training run to fill this in"). A table
    that shows only headings reads like a broken one; the overlay is *placed*
    rather than gridded, so it can never influence any geometry.
    """
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=expand)
    active_bus = bus or _DEFAULT_BUS
    tree = _sortable_table(frame, columns, bus=active_bus)
    tree.pack(fill="both", expand=True)
    attach_overlay_scrollbars(frame, tree, active_bus, scale_px=active_bus.px)
    if fit_columns:
        _fit_table_to_viewport(tree, columns, active_bus)
    if empty_text:
        tree.empty_state = TableEmptyState(tree, active_bus, empty_text)  # type: ignore[attr-defined]
    return tree


def _fit_table_to_viewport(
    tree: ttk.Treeview, columns: tuple[tuple[str, str, int], ...], bus: ThemeBus
) -> None:
    """Fit a wide table using real heading-font widths, not relaxed tests."""
    from tkinter import font as tkfont

    def fit(_event: Any = None) -> None:
        try:
            available = tree.winfo_width() - bus.px(16, minimum=12)
            if available <= 0:
                return
            font_name = ttk.Style(tree).lookup("Treeview.Heading", "font") or "TkHeadingFont"
            font = tkfont.Font(root=tree, font=font_name)
            minimum = [
                max(32, font.measure(title) + bus.px(16, minimum=12))
                for _key, title, _width in columns
            ]
            vp = min(1.0, bus.scale.viewport_scale)
            preferred = [max(32, round(width * vp)) for _key, _title, width in columns]
            fitted = fit_table_column_widths(preferred, minimum, available)
            for (key, _title, _width), width, floor in zip(columns, fitted, minimum):
                tree.column(key, width=width, minwidth=floor)
        except tk.TclError:  # destruction during a density/layout rebuild
            return

    tree.bind("<Configure>", fit, add="+")

    def schedule_fit(_theme: Theme) -> None:
        with contextlib.suppress(tk.TclError):
            tree.after_idle(fit)

    bus.subscribe(schedule_fit, owner=tree)


def refresh_table_empty(tree: Any) -> None:
    """Update a table's empty-state overlay after its rows changed.

    Pages call this right after they fill a table. The overlay also refreshes
    on the table's own ``<Configure>``, so a page that forgets the call still
    shows the hint as soon as the page is laid out.
    """
    state = getattr(tree, "empty_state", None)
    if state is not None:
        state.refresh()


class TableEmptyState:
    """A centred hint shown while a table has no rows.

    It lives in the table's own frame as a *placed* label, so showing or
    hiding it never changes a requested size - the same rule the cards
    follow. Visibility is derived from the tree itself, never tracked
    separately, so a page cannot leave a stale "no data" line over rows.
    """

    def __init__(self, tree: ttk.Treeview, bus: ThemeBus, text: str) -> None:
        self._tree = tree
        self._bus = bus
        self._label = ttk.Label(
            tree.master,
            text=text,
            style="Empty.TLabel",
            anchor="center",
            justify="center",
            wraplength=bus.px(420, minimum=240),
        )
        self._shown = False
        tree.bind("<Configure>", lambda _event: self.refresh(), add="+")

    @property
    def text(self) -> str:
        return str(self._label.cget("text"))

    def set_text(self, text: str) -> None:
        with contextlib.suppress(tk.TclError):
            self._label.configure(text=text)
        self.refresh()

    def is_shown(self) -> bool:
        """Whether the hint is currently placed (used by the smoke harness)."""
        return self._shown

    def refresh(self) -> None:
        try:
            empty = not self._tree.get_children("")
            exists = bool(self._tree.winfo_exists())
        except tk.TclError:  # the table was destroyed by a rebuild
            return
        show = empty and exists
        if show == self._shown:
            return
        self._shown = show
        with contextlib.suppress(tk.TclError):
            if show:
                self._label.place(relx=0.5, rely=0.5, anchor="center")
                self._label.lift()
            else:
                self._label.place_forget()


def _sortable_table(
    parent: tk.Misc,
    columns: tuple[tuple[str, str, int], ...],
    *,
    bus: ThemeBus | None = None,
) -> ttk.Treeview:
    """Builds a Treeview with click-to-sort columns (ascending/descending)."""
    tree = ttk.Treeview(
        parent, columns=tuple(c[0] for c in columns), show="headings", selectmode="extended"
    )
    numeric_columns = {
        "pid",
        "environments",
        "environment_count",
        "workers",
        "env_workers",
        "steps",
        "total_steps",
        "steps_per_second",
        "speedup",
        "episodes_per_second",
        "p50_ms",
        "p95_ms",
        "jitter",
        "startup_seconds",
        "elapsed_seconds",
        "timesteps",
        "progress",
        "progress_percent",
        "episodes",
        "win_rate",
        "loss_rate",
        "reward",
        "mean_episode_reward",
    }
    vp = bus.scale.viewport_scale if bus is not None else 1.0
    for key, title, width in columns:
        anchor = "e" if key in numeric_columns else "w"
        eff_w = max(32, int(round(width * vp))) if vp < 1.0 else width
        # typeshed types anchor as a literal enum; "e"/"w" are valid tk
        # anchors and are what the rest of this module already uses.
        tree.heading(  # type: ignore[call-overload]
            key, text=title, anchor=anchor, command=lambda k=key: _sort_tree(tree, k, False)
        )
        tree.column(  # type: ignore[call-overload]
            key,
            width=eff_w,
            minwidth=max(28, min(eff_w, 48)),
            anchor=anchor,
            stretch=True,
        )
    if bus is not None:

        def _rescale_columns(_theme: Theme) -> None:
            cur_vp = bus.scale.viewport_scale
            for col_key, _col_title, base_width in columns:
                scaled_w = max(32, int(round(base_width * cur_vp))) if cur_vp < 1.0 else base_width
                with contextlib.suppress(tk.TclError):
                    tree.column(
                        col_key,
                        width=scaled_w,
                        minwidth=max(28, min(scaled_w, 48)),
                    )

        bus.subscribe(_rescale_columns, owner=tree)
    return tree


def _sort_tree(tree: ttk.Treeview, column: str, descending: bool) -> None:
    def sort_key(item_id: str) -> Any:
        value = str(tree.set(item_id, column)).strip()
        cleaned = value.replace(",", "")
        if cleaned.endswith(("%", "x", "s")) and len(cleaned) > 1:
            cleaned = cleaned[:-1].strip()
        try:
            return (0, float(cleaned))
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


class CaptureView(tk.Canvas):
    """One captured frame, with the boxes the detector drew on it.

    The TTK captures used to have no surface in the GUI at all: the
    operator pressed Capture, was told a file had been written, and had to
    go and open it somewhere else to learn whether the enemy was even in
    the picture. This draws the frame where the button is.

    It shows whichever frame the caller hands it - the raw capture or the
    shadow-lifted one - so "lift shadows" is a page-level choice and this
    widget stays a dumb surface. Boxes are drawn here rather than baked
    into the image for the same reason: re-drawing a rectangle is free,
    re-encoding a 1080p PNG to move one box is not.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        bus: ThemeBus | None = None,
        height: int = 210,
        empty_text: str = "No capture yet — press Capture Screenshot.",
    ) -> None:
        self._bus = bus or _DEFAULT_BUS
        self._cc_bus = self._bus
        self._theme = self._bus.theme
        self._base_height = height
        effective = (
            max(96, int(round(height * self._bus.scale.viewport_scale)))
            if self._bus.scale.viewport_scale < 1.0
            else height
        )
        super().__init__(
            parent,
            height=effective,
            background=self._theme.card,
            highlightthickness=1,
            highlightbackground=self._theme.border,
        )
        self._empty_text = empty_text
        self._message = empty_text
        self._image: Any = None
        self._boxes: tuple[dict[str, Any], ...] = ()
        self._photo: tk.PhotoImage | None = None
        self._scale: float = 1.0
        self._offset: tuple[int, int] = (0, 0)
        self.bind("<Configure>", lambda _event: self._redraw())
        self._unsubscribe = self._bus.subscribe(self.apply_theme, owner=self)

    # -- content ----------------------------------------------------------

    def set_frame(
        self,
        image: Any,
        boxes: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        *,
        message: str | None = None,
    ) -> None:
        """Show ``image`` (an ``(H, W, 3)`` uint8 array, or ``None`` to clear)."""
        self._image = None if image is None else image
        self._boxes = tuple(boxes)
        if message is not None:
            self._message = message
        elif image is None:
            self._message = self._empty_text
        self._redraw()

    def clear(self, message: str | None = None) -> None:
        self.set_frame(None, (), message=message or self._empty_text)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            effective = (
                max(96, int(round(self._base_height * self._bus.scale.viewport_scale)))
                if self._bus.scale.viewport_scale < 1.0
                else self._base_height
            )
            self.configure(
                background=theme.card,
                highlightbackground=theme.border,
                height=effective,
            )
        self._redraw()

    # -- painting ---------------------------------------------------------

    def _redraw(self) -> None:
        with contextlib.suppress(tk.TclError):
            self.delete("all")
            width = max(1, int(self.winfo_width()))
            height = max(1, int(self.winfo_height()))
            if self._image is None:
                self._offset = (0, 0)
                self._scale = 1.0
                self._photo = None
                self.create_text(
                    width // 2,
                    height // 2,
                    text=self._message,
                    fill=self._theme.text_dim,
                    font=self._bus.font("body"),
                    width=max(80, width - 24),
                    justify="center",
                )
                return
            self._paint_frame(width, height)
            self._paint_boxes()
            self._paint_message(width, height)

    def _paint_frame(self, width: int, height: int) -> None:
        """Scale the frame down by whole pixels and put it on the canvas.

        The array is strided rather than the photo subsampled: Tk would
        still hold the full-resolution image either way, and a 1080p
        capture in a 400 px card is a lot of Tcl memory spent on pixels
        nobody can see.
        """
        rows, columns = int(self._image.shape[0]), int(self._image.shape[1])
        stride = max(1, int(-(-columns // max(1, width))), int(-(-rows // max(1, height))))
        small = self._image[::stride, ::stride]
        self._scale = 1.0 / float(stride)
        drawn_width = max(1, int(small.shape[1]))
        drawn_height = max(1, int(small.shape[0]))
        self._offset = (
            max(0, (width - drawn_width) // 2),
            max(0, (height - drawn_height) // 2),
        )
        try:
            self._photo = tk.PhotoImage(data=encode_png(small))
        except tk.TclError:
            self._photo = None
            self.create_text(
                width // 2,
                height // 2,
                text="this frame cannot be shown",
                fill=self._theme.text_dim,
                font=self._bus.font("body"),
            )
            return
        self.create_image(self._offset[0], self._offset[1], anchor="nw", image=self._photo)

    def _paint_boxes(self) -> None:
        for index, box in enumerate(self._boxes, start=1):
            x0 = self._offset[0] + float(box.get("x", 0)) * self._scale
            y0 = self._offset[1] + float(box.get("y", 0)) * self._scale
            x1 = x0 + float(box.get("width", 0)) * self._scale
            y1 = y0 + float(box.get("height", 0)) * self._scale
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            colour = self._theme.error if box.get("clipped") else self._theme.warn
            self.create_rectangle(x0, y0, x1, y1, outline=colour, width=2)
            self.create_text(
                x0 + 3,
                max(10.0, y0 - 8.0),
                anchor="sw",
                text=f"#{index}",
                fill=colour,
                font=self._bus.font("micro", mono=True),
            )

    def _paint_message(self, width: int, height: int) -> None:
        if not self._message:
            return
        self.create_rectangle(
            0,
            height - 18,
            min(width, 8 + 7 * len(self._message)),
            height,
            fill=self._theme.panel,
            outline="",
        )
        self.create_text(
            4,
            height - 9,
            anchor="w",
            text=self._message,
            fill=self._theme.text,
            font=self._bus.font("micro", mono=True),
        )


class AgentView(tk.Canvas):
    """The agent's own screen, drawn from the numbers it was given.

    The Stats page can list all 126 values the policy receives and still
    not answer "what did it see". This is that answer in one picture: the
    three contacts where they fall in the agent's field of view - left of
    centre, dead ahead, off to the side - each drawn as the box the engine
    reported and lit by how readable it was.

    It is deliberately *not* a map. The coordinates are the agent's view,
    so x=0 is under the crosshair whatever the agent happens to be facing,
    and a contact behind it has no box and is not drawn at all.
    """

    #: Height of the note strip at the bottom of the canvas.
    NOTE_HEIGHT = 16

    def __init__(
        self,
        parent: tk.Misc,
        *,
        bus: ThemeBus | None = None,
        height: int = 200,
    ) -> None:
        self._bus = bus or _DEFAULT_BUS
        self._cc_bus = self._bus
        self._theme = self._bus.theme
        self._base_height = height
        effective = (
            max(96, int(round(height * self._bus.scale.viewport_scale)))
            if self._bus.scale.viewport_scale < 1.0
            else height
        )
        super().__init__(
            parent,
            height=effective,
            background=self._theme.panel,
            highlightthickness=1,
            highlightbackground=self._theme.border,
        )
        self._contacts: tuple[dict[str, Any], ...] = ()
        self._reticle = False
        self._note = "no observation loaded"
        self.bind("<Configure>", lambda _event: self._redraw())
        self._unsubscribe = self._bus.subscribe(self.apply_theme, owner=self)

    def set_contacts(
        self,
        contacts: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        *,
        reticle: bool = False,
        note: str = "",
    ) -> None:
        self._contacts = tuple(contacts)
        self._reticle = bool(reticle)
        self._note = note
        self._redraw()

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        with contextlib.suppress(tk.TclError):
            effective = (
                max(96, int(round(self._base_height * self._bus.scale.viewport_scale)))
                if self._bus.scale.viewport_scale < 1.0
                else self._base_height
            )
            self.configure(
                background=theme.panel,
                highlightbackground=theme.border,
                height=effective,
            )
        self._redraw()

    # -- painting ---------------------------------------------------------

    def _redraw(self) -> None:
        with contextlib.suppress(tk.TclError):
            self.delete("all")
            width = max(1, int(self.winfo_width()))
            height = max(1, int(self.winfo_height()))
            frame_height = max(1, height - self.NOTE_HEIGHT)
            # The view is the agent's cone, so it keeps the engine's aspect:
            # a wider card letterboxes instead of stretching every box.
            frame_width = min(width, int(round(frame_height * AGENT_VIEW_ASPECT)))
            left = max(0, (width - frame_width) // 2)
            self._paint_frame(left, 0, frame_width, frame_height)
            self._paint_contacts(left, 0, frame_width, frame_height)
            self._paint_note(width, height)

    def _paint_frame(self, left: int, top: int, frame_width: int, frame_height: int) -> None:
        self.create_rectangle(
            left,
            top,
            left + frame_width,
            top + frame_height,
            fill=self._theme.card,
            outline=self._theme.border,
        )
        centre_y = top + frame_height / 2.0
        self.create_line(
            left,
            centre_y,
            left + frame_width,
            centre_y,
            fill=self._theme.border,
            dash=(3, 3),
        )

    def _paint_contacts(self, left: int, top: int, frame_width: int, frame_height: int) -> None:
        centre_x = left + frame_width / 2.0
        centre_y = top + frame_height / 2.0
        unit_x = frame_width / 2.0
        unit_y = frame_height / 2.0
        # The crosshair: bright when it is on the closest contact, because
        # "the policy is aiming at it" is the question behind the box.
        reticle_colour = self._theme.accent if self._reticle else self._theme.text_muted
        for offset in (-6.0, 6.0):
            self.create_line(
                centre_x + offset,
                centre_y - 2,
                centre_x + offset,
                centre_y + 2,
                fill=reticle_colour,
            )
            self.create_line(
                centre_x - 2,
                centre_y + offset,
                centre_x + 2,
                centre_y + offset,
                fill=reticle_colour,
            )
        for contact in self._contacts:
            if not contact.get("on_screen"):
                continue
            half_w = max(1.5, float(contact.get("half_width", 0.0)) * unit_x)
            half_h = max(1.5, float(contact.get("half_height", 0.0)) * unit_y)
            x0 = centre_x + float(contact.get("screen_x", 0.0)) * unit_x - half_w
            x1 = centre_x + float(contact.get("screen_x", 0.0)) * unit_x + half_w
            y0 = centre_y - float(contact.get("screen_y", 0.0)) * unit_y - half_h
            y1 = centre_y - float(contact.get("screen_y", 0.0)) * unit_y + half_h
            visibility = _contact_visibility(contact)
            colour = lerp_color(self._theme.text_dim, self._theme.warn, visibility)
            self.create_rectangle(x0, y0, x1, y1, outline=colour, width=2)
            label = (
                f"#{int(contact.get('index', 0)) + 1} {contact.get('distance_text', '')}".strip()
            )
            self.create_text(
                x0,
                max(top + 8.0, y0 - 4.0),
                anchor="sw",
                text=label,
                fill=colour,
                font=self._bus.font("micro", mono=True),
            )

    def _paint_note(self, width: int, height: int) -> None:
        if not self._note:
            return
        self.create_text(
            4,
            height - self.NOTE_HEIGHT / 2.0,
            anchor="w",
            text=self._note,
            fill=self._theme.text_dim,
            font=self._bus.font("micro", mono=True),
        )


def _contact_visibility(contact: Mapping[str, Any]) -> float:
    """How readable a contact was, in ``[0, 1]``, from what the vector says.

    Clarity when the engine reported one (it folds the light on the target
    and the distance into a single number), otherwise the product of how
    much of the body was exposed and how brightly it was lit. A remembered
    contact has neither, and is drawn as the dim box that it is.
    """
    clarity = contact.get("clarity")
    if isinstance(clarity, (int, float)):
        return max(0.0, min(1.0, float(clarity)))
    exposure = float(contact.get("exposure", 0.0) or 0.0)
    illumination = float(contact.get("illumination", 0.0) or 0.0)
    return max(0.0, min(1.0, exposure * illumination))

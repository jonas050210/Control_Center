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
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from . import control_center_viewmodel as vm

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

# Shared desktop Control Center palette: Cyber-HUD tactical telemetry theme.
_FONT_FAMILY = "Segoe UI"
COLOR_BG = "#030711"
COLOR_SURFACE = "#081121"
COLOR_SURFACE_RAISED = "#0d1b33"
COLOR_HOVER = "#14294a"
COLOR_TEXT = "#ecf6ff"
COLOR_ACCENT = "#00f0ff"
COLOR_ACCENT_SECONDARY = "#8b5cf6"
COLOR_OK = "#00ff9d"
COLOR_WARN = "#ffb800"
COLOR_ERROR = "#ff2e63"
COLOR_MUTED = "#7c96b8"
COLOR_BORDER = "#1b355a"
COLOR_GRID = "#10223d"


class ToolTip:
    """Small, delayed keyboard/mouse help bubble for otherwise terse controls."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
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
        self._after_id = self.widget.after(450, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window is not None or not self.text or not self.widget.winfo_exists():
            return
        self._after_id = None
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(
            f"+{self.widget.winfo_rootx() + 14}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 8}"
        )
        tk.Label(
            tip,
            text=self.text,
            justify="left",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            relief="solid",
            borderwidth=1,
            padx=9,
            pady=6,
            font=(_FONT_FAMILY, 9),
            wraplength=320,
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
    """One labelled value in a Dashboard/Benchmark/System HUD telemetry grid."""

    def __init__(self, parent: tk.Misc, label: str) -> None:
        super().__init__(parent, style="Card.TFrame", padding=(0, 0))
        self._accent_bar = tk.Frame(self, height=2, background=COLOR_ACCENT, borderwidth=0)
        self._accent_bar.pack(fill="x", side="top")
        body = ttk.Frame(self, style="CardInner.TFrame", padding=(12, 8))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=label.upper(), style="CardLabel.TLabel").pack(anchor="w")
        self._value = ttk.Label(body, text="n/a", style="CardValue.TLabel")
        self._value.pack(anchor="w", pady=(2, 0))

    def set(self, text: str, color: str | None = None) -> None:
        self._value.configure(text=text, foreground=color or COLOR_TEXT)
        self._accent_bar.configure(background=color or COLOR_ACCENT)


class StatRow(ttk.Frame):
    """A grid of :class:`StatCard` modules built from an ordered label tuple."""

    def __init__(
        self, parent: tk.Misc, labels: tuple[str, ...], *, max_columns: int = 5
    ) -> None:
        super().__init__(parent)
        self._cards: dict[str, StatCard] = {}
        columns = min(max_columns, max(1, len(labels)))
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


class PhaseStepper(tk.Canvas):
    """Futuristic horizontal HUD pipeline stepper + integrated progress bar."""

    _STATUS_PALETTE: dict[str, tuple[str, str, str]] = {
        "pending": (COLOR_SURFACE, COLOR_BORDER, COLOR_MUTED),
        "active": ("#062c43", COLOR_ACCENT, COLOR_ACCENT),
        "done": ("#052e24", COLOR_OK, COLOR_OK),
        "skipped": ("#2b2108", COLOR_WARN, COLOR_WARN),
        "failed": ("#360b1b", COLOR_ERROR, COLOR_ERROR),
    }

    def __init__(self, parent: tk.Misc, height: int = 54) -> None:
        super().__init__(
            parent,
            height=height,
            background=COLOR_SURFACE,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        self._phases: list[dict[str, str]] = []
        self._fraction: float = 0.0
        self.bind("<Configure>", lambda _event: self._redraw())

    def set_state(self, phases: list[dict[str, str]], fraction: float = 0.0) -> None:
        self._phases = list(phases)
        self._fraction = max(0.0, min(1.0, float(fraction)))
        self._redraw()

    def set_phases(self, phases: list[dict[str, str]], fraction: float = 0.0) -> None:
        self.set_state(phases, fraction)

    def _redraw(self) -> None:
        self.delete("all")
        width = max(int(self.winfo_width()), 1)
        height = max(int(self.winfo_height()), 1)
        if not self._phases:
            return
        count = len(self._phases)
        pad_x = 12
        gap = 10
        box_h = 30
        box_y0 = 8
        box_y1 = box_y0 + box_h
        slot_w = max(40.0, (width - 2 * pad_x - (count - 1) * gap) / count)
        for idx, phase in enumerate(self._phases):
            status = phase.get("status", "pending")
            bg, border, fg = self._STATUS_PALETTE.get(status, self._STATUS_PALETTE["pending"])
            x0 = pad_x + idx * (slot_w + gap)
            x1 = x0 + slot_w
            if idx < count - 1:
                conn_color = COLOR_OK if status == "done" else COLOR_BORDER
                self.create_line(
                    x1, (box_y0 + box_y1) / 2, x1 + gap, (box_y0 + box_y1) / 2, fill=conn_color, width=2
                )
            self.create_rectangle(x0, box_y0, x1, box_y1, fill=bg, outline=border, width=1)
            self.create_rectangle(x0, box_y0, x0 + 4, box_y1, fill=border, outline="")
            prefix = f"0{idx + 1}"
            label = str(phase.get("label", "")).upper()
            self.create_text(
                x0 + 10,
                (box_y0 + box_y1) / 2,
                anchor="w",
                text=f"{prefix} // {label}",
                fill=fg,
                font=(_FONT_FAMILY, 8, "bold"),
            )
        # Integrated bottom progress bar
        bar_y0 = height - 8
        bar_y1 = height - 3
        self.create_rectangle(
            pad_x, bar_y0, width - pad_x, bar_y1, fill=COLOR_BG, outline=COLOR_BORDER
        )
        if self._fraction > 0.0:
            fill_w = (width - 2 * pad_x) * self._fraction
            bar_color = COLOR_OK if self._fraction >= 0.999 else COLOR_ACCENT
            self.create_rectangle(
                pad_x, bar_y0, pad_x + fill_w, bar_y1, fill=bar_color, outline=""
            )


class LineChart(tk.Canvas):
    """A dependency-free bounded HUD oscilloscope chart with interactive crosshair.

    Draws one series of ``(x, y)`` points already bounded by the adapter
    (``deque(maxlen=...)``) and again decimated to the canvas width by
    :func:`control_center_viewmodel.downsample_series`, so render cost never
    grows with run length. Moving the pointer over the chart snaps a
    crosshair to the nearest measured point and displays its exact ``(x, y)``
    coordinates in the header.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        height: int = 148,
        *,
        color: str = COLOR_ACCENT,
    ) -> None:
        super().__init__(
            parent,
            height=height,
            background=COLOR_SURFACE_RAISED,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        self._title = title
        self._color = color
        self._points: list[tuple[float, float]] = []
        self._hover_x: int | None = None
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = points
        self._redraw()

    def _on_motion(self, event: tk.Event[Any]) -> None:
        self._hover_x = int(getattr(event, "x", 0))
        if len(self._points) >= 2:
            self._redraw()

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        if self._hover_x is not None:
            self._hover_x = None
            self._redraw()

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
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom
        for v_frac in (0.25, 0.5, 0.75):
            gx = pad_left + v_frac * plot_w
            self.create_line(gx, pad_top, gx, height - pad_bottom, fill=COLOR_GRID)
        for fraction in (0.0, 0.5, 1.0):
            gy = pad_top + fraction * plot_h
            self.create_line(pad_left, gy, width - pad_right, gy, fill=COLOR_BORDER)
            value = y_max - fraction * (y_max - y_min)
            self.create_text(
                pad_left - 6,
                gy,
                anchor="e",
                text=vm.format_number(value, 2),
                fill=COLOR_MUTED,
                font=("Consolas", 8),
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
        best_idx = min(
            range(len(points)),
            key=lambda i: abs(coords[2 * i] - hover_x),
        )
        hx, hy = coords[2 * best_idx], coords[2 * best_idx + 1]
        px, py = points[best_idx]
        self.create_line(hx, pad_top, hx, height - pad_bottom, fill=COLOR_ACCENT, dash=(2, 2))
        self.create_line(pad_left, hy, width - pad_right, hy, fill=COLOR_BORDER, dash=(2, 2))
        self.create_oval(
            hx - 4, hy - 4, hx + 4, hy + 4, fill=COLOR_WARN, outline=COLOR_TEXT, width=1
        )
        return f"CURSOR [{vm.format_number(px, 0)} ▸ {vm.format_number(py, 3)}]  │  "

    def _redraw(self) -> None:
        self.delete("all")
        width = max(int(self.winfo_width()), 1)
        height = max(int(self.winfo_height()), 1)
        pad_left, pad_right, pad_top, pad_bottom = 52, 12, 24, 18
        self.create_rectangle(0, 0, width, 20, fill=COLOR_SURFACE, outline="")
        self.create_text(
            8,
            4,
            anchor="nw",
            text=f"// {self._title.upper()}",
            font=(_FONT_FAMILY, 8, "bold"),
            fill=self._color,
        )
        points = vm.downsample_series(
            self._points, max_points=max(width - pad_left - pad_right, 10)
        )
        if len(points) < 2:
            self.create_text(
                width / 2,
                height / 2 + 6,
                text="AWAITING LIVE TELEMETRY STREAM...",
                fill=COLOR_MUTED,
                font=("Consolas", 9),
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
        self.create_polygon(*poly_coords, fill="#07364d", stipple="gray25", outline="")
        self.create_line(*coords, fill="#0a5c7d", width=4, smooth=False)
        self.create_line(*coords, fill=self._color, width=2, smooth=False)
        last_x, last_y = coords[-2], coords[-1]
        self.create_oval(
            last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill=COLOR_OK, outline=COLOR_BG
        )
        cursor_prefix = (
            self._draw_hover(points, coords, width, height, pad_left, pad_right, pad_top, pad_bottom)
            or ""
        )
        summary = (
            f"{cursor_prefix}MIN {vm.format_number(raw_y_min, 2)}  │  "
            f"MAX {vm.format_number(raw_y_max, 2)}  │  "
            f"NOW {vm.format_number(ys[-1], 3)}"
        )
        self.create_text(
            width - pad_right,
            4,
            anchor="ne",
            text=summary,
            fill=COLOR_WARN if cursor_prefix else COLOR_TEXT,
            font=("Consolas", 8, "bold"),
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

    def __init__(self, parent: tk.Misc, max_lines: int = 4000) -> None:
        super().__init__(parent)
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
        ttk.Button(toolbar, text="Clear view", command=self.clear).pack(side="left", padx=(8, 0))
        self._truncated_label = ttk.Label(toolbar, text="", foreground=COLOR_WARN)
        self._truncated_label.pack(side="right")
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.text = tk.Text(
            text_frame,
            height=16,
            wrap="none",
            state="disabled",
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            font=("Consolas", 9),
        )
        # Process output routinely contains wide commands, paths and tracebacks.
        # With ``wrap=\"none\"`` a horizontal scrollbar is therefore a
        # readability requirement, not a decorative extra. Grid lets both
        # native scrollbars share the same data surface without clipping each
        # other on compact windows.
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self._yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self._scroll_text_y)
        self._xscroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=self._on_text_scroll, xscrollcommand=self._xscroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        self._yscroll.grid(row=0, column=1, sticky="ns")
        self._xscroll.grid(row=1, column=0, sticky="ew")
        self.text.tag_configure("stderr", foreground=COLOR_ERROR)
        self.text.tag_configure("meta", foreground=COLOR_MUTED)
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
    parent: tk.Misc, columns: tuple[tuple[str, str, int], ...], *, expand: bool = True
) -> ttk.Treeview:
    """Build a sortable table with both axes reachable on a narrow window.

    Result inventories have deliberately descriptive columns (run location,
    timestamp, error, checkpoint kind). Shrinking them to fit a fixed window
    turns their values into ellipses, while a vertical-only scrollbar made the
    rightmost columns unreachable. The wrapper keeps a conventional native
    table and makes its full width explicitly reachable instead.
    """
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=expand)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    tree = _sortable_table(frame, columns)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    xscroll.grid(row=1, column=0, sticky="ew")
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    # The attributes are intentionally private presentation seams: desktop
    # tests verify that every dense inventory preserves horizontal reachability
    # without teaching pages Tk grid details.
    tree._horizontal_scrollbar = xscroll  # type: ignore[attr-defined]
    tree._vertical_scrollbar = yscroll  # type: ignore[attr-defined]
    return tree


def _sortable_table(parent: tk.Misc, columns: tuple[tuple[str, str, int], ...]) -> ttk.Treeview:
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
    for key, title, width in columns:
        anchor = "e" if key in numeric_columns else "w"
        # typeshed types anchor as a literal enum; "e"/"w" are valid tk
        # anchors and are what the rest of this module already uses.
        tree.heading(  # type: ignore[call-overload]
            key, text=title, anchor=anchor, command=lambda k=key: _sort_tree(tree, k, False)
        )
        tree.column(key, width=width, anchor=anchor, stretch=True)  # type: ignore[call-overload]
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

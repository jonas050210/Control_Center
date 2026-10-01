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
    ) -> None:
        if self._closed:
            return

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
        if not self._closed:
            self._root.after(self._poll_ms, self._pump)

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

# Shared desktop Control Center palette. This is an operator-facing local
# calibration surface, not a reconstruction of a TTK Testing player HUD.
_FONT_FAMILY = "Segoe UI"
COLOR_BG = "#050a16"
COLOR_SURFACE = "#0b1427"
COLOR_SURFACE_RAISED = "#101d32"
COLOR_HOVER = "#162943"
COLOR_TEXT = "#e7f1ff"
COLOR_ACCENT = "#35d7ff"
COLOR_ACCENT_SECONDARY = "#9c8cff"
COLOR_OK = "#4ee6a1"
COLOR_WARN = "#ffc861"
COLOR_ERROR = "#ff718d"
COLOR_MUTED = "#91a7bf"
COLOR_BORDER = "#29445f"


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
        super().__init__(
            parent,
            height=height,
            background=COLOR_SURFACE_RAISED,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
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
        self.create_text(
            8, 6, anchor="nw", text=self._title, font=(_FONT_FAMILY, 9, "bold"), fill=COLOR_MUTED
        )
        points = vm.downsample_series(
            self._points, max_points=max(width - pad_left - pad_right, 10)
        )
        if len(points) < 2:
            self.create_text(
                width / 2,
                height / 2,
                text="not enough data yet",
                fill=COLOR_MUTED,
                font=(_FONT_FAMILY, 9),
            )
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
            self.create_line(pad_left, gy, width - pad_right, gy, fill=COLOR_BORDER)
            value = y_max - fraction * (y_max - y_min)
            self.create_text(
                pad_left - 6,
                gy,
                anchor="e",
                text=vm.format_number(value, 2),
                fill=COLOR_MUTED,
                font=(_FONT_FAMILY, 8),
            )
        coords: list[float] = []
        for x, y in points:
            cx, cy = to_canvas(x, y)
            coords.extend((cx, cy))
        self.create_line(*coords, fill=COLOR_ACCENT, width=2, smooth=False)
        self.create_text(
            width - pad_right,
            height - 4,
            anchor="se",
            text=f"latest: {vm.format_number(ys[-1], 3)}",
            fill=COLOR_MUTED,
            font=(_FONT_FAMILY, 8),
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
        self._yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self._on_text_scroll)
        self.text.pack(side="left", fill="both", expand=True)
        self._yscroll.pack(side="right", fill="y")
        self.text.tag_configure("stderr", foreground=COLOR_ERROR)
        self.text.tag_configure("meta", foreground=COLOR_MUTED)
        self.text.bind("<MouseWheel>", self._on_manual_scroll)
        self.text.bind("<Button-4>", self._on_manual_scroll)
        self.text.bind("<Button-5>", self._on_manual_scroll)

    def _on_manual_scroll(self, _event: object) -> None:
        self._user_scrolled_up = True

    def _on_text_scroll(self, first: str, last: str) -> None:
        """Synchronize the native scrollbar and restore follow-at-bottom.

        A manual wheel event must pause live-follow while an operator reads
        historical output, but reaching the newest line again should make
        the already-enabled Auto-scroll checkbox useful immediately.
        """
        self._yscroll.set(first, last)
        try:
            if float(last) >= 0.999:
                self._user_scrolled_up = False
        except ValueError:  # pragma: no cover - Tk always sends float strings
            pass

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


def _sortable_table(parent: tk.Misc, columns: tuple[tuple[str, str, int], ...]) -> ttk.Treeview:
    """Builds a Treeview with click-to-sort columns (ascending/descending)."""
    tree = ttk.Treeview(
        parent, columns=tuple(c[0] for c in columns), show="headings", selectmode="extended"
    )
    numeric_columns = {
        "environments",
        "workers",
        "total_steps",
        "steps_per_second",
        "episodes_per_second",
        "p50_ms",
        "p95_ms",
        "elapsed_seconds",
        "timesteps",
        "episodes",
        "win_rate",
        "loss_rate",
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

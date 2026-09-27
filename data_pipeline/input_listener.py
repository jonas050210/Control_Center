"""Keyboard and Mouse input listener for SandboxAI (M1).

Captures OS-level input events non-intrusively, records microsecond-accurate
monotonic timestamps, and buffers them for synchronization with video frames.
"""

from __future__ import annotations

import collections
import dataclasses
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


@dataclasses.dataclass
class RawInputEvent:
    """An asynchronous input event with high-precision monotonic timestamp."""

    t: float  # time.perf_counter() timestamp
    event_type: str  # "key_down", "key_up", "mouse_move", "mouse_click", "mouse_scroll"
    data: Dict[str, Any] = dataclasses.field(default_factory=dict)


class InputListener:
    """Thread-safe input listener recording keyboard and mouse events."""

    def __init__(self, max_buffer_size: int = 50000) -> None:
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be positive")
        self.max_buffer_size = int(max_buffer_size)
        self._events: Deque[RawInputEvent] = collections.deque(maxlen=self.max_buffer_size)
        self._dropped_events = 0
        self._lock = threading.Lock()
        self._running = False

        self._keyboard_listener: Optional[Any] = None
        self._mouse_listener: Optional[Any] = None

        # Current instantaneous state
        self._currently_pressed_keys: Set[str] = set()
        self._current_mouse_buttons: Dict[str, bool] = {
            "left": False,
            "right": False,
            "middle": False,
        }
        self._last_mouse_pos: Optional[Tuple[int, int]] = None

    def start(self) -> None:
        """Starts background keyboard and mouse listeners."""
        if not PYNPUT_AVAILABLE:
            raise RuntimeError(
                "pynput is not installed or available. Cannot start real input listener."
            )

        with self._lock:
            if self._running:
                return
            self._running = True
            self._events.clear()
            self._dropped_events = 0
            self._currently_pressed_keys.clear()
            self._current_mouse_buttons = {"left": False, "right": False, "middle": False}
            self._last_mouse_pos = None

        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
        )

        self._keyboard_listener.start()
        self._mouse_listener.start()

    def stop(self) -> None:
        """Stops background listeners."""
        self._running = False
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None

        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None

    def _append_event(self, event: RawInputEvent) -> None:
        """Append an event while making buffer overflow observable."""
        if len(self._events) >= self.max_buffer_size:
            self._dropped_events += 1
        self._events.append(event)

    def _normalize_key(self, key: Any) -> str:
        """Normalizes pynput Key / KeyCode into lowercase canonical string."""
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        elif hasattr(key, "name") and key.name is not None:
            return key.name.lower()
        else:
            k_str = str(key).lower().replace("key.", "").strip("'")
            return k_str

    def _on_key_press(self, key: Any) -> None:
        t = time.perf_counter()
        k_str = self._normalize_key(key)
        with self._lock:
            self._currently_pressed_keys.add(k_str)
            self._append_event(
                RawInputEvent(
                    t=t,
                    event_type="key_down",
                    data={"key": k_str},
                )
            )

    def _on_key_release(self, key: Any) -> None:
        t = time.perf_counter()
        k_str = self._normalize_key(key)
        with self._lock:
            self._currently_pressed_keys.discard(k_str)
            self._append_event(
                RawInputEvent(
                    t=t,
                    event_type="key_up",
                    data={"key": k_str},
                )
            )

    def _on_mouse_move(self, x: int, y: int) -> None:
        t = time.perf_counter()
        with self._lock:
            if self._last_mouse_pos is not None:
                dx = x - self._last_mouse_pos[0]
                dy = y - self._last_mouse_pos[1]
            else:
                dx = 0
                dy = 0
            self._last_mouse_pos = (x, y)

            self._append_event(
                RawInputEvent(
                    t=t,
                    event_type="mouse_move",
                    data={"x": x, "y": y, "dx": dx, "dy": dy},
                )
            )

    def _on_mouse_click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        t = time.perf_counter()
        btn_str = "left"
        if hasattr(button, "name"):
            btn_str = button.name.lower()
        elif "left" in str(button).lower():
            btn_str = "left"
        elif "right" in str(button).lower():
            btn_str = "right"
        elif "middle" in str(button).lower():
            btn_str = "middle"

        with self._lock:
            self._current_mouse_buttons[btn_str] = pressed
            self._append_event(
                RawInputEvent(
                    t=t,
                    event_type="mouse_click",
                    data={"button": btn_str, "pressed": pressed, "x": x, "y": y},
                )
            )

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        t = time.perf_counter()
        with self._lock:
            self._append_event(
                RawInputEvent(
                    t=t,
                    event_type="mouse_scroll",
                    data={"dx": dx, "dy": dy, "x": x, "y": y},
                )
            )

    def push_event(self, event: RawInputEvent) -> None:
        """Manually push an event (useful for mocking and testing)."""
        with self._lock:
            self._append_event(event)
            if event.event_type == "key_down":
                self._currently_pressed_keys.add(event.data.get("key", ""))
            elif event.event_type == "key_up":
                self._currently_pressed_keys.discard(event.data.get("key", ""))
            elif event.event_type == "mouse_click":
                btn = event.data.get("button", "left")
                self._current_mouse_buttons[btn] = bool(event.data.get("pressed", False))

    def drain_events(self) -> List[RawInputEvent]:
        """Atomically drains and returns all buffered events."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def get_dropped_event_count(self) -> int:
        """Return the number of events discarded because the buffer was full."""
        with self._lock:
            return int(self._dropped_events)

    def get_instantaneous_state(self) -> Tuple[Set[str], Dict[str, bool]]:
        """Returns snapshot of current pressed keys and mouse button states."""
        with self._lock:
            return set(self._currently_pressed_keys), dict(self._current_mouse_buttons)

    def __enter__(self) -> InputListener:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

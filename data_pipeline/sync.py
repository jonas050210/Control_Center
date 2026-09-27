"""Lossless timestamp alignment of asynchronous input events and video frames."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from data_pipeline.actions import MouseBinner, build_action_state
from data_pipeline.input_listener import RawInputEvent
from data_pipeline.schema import DatasetSample, MouseConfig


class ActionSynchronizer:
    """Integrate inputs over frame intervals while preserving taps and future events."""

    def __init__(
        self,
        binner_x: Optional[MouseBinner] = None,
        binner_y: Optional[MouseBinner] = None,
        mouse_config: Optional[MouseConfig] = None,
    ) -> None:
        def build(axis: str, supplied: Optional[MouseBinner]) -> MouseBinner:
            if supplied is not None:
                return supplied
            if mouse_config is not None:
                count = mouse_config.num_bins_x if axis == "x" else mouse_config.num_bins_y
                edges = mouse_config.bin_edges_x if axis == "x" else mouse_config.bin_edges_y
                return MouseBinner(
                    num_bins=count,
                    strategy=mouse_config.binning_strategy,
                    custom_edges=edges or None,
                )
            return MouseBinner()

        self.binner_x = build("x", binner_x)
        self.binner_y = build("y", binner_y)
        self._held_keys: Set[str] = set()
        self._mouse_buttons: Dict[str, bool] = {
            "left": False,
            "right": False,
            "middle": False,
        }
        self._pending_events: List[RawInputEvent] = []

    def reset(self) -> None:
        self._held_keys.clear()
        self._mouse_buttons = {"left": False, "right": False, "middle": False}
        self._pending_events.clear()

    def _apply_state_event(self, event: RawInputEvent) -> None:
        if event.event_type == "key_down":
            self._held_keys.add(str(event.data.get("key", "")).lower())
        elif event.event_type == "key_up":
            self._held_keys.discard(str(event.data.get("key", "")).lower())
        elif event.event_type == "mouse_click":
            button = str(event.data.get("button", "left"))
            self._mouse_buttons[button] = bool(event.data.get("pressed", False))

    def process_interval(
        self,
        events: Sequence[RawInputEvent],
        t_prev: float,
        t_curr: float,
    ) -> Tuple[Set[str], Dict[str, bool], float, float, int]:
        if t_curr < t_prev:
            raise ValueError(f"Frame interval is backwards: {t_prev} -> {t_curr}")
        combined = sorted([*self._pending_events, *events], key=lambda event: event.t)
        self._pending_events = []

        # Bring persistent state up to the start boundary before taking the
        # interval snapshot. This matters when listeners are started slightly
        # before the first captured frame.
        index = 0
        while index < len(combined) and combined[index].t <= t_prev:
            self._apply_state_event(combined[index])
            index += 1
        window_keys = set(self._held_keys)
        window_buttons = dict(self._mouse_buttons)
        dx = dy = 0.0
        wheel = 0

        for event in combined[index:]:
            if event.t > t_curr:
                self._pending_events.append(event)
                continue
            if event.event_type == "key_down":
                key = str(event.data.get("key", "")).lower()
                self._held_keys.add(key)
                window_keys.add(key)
            elif event.event_type == "key_up":
                key = str(event.data.get("key", "")).lower()
                # Keep key in window_keys: a press/release tap during one frame
                # must remain a positive label for that frame.
                self._held_keys.discard(key)
            elif event.event_type == "mouse_move":
                dx += float(event.data.get("dx", 0.0))
                dy += float(event.data.get("dy", 0.0))
            elif event.event_type == "mouse_click":
                button = str(event.data.get("button", "left"))
                pressed = bool(event.data.get("pressed", False))
                self._mouse_buttons[button] = pressed
                if pressed:
                    window_buttons[button] = True
            elif event.event_type == "mouse_scroll":
                wheel += int(event.data.get("dy", 0))
        return window_keys, window_buttons, dx, dy, wheel

    def create_sample(
        self,
        step_idx: int,
        t_curr: float,
        t_prev: float,
        frame_file: str,
        events: Sequence[RawInputEvent],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetSample:
        keys, buttons, dx, dy, wheel = self.process_interval(events, t_prev, t_curr)
        action = build_action_state(
            keys,
            buttons,
            dx,
            dy,
            self.binner_x,
            self.binner_y,
            wheel,
        )
        return DatasetSample(
            step_idx=int(step_idx),
            timestamp=float(t_curr),
            iso_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            dt=max(0.0, t_curr - t_prev) if step_idx > 0 else 0.0,
            frame_file=frame_file,
            actions=action,
            is_valid=True,
            metadata=metadata or {},
        )

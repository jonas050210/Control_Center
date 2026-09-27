"""Timestamp synchronization and alignment engine for SandboxAI (M1).

Aligns continuous asynchronous keyboard/mouse input events with discrete
video frame capture timestamps to produce consistent (observation, action) pairs.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from data_pipeline.actions import MouseBinner, build_action_state
from data_pipeline.input_listener import RawInputEvent
from data_pipeline.schema import ActionState, DatasetSample, MouseConfig


class ActionSynchronizer:
    """Synchronizes input events over frame time intervals [t_prev, t_curr]."""

    def __init__(
        self,
        binner_x: Optional[MouseBinner] = None,
        binner_y: Optional[MouseBinner] = None,
        mouse_config: Optional[MouseConfig] = None,
    ) -> None:
        if binner_x is not None:
            self.binner_x = binner_x
        elif mouse_config is not None:
            self.binner_x = MouseBinner(
                num_bins=mouse_config.num_bins_x,
                strategy=mouse_config.binning_strategy,
                custom_edges=mouse_config.bin_edges_x if mouse_config.bin_edges_x else None,
            )
        else:
            self.binner_x = MouseBinner(num_bins=21, strategy="symmetric_log")

        if binner_y is not None:
            self.binner_y = binner_y
        elif mouse_config is not None:
            self.binner_y = MouseBinner(
                num_bins=mouse_config.num_bins_y,
                strategy=mouse_config.binning_strategy,
                custom_edges=mouse_config.bin_edges_y if mouse_config.bin_edges_y else None,
            )
        else:
            self.binner_y = MouseBinner(num_bins=21, strategy="symmetric_log")

        self._held_keys: Set[str] = set()
        self._mouse_buttons: Dict[str, bool] = {"left": False, "right": False, "middle": False}

    def process_interval(
        self,
        events: Sequence[RawInputEvent],
        t_prev: float,
        t_curr: float,
    ) -> Tuple[Set[str], Dict[str, bool], float, float, int]:
        """Integrates all events strictly within (t_prev, t_curr].

        Returns:
            (active_keys, active_buttons, accumulated_dx, accumulated_dy, accumulated_wheel)
        """
        # Active keys during interval: includes previously held keys plus any pressed in window
        window_keys = set(self._held_keys)
        window_buttons = dict(self._mouse_buttons)

        acc_dx = 0.0
        acc_dy = 0.0
        acc_wheel = 0

        # Sort events by monotonic timestamp just in case
        sorted_events = sorted(events, key=lambda e: e.t)

        for event in sorted_events:
            # Only process events up to t_curr
            if event.t <= t_prev:
                # Update persistent state if not already recorded
                if event.event_type == "key_down":
                    self._held_keys.add(event.data.get("key", "").lower())
                elif event.event_type == "key_up":
                    self._held_keys.discard(event.data.get("key", "").lower())
                elif event.event_type == "mouse_click":
                    btn = event.data.get("button", "left")
                    self._mouse_buttons[btn] = bool(event.data.get("pressed", False))
                continue

            if event.t > t_curr:
                # Future event for next interval; do not process yet
                break

            if event.event_type == "key_down":
                k = event.data.get("key", "").lower()
                self._held_keys.add(k)
                window_keys.add(k)
            elif event.event_type == "key_up":
                k = event.data.get("key", "").lower()
                self._held_keys.discard(k)
                # Note: window_keys keeps k marked if it was active during this frame interval
            elif event.event_type == "mouse_move":
                acc_dx += float(event.data.get("dx", 0.0))
                acc_dy += float(event.data.get("dy", 0.0))
            elif event.event_type == "mouse_click":
                btn = event.data.get("button", "left")
                is_pressed = bool(event.data.get("pressed", False))
                self._mouse_buttons[btn] = is_pressed
                if is_pressed:
                    window_buttons[btn] = True
            elif event.event_type == "mouse_scroll":
                acc_wheel += int(event.data.get("dy", 0))

        return window_keys, window_buttons, acc_dx, acc_dy, acc_wheel

    def create_sample(
        self,
        step_idx: int,
        t_curr: float,
        t_prev: float,
        frame_file: str,
        events: Sequence[RawInputEvent],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetSample:
        """Constructs a synchronized DatasetSample for step_idx at timestamp t_curr."""
        dt = max(0.0, t_curr - t_prev) if step_idx > 0 else 0.0
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        keys, buttons, dx, dy, wheel = self.process_interval(events, t_prev, t_curr)

        actions = build_action_state(
            active_keys=keys,
            mouse_buttons=buttons,
            mouse_dx=dx,
            mouse_dy=dy,
            binner_x=self.binner_x,
            binner_y=self.binner_y,
            wheel_dy=wheel,
        )

        return DatasetSample(
            step_idx=step_idx,
            timestamp=t_curr,
            iso_timestamp=iso_now,
            dt=dt,
            frame_file=frame_file,
            actions=actions,
            is_valid=True,
            metadata=metadata or {},
        )

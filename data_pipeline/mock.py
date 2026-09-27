"""Synthetic and Mock Recording Generator for SandboxAI (M1).

Allows full end-to-end testing of the capture, synchronization, serialization,
validation, and inspection pipeline without requiring Roblox, Godot, or an active display.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data_pipeline.capture import CaptureSource
from data_pipeline.input_listener import RawInputEvent


class MockScreenCapture(CaptureSource):
    """Generates synthetic 3D-like FPS frames for testing and development."""

    def __init__(
        self,
        target_width: int = 160,
        target_height: int = 120,
        seed: Optional[int] = 42,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self._rng = random.Random(seed)
        self._step = 0
        self._player_x = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._firing = False
        self._opened = False

    def start(self) -> None:
        self._opened = True
        self._step = 0

    def close(self) -> None:
        self._opened = False

    def update_state(self, dx: float, dy: float, firing: bool = False) -> None:
        """Updates internal simulated view angles from mouse deltas."""
        self._yaw += dx * 0.005
        self._pitch = max(-1.0, min(1.0, self._pitch - dy * 0.005))
        self._firing = firing

    def grab_frame(self) -> Tuple[Image.Image, float]:
        """Renders a synthetic frame representing a first-person tactical view."""
        t_capture = time.perf_counter()
        self._step += 1

        w, h = self.target_width, self.target_height
        img = Image.new("RGB", (w, h), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        # 1. Horizon & Ground
        horizon_y = int(h / 2 + self._pitch * (h / 4))
        horizon_y = max(10, min(h - 10, horizon_y))

        # Sky
        draw.rectangle([0, 0, w, horizon_y], fill=(45, 55, 75))
        # Ground
        draw.rectangle([0, horizon_y, w, h], fill=(60, 65, 55))

        # 2. Grid lines / perspective floor
        for offset in range(-3, 4):
            x_top = int(w / 2 + (offset + self._yaw * 2.0) * (w / 8))
            x_bot = int(w / 2 + (offset + self._yaw * 2.0) * (w / 3))
            if 0 <= x_top <= w or 0 <= x_bot <= w:
                draw.line([(x_top, horizon_y), (x_bot, h)], fill=(75, 80, 70), width=1)

        # 3. Simulated enemy / target box
        target_phase = (self._step * 0.05) % (2 * math.pi)
        target_x = int(w / 2 + math.sin(target_phase) * (w / 3) + math.sin(self._yaw) * 20)
        target_y = int(horizon_y - 15)
        target_w = max(8, int(w / 12))
        target_h = max(12, int(h / 6))

        draw.rectangle(
            [target_x - target_w // 2, target_y - target_h, target_x + target_w // 2, target_y],
            fill=(180, 50, 50),
            outline=(220, 80, 80),
        )

        # 4. Aim Crosshair
        cx, cy = w // 2, h // 2
        ch_len = max(3, int(w / 40))
        ch_color = (0, 255, 0)
        draw.line([(cx - ch_len, cy), (cx - 1, cy)], fill=ch_color, width=1)
        draw.line([(cx + 1, cy), (cx + ch_len, cy)], fill=ch_color, width=1)
        draw.line([(cx, cy - ch_len), (cx, cy - 1)], fill=ch_color, width=1)
        draw.line([(cx, cy + 1), (cx, cy + ch_len)], fill=ch_color, width=1)

        # 5. Muzzle flash if firing
        if self._firing:
            flash_color = (255, 230, 100)
            draw.ellipse([cx - 4, cy + 10, cx + 4, cy + 18], fill=flash_color)

        return img, t_capture


class MockInputGenerator:
    """Generates synthetic human-like WASD movements, mouse aiming, and clicks."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        self._rng = random.Random(seed)
        self._held_keys: Set[str] = set()
        self._held_mouse_buttons: Set[str] = set()
        self._current_movement: str = "idle"
        self._move_steps_remaining: int = 0
        self._fire_steps_remaining: int = 0
        self._ads_steps_remaining: int = 0

    def generate_events_for_step(
        self,
        t_start: float,
        t_end: float,
        num_sub_events: int = 5,
    ) -> List[RawInputEvent]:
        """Emits a series of realistic input events within the time window [t_start, t_end]."""
        events: List[RawInputEvent] = []
        dt = t_end - t_start
        if dt <= 0:
            return events

        # 1. State machine for WASD movement
        if self._move_steps_remaining <= 0:
            # Release old movement keys
            for k in list(self._held_keys):
                if k in ("w", "a", "s", "d", "shift"):
                    self._held_keys.remove(k)
                    events.append(
                        RawInputEvent(
                            t=t_start + 0.05 * dt,
                            event_type="key_up",
                            data={"key": k},
                        )
                    )

            # Choose new movement mode
            r_move = self._rng.random()
            self._move_steps_remaining = self._rng.randint(3, 12)  # hold for 3-12 frames (~0.2 - 0.8s)

            if r_move < 0.40:
                # Advance forward (+ optional sprint)
                new_keys = ["w"]
                if self._rng.random() < 0.35:
                    new_keys.append("d")
                elif self._rng.random() < 0.35:
                    new_keys.append("a")
                if self._rng.random() < 0.40:
                    new_keys.append("shift")
            elif r_move < 0.65:
                # Strafe left or right
                new_keys = ["a"] if self._rng.random() < 0.5 else ["d"]
            elif r_move < 0.80:
                # Backward
                new_keys = ["s"]
            else:
                # Idle
                new_keys = []

            for k in new_keys:
                self._held_keys.add(k)
                events.append(
                    RawInputEvent(
                        t=t_start + self._rng.uniform(0.05 * dt, 0.2 * dt),
                        event_type="key_down",
                        data={"key": k},
                    )
                )
        else:
            self._move_steps_remaining -= 1

        # 2. Discrete action taps (Jump, Reload, Crouch)
        if "space" not in self._held_keys and self._rng.random() < 0.04:
            # Quick jump tap
            t_down = t_start + self._rng.uniform(0.1 * dt, 0.3 * dt)
            t_up = t_down + self._rng.uniform(0.2 * dt, 0.5 * dt)
            events.append(RawInputEvent(t=t_down, event_type="key_down", data={"key": "space"}))
            events.append(RawInputEvent(t=t_up, event_type="key_up", data={"key": "space"}))

        if "r" not in self._held_keys and self._rng.random() < 0.02:
            # Quick reload tap
            t_down = t_start + self._rng.uniform(0.1 * dt, 0.3 * dt)
            t_up = t_down + self._rng.uniform(0.2 * dt, 0.5 * dt)
            events.append(RawInputEvent(t=t_down, event_type="key_down", data={"key": "r"}))
            events.append(RawInputEvent(t=t_up, event_type="key_up", data={"key": "r"}))

        # 3. Mouse Aiming & Trajectories
        # Target tracking + micro noise
        flick = self._rng.random() < 0.15
        flick_scale = 15.0 if flick else 2.5
        total_dx = self._rng.gauss(0.0, flick_scale)
        total_dy = self._rng.gauss(0.0, flick_scale * 0.5)

        for i in range(num_sub_events):
            t_sub = t_start + (i + 1) * (dt / (num_sub_events + 1))
            sub_dx = total_dx / num_sub_events + self._rng.gauss(0.0, 0.5)
            sub_dy = total_dy / num_sub_events + self._rng.gauss(0.0, 0.3)
            events.append(
                RawInputEvent(
                    t=t_sub,
                    event_type="mouse_move",
                    data={"x": 100, "y": 100, "dx": sub_dx, "dy": sub_dy},
                )
            )

        # 4. Firing (LMB bursts)
        if self._fire_steps_remaining <= 0:
            if "left" in self._held_mouse_buttons:
                self._held_mouse_buttons.remove("left")
                events.append(
                    RawInputEvent(
                        t=t_start + 0.1 * dt,
                        event_type="mouse_click",
                        data={"button": "left", "pressed": False, "x": 100, "y": 100},
                    )
                )
            if self._rng.random() < 0.20:
                # Start firing burst for 2-5 frames
                self._fire_steps_remaining = self._rng.randint(2, 5)
                self._held_mouse_buttons.add("left")
                events.append(
                    RawInputEvent(
                        t=t_start + self._rng.uniform(0.1 * dt, 0.3 * dt),
                        event_type="mouse_click",
                        data={"button": "left", "pressed": True, "x": 100, "y": 100},
                    )
                )
        else:
            self._fire_steps_remaining -= 1

        # 5. ADS (RMB toggle/hold)
        if self._ads_steps_remaining <= 0:
            if "right" in self._held_mouse_buttons:
                self._held_mouse_buttons.remove("right")
                events.append(
                    RawInputEvent(
                        t=t_start + 0.1 * dt,
                        event_type="mouse_click",
                        data={"button": "right", "pressed": False, "x": 100, "y": 100},
                    )
                )
            if self._rng.random() < 0.15:
                self._ads_steps_remaining = self._rng.randint(4, 15)
                self._held_mouse_buttons.add("right")
                events.append(
                    RawInputEvent(
                        t=t_start + self._rng.uniform(0.1 * dt, 0.3 * dt),
                        event_type="mouse_click",
                        data={"button": "right", "pressed": True, "x": 100, "y": 100},
                    )
                )
        else:
            self._ads_steps_remaining -= 1

        return sorted(events, key=lambda e: e.t)

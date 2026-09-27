"""Gymnasium Environment Wrapper for SandboxAI Tactical FPS Arena (Phase 3/4/5).

Supports both real exported Godot 4.3 binary execution via godot_rl_agents
and an in-process simulated tactical arena for deterministic testing, evaluation, and CI.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw


class MockTacticalArenaEnv(gym.Env):
    """In-process Gymnasium tactical FPS sandbox environment with visual observations."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        width: int = 84,
        height: int = 84,
        max_steps: int = 500,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        # Observation space: 84x84 RGB pixel observation [3, H, W]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, height, width),
            dtype=np.uint8,
        )

        # Action space: [move_x (3), move_y (3), fire (2), turn_yaw (21), turn_pitch (21)]
        # move_x: 0=Left (-1), 1=None (0), 2=Right (+1)
        # move_y: 0=Back (-1), 1=None (0), 2=Forward (+1)
        # fire: 0=No, 1=Shoot
        # turn_yaw: 0..20 (bin 10 is 0.0)
        # turn_pitch: 0..20 (bin 10 is 0.0)
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 21, 21])

        # State variables
        self.player_x = 0.0
        self.player_z = 0.0
        self.player_yaw = 0.0  # Radians
        self.player_pitch = 0.0  # Radians
        self.targets: List[Dict[str, float]] = []  # List of {x, z, radius, health}
        self.step_count = 0
        self.targets_hit_count = 0
        self.last_shot_hit = False
        self.last_action_fired = False

        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self.step_count = 0
        self.targets_hit_count = 0
        self.last_shot_hit = False
        self.last_action_fired = False

        # Randomize player position near center
        p_angle = self._rng.uniform(0.0, 2 * math.pi)
        self.player_x = math.cos(p_angle) * 2.0
        self.player_z = math.sin(p_angle) * 2.0
        self.player_yaw = self._rng.uniform(-math.pi, math.pi)
        self.player_pitch = 0.0

        # Place 3 target dummies around the arena
        self.targets = []
        for _ in range(3):
            t_angle = self._rng.uniform(0.0, 2 * math.pi)
            t_dist = self._rng.uniform(6.0, 15.0)
            self.targets.append(
                {
                    "x": math.cos(t_angle) * t_dist,
                    "z": math.sin(t_angle) * t_dist,
                    "radius": 1.0,
                    "health": 100.0,
                }
            )

        obs = self._render_observation()
        info = {"targets_hit": 0, "step": 0}
        return obs, info

    def step(
        self,
        action: Union[np.ndarray, List[int], Dict[str, Any]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = -0.005  # Step penalty
        self.last_shot_hit = False

        # Parse action
        if isinstance(action, dict):
            move_x_idx = int(action.get("move_x", 1))
            move_y_idx = int(action.get("move_y", 1))
            fire = int(action.get("fire", 0))
            yaw_bin = int(action.get("turn_yaw", 10))
            pitch_bin = int(action.get("turn_pitch", 10))
        elif isinstance(action, (list, tuple, np.ndarray)):
            act_arr = np.asarray(action, dtype=np.int32).flatten()
            move_x_idx = int(act_arr[0]) if len(act_arr) > 0 else 1
            move_y_idx = int(act_arr[1]) if len(act_arr) > 1 else 1
            fire = int(act_arr[2]) if len(act_arr) > 2 else 0
            yaw_bin = int(act_arr[3]) if len(act_arr) > 3 else 10
            pitch_bin = int(act_arr[4]) if len(act_arr) > 4 else 10
        else:
            move_x_idx, move_y_idx, fire, yaw_bin, pitch_bin = 1, 1, 0, 10, 10

        self.last_action_fired = fire == 1

        # 1. Update Player Orientation (Yaw/Pitch)
        yaw_delta = (yaw_bin - 10) * 0.04
        pitch_delta = (pitch_bin - 10) * 0.02
        self.player_yaw = (self.player_yaw - yaw_delta) % (2 * math.pi)
        self.player_pitch = max(-1.2, min(1.2, self.player_pitch - pitch_delta))

        # 2. Update Player Position (WASD)
        move_x = float(move_x_idx - 1)  # 0, 1, 2 -> -1, 0, 1
        move_y = float(move_y_idx - 1)  # 0, 1, 2 -> -1, 0, 1

        # Forward vector
        fwd_x = -math.sin(self.player_yaw)
        fwd_z = -math.cos(self.player_yaw)
        # Right vector
        right_x = math.cos(self.player_yaw)
        right_z = -math.sin(self.player_yaw)

        speed = 0.4
        dx = (right_x * move_x + fwd_x * move_y) * speed
        dz = (right_z * move_x + fwd_z * move_y) * speed

        # Boundary collision check [-18, 18]
        self.player_x = max(-18.0, min(18.0, self.player_x + dx))
        self.player_z = max(-18.0, min(18.0, self.player_z + dz))

        # 3. Raycast Weapon Shooting
        if fire == 1:
            hit_any = False
            for target in self.targets:
                # Vector from player to target
                tx = target["x"] - self.player_x
                tz = target["z"] - self.player_z
                dist = math.hypot(tx, tz)

                if dist < 40.0:
                    # Angle to target in XZ plane
                    target_angle = math.atan2(-tx, -tz) % (2 * math.pi)
                    angle_diff = abs((self.player_yaw - target_angle + math.pi) % (2 * math.pi) - math.pi)

                    # Angular hitbox tolerance (tighter at distance)
                    ang_tolerance = max(0.08, target["radius"] / max(1.0, dist))
                    if angle_diff < ang_tolerance and abs(self.player_pitch) < 0.3:
                        target["health"] -= 50.0
                        hit_any = True
                        if target["health"] <= 0.0:
                            # Respawn target
                            new_angle = self._rng.uniform(0.0, 2 * math.pi)
                            new_dist = self._rng.uniform(6.0, 16.0)
                            target["x"] = self.player_x + math.cos(new_angle) * new_dist
                            target["z"] = self.player_z + math.sin(new_angle) * new_dist
                            target["health"] = 100.0

            if hit_any:
                self.last_shot_hit = True
                self.targets_hit_count += 1
                reward += 1.0
            else:
                reward -= 0.02  # Miss penalty

        # Terminate condition
        terminated = self.step_count >= self.max_steps
        truncated = False
        obs = self._render_observation()
        info = {
            "targets_hit": self.targets_hit_count,
            "step": self.step_count,
            "player_pos": (self.player_x, self.player_z),
            "last_shot_hit": self.last_shot_hit,
        }

        return obs, reward, terminated, truncated, info

    def _render_observation(self) -> np.ndarray:
        """Renders 3D perspective visual frame into [3, H, W] uint8 numpy array."""
        w, h = self.width, self.height
        img = Image.new("RGB", (w, h), color=(35, 40, 50))
        draw = ImageDraw.Draw(img)

        # 1. Horizon & Sky/Floor
        horizon_y = int(h / 2 + self.player_pitch * (h / 3))
        horizon_y = max(5, min(h - 5, horizon_y))

        # Sky
        draw.rectangle([0, 0, w, horizon_y], fill=(50, 60, 80))
        # Ground
        draw.rectangle([0, horizon_y, w, h], fill=(65, 70, 60))

        # 2. Floor perspective grid lines
        for offset in range(-4, 5):
            x_top = int(w / 2 + (offset + self.player_yaw * 1.5) * (w / 8))
            x_bot = int(w / 2 + (offset + self.player_yaw * 1.5) * (w / 3))
            if 0 <= x_top <= w or 0 <= x_bot <= w:
                draw.line([(x_top, horizon_y), (x_bot, h)], fill=(75, 80, 70), width=1)

        # 3. Render Targets in Perspective
        for target in self.targets:
            tx = target["x"] - self.player_x
            tz = target["z"] - self.player_z
            dist = math.hypot(tx, tz)

            # Check if in front of player
            target_angle = math.atan2(-tx, -tz) % (2 * math.pi)
            angle_diff = (self.player_yaw - target_angle + math.pi) % (2 * math.pi) - math.pi

            # Within 70 degree field of view
            if abs(angle_diff) < math.radians(45.0) and dist > 0.5:
                # Screen X position
                screen_x = int(w / 2 - (angle_diff / math.radians(45.0)) * (w / 2))
                # Perspective size
                t_size = max(4, int((h / 2) / max(1.0, dist * 0.4)))
                t_height = int(t_size * 1.8)

                screen_y = int(horizon_y - t_height / 2)

                draw.rectangle(
                    [
                        screen_x - t_size // 2,
                        screen_y,
                        screen_x + t_size // 2,
                        screen_y + t_height,
                    ],
                    fill=(200, 50, 40),
                    outline=(240, 90, 80),
                )

        # 4. Aim Crosshair
        cx, cy = w // 2, h // 2
        ch_len = max(2, int(w / 35))
        draw.line([(cx - ch_len, cy), (cx + ch_len, cy)], fill=(0, 255, 0), width=1)
        draw.line([(cx, cy - ch_len), (cx, cy + ch_len)], fill=(0, 255, 0), width=1)

        # 5. Muzzle Flash on shot
        if self.last_action_fired:
            draw.ellipse([cx - 3, cy + 8, cx + 3, cy + 14], fill=(255, 240, 120))

        # Convert PIL to [3, H, W] uint8 numpy array
        arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
        return np.transpose(arr, (2, 0, 1))  # [3, H, W]


def make_sandbox_env(
    env_path: Optional[Union[str, Path]] = None,
    width: int = 84,
    height: int = 84,
    seed: int = 42,
) -> gym.Env:
    """Factory function creating either GodotEnv bridge or MockTacticalArenaEnv."""
    if env_path and Path(env_path).exists() and str(env_path).endswith((".exe", ".x86_64")):
        try:
            from godot_rl.core.godot_env import GodotEnv
            return GodotEnv(env_path=str(env_path), seed=seed)
        except Exception:
            pass

    return MockTacticalArenaEnv(width=width, height=height, seed=seed)

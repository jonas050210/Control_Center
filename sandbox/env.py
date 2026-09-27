"""Deterministic visual tactical-FPS environment used by BC and RL.

``TacticalArenaEnv`` is the fast, self-contained reference simulation.  It is
not a test stub: it implements the same complete action contract as the Godot
sandbox (movement, camera, jump/crouch/sprint, ADS, firing, ammunition and
reload), cover collision/occlusion, enemies, deterministic resets, reward
signals, RGB observations, and rich episode metrics.  It allows training to
continue headlessly when a Godot export is not running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw

from sandbox.actions import ACTION_DIMS, SandboxAction, coerce_sandbox_action


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Obstacle:
    x: float
    z: float
    half_w: float
    half_d: float
    height: float = 2.4


@dataclass
class Enemy:
    x: float
    z: float
    spawn_x: float
    spawn_z: float
    health: float = 100.0
    radius: float = 0.55
    phase: float = 0.0
    fire_cooldown: int = 0


class TacticalArenaEnv(gym.Env[np.ndarray, np.ndarray]):
    """A deterministic, vision-first tactical FPS training environment.

    Observations are channel-first RGB frames.  Privileged state is exposed in
    ``info`` for diagnostics and reward debugging, never in the policy
    observation.  Episodes are reproducible for a given seed.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    ARENA_LIMIT = 19.0
    PLAYER_RADIUS = 0.38
    MAGAZINE_SIZE = 20
    STARTING_RESERVE = 60
    RELOAD_STEPS = 24
    SHOT_COOLDOWN_STEPS = 3
    MAX_HEALTH = 100.0

    def __init__(
        self,
        width: int = 84,
        height: int = 84,
        max_steps: int = 500,
        seed: Optional[int] = 42,
        num_enemies: int = 4,
        difficulty: float = 1.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if width < 32 or height < 32:
            raise ValueError("Visual observations must be at least 32x32")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.num_enemies = max(1, int(num_enemies))
        self.difficulty = max(0.0, float(difficulty))
        self.render_mode = render_mode
        self._initial_seed = seed
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.height, self.width), dtype=np.uint8
        )
        self.action_space = spaces.MultiDiscrete(np.asarray(ACTION_DIMS, dtype=np.int64))

        # Fixed asymmetric cover layout gives lanes, corners, and sightlines.
        self.obstacles: List[Obstacle] = [
            Obstacle(0.0, -5.0, 3.3, 0.65, 2.7),
            Obstacle(-7.0, 1.0, 0.7, 4.0, 3.0),
            Obstacle(7.0, 5.5, 0.7, 3.2, 3.0),
            Obstacle(-2.5, 7.0, 2.2, 0.8, 1.45),
            Obstacle(4.0, -12.0, 2.0, 1.0, 1.8),
            Obstacle(-12.0, -8.0, 1.1, 1.1, 2.8),
            Obstacle(12.0, 11.0, 1.2, 1.2, 2.8),
        ]

        self.player_x = 0.0
        self.player_z = 0.0
        self.player_yaw = 0.0
        self.player_pitch = 0.0
        self.player_health = self.MAX_HEALTH
        self.ammo = self.MAGAZINE_SIZE
        self.reserve_ammo = self.STARTING_RESERVE
        self.reload_timer = 0
        self.shot_cooldown = 0
        self.air_timer = 0
        self.recoil = 0.0
        self.targets: List[Dict[str, float]] = []  # compatibility view
        self.enemies: List[Enemy] = []
        self.step_count = 0
        self.targets_hit_count = 0
        self.kill_count = 0
        self.shots_fired = 0
        self.damage_taken = 0.0
        self.distance_travelled = 0.0
        self.last_shot_hit = False
        self.last_action_fired = False
        self.last_action = SandboxAction()
        self._previous_aim_error = math.pi
        self._visited_cells: set[Tuple[int, int]] = set()
        self._episode_seed = int(seed or 0)
        self.reset(seed=seed)

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self._initial_seed)

        options = options or {}
        self.step_count = 0
        self.targets_hit_count = 0
        self.kill_count = 0
        self.shots_fired = 0
        self.damage_taken = 0.0
        self.distance_travelled = 0.0
        self.last_shot_hit = False
        self.last_action_fired = False
        self.last_action = SandboxAction()
        self.player_health = self.MAX_HEALTH
        self.ammo = self.MAGAZINE_SIZE
        self.reserve_ammo = self.STARTING_RESERVE
        self.reload_timer = 0
        self.shot_cooldown = 0
        self.air_timer = 0
        self.recoil = 0.0
        self._visited_cells = set()

        # Spawn in one of four safe central positions and face a random lane.
        spawn_points = [(-3.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, -1.5)]
        spawn_idx = int(self._rng.integers(0, len(spawn_points)))
        self.player_x, self.player_z = spawn_points[spawn_idx]
        self.player_yaw = float(self._rng.uniform(-math.pi, math.pi))
        self.player_pitch = 0.0

        self.enemies = []
        for idx in range(self.num_enemies):
            x, z = self._sample_free_position(min_player_distance=7.0)
            self.enemies.append(
                Enemy(
                    x=x,
                    z=z,
                    spawn_x=x,
                    spawn_z=z,
                    phase=float(self._rng.uniform(0.0, math.tau)),
                    fire_cooldown=int(self._rng.integers(18, 45)),
                )
            )
        self._sync_targets_compat()
        self._previous_aim_error = self._nearest_aim_error()[0]
        self._mark_visited()
        info = self._build_info(reward_terms={"reset": 0.0})
        return self._render_observation(), info

    # ------------------------------------------------------------------- step
    def step(
        self, action: Union[np.ndarray, Sequence[int], Dict[str, Any], SandboxAction]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        command = coerce_sandbox_action(action)
        self.last_action = command
        self.step_count += 1
        self.last_shot_hit = False
        self.last_action_fired = False
        reward_terms: Dict[str, float] = {"time": -0.002}

        # Timers and reload are stepped exactly once per action command.
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
        if self.air_timer > 0:
            self.air_timer -= 1
        if self.reload_timer > 0:
            self.reload_timer -= 1
            if self.reload_timer == 0:
                needed = self.MAGAZINE_SIZE - self.ammo
                loaded = min(needed, self.reserve_ammo)
                self.ammo += loaded
                self.reserve_ammo -= loaded
                reward_terms["reload_complete"] = 0.01
        elif command.reload and self.ammo < self.MAGAZINE_SIZE and self.reserve_ammo > 0:
            self.reload_timer = self.RELOAD_STEPS
            reward_terms["reload_start"] = 0.005

        # Camera control uses the same bins as recorded mouse motion.  ADS gives
        # finer control and recoil decays naturally over subsequent steps.
        look_scale = 0.55 if command.ads else 1.0
        yaw_delta = (command.mouse_dx_bin - 10) * 0.035 * look_scale
        pitch_delta = (command.mouse_dy_bin - 10) * 0.018 * look_scale
        self.player_yaw = _wrap_angle(self.player_yaw - yaw_delta)
        self.player_pitch = float(
            np.clip(self.player_pitch - pitch_delta + self.recoil, -1.15, 1.15)
        )
        self.recoil *= 0.58

        # Tactical movement and cover collision.
        if command.jump and self.air_timer == 0 and not command.crouch:
            self.air_timer = 12
            reward_terms["jump"] = -0.002
        speed = 0.31
        if command.crouch:
            speed *= 0.55
        elif command.sprint and command.move_y > 0 and not command.ads:
            speed *= 1.55
        if self.air_timer > 0:
            speed *= 0.88

        move_len = math.hypot(command.move_x, command.move_y)
        if move_len > 0:
            move_x = command.move_x / max(1.0, move_len)
            move_y = command.move_y / max(1.0, move_len)
            fwd_x, fwd_z = -math.sin(self.player_yaw), -math.cos(self.player_yaw)
            right_x, right_z = math.cos(self.player_yaw), -math.sin(self.player_yaw)
            dx = (right_x * move_x + fwd_x * move_y) * speed
            dz = (right_z * move_x + fwd_z * move_y) * speed
            old_x, old_z = self.player_x, self.player_z
            self._move_player_with_collision(dx, dz)
            travelled = math.hypot(self.player_x - old_x, self.player_z - old_z)
            self.distance_travelled += travelled
            reward_terms["movement"] = 0.0005 * travelled
            if self._mark_visited():
                reward_terms["exploration"] = 0.01

        # Fire after look/movement so an action can acquire and shoot a target.
        if command.fire:
            if self.reload_timer > 0:
                reward_terms["fire_while_reloading"] = -0.01
            elif self.shot_cooldown > 0:
                reward_terms["fire_cooldown"] = -0.002
            elif self.ammo <= 0:
                reward_terms["dry_fire"] = -0.03
            else:
                hit, killed = self._fire_weapon(command.ads)
                self.last_action_fired = True
                self.shots_fired += 1
                self.ammo -= 1
                self.shot_cooldown = self.SHOT_COOLDOWN_STEPS
                self.recoil = -0.012 * (0.45 if command.ads else 1.0)
                reward_terms["shot_cost"] = -0.008
                if hit:
                    reward_terms["hit"] = 0.45
                else:
                    reward_terms["miss"] = -0.025
                if killed:
                    reward_terms["kill"] = 2.0

        # Enemy motion/fire makes cover and survival matter.
        damage = self._update_enemies()
        if damage > 0:
            self.player_health = max(0.0, self.player_health - damage)
            self.damage_taken += damage
            reward_terms["damage"] = -0.018 * damage

        # Dense aim shaping is bounded and only applies to visible targets.
        aim_error, visible = self._nearest_aim_error()
        if visible:
            improvement = float(np.clip(self._previous_aim_error - aim_error, -0.2, 0.2))
            reward_terms["aim"] = improvement * 0.06
            if aim_error < 0.10:
                reward_terms["on_target"] = 0.003
        self._previous_aim_error = aim_error

        self._sync_targets_compat()
        reward = float(sum(reward_terms.values()))
        dead = self.player_health <= 0.0
        time_limit = self.step_count >= self.max_steps
        terminated = bool(dead)
        truncated = bool(time_limit and not dead)
        if dead:
            reward -= 1.0
            reward_terms["death"] = -1.0
        info = self._build_info(reward_terms)
        return self._render_observation(), reward, terminated, truncated, info

    # ----------------------------------------------------------- game systems
    def _sample_free_position(self, min_player_distance: float = 4.0) -> Tuple[float, float]:
        for _ in range(200):
            x = float(self._rng.uniform(-16.5, 16.5))
            z = float(self._rng.uniform(-16.5, 16.5))
            if math.hypot(x - self.player_x, z - self.player_z) < min_player_distance:
                continue
            if not self._collides(x, z, 0.7):
                return x, z
        return 14.0, 14.0

    def _collides(self, x: float, z: float, radius: float) -> bool:
        if abs(x) + radius > self.ARENA_LIMIT or abs(z) + radius > self.ARENA_LIMIT:
            return True
        return any(
            abs(x - box.x) < box.half_w + radius
            and abs(z - box.z) < box.half_d + radius
            for box in self.obstacles
        )

    def _move_player_with_collision(self, dx: float, dz: float) -> None:
        candidate_x = self.player_x + dx
        if not self._collides(candidate_x, self.player_z, self.PLAYER_RADIUS):
            self.player_x = candidate_x
        candidate_z = self.player_z + dz
        if not self._collides(self.player_x, candidate_z, self.PLAYER_RADIUS):
            self.player_z = candidate_z

    @staticmethod
    def _segment_intersects_box(
        x1: float, z1: float, x2: float, z2: float, box: Obstacle, margin: float = 0.0
    ) -> bool:
        """2-D slab test for cover occlusion."""
        min_x, max_x = box.x - box.half_w - margin, box.x + box.half_w + margin
        min_z, max_z = box.z - box.half_d - margin, box.z + box.half_d + margin
        dx, dz = x2 - x1, z2 - z1
        t_min, t_max = 0.0, 1.0
        for origin, delta, lower, upper in ((x1, dx, min_x, max_x), (z1, dz, min_z, max_z)):
            if abs(delta) < 1e-9:
                if origin < lower or origin > upper:
                    return False
                continue
            t1, t2 = (lower - origin) / delta, (upper - origin) / delta
            if t1 > t2:
                t1, t2 = t2, t1
            t_min, t_max = max(t_min, t1), min(t_max, t2)
            if t_min > t_max:
                return False
        return t_max > 0.01 and t_min < 0.99

    def _has_line_of_sight(self, x: float, z: float) -> bool:
        return not any(
            self._segment_intersects_box(self.player_x, self.player_z, x, z, obstacle)
            for obstacle in self.obstacles
            if obstacle.height >= 1.4
        )

    def _target_angles(self, enemy: Enemy) -> Tuple[float, float, float]:
        dx, dz = enemy.x - self.player_x, enemy.z - self.player_z
        distance = max(0.01, math.hypot(dx, dz))
        yaw = math.atan2(-dx, -dz)
        # Eye is 1.55m; target centre is 1.0m, so targets below horizon at range.
        pitch = math.atan2(1.0 - 1.55, distance)
        return yaw, pitch, distance

    def _nearest_aim_error(self) -> Tuple[float, bool]:
        best = math.pi
        visible = False
        for enemy in self.enemies:
            if enemy.health <= 0 or not self._has_line_of_sight(enemy.x, enemy.z):
                continue
            yaw, pitch, _ = self._target_angles(enemy)
            error = math.hypot(_wrap_angle(self.player_yaw - yaw), self.player_pitch - pitch)
            if error < best:
                best = error
                visible = True
        return best, visible

    def _fire_weapon(self, ads: bool) -> Tuple[bool, bool]:
        best_enemy: Optional[Enemy] = None
        best_error = math.inf
        for enemy in self.enemies:
            if enemy.health <= 0 or not self._has_line_of_sight(enemy.x, enemy.z):
                continue
            yaw, pitch, distance = self._target_angles(enemy)
            yaw_error = abs(_wrap_angle(self.player_yaw - yaw))
            pitch_error = abs(self.player_pitch - pitch)
            tolerance = max(0.022 if ads else 0.032, enemy.radius / max(2.0, distance))
            error = math.hypot(yaw_error, pitch_error)
            if yaw_error <= tolerance and pitch_error <= tolerance * 1.45 and error < best_error:
                best_enemy, best_error = enemy, error
        if best_enemy is None:
            return False, False
        damage = 50.0 if ads else 40.0
        best_enemy.health -= damage
        self.last_shot_hit = True
        self.targets_hit_count += 1
        killed = best_enemy.health <= 0.0
        if killed:
            self.kill_count += 1
            x, z = self._sample_free_position(min_player_distance=6.0)
            best_enemy.x = best_enemy.spawn_x = x
            best_enemy.z = best_enemy.spawn_z = z
            best_enemy.health = 100.0
            best_enemy.phase = float(self._rng.uniform(0.0, math.tau))
            best_enemy.fire_cooldown = int(self._rng.integers(25, 50))
        return True, killed

    def _update_enemies(self) -> float:
        total_damage = 0.0
        for idx, enemy in enumerate(self.enemies):
            enemy.phase += 0.025 + idx * 0.001
            # Short deterministic patrol around spawn; never enter cover.
            desired_x = enemy.spawn_x + math.sin(enemy.phase) * 1.2
            desired_z = enemy.spawn_z + math.cos(enemy.phase * 0.83) * 1.2
            next_x = enemy.x + float(np.clip(desired_x - enemy.x, -0.035, 0.035))
            next_z = enemy.z + float(np.clip(desired_z - enemy.z, -0.035, 0.035))
            if not self._collides(next_x, enemy.z, enemy.radius):
                enemy.x = next_x
            if not self._collides(enemy.x, next_z, enemy.radius):
                enemy.z = next_z

            enemy.fire_cooldown -= 1
            distance = math.hypot(enemy.x - self.player_x, enemy.z - self.player_z)
            if (
                enemy.fire_cooldown <= 0
                and distance < 18.0
                and self._has_line_of_sight(enemy.x, enemy.z)
                and self.difficulty > 0
            ):
                hit_probability = float(np.clip(0.52 - distance * 0.018, 0.12, 0.45))
                if self.last_action.crouch:
                    hit_probability *= 0.78
                if self.last_action.sprint:
                    hit_probability *= 0.86
                if self._rng.random() < hit_probability * min(1.5, self.difficulty):
                    total_damage += 7.0
                enemy.fire_cooldown = int(
                    self._rng.integers(24, 42) / max(0.4, min(2.0, self.difficulty))
                )
        return total_damage

    def _mark_visited(self) -> bool:
        cell = (int((self.player_x + 20) // 4), int((self.player_z + 20) // 4))
        is_new = cell not in self._visited_cells
        self._visited_cells.add(cell)
        return is_new

    def _sync_targets_compat(self) -> None:
        self.targets = [
            {"x": e.x, "z": e.z, "radius": e.radius, "health": e.health}
            for e in self.enemies
        ]

    # --------------------------------------------------------------- telemetry
    def _build_info(self, reward_terms: Dict[str, float]) -> Dict[str, Any]:
        accuracy = self.targets_hit_count / max(1, self.shots_fired)
        return {
            "episode_seed": self._episode_seed,
            "step": self.step_count,
            "targets_hit": self.targets_hit_count,
            "kills": self.kill_count,
            "shots_fired": self.shots_fired,
            "accuracy": round(accuracy, 4),
            "last_shot_hit": self.last_shot_hit,
            "player_pos": (round(self.player_x, 4), round(self.player_z, 4)),
            "yaw": round(self.player_yaw, 5),
            "pitch": round(self.player_pitch, 5),
            "health": round(self.player_health, 2),
            "ammo": self.ammo,
            "reserve_ammo": self.reserve_ammo,
            "reloading": self.reload_timer > 0,
            "reload_steps_remaining": self.reload_timer,
            "damage_taken": round(self.damage_taken, 2),
            "distance_travelled": round(self.distance_travelled, 3),
            "visited_cells": len(self._visited_cells),
            "reward_terms": {key: round(value, 6) for key, value in reward_terms.items()},
            "action": self.last_action.to_dict(),
        }

    def state_snapshot(self) -> Dict[str, Any]:
        """Return a restorable deterministic state for debugging and replay tests."""
        return {
            "player": [
                self.player_x,
                self.player_z,
                self.player_yaw,
                self.player_pitch,
                self.player_health,
                self.ammo,
                self.reserve_ammo,
            ],
            "enemies": [
                [e.x, e.z, e.spawn_x, e.spawn_z, e.health, e.radius, e.phase, e.fire_cooldown]
                for e in self.enemies
            ],
            "step": self.step_count,
            "episode_seed": self._episode_seed,
            "timers": [self.reload_timer, self.shot_cooldown, self.air_timer, self.recoil],
            "counters": [
                self.targets_hit_count,
                self.kill_count,
                self.shots_fired,
                self.damage_taken,
                self.distance_travelled,
            ],
            "last_action": self.last_action.to_array().tolist(),
            "last_shot_hit": self.last_shot_hit,
            "last_action_fired": self.last_action_fired,
            "previous_aim_error": self._previous_aim_error,
            "visited_cells": [list(cell) for cell in sorted(self._visited_cells)],
            "rng_state": self._rng.bit_generator.state,
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore a state previously returned by :meth:`state_snapshot`."""
        if not isinstance(snapshot, dict) or "player" not in snapshot or "enemies" not in snapshot:
            raise ValueError("invalid TacticalArenaEnv state snapshot")
        player = snapshot["player"]
        if not isinstance(player, list) or len(player) != 7:
            raise ValueError("snapshot player state must contain seven values")
        (
            self.player_x,
            self.player_z,
            self.player_yaw,
            self.player_pitch,
            self.player_health,
            self.ammo,
            self.reserve_ammo,
        ) = player
        restored: List[Enemy] = []
        for values in snapshot["enemies"]:
            if not isinstance(values, list) or len(values) != 8:
                raise ValueError("snapshot enemy state is malformed")
            restored.append(Enemy(*values))
        self.enemies = restored
        self.step_count = int(snapshot.get("step", 0))
        self._episode_seed = int(snapshot.get("episode_seed", self._episode_seed))
        timers = snapshot.get("timers", [0, 0, 0, 0.0])
        if isinstance(timers, list) and len(timers) == 4:
            self.reload_timer, self.shot_cooldown, self.air_timer, self.recoil = timers
        counters = snapshot.get("counters", [0, 0, 0, 0.0, 0.0])
        if isinstance(counters, list) and len(counters) == 5:
            (
                self.targets_hit_count,
                self.kill_count,
                self.shots_fired,
                self.damage_taken,
                self.distance_travelled,
            ) = counters
        self.last_action = SandboxAction.from_array(snapshot.get("last_action", SandboxAction().to_array()))
        self.last_shot_hit = bool(snapshot.get("last_shot_hit", False))
        self.last_action_fired = bool(snapshot.get("last_action_fired", False))
        self._previous_aim_error = float(snapshot.get("previous_aim_error", math.pi))
        self._visited_cells = {tuple(cell) for cell in snapshot.get("visited_cells", [])}
        if "rng_state" in snapshot:
            self._rng.bit_generator.state = snapshot["rng_state"]
        self._sync_targets_compat()

    # ----------------------------------------------------------------- render
    def _project(self, x: float, z: float) -> Optional[Tuple[int, float, float]]:
        dx, dz = x - self.player_x, z - self.player_z
        distance = math.hypot(dx, dz)
        angle = _wrap_angle(self.player_yaw - math.atan2(-dx, -dz))
        half_fov = math.radians(46.0 if not self.last_action.ads else 34.0)
        if abs(angle) > half_fov or distance < 0.3:
            return None
        screen_x = int(self.width / 2 - angle / half_fov * self.width / 2)
        return screen_x, distance, angle

    def _render_observation(self) -> np.ndarray:
        w, h = self.width, self.height
        horizon = int(np.clip(h / 2 + self.player_pitch * h * 0.42, 7, h - 8))
        img = Image.new("RGB", (w, h), (42, 55, 78))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, horizon, w, h), fill=(62, 67, 58))

        # Perspective lane markings communicate rotation and movement visually.
        yaw_phase = self.player_yaw / math.tau
        for row in range(1, 6):
            y = horizon + int((h - horizon) * (row / 6.0) ** 1.7)
            shade = 74 + row * 2
            draw.line((0, y, w, y), fill=(shade, shade + 3, shade - 4), width=1)
        for offset in range(-5, 6):
            bottom_x = int(w / 2 + (offset + yaw_phase * 2.0) * w / 3.2)
            draw.line((w // 2, horizon, bottom_x, h), fill=(75, 79, 68), width=1)

        objects: List[Tuple[float, str, Any, Tuple[int, float, float]]] = []
        for box in self.obstacles:
            projected = self._project(box.x, box.z)
            if projected:
                objects.append((projected[1], "cover", box, projected))
        for enemy in self.enemies:
            projected = self._project(enemy.x, enemy.z)
            if projected and self._has_line_of_sight(enemy.x, enemy.z):
                objects.append((projected[1], "enemy", enemy, projected))

        # Painter's algorithm: distant geometry first.
        for distance, kind, obj, projected in sorted(objects, reverse=True, key=lambda item: item[0]):
            screen_x = projected[0]
            if kind == "cover":
                box: Obstacle = obj
                rect_h = int(np.clip(h * box.height / (distance * 1.7), 4, h * 0.75))
                rect_w = int(np.clip(w * (box.half_w + box.half_d) / (distance * 1.7), 3, w))
                bottom = horizon + int(h * 0.12 / max(1.0, distance / 4.0))
                color = (103, 108, 113) if box.height > 2 else (116, 111, 98)
                draw.rectangle(
                    (screen_x - rect_w // 2, bottom - rect_h, screen_x + rect_w // 2, bottom),
                    fill=color,
                    outline=(137, 142, 146),
                )
            else:
                enemy: Enemy = obj
                body_h = int(np.clip(h * 1.9 / (distance * 1.45), 5, h * 0.58))
                body_w = max(3, int(body_h * 0.43))
                centre_y = int(horizon + self.player_pitch * h * 0.02)
                top = centre_y - body_h // 2
                health_ratio = max(0.0, enemy.health / 100.0)
                draw.rectangle(
                    (screen_x - body_w // 2, top, screen_x + body_w // 2, top + body_h),
                    fill=(198, 52, 43),
                    outline=(241, 96, 76),
                )
                if body_h >= 9:
                    bar_y = max(0, top - 3)
                    draw.rectangle((screen_x - body_w // 2, bar_y, screen_x + body_w // 2, bar_y + 1), fill=(45, 20, 20))
                    draw.line((screen_x - body_w // 2, bar_y, screen_x - body_w // 2 + int(body_w * health_ratio), bar_y), fill=(80, 230, 95))

        # HUD stays deliberately small so vision training cannot rely only on it.
        draw.rectangle((2, h - 6, int(2 + (w * 0.25) * self.player_health / 100.0), h - 4), fill=(65, 220, 90))
        ammo_width = int((w * 0.18) * self.ammo / self.MAGAZINE_SIZE)
        draw.rectangle((w - 2 - ammo_width, h - 6, w - 2, h - 4), fill=(235, 204, 72))
        if self.reload_timer > 0:
            progress = 1.0 - self.reload_timer / self.RELOAD_STEPS
            draw.rectangle((w // 2 - 10, h - 4, w // 2 - 10 + int(20 * progress), h - 3), fill=(100, 180, 250))

        cx, cy = w // 2, h // 2
        gap = 1 if self.last_action.ads else 2
        length = max(2, w // 32)
        cross = (120, 255, 140) if self.last_shot_hit else (230, 235, 225)
        draw.line((cx - gap - length, cy, cx - gap, cy), fill=cross)
        draw.line((cx + gap, cy, cx + gap + length, cy), fill=cross)
        draw.line((cx, cy - gap - length, cx, cy - gap), fill=cross)
        draw.line((cx, cy + gap, cx, cy + gap + length), fill=cross)
        if self.last_action_fired:
            draw.ellipse((cx - 2, cy + 7, cx + 2, cy + 11), fill=(255, 224, 96))

        return np.transpose(np.asarray(img, dtype=np.uint8), (2, 0, 1)).copy()

    def render(self) -> np.ndarray:
        return np.transpose(self._render_observation(), (1, 2, 0))

    def close(self) -> None:
        return None


# Backward-compatible import name.  The implementation is now the production
# headless reference environment, not a reduced mock.
MockTacticalArenaEnv = TacticalArenaEnv


def make_sandbox_env(
    env_path: Optional[Union[str, Path]] = None,
    width: int = 84,
    height: int = 84,
    seed: int = 42,
    max_steps: int = 500,
    backend: str = "auto",
    **kwargs: Any,
) -> gym.Env:
    """Create the deterministic Python sandbox or an exported Godot sandbox.

    ``backend='auto'`` selects Godot only when ``env_path`` is supplied.  A
    requested Godot binary that cannot start raises a useful error instead of
    silently training against a different environment.
    """
    if backend not in {"auto", "python", "godot"}:
        raise ValueError("backend must be one of: auto, python, godot")
    use_godot = backend == "godot" or (backend == "auto" and env_path is not None)
    if use_godot:
        if env_path is None:
            raise ValueError("Godot backend requires env_path")
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"Godot sandbox export does not exist: {path}")
        from sandbox.godot_env import GodotSandboxEnv

        return GodotSandboxEnv(
            env_path=path,
            width=width,
            height=height,
            seed=seed,
            max_steps=max_steps,
            **kwargs,
        )
    return TacticalArenaEnv(
        width=width,
        height=height,
        seed=seed,
        max_steps=max_steps,
        **kwargs,
    )

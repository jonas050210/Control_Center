"""Headless 3D one-versus-one shooter environment for Gymnasium and SB3.

The environment simulates a first-person agent against a deterministic bot by
default. It uses a MultiDiscrete action vector so movement, look, stance, lean,
jump, reload, and firing can be combined in one policy action. ``step_duel`` is
also exposed for browser-side model-versus-model playback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.maps import ArenaMap, create_map, serialize_map
from env.physics import (
    VISION_FOV_DEGREES,
    AgentBody,
    can_see,
    direction_from_angles,
    ground_height,
    has_line_of_sight,
    in_field_of_view,
    move_body,
    position_is_free,
    position_is_free,
    ray_cylinder_hit,
    raycast_distance,
    raycast_scene,
    wrap_angle,
)
from env.weapons import WEAPON_NAMES, WeaponRuntime, get_weapon
from training.rewards import RewardEvents, shape_reward


# MultiDiscrete action fields: forward/back, strafe, yaw, pitch, fire, sprint,
# stance (stand/crouch/prone), lean, jump, reload.
ACTION_NVECS = (3, 3, 3, 3, 2, 2, 3, 3, 2, 2)
ACTION_SIZE = len(ACTION_NVECS)

# ---------------------------------------------------------------- perception
# The agent never receives exact enemy coordinates. It perceives
#   * a coarse bearing (12 sectors of 30 degrees),
#   * a coarse distance band,
#   * a coarse enemy health band,
#   * whether it currently has visual contact (view cone + line of sight)
#   * and a short-term memory ("seen 1.8 s ago in that direction").
# ``vision_mode`` selects how much of that is quantised:
#   "exact"     - legacy behaviour: precise bearing/distance/health, no blind state
#   "noisy"     - precise values plus seeded Gaussian noise
#   "coarse"    - sector/bands, but the enemy is always treated as tracked
#   "coarse_los"- sector/bands *and* real vision: cover hides the enemy
# Opponent behaviours for the scripted bot. ``mover`` is the gentle rung of the
# curriculum ladder: it moves and aims like the walker but never fires, so the
# policy can learn to track a moving target before it has to survive return fire.
OPPONENT_MODES: tuple[str, ...] = ("stationary", "mover", "walker", "shooter", "full")
# Which opponent each curriculum phase uses (``None`` = automatic by phase).
CURRICULUM_OPPONENTS: dict[int, str] = {
    1: "stationary",
    2: "mover",
    3: "walker",
    4: "full",
}
VISION_MODES = ("exact", "noisy", "coarse", "coarse_los")
DEFAULT_VISION_MODE = "coarse_los"
OBSERVATION_VERSION = 2
BEARING_SECTORS = 12
BEARING_NOISE_SIGMA = 0.35          # radians, "noisy" mode
DISTANCE_NOISE_SIGMA = 0.08         # fraction of the map diagonal
HP_NOISE_SIGMA = 0.10               # fraction of full health
MEMORY_SECONDS = 5.0                # how long a sighting stays in memory
# Upper bounds of the distance bands in metres (last band is open ended).
DISTANCE_BAND_EDGES = (5.0, 10.0, 20.0, 35.0, 60.0)
# Bearing of the empty state when nothing was ever seen.
UNKNOWN_BEARING = 0.0

OBSERVATION_FIELDS = (
    "self_x", "self_y", "self_z", "self_yaw", "self_pitch",
    "enemy_bearing_sin", "enemy_bearing_cos", "enemy_distance_band",
    "enemy_visible", "enemy_hp_band", "enemy_time_since_seen",
    "enemy_memory_sin", "enemy_memory_cos",
    "ray_0", "ray_1", "ray_2", "ray_3", "ray_4", "ray_5", "ray_6", "ray_7",
    "self_hp", "enemy_alive",
    "self_ammo_fraction", "self_mag_size",
    "self_sprinting", "self_crouched", "self_prone", "self_in_air",
    "self_time_since_shot", "self_time_since_reload",
)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)   # 31 Werte, Layout-Version 2


def bearing_sector(angle: float, sectors: int = BEARING_SECTORS) -> float:
    """Snap an angle in radians to the centre of its sector (360/sectors wide).

    Sector 0 is centred on 0 radians (straight ahead), so a target the agent is
    already facing is reported exactly and everything else is rounded to the
    nearest 30 degrees. The result is wrapped into [-pi, pi).
    """
    width = 2.0 * math.pi / sectors
    index = int(math.floor((wrap_angle(angle) + width * 0.5) / width)) % sectors
    return wrap_angle(index * width)


def sector_angles(sectors: int = BEARING_SECTORS) -> tuple[float, ...]:
    """Centre angle of every sector, in a stable order from -180 to 180 degrees."""
    width = 2.0 * math.pi / sectors
    return tuple(wrap_angle((index - sectors // 2) * width) for index in range(sectors))


def distance_band_value(distance: float, max_distance: float) -> float:
    """Map a distance onto one of the bands, returned in [-1, 1]."""
    edge = 0.0
    centre = DISTANCE_BAND_EDGES[0] * 0.5
    for index, limit in enumerate(DISTANCE_BAND_EDGES):
        if distance <= limit:
            lower = edge
            centre = (lower + limit) * 0.5
            break
        edge = limit
    else:
        centre = DISTANCE_BAND_EDGES[-1] + 20.0
    return float(2.0 * np.clip(centre / max(1e-6, max_distance), 0.0, 1.0) - 1.0)


def hp_band_value(fraction: float, bands: int = 4) -> float:
    """Map a health fraction onto one of ``bands`` buckets, returned in [-1, 1]."""
    bucket = int(np.clip(np.floor(fraction * bands), 0, bands - 1))
    centre = (bucket + 0.5) / bands
    return float(2.0 * centre - 1.0)


def quantize_noise(rng: np.random.Generator, value: float, sigma: float) -> float:
    """Add seeded Gaussian noise, clamped to the observation range."""
    return float(np.clip(value + float(rng.normal(0.0, sigma)), -1.0, 1.0))
ACTION_LABELS = (
    "move_forward_back", "move_left_right", "look_left_right", "look_down_up",
    "shoot", "sprint", "stance", "lean", "jump", "reload",
)


@dataclass
class Combatant:
    body: AgentBody
    weapon: WeaponRuntime
    shots_fired: int = 0
    bullets_fired: int = 0
    hits: int = 0
    headshots: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    kills: int = 0
    deaths: int = 0
    kill_distances: list[float] = field(default_factory=list)
    # Short-term perception memory: what this fighter last had visual contact with.
    last_seen_time: float | None = None
    last_seen_bearing: float | None = None
    last_seen_distance: float | None = None
    last_seen_hp_fraction: float | None = None
    blind_steps: int = 0
    visible_steps: int = 0
    last_view_frame: int = -1
    # Obstacle handling for the scripted opponent (it has no path planning).
    bot_stuck: int = 0
    bot_detour: int = 0
    bot_detour_sign: int = 1
    bot_steer: float = 0.0
    bot_last_position: tuple[float, float] | None = None

    def forget(self) -> None:
        """Clear the perception memory (called on every reset)."""
        self.last_seen_time = None
        self.last_seen_bearing = None
        self.last_seen_distance = None
        self.last_seen_hp_fraction = None
        self.blind_steps = 0
        self.visible_steps = 0
        self.last_view_frame = -1
        self.bot_stuck = 0
        self.bot_detour = 0
        self.bot_detour_sign = 1
        self.bot_steer = 0.0
        self.bot_last_position = None

    def seconds_since_seen(self, now: float) -> float | None:
        if self.last_seen_time is None:
            return None
        return max(0.0, now - self.last_seen_time)


class ShooterEnv(gym.Env[np.ndarray, np.ndarray]):
    """A fast, renderer-free 3D arena environment with a Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str | ArenaMap = "Dust",
        weapon_name: str | None = None,
        opponent_weapon: str | None = None,
        frame_skip: int = 4,
        max_episode_seconds: float = 120.0,
        dt: float = 1.0 / 60.0,
        curriculum: bool = False,
        curriculum_phase: int = 1,
        opponent_mode: str | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
        vision_mode: str = DEFAULT_VISION_MODE,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError(
                "ShooterEnv is headless and has no display render mode; use get_snapshot() and "
                "the browser renderer of the control center."
            )
        if vision_mode not in VISION_MODES:
            raise ValueError(
                f"Unknown vision_mode {vision_mode!r}. Choose one of: {', '.join(VISION_MODES)}"
            )
        self.vision_mode = vision_mode
        self.max_episode_seconds = float(max_episode_seconds)
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if max_episode_seconds <= 0 or dt <= 0:
            raise ValueError("episode duration and physics dt must be positive")
        self.render_mode = render_mode
        self.map_template = map_name.copy() if isinstance(map_name, ArenaMap) else create_map(map_name)
        self.arena_map = self.map_template.copy()
        self._fixed_weapon_name = self._canonical_weapon(weapon_name) if weapon_name else None
        self._fixed_opponent_weapon = self._canonical_weapon(opponent_weapon) if opponent_weapon else None
        self.frame_skip = int(frame_skip)
        self.dt = float(dt)
        self.max_episode_seconds = float(max_episode_seconds)
        self.max_frames = max(1, int(math.ceil(self.max_episode_seconds / self.dt)))
        # Diagonal of the current layout: used to normalise distances.
        self._max_distance = max(1.0, math.hypot(self.arena_map.width, self.arena_map.depth))
        self.curriculum = bool(curriculum)
        self.curriculum_phase = min(4, max(1, int(curriculum_phase)))
        if opponent_mode is not None and opponent_mode not in OPPONENT_MODES:
            raise ValueError(
                f"Unknown opponent mode {opponent_mode!r}. Choose one of: "
                f"{', '.join(OPPONENT_MODES)}"
            )
        self.opponent_mode = opponent_mode
        self.opponent_snapshot: dict[str, Any] | None = None
        self.action_space = spaces.MultiDiscrete(np.asarray(ACTION_NVECS, dtype=np.int64))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self._combatants: list[Combatant] = []
        self.elapsed = 0.0
        self.physics_frames = 0
        self.episode_index = 0
        self._done = False
        self._closed = False
        self.bullet_trails: list[dict[str, Any]] = []
        self.last_events: list[dict[str, Any]] = []
        if seed is not None:
            self.action_space.seed(seed)
        self.reset(seed=seed)
        # Constructor initialization is not a completed user-visible episode.
        self.episode_index = 0

    @staticmethod
    def _canonical_weapon(name: str) -> str:
        return next((weapon for weapon in WEAPON_NAMES if weapon.lower() == name.lower()), name)

    @property
    def map(self) -> ArenaMap:
        return self.arena_map

    @property
    def done(self) -> bool:
        """Whether this episode has reached termination or its time limit."""
        return self._done

    @property
    def player(self) -> Combatant:
        return self._combatants[0]

    @property
    def opponent(self) -> Combatant:
        return self._combatants[1]

    def set_curriculum_phase(self, phase: int) -> None:
        """Set the next reset's curriculum phase and its opponent behavior."""
        self.curriculum_phase = min(4, max(1, int(phase)))
        self.curriculum = True
        if self.curriculum_phase < 4:
            self.opponent_snapshot = None

    def set_vision_mode(self, mode: str) -> None:
        """Switch the perception model (used by the trainer's vision curriculum).

        Memory is dropped on a switch: what was "sighted" under one perception
        model must not be carried into a different one.
        """
        if mode not in VISION_MODES:
            raise ValueError(
                f"Unknown vision_mode {mode!r}. Choose one of: {', '.join(VISION_MODES)}"
            )
        if mode == self.vision_mode:
            return
        self.vision_mode = mode
        for fighter in self._combatants:
            fighter.forget()

    def set_opponent_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        """Install a frozen, NumPy-only PPO policy copy for self-play episodes."""
        self.opponent_snapshot = snapshot

    def set_map(self, arena_map: ArenaMap) -> None:
        """Replace the map template; the change takes effect on the next reset."""
        self.map_template = arena_map.copy()

    # Curriculum spawn distance as a share of the map's spawn separation. The
    # weapons are range limited (Pistol 35 m, damage falls off with distance), so
    # on wide maps the early phases have to start closer: measured on Dust, a
    # perfectly aimed Pistol hits 0 % of its shots at 36 m but ~24 % at 8 m.
    # A first run with 45 % for phase 2 (16.2 m, SMG vs. a bot that shoots back)
    # was a cliff: ~500 consecutive episodes without a single win or kill. The
    # ramp is therefore flatter. Phase 4 always uses the real spawn points.
    CURRICULUM_DISTANCE_FACTORS: dict[int, float] = {1: 0.25, 2: 0.35, 3: 0.6, 4: 1.0}

    def _curriculum_spawns(
        self, first: tuple[float, float], second: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Pull the fighters closer together in the early curriculum phases."""
        if not self.curriculum or self.curriculum_phase >= 4:
            return first, second
        factor = self.CURRICULUM_DISTANCE_FACTORS.get(self.curriculum_phase, 1.0)
        center_x = (first[0] + second[0]) / 2.0
        center_y = (first[1] + second[1]) / 2.0
        # If cover sits on the pulled-in point, try slightly further out before
        # falling back to the real spawn points.
        for candidate in (factor, min(1.0, factor + 0.1), min(1.0, factor + 0.25)):
            moved = [
                (center_x + (point[0] - center_x) * candidate,
                 center_y + (point[1] - center_y) * candidate)
                for point in (first, second)
            ]
            if all(position_is_free(self.arena_map, x, y) for x, y in moved):
                return (moved[0], moved[1])
        return first, second

    def _phase_weapon_names(self) -> tuple[str, str]:
        if not self.curriculum:
            player_name = self._fixed_weapon_name or str(self.np_random.choice(WEAPON_NAMES))
            enemy_name = self._fixed_opponent_weapon or str(self.np_random.choice(WEAPON_NAMES))
            return player_name, enemy_name
        if self.curriculum_phase == 1:
            return "Pistol", "Pistol"
        if self.curriculum_phase == 2:
            return "SMG", "SMG"
        if self.curriculum_phase == 3:
            return "AK-47", "AK-47"
        player_name = self._fixed_weapon_name or str(self.np_random.choice(WEAPON_NAMES))
        enemy_name = self._fixed_opponent_weapon or str(self.np_random.choice(WEAPON_NAMES))
        return player_name, enemy_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.arena_map = self.map_template.copy()
        self.elapsed = 0.0
        self.physics_frames = 0
        self._done = False
        self._closed = False
        self.episode_index += 1
        self.bullet_trails = []
        self.last_events = []
        player_weapon_name, enemy_weapon_name = self._phase_weapon_names()
        spawns = self.arena_map.spawn_points
        first_pos, second_pos = self._curriculum_spawns(*spawns)
        first_ground = ground_height(self.arena_map, first_pos[0], first_pos[1])
        second_ground = ground_height(self.arena_map, second_pos[0], second_pos[1])
        first_yaw = math.atan2(second_pos[0] - first_pos[0], second_pos[1] - first_pos[1])
        second_yaw = math.atan2(first_pos[0] - second_pos[0], first_pos[1] - second_pos[1])
        first = Combatant(
            AgentBody(first_pos[0], first_pos[1], first_ground, first_yaw),
            get_weapon(player_weapon_name).create_runtime(),
        )
        second = Combatant(
            AgentBody(second_pos[0], second_pos[1], second_ground, second_yaw),
            get_weapon(enemy_weapon_name).create_runtime(),
        )
        for fighter in (first, second):
            fighter.forget()
        self._combatants = [first, second]
        obs = self._observation(0)
        info = {
            "map": self.arena_map.name,
            "phase": self.curriculum_phase,
            "weapon": player_weapon_name,
            "opponent_weapon": enemy_weapon_name,
            "episode": self.episode_index,
            "vision_mode": self.vision_mode,
            "perception": {"player": self.enemy_view(0), "opponent": self.enemy_view(1)},
        }
        return obs, info

    def _sanitize_action(self, action: Sequence[int] | np.ndarray) -> np.ndarray:
        result = np.asarray(action, dtype=np.int64).reshape(-1)
        if result.size != ACTION_SIZE:
            raise ValueError(f"Expected {ACTION_SIZE} action values, received {result.size}")
        return np.asarray([np.clip(result[index], 0, ACTION_NVECS[index] - 1)
                           for index in range(ACTION_SIZE)], dtype=np.int64)

    @staticmethod
    def _decode_action(action: np.ndarray) -> dict[str, int | bool]:
        return {
            "forward": int(action[0]) - 1,
            "strafe": int(action[1]) - 1,
            "look_yaw": int(action[2]) - 1,
            "look_pitch": int(action[3]) - 1,
            "shoot": bool(action[4]),
            "sprint": bool(action[5]),
            "stance": int(action[6]),
            "lean": int(action[7]) - 1,
            "jump": bool(action[8]),
            "reload": bool(action[9]),
        }

    def step(self, action: Sequence[int] | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the learning agent against the configured scripted opponent."""
        if self._done:
            raise RuntimeError("Episode is over; call reset() before stepping again.")
        player_action = self._sanitize_action(action)
        return self._advance(player_action, None)

    def step_duel(
        self,
        action_a: Sequence[int] | np.ndarray,
        action_b: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance both agents once; the returned reward is from Agent 1's view."""
        if self._done:
            raise RuntimeError("Episode is over; call reset() before stepping again.")
        return self._advance(self._sanitize_action(action_a), self._sanitize_action(action_b))

    def _advance(
        self,
        action_a: np.ndarray,
        action_b: np.ndarray | None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        rewards = [0.0, 0.0]
        action_events: list[dict[str, Any]] = []
        terminal = False
        truncated = False
        for _ in range(self.frame_skip):
            before_distance = self._distance_between()
            moved_before = [(item.body.x, item.body.y) for item in self._combatants]
            actions = [action_a, action_b if action_b is not None else self.heuristic_action(1)]
            decoded = [self._decode_action(actions[index]) for index in range(2)]

            for index, combatant in enumerate(self._combatants):
                if not combatant.body.alive:
                    continue
                controls = decoded[index]
                move_body(
                    combatant.body,
                    self.arena_map,
                    float(controls["forward"]),
                    float(controls["strafe"]),
                    float(controls["look_yaw"]),
                    float(controls["look_pitch"]),
                    bool(controls["sprint"]),
                    int(controls["stance"]),
                    int(controls["lean"]),
                    bool(controls["jump"]),
                    self.dt,
                )
                combatant.weapon.tick(self.dt, firing=bool(controls["shoot"]))
                if bool(controls["reload"]):
                    combatant.weapon.begin_reload()

            frame_events = [
                {"hits": 0, "headshots": 0, "kills": 0, "incoming_hits": 0,
                 "deaths": 0, "dodges": 0, "wasted_ammo": 0, "fired": False}
                for _ in range(2)
            ]
            # Alternate which agent fires first on consecutive frames to reduce
            # the tiny same-frame advantage of one side in close-range duels.
            order = (0, 1) if self.physics_frames % 2 == 0 else (1, 0)
            for attacker_index in order:
                attacker = self._combatants[attacker_index]
                defender_index = 1 - attacker_index
                if not attacker.body.alive or not self._combatants[defender_index].body.alive:
                    continue
                if decoded[attacker_index]["shoot"]:
                    shot, pellets = self._fire(attacker_index)
                    if shot is not None:
                        frame_events[attacker_index]["fired"] = True
                        frame_events[attacker_index]["wasted_ammo"] += int(pellets == 0)
                        frame_events[attacker_index]["hits"] += pellets
                        frame_events[defender_index]["incoming_hits"] += shot["incoming_hits"]
                        frame_events[attacker_index]["headshots"] += shot["headshots"]
                        frame_events[attacker_index]["kills"] += shot["kills"]
                        action_events.extend(shot["events"])
                        if attacker_index == 1 and pellets == 0:
                            moved = math.hypot(
                                self._combatants[0].body.x - moved_before[0][0],
                                self._combatants[0].body.y - moved_before[0][1],
                            )
                            if moved > 0.025:
                                frame_events[0]["dodges"] += 1

            self.elapsed += self.dt
            self.physics_frames += 1
            after_distance = self._distance_between()
            for index in range(2):
                own = self._combatants[index]
                enemy = self._combatants[1 - index]
                # The aim bonus must not reward knowledge the agent cannot have:
                # without visual contact the directional signal is switched off.
                if enemy.body.alive and self._tracked(index):
                    aim_error = self._aim_error(index)
                else:
                    aim_error = math.pi
                if not own.body.alive and own.deaths == 0:
                    own.deaths += 1
                    frame_events[index]["deaths"] += 1
                breakdown = shape_reward(RewardEvents(
                    distance_before=before_distance,
                    distance_after=after_distance,
                    aim_error_radians=aim_error,
                    enemy_visible=bool(self._tracked(index)),
                    elapsed_seconds=self.elapsed,
                    time_limit_seconds=self.max_episode_seconds,
                    hits=frame_events[index]["hits"],
                    headshots=frame_events[index]["headshots"],
                    kills=frame_events[index]["kills"],
                    incoming_hits=frame_events[index]["incoming_hits"],
                    deaths=frame_events[index]["deaths"],
                    successful_dodges=frame_events[index]["dodges"],
                    wasted_ammo=frame_events[index]["wasted_ammo"],
                    physics_steps=1,
                ))
                rewards[index] += breakdown.total

            if not self._combatants[0].body.alive or not self._combatants[1].body.alive:
                terminal = True
                break
            if self.physics_frames >= self.max_frames:
                truncated = True
                break

        self.last_events = action_events[-80:]
        info: dict[str, Any] = {
            "map": self.arena_map.name,
            "phase": self.curriculum_phase,
            "distance": self._distance_between(),
            "elapsed": self.elapsed,
            "vision_mode": self.vision_mode,
            "perception": {
                "player": self.enemy_view(0),
                "opponent": self.enemy_view(1),
            },
            "reward_agent_2": float(rewards[1]),
            "combat_events": action_events,
            "player": self._combatant_summary(0),
            "opponent": self._combatant_summary(1),
        }
        if terminal or truncated:
            self._done = True
            info.update(self._episode_summary())
        return self._observation(0), float(rewards[0]), terminal, truncated, info

    def _distance_between(self) -> float:
        first, second = self._combatants
        dx = first.body.x - second.body.x
        dy = first.body.y - second.body.y
        dz = (first.body.z + first.body.height * 0.5) - (second.body.z + second.body.height * 0.5)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _aim_error(self, attacker_index: int) -> float:
        attacker = self._combatants[attacker_index].body
        target = self._combatants[1 - attacker_index].body
        dx, dy = target.x - attacker.x, target.y - attacker.y
        horizontal_distance = max(1e-6, math.hypot(dx, dy))
        target_yaw = math.atan2(dx, dy)
        target_height = target.z + target.height * 0.75
        source_height = attacker.z + attacker.eye_height
        target_pitch = math.atan2(target_height - source_height, horizontal_distance)
        yaw_error = wrap_angle(target_yaw - attacker.yaw)
        pitch_error = target_pitch - attacker.pitch
        return math.sqrt(yaw_error * yaw_error + pitch_error * pitch_error)

    def _fire(self, attacker_index: int) -> tuple[dict[str, Any] | None, int]:
        attacker = self._combatants[attacker_index]
        defender_index = 1 - attacker_index
        defender = self._combatants[defender_index]
        fired, _reason = attacker.weapon.try_fire()
        if not fired:
            return None, 0

        attacker.shots_fired += 1
        spec = attacker.weapon.spec
        pellets = max(1, spec.pellets)
        attacker.bullets_fired += pellets
        origin = (attacker.body.x, attacker.body.y, attacker.body.z + attacker.body.eye_height)
        sigma = attacker.weapon.weapon.spread_radians(
            attacker.weapon.recoil_level,
            moving=attacker.body.sprinting or attacker.body.movement_speed > 4.6,
        )
        hit_count = 0
        headshot_count = 0
        incoming_hits = 0
        kills = 0
        shot_events: list[dict[str, Any]] = []
        for pellet_index in range(pellets):
            yaw = attacker.body.yaw + float(self.np_random.normal(0.0, sigma))
            pitch = attacker.body.pitch + float(self.np_random.normal(0.0, sigma))
            direction = direction_from_angles(yaw, pitch)
            obstacle_distance, obstacle = raycast_scene(
                origin, direction, self.arena_map, max_distance=spec.range
            )
            target_hit = ray_cylinder_hit(origin, direction, defender.body)
            hit_distance: float | None = None
            is_headshot = False
            if target_hit is not None:
                target_distance, head = target_hit
                if target_distance <= spec.range and (
                    obstacle_distance is None or target_distance < obstacle_distance - 1e-4
                ):
                    hit_distance, is_headshot = target_distance, head

            endpoint_distance = spec.range
            if obstacle_distance is not None:
                endpoint_distance = min(endpoint_distance, obstacle_distance)
            if hit_distance is not None:
                endpoint_distance = min(endpoint_distance, hit_distance)
                damage = attacker.weapon.weapon.damage_at(hit_distance, is_headshot)
                if damage > 0.0 and defender.body.alive:
                    defender.body.hp = max(0.0, defender.body.hp - damage)
                    attacker.damage_dealt += damage
                    defender.damage_taken += damage
                    attacker.hits += 1
                    hit_count += 1
                    incoming_hits += 1
                    if is_headshot:
                        attacker.headshots += 1
                        headshot_count += 1
                    shot_events.append({
                        "type": "hit",
                        "attacker": attacker_index + 1,
                        "target": defender_index + 1,
                        "weapon": spec.name,
                        "damage": round(damage, 2),
                        "headshot": bool(is_headshot),
                        "distance": round(hit_distance, 2),
                    })
                    if defender.body.hp <= 0.0 and defender.deaths == 0:
                        attacker.kills += 1
                        attacker.kill_distances.append(hit_distance)
                        kills += 1
                        shot_events.append({
                            "type": "kill",
                            "attacker": attacker_index + 1,
                            "target": defender_index + 1,
                            "weapon": spec.name,
                            "headshot": bool(is_headshot),
                            "distance": round(hit_distance, 2),
                            "x": defender.body.x,
                            "y": defender.body.y,
                        })
            end_x = origin[0] + direction[0] * endpoint_distance
            end_y = origin[1] + direction[1] * endpoint_distance
            end_z = origin[2] + direction[2] * endpoint_distance
            self.bullet_trails.append({
                "start": origin,
                "end": (end_x, end_y, end_z),
                "hit": hit_distance is not None,
                "attacker": attacker_index + 1,
                "time": self.elapsed,
            })

        if len(self.bullet_trails) > 160:
            self.bullet_trails = self.bullet_trails[-160:]
        result = {
            "incoming_hits": incoming_hits,
            "headshots": headshot_count,
            "kills": kills,
            "events": shot_events,
        }
        return result, hit_count

    def heuristic_action(self, agent_index: int = 1) -> np.ndarray:
        """Return an aim-and-move action for the built-in opponent controller."""
        agent_index = 0 if agent_index <= 0 else 1
        own = self._combatants[agent_index]
        target = self._combatants[1 - agent_index]
        default = np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
        if not target.body.alive or not own.body.alive:
            return default
        if agent_index == 1 and self.curriculum_phase >= 4 and self.opponent_snapshot is not None:
            return self._snapshot_action(self._observation(1), self.opponent_snapshot)

        mode = self.opponent_mode
        if mode is None:
            mode = (CURRICULUM_OPPONENTS.get(self.curriculum_phase, "full")
                    if self.curriculum else "full")
        dx, dy = target.body.x - own.body.x, target.body.y - own.body.y
        distance = max(1e-6, math.hypot(dx, dy))
        desired_yaw = math.atan2(dx, dy)
        yaw_error = wrap_angle(desired_yaw - own.body.yaw)
        desired_pitch = math.atan2(
            (target.body.z + target.body.height * 0.75) - (own.body.z + own.body.eye_height), distance
        )
        pitch_error = desired_pitch - own.body.pitch
        if mode in {"mover", "walker", "shooter", "full"}:
            if abs(yaw_error) > 0.02:
                default[2] = 2 if yaw_error > 0 else 0
            if abs(pitch_error) > 0.02:
                default[3] = 2 if pitch_error > 0 else 0

        if mode in {"mover", "walker", "full"}:
            if distance > 8.0:
                default[0] = 2
            elif distance < 4.0:
                default[0] = 0
            else:
                # Alternate strafing direction over time to avoid a static target.
                default[1] = 2 if (self.physics_frames // 45 + agent_index) % 2 else 0
            if mode == "full" and distance > 13.0:
                default[5] = 1
            if mode == "full" and self.physics_frames % 120 < 18:
                default[8] = 1
            if mode == "full" and self.physics_frames % 90 < 25:
                default[6] = 1

        self._avoid_obstacles(agent_index, default, target_bearing=desired_yaw)

        # ``mover`` deliberately never fires: it exists only in early curriculum
        # phases, where a harmless opponent is the point. ``walker`` is the first
        # one that shoots back (every third chance, wider tolerance). Normalising
        # the aim bonus (0.005/step) removed the reason the harmless variant was
        # abandoned earlier - it can no longer be farmed for more than a kill.
        may_fire = mode in {"walker", "shooter", "full"}
        aligned = math.sqrt(yaw_error * yaw_error + pitch_error * pitch_error) < math.radians(
            {"walker": 14, "shooter": 12}.get(mode, 9)
        )
        sight = has_line_of_sight(own.body, target.body, self.arena_map)
        if may_fire and aligned and sight and (mode != "walker" or self.physics_frames % 3 == 0):
            default[4] = 1
        if own.weapon.ammo <= max(2, int(own.weapon.spec.mag_size * 0.15)):
            default[9] = 1
        return default

    def _avoid_obstacles(self, agent_index: int, action: np.ndarray,
                         target_bearing: float | None = None, notice: float = 0.004,
                         detour_frames: int = 70) -> None:
        """Steer the scripted opponent around cover instead of freezing.

        The bot has no path planner. Walking straight into a crate or wall used to
        freeze the whole duel - on Warehouse the opponent never reached the player
        within the episode, so that map produced *no* sighting, *no* reward and
        *no* training signal at all. When "wants to advance but does not move" is
        detected, the bot now looks for the most promising free direction from its
        ray fan, turns there and keeps moving; the normal aiming logic (and with
        it the firing decision) resumes as soon as it is unstuck.
        """
        fighter = self._combatants[0 if agent_index <= 0 else 1]
        position = (fighter.body.x, fighter.body.y)
        previous = fighter.bot_last_position
        moved = math.inf if previous is None else math.hypot(
            position[0] - previous[0], position[1] - previous[1]
        )
        fighter.bot_last_position = position
        if int(action[0]) == 2 and moved < notice:
            fighter.bot_stuck += 1
        else:
            fighter.bot_stuck = 0
        if fighter.bot_detour <= 0 and fighter.bot_stuck >= 6:
            origin = (fighter.body.x, fighter.body.y, fighter.body.z + fighter.body.eye_height)
            base = fighter.body.yaw if target_bearing is None else target_bearing
            best_angle: float | None = None
            best_score = float("-inf")
            for step in (-3, -2, -1, 1, 2, 3):
                angle = wrap_angle(base + step * math.pi / 4.0)
                free = min(raycast_distance(origin, angle, 0.0, self.arena_map, 18.0), 18.0)
                if free < 1.5:
                    continue
                # Prefer open space that still heads roughly toward the target.
                score = min(free, 12.0) - 2.4 * abs(step)
                if score > best_score:
                    best_score, best_angle = score, angle
            if best_angle is None:
                fighter.bot_detour = 0
                fighter.bot_stuck = 0
                return
            fighter.bot_steer = best_angle
            fighter.bot_detour = detour_frames
            fighter.bot_stuck = 0
        if fighter.bot_detour > 0:
            fighter.bot_detour -= 1
            error = wrap_angle(fighter.bot_steer - fighter.body.yaw)
            action[2] = 2 if error > 0.06 else (0 if error < -0.06 else 1)
            action[0] = 2 if abs(error) < math.radians(60) else 1

    @staticmethod
    def _snapshot_action(observation: np.ndarray, snapshot: dict[str, Any] | None = None) -> np.ndarray:
        """Run an exported frozen MLP policy without importing torch in an env worker."""
        if snapshot is None:
            return np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
        obs = np.asarray(observation, dtype=np.float32)
        mean = np.asarray(snapshot.get("obs_mean", np.zeros_like(obs)), dtype=np.float32)
        var = np.asarray(snapshot.get("obs_var", np.ones_like(obs)), dtype=np.float32)
        normalized = np.clip((obs - mean) / np.sqrt(np.maximum(var, 1e-8) + 1e-8), -10.0, 10.0)
        value = normalized
        layers = snapshot.get("layers", [])
        for index, layer in enumerate(layers):
            weight = np.asarray(layer["weight"], dtype=np.float32)
            bias = np.asarray(layer["bias"], dtype=np.float32)
            value = value @ weight.T + bias
            if index < len(layers) - 1:
                value = np.tanh(value)
        result: list[int] = []
        cursor = 0
        for categories in snapshot.get("action_nvec", ACTION_NVECS):
            block = value[cursor:cursor + int(categories)]
            result.append(int(np.argmax(block)) if block.size else 0)
            cursor += int(categories)
        if len(result) != ACTION_SIZE:
            return np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
        return np.asarray(result, dtype=np.int64)

    # ------------------------------------------------------------ perception
    def enemy_view(self, agent_index: int = 0) -> dict[str, Any]:
        """What agent ``agent_index`` currently perceives about the opponent.

        This is the single source of truth for the observation *and* for the
        reward gating: whatever is not in here cannot be known by the policy.
        """
        observer = self._combatants[0 if agent_index <= 0 else 1]
        target = self._combatants[1 if agent_index <= 0 else 0]
        own, enemy = observer.body, target.body
        dx, dy = enemy.x - own.x, enemy.y - own.y
        bearing = math.atan2(dx, dy)
        distance = self._distance_between()
        hp_fraction = float(np.clip(enemy.hp / 100.0, 0.0, 1.0))

        # The legacy and the "noisy" mode never hide the opponent; the two
        # coarse modes respect cover, the view cone and memory.
        visible = self._tracked(agent_index)

        # Record the sighting. Memory always takes the newest information (that
        # is idempotent while nothing moves), the counters only move once per
        # physics frame so repeated observation calls cannot inflate them.
        if visible:
            observer.last_seen_time = self.elapsed
            observer.last_seen_bearing = bearing
            observer.last_seen_distance = distance
            observer.last_seen_hp_fraction = hp_fraction
        if observer.last_view_frame != self.physics_frames:
            observer.last_view_frame = self.physics_frames
            if visible:
                observer.visible_steps += 1
            else:
                observer.blind_steps += 1

        now = self.elapsed
        seconds_since = observer.seconds_since_seen(now)
        if seconds_since is not None and seconds_since > MEMORY_SECONDS:
            # Stale contacts fade out completely.
            observer.last_seen_bearing = None
            observer.last_seen_distance = None
            observer.last_seen_hp_fraction = None
            seconds_since = None

        # Bearing: memory keeps the last known direction, world-anchored, so it
        # rotates with the agent's own look direction instead of pointing at the
        # hidden opponent's true position.
        memory_bearing = observer.last_seen_bearing if seconds_since is not None else None
        relative_bearing = wrap_angle((memory_bearing if memory_bearing is not None else bearing)
                                      - own.yaw)
        coarse = self.vision_mode in {"coarse", "coarse_los"}

        rng = self.np_random
        known = seconds_since is not None
        if coarse and self.vision_mode != "exact":
            bearing_used = bearing_sector(relative_bearing)
        else:
            bearing_used = relative_bearing
        # Distance and health come from the live sighting or from memory - never
        # from the current (hidden) state of the opponent.
        band_distance = 0.0
        band_hp = 0.0
        remembered = observer.last_seen_distance if known else None
        if not coarse or self.vision_mode == "exact":
            known_distance = distance if visible else remembered
            if known_distance is not None:
                band_distance = float(2.0 * np.clip(known_distance / self._max_distance, 0.0, 1.0) - 1.0)
            known_hp = hp_fraction if visible else (observer.last_seen_hp_fraction if known else None)
            if known_hp is not None:
                band_hp = float(2.0 * known_hp - 1.0)
        else:
            if remembered is not None:
                band_distance = distance_band_value(remembered, self._max_distance)
            if known and observer.last_seen_hp_fraction is not None:
                band_hp = hp_band_value(observer.last_seen_hp_fraction)

        if self.vision_mode == "noisy":
            bearing_used = quantize_noise(rng, bearing_used, BEARING_NOISE_SIGMA)
            band_distance = quantize_noise(rng, band_distance, DISTANCE_NOISE_SIGMA)
            band_hp = quantize_noise(rng, band_hp, HP_NOISE_SIGMA)

        # Encoding: -1 = never seen (or memory faded), 0 = seen right now,
        # +1 = the sighting is MEMORY_SECONDS old.
        if seconds_since is None:
            time_since_value = -1.0
        else:
            time_since_value = float(min(seconds_since, MEMORY_SECONDS) / MEMORY_SECONDS)

        memory_relative = (wrap_angle(memory_bearing - own.yaw)
                           if memory_bearing is not None else None)
        return {
            "visible": bool(visible),
            "memory_sin": float(math.sin(memory_relative)) if memory_relative is not None else 0.0,
            "memory_cos": float(math.cos(memory_relative)) if memory_relative is not None else 0.0,
            # Direction the agent would aim at right now: live sighting, memory
            # or - if it never saw the enemy - no information at all.
            "bearing": float(bearing_used),
            "bearing_sin": float(math.sin(bearing_used) if known else 0.0),
            "bearing_cos": float(math.cos(bearing_used) if known else 0.0),
            "distance_band": float(band_distance),
            "distance": float(distance),
            "hp_band": float(band_hp),
            "hp_fraction": hp_fraction,
            "time_since_seen": float(time_since_value),
            "seconds_since_seen": seconds_since,
            "memory": bool(known),
            "blind_steps": observer.blind_steps,
            "visible_steps": observer.visible_steps,
        }

    def _tracked(self, agent_index: int) -> bool:
        """Whether the agent has the opponent in view right now.

        Without a real vision model the opponent is always tracked, which is the
        legacy behaviour; ``coarse_los`` is the only mode that respects cover.
        """
        observer = self._combatants[0 if agent_index <= 0 else 1]
        target = self._combatants[1 if agent_index <= 0 else 0]
        if not observer.body.alive or not target.body.alive:
            return False
        if self.vision_mode != "coarse_los":
            return True
        return can_see(observer.body, target.body, self.arena_map)

    def _observation(self, agent_index: int) -> np.ndarray:
        own = self._combatants[agent_index]
        enemy = self._combatants[1 - agent_index]
        body, target = own.body, enemy.body
        view = self.enemy_view(agent_index)
        max_ray = max(10.0, max(self.arena_map.width, self.arena_map.depth) * 1.5)
        position_values = [
            np.clip(body.x / (self.arena_map.width / 2.0), -1.0, 1.0),
            np.clip(body.y / (self.arena_map.depth / 2.0), -1.0, 1.0),
            2.0 * np.clip(body.z / 5.0, 0.0, 1.0) - 1.0,
            body.yaw / math.pi,
            body.pitch / (math.pi / 2.0),
            view["bearing_sin"],
            view["bearing_cos"],
            view["distance_band"],
            ####################################################################
            # Perception block (layout version 2): the opponent is never
            # reported as an exact position/angle/health anymore.
            ####################################################################
            1.0 if view["visible"] else -1.0,
            view["hp_band"],
            view["time_since_seen"],
            # Memory channel: the exact direction of the last sighting (0/0 when
            # nothing is remembered). It only ever contains seen information.
            view["memory_sin"],
            view["memory_cos"],
        ]
        ray_values: list[float] = []
        origin = (body.x, body.y, body.z + body.eye_height)
        for ray_index in range(8):
            ray_yaw = body.yaw + ray_index * math.pi / 4.0
            distance = raycast_distance(origin, ray_yaw, 0.0, self.arena_map, max_ray)
            ray_values.append(2.0 * np.clip(distance / max_ray, 0.0, 1.0) - 1.0)
        spec = own.weapon.spec
        ammo_fraction = own.weapon.ammo / max(1, spec.mag_size)
        state_values = [
            2.0 * np.clip(body.hp / 100.0, 0.0, 1.0) - 1.0,
            1.0 if target.alive else -1.0,
            2.0 * np.clip(ammo_fraction, 0.0, 1.0) - 1.0,
            2.0 * np.clip(spec.mag_size / 100.0, 0.0, 1.0) - 1.0,
            1.0 if body.sprinting else -1.0,
            1.0 if body.stance == 1 else -1.0,
            1.0 if body.stance == 2 else -1.0,
            1.0 if body.in_air else -1.0,
            2.0 * np.clip(own.weapon.time_since_last_shot / 5.0, 0.0, 1.0) - 1.0,
            2.0 * np.clip(own.weapon.time_since_reload / 8.0, 0.0, 1.0) - 1.0,
        ]
        values = position_values + ray_values + state_values
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                f"Observation contract mismatch: expected {OBSERVATION_SIZE}, got {result.shape}"
            )
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def get_observation(self, agent_index: int = 0) -> np.ndarray:
        """Return the current observation from Agent 1 (0) or Agent 2 (1)."""
        if not self._combatants:
            raise RuntimeError("Call reset() before requesting an observation")
        return self._observation(0 if agent_index <= 0 else 1)

    def _combatant_summary(self, index: int) -> dict[str, Any]:
        combatant = self._combatants[index]
        body = combatant.body
        return {
            "x": body.x,
            "y": body.y,
            "z": body.z,
            "yaw": body.yaw,
            "pitch": body.pitch,
            "hp": max(0.0, body.hp),
            "height": body.height,
            "alive": body.alive,
            "weapon": combatant.weapon.spec.name,
            "ammo": combatant.weapon.ammo,
            "mag_size": combatant.weapon.spec.mag_size,
            "stance": body.stance,
            "sprinting": body.sprinting,
            "in_air": body.in_air,
            "shots_fired": combatant.shots_fired,
            "bullets_fired": combatant.bullets_fired,
            "hits": combatant.hits,
            "headshots": combatant.headshots,
            "damage_dealt": combatant.damage_dealt,
        }

    def perception_snapshot(self) -> dict[str, Any]:
        """Perception state of both fighters, for the UI's 'what does the AI see' box."""
        if not self._combatants:
            return {}
        return {
            "vision_mode": self.vision_mode,
            "field_of_view_degrees": VISION_FOV_DEGREES,
            "player": self.enemy_view(0),
            "opponent": self.enemy_view(1),
        }

    def get_snapshot(self) -> dict[str, Any]:
        """Return JSON-friendly world state for the browser renderer."""
        agents = [self._combatant_summary(index) for index in range(2)]
        for index, agent in enumerate(agents):
            agent["agent"] = index + 1
        return {
            "map": serialize_map(self.arena_map),
            "agents": agents,
            "trails": list(self.bullet_trails[-60:]),
            "elapsed": self.elapsed,
            "frame": self.physics_frames,
            "episode": self.episode_index,
            "done": self._done,
        }

    def _episode_summary(self) -> dict[str, Any]:
        first, second = self._combatants
        # At the time limit, the fighter with more health wins; a tie is a draw.
        if first.body.alive and second.body.alive:
            win = float(first.body.hp > second.body.hp)
            draw = bool(math.isclose(first.body.hp, second.body.hp, abs_tol=1e-6))
        else:
            win = float(first.body.alive and not second.body.alive)
            draw = bool(not first.body.alive and not second.body.alive)
        first_accuracy = first.hits / max(1, first.bullets_fired)
        headshot_pct = first.headshots / max(1, first.hits)
        if win:
            kill_x, kill_y = second.body.x, second.body.y
            death_x, death_y = math.nan, math.nan
        elif not first.body.alive:
            kill_x, kill_y = math.nan, math.nan
            death_x, death_y = first.body.x, first.body.y
        else:
            kill_x, kill_y = math.nan, math.nan
            death_x, death_y = math.nan, math.nan
        metrics = {
            "win": win,
            "draw": draw,
            # ``ttk`` is only a real time-to-kill when the opponent actually died.
            # On a time-limit decision (or a defeat/draw) the match clock is logged
            # instead, so analytics must filter on this flag.
            "killed": bool(not second.body.alive),
            "ttk": self.elapsed,
            "distance": self._distance_between(),
            "shots_fired": first.shots_fired,
            "bullets_fired": first.bullets_fired,
            "hits": first.hits,
            "headshots": first.headshots,
            "accuracy": first_accuracy,
            "headshot_pct": headshot_pct,
            "damage_dealt": first.damage_dealt,
            "damage_taken": first.damage_taken,
            "avg_kill_distance": float(np.mean(first.kill_distances)) if first.kill_distances else math.nan,
            "weapon": first.weapon.spec.name,
            "opponent_weapon": second.weapon.spec.name,
            "map": self.arena_map.name,
            "death_x": death_x,
            "death_y": death_y,
            "kill_x": kill_x,
            "kill_y": kill_y,
            "episode": self.episode_index,
            "vision_mode": self.vision_mode,
            "player_blind_steps": first.blind_steps,
            "player_visible_steps": first.visible_steps,
            "player_blind_ratio": first.blind_steps / max(1, first.blind_steps + first.visible_steps),
        }
        return {
            "win": win,
            "draw": draw,
            "ttk": self.elapsed,
            "episode_metrics": metrics,
            "terminal_observation": self._observation(0),
        }

    def render(self) -> dict[str, Any]:
        """Return a frame snapshot; no graphical window is opened."""
        return self.get_snapshot()

    def close(self) -> None:
        self._closed = True

"""Translate RLBot v5 game packets into RLGym ``GameState`` objects (and actions back).

This is what lets a policy trained in RocketSim drive in the real game: the
bot rebuilds exactly the observation it saw during training from each packet.
Field access is duck-typed, so tests can feed plain objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject
from rlgym.rocket_league.common_values import (
    BIG_PAD_RECHARGE_SECONDS,
    BOOST_LOCATIONS,
    DOUBLEJUMP_MAX_DELAY,
    OCTANE,
    SMALL_PAD_RECHARGE_SECONDS,
)

# RLBot AirState values.
ON_GROUND, JUMPING, DOUBLE_JUMPING, DODGING, IN_AIR = range(5)

_PAD_POSITIONS = np.asarray(BOOST_LOCATIONS, dtype=np.float32)
_BIG_PAD = _PAD_POSITIONS[:, 2] > 72  # big pads sit at z=73, small ones at z=70


def _vec(v: Any) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def physics_object(physics: Any) -> PhysicsObject:
    obj = PhysicsObject()
    obj.position = _vec(physics.location)
    obj.linear_velocity = _vec(physics.velocity)
    obj.angular_velocity = _vec(physics.angular_velocity)
    rotation = physics.rotation
    obj.euler_angles = np.array([rotation.pitch, rotation.yaw, rotation.roll], dtype=np.float32)
    return obj


def pad_index_map(field_pads: list[Any]) -> list[int | None]:
    """For each RLBot pad, the index of the nearest RLGym pad (None if nothing is close)."""
    mapping: list[int | None] = []
    for pad in field_pads:
        location = _vec(pad.location)
        distances = np.linalg.norm(_PAD_POSITIONS[:, :2] - location[:2], axis=1)
        nearest = int(np.argmin(distances))
        mapping.append(nearest if distances[nearest] < 250 else None)
    return mapping


def _flag(value: Any) -> int:
    return int(getattr(value, "value", value))


class PacketConverter:
    """Stateful only for ball touches (counted from changes of ``latest_touch``)."""

    def __init__(self, field_pads: list[Any] | None = None):
        self.pad_map = pad_index_map(field_pads or [])
        self._last_touch: dict[int, float] = {}

    @staticmethod
    def agent_id(index: int) -> str:
        return f"player-{index}"

    def boost_timers(self, pads: list[Any]) -> np.ndarray:
        timers = np.zeros(len(_PAD_POSITIONS), dtype=np.float32)
        for state, index in zip(pads, self.pad_map, strict=False):
            if index is None or state.is_active:
                continue
            cooldown = BIG_PAD_RECHARGE_SECONDS if _BIG_PAD[index] else SMALL_PAD_RECHARGE_SECONDS
            timers[index] = max(0.0, cooldown - float(state.timer))
        return timers

    def car(self, index: int, player: Any) -> Car:
        car = Car()
        air_state = _flag(player.air_state)
        on_ground = air_state == ON_GROUND
        controls = player.last_input
        car.team_num = int(player.team)
        car.hitbox_type = OCTANE
        car.bump_victim_id = None
        touch = player.latest_touch
        touches = 0
        if touch is not None:
            previous = self._last_touch.get(index)
            if previous is not None and touch.game_seconds > previous:
                touches = 1
            self._last_touch[index] = float(touch.game_seconds)
        car.ball_touches = touches
        car.demo_respawn_timer = max(0.0, float(player.demolished_timeout))
        car.wheels_with_contact = (on_ground,) * 4
        car.supersonic_time = 0.5 if player.is_supersonic else 0.0
        car.boost_amount = float(player.boost)
        car.boost_active_time = 0.05 if controls.boost and player.boost > 0 else 0.0
        car.handbrake = 1.0 if controls.handbrake else 0.0
        car.is_jumping = air_state == JUMPING
        car.has_jumped = bool(player.has_jumped)
        car.is_holding_jump = bool(controls.jump)
        car.jump_time = 0.0
        car.has_flipped = bool(player.has_dodged)
        car.has_double_jumped = bool(player.has_double_jumped)
        if on_ground or not player.has_jumped or car.is_jumping:
            car.air_time_since_jump = 0.0
        elif player.dodge_timeout >= 0:
            car.air_time_since_jump = max(0.0, DOUBLEJUMP_MAX_DELAY - float(player.dodge_timeout))
        else:
            car.air_time_since_jump = DOUBLEJUMP_MAX_DELAY
        car.flip_time = float(player.dodge_elapsed) if player.has_dodged else 0.0
        car.flip_torque = np.zeros(3, dtype=np.float32)
        car.is_autoflipping = False
        car.autoflip_timer = 0.0
        car.autoflip_direction = 1.0
        car.physics = physics_object(player.physics)
        car._inverted_physics = None
        return car

    def convert(self, packet: Any) -> GameState:
        state = GameState()
        state.tick_count = int(getattr(packet.match_info, "frame_num", 0))
        state.goal_scored = False
        config = GameConfig()
        config.gravity = 1.0
        config.boost_consumption = 1.0
        config.dodge_deadzone = 0.5
        state.config = config
        state.cars = {self.agent_id(i): self.car(i, p) for i, p in enumerate(packet.players)}
        if packet.balls:
            state.ball = physics_object(packet.balls[0].physics)
        else:  # between goals there can be no ball
            ball = PhysicsObject()
            ball.position = np.array([0, 0, 93], dtype=np.float32)
            ball.linear_velocity = np.zeros(3, dtype=np.float32)
            ball.angular_velocity = np.zeros(3, dtype=np.float32)
            ball.rotation_mtx = np.eye(3, dtype=np.float32)
            state.ball = ball
        state._inverted_ball = None
        state.boost_pad_timers = self.boost_timers(list(packet.boost_pads))
        state._inverted_boost_pad_timers = None
        return state


def controls_from_action(row: np.ndarray) -> dict[str, Any]:
    """A LookupTableAction row [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]."""
    throttle, steer, pitch, yaw, roll, jump, boost, handbrake = (float(v) for v in row)
    return {
        "throttle": throttle,
        "steer": steer,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "jump": jump > 0,
        "boost": boost > 0,
        "handbrake": handbrake > 0,
    }

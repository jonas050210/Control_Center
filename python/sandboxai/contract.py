"""Observation/Action contract description and the external-adapter boundary.

This module is documentation-as-code: it mirrors, in Python, the same
Observation/Action contract implemented in Godot by
``scripts/core/observation.gd`` and ``scripts/core/action.gd``. It exists so
a future adapter for a *different* game/engine (the planned external Roblox
Player adapter) has a single, explicit, versioned target to implement
against instead of reverse-engineering the Godot bridge protocol.

Nothing in this file talks to Roblox, opens a network socket, or depends on
any paid/cloud service. ``GameAdapter`` below is an abstract interface only:
a future concrete implementation (e.g. ``RobloxPlayerAdapter``) would live in
a separate module and translate a *real* Roblox game/session into this same
observation/action shape. Implementing that translation is explicitly out of
scope for this milestone — see docs/ROBLOX_ADAPTER.md.

IMPORTANT: keep ``OBSERVATION_FIELD_COUNT`` and ``OBSERVATION_SPEC`` in sync
with ``Observation.FIELD_COUNT`` / the field table documented at the top of
scripts/core/observation.gd whenever that file changes. There is no build
step that generates one from the other (they are two different languages),
so this is a manual, but small and explicit, contract to maintain.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ObservationField:
    """One (or a small fixed group of) observation vector entries."""

    index: int
    width: int
    name: str
    description: str
    normalization: str


# Keep this list's total width equal to OBSERVATION_FIELD_COUNT and each
# field's `index` contiguous — a test asserts both. Order matches
# Observation.to_array() in scripts/core/observation.gd exactly.
OBSERVATION_SPEC: tuple[ObservationField, ...] = (
    ObservationField(0, 3, "agent_position_norm", "Agent world position (x,y,z)", "divided by arena half-extent (x,z) / wall height (y)"),
    ObservationField(3, 3, "agent_velocity_norm", "Agent velocity (x,y,z)", "divided by agent move speed"),
    ObservationField(6, 3, "agent_forward", "Agent look/forward unit vector (x,y,z)", "unit vector, already in [-1, 1]"),
    ObservationField(9, 1, "agent_health_norm", "Agent health", "health / max_health, in [0, 1]"),
    ObservationField(10, 3, "primary_enemy_relative_position_norm", "Nearest alive enemy position relative to the agent (x,y,z)", "divided by max arena diagonal distance"),
    ObservationField(13, 1, "primary_enemy_distance_norm", "Distance to the nearest alive enemy", "divided by max arena diagonal distance, in [0, 1]"),
    ObservationField(14, 1, "primary_enemy_health_norm", "Nearest alive enemy health", "health / max_health, in [0, 1]"),
    ObservationField(15, 1, "weapon_ready", "Whether the weapon can fire this tick", "boolean 0/1"),
    ObservationField(16, 1, "in_combat", "Whether the primary enemy is alive and within weapon range", "boolean 0/1"),
    ObservationField(17, 1, "primary_enemy_bearing_norm", "Signed horizontal aim offset to the primary enemy", "angle / 180 degrees, in [-1, 1]; 0 = dead-center"),
    ObservationField(18, 1, "alive_enemy_count_norm", "How many configured enemies are currently alive", "alive_count / total_configured_enemy_count, in [0, 1]"),
    ObservationField(19, 3, "secondary_enemy_relative_position_norm", "2nd-nearest alive enemy position relative to the agent (x,y,z)", "divided by max arena diagonal distance; zero if absent"),
    ObservationField(22, 1, "secondary_enemy_distance_norm", "Distance to the 2nd-nearest alive enemy", "divided by max arena diagonal distance; 1.0 (max) if absent"),
    ObservationField(23, 1, "secondary_enemy_bearing_norm", "Signed horizontal aim offset to the 2nd-nearest alive enemy", "angle / 180 degrees, in [-1, 1]; 0 if absent"),
    ObservationField(24, 1, "secondary_enemy_health_norm", "2nd-nearest alive enemy health", "health / max_health; 0 if absent"),
    ObservationField(25, 1, "secondary_enemy_alive", "Whether a 2nd enemy is currently alive/tracked", "boolean 0/1"),
    ObservationField(26, 3, "tertiary_enemy_relative_position_norm", "3rd-nearest alive enemy position relative to the agent (x,y,z)", "divided by max arena diagonal distance; zero if absent"),
    ObservationField(29, 1, "tertiary_enemy_distance_norm", "Distance to the 3rd-nearest alive enemy", "divided by max arena diagonal distance; 1.0 (max) if absent"),
    ObservationField(30, 1, "tertiary_enemy_bearing_norm", "Signed horizontal aim offset to the 3rd-nearest alive enemy", "angle / 180 degrees, in [-1, 1]; 0 if absent"),
    ObservationField(31, 1, "tertiary_enemy_health_norm", "3rd-nearest alive enemy health", "health / max_health; 0 if absent"),
    ObservationField(32, 1, "tertiary_enemy_alive", "Whether a 3rd enemy is currently alive/tracked", "boolean 0/1"),
)

OBSERVATION_FIELD_COUNT: int = sum(field.width for field in OBSERVATION_SPEC)
OBSERVATION_LOW: float = -1.0
OBSERVATION_HIGH: float = 1.0
## Enemies beyond this rank still exist and affect reward/simulation, but are
## not individually reported in the observation vector.
OBSERVATION_MAX_TRACKED_ENEMIES: int = 3


@dataclass(frozen=True)
class ActionField:
    index: int
    name: str
    cardinality: int
    description: str


# MultiDiscrete([3, 3, 3, 3, 2]); values are shifted by +1 relative to the
# canonical -1/0/1 Godot Action fields (see Action.from_multidiscrete()).
ACTION_SPEC: tuple[ActionField, ...] = (
    ActionField(0, "move_axis", 3, "Forward/back: 0=backward, 1=idle, 2=forward"),
    ActionField(1, "strafe_axis", 3, "Left/right strafe: 0=left, 1=idle, 2=right"),
    ActionField(2, "look_yaw_axis", 3, "Turn: 0=left, 1=idle, 2=right"),
    ActionField(3, "look_pitch_axis", 3, "Look: 0=down, 1=idle, 2=up"),
    ActionField(4, "shoot", 2, "Trigger: 0=not firing, 1=firing"),
)
ACTION_NVEC: tuple[int, ...] = tuple(field.cardinality for field in ACTION_SPEC)


def validate_observation_spec() -> None:
    """Raises AssertionError if OBSERVATION_SPEC is internally inconsistent."""
    expected_index = 0
    for field in OBSERVATION_SPEC:
        assert field.index == expected_index, (
            f"observation field {field.name!r} starts at {field.index}, expected {expected_index}"
        )
        assert field.width > 0
        expected_index += field.width
    assert expected_index == OBSERVATION_FIELD_COUNT


class GameAdapter(ABC):
    """Abstract boundary a future external game adapter (e.g. Roblox) implements.

    This is a *hook*, not an implementation. It exists so that:
      1. The observation/action semantics used by PPO never implicitly leak
         Godot-only concepts (Node references, scene-tree state, etc.).
      2. A future Roblox Player adapter can be developed and tested against
         this exact interface, independently of the Godot simulator, using
         the same trained policy without retraining the observation head.

    A concrete implementation MUST:
      - Only expose information a player in that position could reasonably
        perceive (own state, relative enemy positions/health/aliveness/
        direction) — no server-only/privileged state.
      - Produce a float32 vector of exactly OBSERVATION_FIELD_COUNT values,
        each within [OBSERVATION_LOW, OBSERVATION_HIGH].
      - Accept an action in the ACTION_SPEC MultiDiscrete shape (ACTION_NVEC).

    No concrete Roblox implementation exists yet. Do not claim Roblox
    integration is implemented until a subclass actually connects to a real
    Roblox session and this docstring is updated to link to it.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> Sequence[float]:
        """Starts a new episode and returns the first observation vector."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Sequence[int]) -> tuple[Sequence[float], float, bool, dict[str, Any]]:
        """Applies one action; returns (observation, reward, done, info)."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Releases any external resources (sockets, processes, handles)."""
        raise NotImplementedError

    @staticmethod
    def observation_spec() -> tuple[ObservationField, ...]:
        return OBSERVATION_SPEC

    @staticmethod
    def action_spec() -> tuple[ActionField, ...]:
        return ACTION_SPEC

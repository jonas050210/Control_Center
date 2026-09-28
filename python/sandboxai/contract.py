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
``python/tests/test_contract.py`` (GodotSourceDriftTests) narrows the gap by
statically parsing the Godot sources and failing when the declared
constants or the ``to_array()`` index layout no longer match this module.
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
    # --- Contract v2 -------------------------------------------------------
    # Appended, never reordered: indices 0-32 keep their v1 meaning so a v1
    # policy's input weights still line up. Everything below is produced by
    # the perception layer (FOV/LOS gating, decaying memory, sound events)
    # and the vertical-movement layer, and is still strictly player-
    # perceivable information: no ground-truth coordinates of an enemy the
    # agent cannot currently see are ever exposed. When the primary enemy is
    # not visible, indices 10-17 describe the agent's remembered BELIEF and
    # index 37 tells the policy so.
    ObservationField(33, 1, "agent_on_ground", "Whether the agent is standing on the floor or a box", "boolean 0/1"),
    ObservationField(34, 1, "agent_vertical_velocity_norm", "Agent vertical velocity", "divided by jump velocity, clamped to [-1, 1]"),
    ObservationField(35, 1, "agent_in_cover", "No living enemy currently has line of sight to the agent", "boolean 0/1"),
    ObservationField(36, 1, "agent_forward_clearance_norm", "Distance to the first sight-blocking surface straight ahead", "divided by vision range, in [0, 1]"),
    ObservationField(37, 1, "primary_enemy_visible", "Primary contact is a live sighting (else a decaying memory)", "boolean 0/1"),
    ObservationField(38, 1, "primary_enemy_in_fov", "Primary contact lies inside the agent's FOV cone", "boolean 0/1"),
    ObservationField(39, 1, "primary_enemy_los_clear", "Geometry does not occlude the primary contact", "boolean 0/1"),
    ObservationField(40, 1, "primary_enemy_elevation_norm", "Signed vertical angle from the agent's eye to the primary contact", "angle / 90 degrees, in [-1, 1]"),
    ObservationField(41, 1, "primary_enemy_info_age_norm", "Time since the primary contact information was acquired", "seconds / MEMORY_MAX_AGE, in [0, 1]"),
    ObservationField(42, 1, "primary_enemy_confidence", "Confidence in the primary contact's position", "exponentially decayed, in [0, 1]"),
    ObservationField(43, 1, "primary_enemy_source_visual", "Primary contact information came from vision", "boolean 0/1"),
    ObservationField(44, 1, "primary_enemy_source_sound", "Primary contact information came from hearing", "boolean 0/1"),
    ObservationField(45, 1, "secondary_enemy_visible", "Secondary contact is a live sighting", "boolean 0/1"),
    ObservationField(46, 1, "secondary_enemy_info_age_norm", "Age of the secondary contact information", "seconds / MEMORY_MAX_AGE, in [0, 1]"),
    ObservationField(47, 1, "secondary_enemy_elevation_norm", "Signed vertical angle to the secondary contact", "angle / 90 degrees, in [-1, 1]"),
    ObservationField(48, 1, "tertiary_enemy_visible", "Tertiary contact is a live sighting", "boolean 0/1"),
    ObservationField(49, 1, "tertiary_enemy_info_age_norm", "Age of the tertiary contact information", "seconds / MEMORY_MAX_AGE, in [0, 1]"),
    ObservationField(50, 1, "tertiary_enemy_elevation_norm", "Signed vertical angle to the tertiary contact", "angle / 90 degrees, in [-1, 1]"),
    ObservationField(51, 3, "last_sound_direction", "Perceived direction of the loudest audible event (x,y,z)", "unit vector with directional error applied; zero if silent"),
    ObservationField(54, 1, "last_sound_distance_norm", "Perceived distance of the loudest audible event", "divided by max arena diagonal distance, in [0, 1]"),
    ObservationField(55, 1, "last_sound_bearing_norm", "Signed horizontal offset to the loudest audible event", "angle / 180 degrees, in [-1, 1]"),
    ObservationField(56, 1, "last_sound_age_norm", "Age of the loudest audible event", "seconds / SOUND_EVENT_LIFETIME, in [0, 1]"),
    ObservationField(57, 1, "last_sound_loudness", "Perceived loudness after distance and occlusion attenuation", "in [0, 1]"),
    ObservationField(58, 1, "last_sound_category_norm", "Event category ordinal (footstep/jump/land/shot/impact/death/environment)", "ordinal / 6, in [0, 1]"),
    ObservationField(59, 1, "audible_event_count_norm", "How many events are audible this tick", "count / 8, clamped to [0, 1]"),
    ObservationField(60, 1, "nearest_obstacle_distance_norm", "Distance to the nearest piece of cover/geometry", "divided by max arena diagonal distance, in [0, 1]"),
    ObservationField(61, 1, "nearest_obstacle_bearing_norm", "Signed horizontal offset to the nearest obstacle", "angle / 180 degrees, in [-1, 1]"),
    ObservationField(62, 1, "visible_enemy_count_norm", "How many enemies are currently visible", "count / 8, clamped to [0, 1]"),
    ObservationField(63, 1, "remembered_enemy_count_norm", "How many contacts are remembered but not visible", "count / 8, clamped to [0, 1]"),
    ObservationField(64, 1, "corpse_count_norm", "How many corpses exist in the arena", "count / 8, clamped to [0, 1]"),
)

OBSERVATION_FIELD_COUNT: int = sum(field.width for field in OBSERVATION_SPEC)
OBSERVATION_LOW: float = -1.0
OBSERVATION_HIGH: float = 1.0
## Enemies beyond this rank still exist and affect reward/simulation, but are
## not individually reported in the observation vector.
OBSERVATION_MAX_TRACKED_ENEMIES: int = 3
## Length of the immutable v1 prefix of OBSERVATION_SPEC.
OBSERVATION_LEGACY_FIELD_COUNT: int = 33


# Semantic channels of the observation vector. An external adapter has to
# be able to produce each channel independently and honestly; when a target
# game cannot supply one, the adapter must emit that channel's neutral
# "no information" encoding (zeros, with the corresponding `*_visible` /
# `*_confidence` flags at 0) rather than substituting privileged data.
#
# Keys are channel names; values are the OBSERVATION_SPEC field names in
# that channel. Every field belongs to exactly one channel (asserted by
# validate_observation_spec()).
OBSERVATION_GROUPS: dict[str, tuple[str, ...]] = {
    "self_state": (
        "agent_position_norm",
        "agent_forward",
        "agent_health_norm",
    ),
    "movement": (
        "agent_velocity_norm",
        "agent_on_ground",
        "agent_vertical_velocity_norm",
    ),
    "combat": (
        "weapon_ready",
        "in_combat",
    ),
    "targets": (
        "primary_enemy_relative_position_norm",
        "primary_enemy_distance_norm",
        "primary_enemy_health_norm",
        "primary_enemy_bearing_norm",
        "alive_enemy_count_norm",
        "secondary_enemy_relative_position_norm",
        "secondary_enemy_distance_norm",
        "secondary_enemy_bearing_norm",
        "secondary_enemy_health_norm",
        "secondary_enemy_alive",
        "tertiary_enemy_relative_position_norm",
        "tertiary_enemy_distance_norm",
        "tertiary_enemy_bearing_norm",
        "tertiary_enemy_health_norm",
        "tertiary_enemy_alive",
    ),
    "perception": (
        "primary_enemy_visible",
        "primary_enemy_in_fov",
        "primary_enemy_los_clear",
        "primary_enemy_elevation_norm",
        "secondary_enemy_visible",
        "secondary_enemy_elevation_norm",
        "tertiary_enemy_visible",
        "tertiary_enemy_elevation_norm",
        "visible_enemy_count_norm",
        "agent_in_cover",
        "agent_forward_clearance_norm",
    ),
    "memory": (
        "primary_enemy_info_age_norm",
        "primary_enemy_confidence",
        "primary_enemy_source_visual",
        "primary_enemy_source_sound",
        "secondary_enemy_info_age_norm",
        "tertiary_enemy_info_age_norm",
        "remembered_enemy_count_norm",
    ),
    "sound": (
        "last_sound_direction",
        "last_sound_distance_norm",
        "last_sound_bearing_norm",
        "last_sound_age_norm",
        "last_sound_loudness",
        "last_sound_category_norm",
        "audible_event_count_norm",
    ),
    "world": (
        "nearest_obstacle_distance_norm",
        "nearest_obstacle_bearing_norm",
        "corpse_count_norm",
    ),
}


@dataclass(frozen=True)
class ActionField:
    index: int
    name: str
    cardinality: int
    description: str


# MultiDiscrete([3, 3, 3, 3, 2, 2]); values are shifted by +1 relative to the
# canonical -1/0/1 Godot Action fields (see Action.from_multidiscrete()).
# `jump` was APPENDED in contract v2 rather than inserted, so the first five
# components keep their v1 meaning. Godot still accepts a 5-component action
# (it simply never jumps), which keeps recorded human demonstrations valid.
ACTION_SPEC: tuple[ActionField, ...] = (
    ActionField(0, "move_axis", 3, "Forward/back: 0=backward, 1=idle, 2=forward"),
    ActionField(1, "strafe_axis", 3, "Left/right strafe: 0=left, 1=idle, 2=right"),
    ActionField(2, "look_yaw_axis", 3, "Turn: 0=left, 1=idle, 2=right"),
    ActionField(3, "look_pitch_axis", 3, "Look: 0=down, 1=idle, 2=up"),
    ActionField(4, "shoot", 2, "Trigger: 0=not firing, 1=firing"),
    ActionField(5, "jump", 2, "Jump: 0=grounded, 1=jump if on the ground"),
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

    # Every field belongs to exactly one adapter channel. This is what makes
    # OBSERVATION_GROUPS a usable implementation checklist rather than
    # decorative documentation.
    grouped: list[str] = [name for names in OBSERVATION_GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped)), "a field appears in more than one observation group"
    declared = {field.name for field in OBSERVATION_SPEC}
    missing = declared - set(grouped)
    unknown = set(grouped) - declared
    assert not missing, f"observation fields not assigned to a group: {sorted(missing)}"
    assert not unknown, f"OBSERVATION_GROUPS references unknown fields: {sorted(unknown)}"


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

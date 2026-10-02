"""Local Godot/Python observation and action contract description.

This module is documentation-as-code: it mirrors, in Python, the same
Observation/Action contract implemented in Godot by
``scripts/core/observation.gd`` and ``scripts/core/action.gd``. It gives the
local Godot and Python layers one explicit, versioned target instead of
reverse-engineering the bridge protocol.

Nothing in this file talks to Roblox, opens a network socket, or depends on
any paid/cloud service. It describes the local simulator contract only.

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

from collections.abc import Sequence
from dataclasses import dataclass

# Semantic compatibility boundary shared by training, datasets, checkpoints,
# evaluation, and replay provenance. Shape checks alone cannot detect reordered
# or reinterpreted fields.
CONTRACT_VERSION: int = 4


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
    ObservationField(
        0,
        3,
        "agent_position_norm",
        "Agent world position (x,y,z)",
        "divided by arena half-extent (x,z) / wall height (y)",
    ),
    ObservationField(
        3, 3, "agent_velocity_norm", "Agent velocity (x,y,z)", "divided by agent move speed"
    ),
    ObservationField(
        6,
        3,
        "agent_forward",
        "Agent look/forward unit vector (x,y,z)",
        "unit vector, already in [-1, 1]",
    ),
    ObservationField(9, 1, "agent_health_norm", "Agent health", "health / max_health, in [0, 1]"),
    ObservationField(
        10,
        3,
        "primary_enemy_relative_position_norm",
        "Nearest alive enemy position relative to the agent (x,y,z)",
        "divided by max arena diagonal distance",
    ),
    ObservationField(
        13,
        1,
        "primary_enemy_distance_norm",
        "Distance to the nearest alive enemy",
        "divided by max arena diagonal distance, in [0, 1]",
    ),
    ObservationField(
        14,
        1,
        "primary_enemy_health_norm",
        "Nearest alive enemy health",
        "health / max_health, in [0, 1]",
    ),
    ObservationField(15, 1, "weapon_ready", "Whether the weapon can fire this tick", "boolean 0/1"),
    ObservationField(
        16,
        1,
        "in_combat",
        "Whether the primary enemy is alive and within weapon range",
        "boolean 0/1",
    ),
    ObservationField(
        17,
        1,
        "primary_enemy_bearing_norm",
        "Signed horizontal aim offset to the primary enemy",
        "angle / 180 degrees, in [-1, 1]; 0 = dead-center",
    ),
    ObservationField(
        18,
        1,
        "alive_enemy_count_norm",
        "How many configured enemies are currently alive",
        "alive_count / total_configured_enemy_count, in [0, 1]",
    ),
    ObservationField(
        19,
        3,
        "secondary_enemy_relative_position_norm",
        "2nd-nearest alive enemy position relative to the agent (x,y,z)",
        "divided by max arena diagonal distance; zero if absent",
    ),
    ObservationField(
        22,
        1,
        "secondary_enemy_distance_norm",
        "Distance to the 2nd-nearest alive enemy",
        "divided by max arena diagonal distance; 1.0 (max) if absent",
    ),
    ObservationField(
        23,
        1,
        "secondary_enemy_bearing_norm",
        "Signed horizontal aim offset to the 2nd-nearest alive enemy",
        "angle / 180 degrees, in [-1, 1]; 0 if absent",
    ),
    ObservationField(
        24,
        1,
        "secondary_enemy_health_norm",
        "2nd-nearest alive enemy health",
        "health / max_health; 0 if absent",
    ),
    ObservationField(
        25,
        1,
        "secondary_enemy_alive",
        "Whether a 2nd enemy is currently alive/tracked",
        "boolean 0/1",
    ),
    ObservationField(
        26,
        3,
        "tertiary_enemy_relative_position_norm",
        "3rd-nearest alive enemy position relative to the agent (x,y,z)",
        "divided by max arena diagonal distance; zero if absent",
    ),
    ObservationField(
        29,
        1,
        "tertiary_enemy_distance_norm",
        "Distance to the 3rd-nearest alive enemy",
        "divided by max arena diagonal distance; 1.0 (max) if absent",
    ),
    ObservationField(
        30,
        1,
        "tertiary_enemy_bearing_norm",
        "Signed horizontal aim offset to the 3rd-nearest alive enemy",
        "angle / 180 degrees, in [-1, 1]; 0 if absent",
    ),
    ObservationField(
        31,
        1,
        "tertiary_enemy_health_norm",
        "3rd-nearest alive enemy health",
        "health / max_health; 0 if absent",
    ),
    ObservationField(
        32,
        1,
        "tertiary_enemy_alive",
        "Whether a 3rd enemy is currently alive/tracked",
        "boolean 0/1",
    ),
    # --- Contract v2 -------------------------------------------------------
    # Appended, never reordered: indices 0-32 keep their v1 meaning so a v1
    # policy's input weights still line up. Everything below is produced by
    # the perception layer (FOV/LOS gating, decaying memory, sound events)
    # and the vertical-movement layer, and is still strictly player-
    # perceivable information: no ground-truth coordinates of an enemy the
    # agent cannot currently see are ever exposed. When the primary enemy is
    # not visible, indices 10-17 describe the agent's remembered BELIEF and
    # index 37 tells the policy so.
    ObservationField(
        33,
        1,
        "agent_on_ground",
        "Whether the agent is standing on the floor or a box",
        "boolean 0/1",
    ),
    ObservationField(
        34,
        1,
        "agent_vertical_velocity_norm",
        "Agent vertical velocity",
        "divided by jump velocity, clamped to [-1, 1]",
    ),
    ObservationField(
        35,
        1,
        "agent_in_cover",
        "No living enemy currently has line of sight to the agent",
        "boolean 0/1",
    ),
    ObservationField(
        36,
        1,
        "agent_forward_clearance_norm",
        "Distance to the first sight-blocking surface straight ahead",
        "divided by vision range, in [0, 1]",
    ),
    ObservationField(
        37,
        1,
        "primary_enemy_visible",
        "Primary contact is a live sighting (else a decaying memory)",
        "boolean 0/1",
    ),
    ObservationField(
        38,
        1,
        "primary_enemy_in_fov",
        "Primary contact lies inside the agent's FOV cone",
        "boolean 0/1",
    ),
    ObservationField(
        39,
        1,
        "primary_enemy_los_clear",
        "Geometry does not occlude the primary contact",
        "boolean 0/1",
    ),
    ObservationField(
        40,
        1,
        "primary_enemy_elevation_norm",
        "Signed vertical angle from the agent's eye to the primary contact",
        "angle / 90 degrees, in [-1, 1]",
    ),
    ObservationField(
        41,
        1,
        "primary_enemy_info_age_norm",
        "Time since the primary contact information was acquired",
        "seconds / MEMORY_MAX_AGE, in [0, 1]",
    ),
    ObservationField(
        42,
        1,
        "primary_enemy_confidence",
        "Confidence in the primary contact's position",
        "exponentially decayed, in [0, 1]",
    ),
    ObservationField(
        43,
        1,
        "primary_enemy_source_visual",
        "Primary contact information came from vision",
        "boolean 0/1",
    ),
    ObservationField(
        44,
        1,
        "primary_enemy_source_sound",
        "Primary contact information came from hearing",
        "boolean 0/1",
    ),
    ObservationField(
        45, 1, "secondary_enemy_visible", "Secondary contact is a live sighting", "boolean 0/1"
    ),
    ObservationField(
        46,
        1,
        "secondary_enemy_info_age_norm",
        "Age of the secondary contact information",
        "seconds / MEMORY_MAX_AGE, in [0, 1]",
    ),
    ObservationField(
        47,
        1,
        "secondary_enemy_elevation_norm",
        "Signed vertical angle to the secondary contact",
        "angle / 90 degrees, in [-1, 1]",
    ),
    ObservationField(
        48, 1, "tertiary_enemy_visible", "Tertiary contact is a live sighting", "boolean 0/1"
    ),
    ObservationField(
        49,
        1,
        "tertiary_enemy_info_age_norm",
        "Age of the tertiary contact information",
        "seconds / MEMORY_MAX_AGE, in [0, 1]",
    ),
    ObservationField(
        50,
        1,
        "tertiary_enemy_elevation_norm",
        "Signed vertical angle to the tertiary contact",
        "angle / 90 degrees, in [-1, 1]",
    ),
    ObservationField(
        51,
        3,
        "last_sound_direction",
        "Perceived direction of the loudest audible event (x,y,z)",
        "unit vector with directional error applied; zero if silent",
    ),
    ObservationField(
        54,
        1,
        "last_sound_distance_norm",
        "Perceived distance of the loudest audible event",
        "divided by max arena diagonal distance, in [0, 1]",
    ),
    ObservationField(
        55,
        1,
        "last_sound_bearing_norm",
        "Signed horizontal offset to the loudest audible event",
        "angle / 180 degrees, in [-1, 1]",
    ),
    ObservationField(
        56,
        1,
        "last_sound_age_norm",
        "Age of the loudest audible event",
        "seconds / SOUND_EVENT_LIFETIME, in [0, 1]",
    ),
    ObservationField(
        57,
        1,
        "last_sound_loudness",
        "Perceived loudness after distance and occlusion attenuation",
        "in [0, 1]",
    ),
    ObservationField(
        58,
        1,
        "last_sound_category_norm",
        "Event category ordinal (footstep/jump/land/shot/impact/death/environment)",
        "ordinal / 6, in [0, 1]",
    ),
    ObservationField(
        59,
        1,
        "audible_event_count_norm",
        "How many events are audible this tick",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        60,
        1,
        "nearest_obstacle_distance_norm",
        "Distance to the nearest piece of cover/geometry",
        "divided by max arena diagonal distance, in [0, 1]",
    ),
    ObservationField(
        61,
        1,
        "nearest_obstacle_bearing_norm",
        "Signed horizontal offset to the nearest obstacle",
        "angle / 180 degrees, in [-1, 1]",
    ),
    ObservationField(
        62,
        1,
        "visible_enemy_count_norm",
        "How many enemies are currently visible",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        63,
        1,
        "remembered_enemy_count_norm",
        "How many contacts are remembered but not visible",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        64,
        1,
        "corpse_count_norm",
        "How many corpses exist in the arena",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        65,
        1,
        "local_illumination",
        "Perceived brightness where the agent stands",
        "in [0, 1]; the lighting MODE is never exposed",
    ),
    ObservationField(
        66,
        1,
        "overflow_contact_count_norm",
        "Contacts beyond the individually tracked slots",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        67,
        1,
        "overflow_visible_count_norm",
        "How many of those overflow contacts are visible",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        68,
        1,
        "overflow_mean_distance_norm",
        "Mean distance of the overflow contacts",
        "divided by max arena diagonal distance, in [0, 1]",
    ),
    ObservationField(
        69,
        1,
        "overflow_min_distance_norm",
        "Distance of the nearest overflow contact",
        "divided by max arena diagonal distance, in [0, 1]",
    ),
    ObservationField(
        70,
        1,
        "target_priority_norm",
        "Selection score of the primary contact",
        "score / TargetSelector.max_score(), in [0, 1]",
    ),
    ObservationField(
        71,
        1,
        "target_switch_recent",
        "The primary contact changed within TARGET_SWITCH_RECENT_WINDOW",
        "boolean 0/1",
    ),
    ObservationField(
        72,
        1,
        "second_sound_bearing_norm",
        "Signed horizontal offset to the second loudest audible event",
        "angle / 180 degrees, in [-1, 1]",
    ),
    ObservationField(
        73,
        1,
        "second_sound_loudness",
        "Perceived loudness of the second loudest audible event",
        "in [0, 1]",
    ),
    ObservationField(
        74,
        1,
        "sound_direction_error_norm",
        "How imprecisely the loudest event can be placed",
        "degrees / 90, in [0, 1]",
    ),
    ObservationField(
        75,
        1,
        "distinct_sound_source_count_norm",
        "How many separate directions noise is coming from",
        "count / 8, clamped to [0, 1]",
    ),
    ObservationField(
        76,
        1,
        "explored_fraction",
        "Share of the map the agent has actually observed",
        "in [0, 1]; 0 when map tracking is off",
    ),
    ObservationField(
        77,
        1,
        "current_area_known",
        "The agent has observed the cell it is standing in",
        "boolean 0/1",
    ),
    ObservationField(
        78,
        1,
        "time_since_area_visited_norm",
        "How long since the agent last stood here",
        "seconds / EXPLORATION_MAX_RECALL_AGE, in [0, 1]; 1 if never",
    ),
    ObservationField(
        79,
        1,
        "remembered_cover_distance_norm",
        "Distance to the nearest remembered cover",
        "divided by max arena diagonal distance, in [0, 1]; 0 if none remembered",
    ),
    ObservationField(
        80,
        1,
        "remembered_cover_bearing_norm",
        "Signed horizontal offset to the nearest remembered cover",
        "angle / 180 degrees, in [-1, 1]",
    ),
    ObservationField(
        81,
        1,
        "remembered_danger_distance_norm",
        "Distance to the nearest place the agent was hurt",
        "divided by max arena diagonal distance, in [0, 1]; 0 if none remembered",
    ),
    ObservationField(
        82,
        1,
        "remembered_danger_bearing_norm",
        "Signed horizontal offset to that place",
        "angle / 180 degrees, in [-1, 1]",
    ),
    ObservationField(
        83,
        1,
        "contact_uncertainty_norm",
        "Mean staleness of the contacts held from memory only",
        "1 - confidence, averaged, in [0, 1]",
    ),
    # Every bearing field (17, 23, 30, 55, 61, 72, 80, 82, 88, 95, 102) uses
    # ONE sign convention: positive is to the agent's right, the direction a
    # positive `Action.look_yaw_axis` turns ("+ = turn right to face it" for
    # index 17). The GDScript side implements it exactly once, in
    # `scripts/core/vector_math.gd` (`VectorMath.signed_bearing_*`), because
    # the enemy field once used a yaw difference while the world/sound/memory
    # queries took the opposite sign of `forward.cross(direction).y` - which
    # put an enemy and the crate next to it on opposite sides of the vector.
    # `python/tests/test_contract.py` fails if a second implementation appears.
    # --- contract v4: world objects the agent can see ---------------------
    # Three ranked slots (nearest visible first) plus the visible total. A
    # slot is only filled by geometry the agent could see from where it
    # stands (inside its FOV cone, within vision range, not hidden behind
    # another box); the fence around the arena is excluded. Empty slots
    # keep the neutral encoding: zero position, distance 1.0, bearing 0.0,
    # kind 0.0, visible 0.0. The slots never contain a hidden object, an
    # unseen layout or any part of the map the agent has not looked at.
    ObservationField(
        84,
        3,
        "object_1_relative_position_norm",
        "Nearest visible object's closest surface point, relative to the agent",
        "x/z divided by max arena diagonal distance, y by wall height, each in [-1, 1]",
    ),
    ObservationField(
        87,
        1,
        "object_1_distance_norm",
        "Distance to that object",
        "divided by max arena diagonal distance, in [0, 1]; 1 if no object is visible",
    ),
    ObservationField(
        88,
        1,
        "object_1_bearing_norm",
        "Signed horizontal offset to that object",
        "angle / 180 degrees, in [-1, 1]; 0 if no object is visible",
    ),
    ObservationField(
        89,
        1,
        "object_1_kind_norm",
        "Object kind as a normalized ordinal (crate, pillar, wall, ...)",
        "kind / (OBJECT_KIND_COUNT - 1), in [0, 1]; 0 if no object is visible",
    ),
    ObservationField(
        90,
        1,
        "object_1_visible",
        "1 when this slot holds a real sighting, 0 when it is empty",
        "0/1",
    ),
    ObservationField(
        91,
        3,
        "object_2_relative_position_norm",
        "Second-nearest visible object's closest surface point, relative to the agent",
        "x/z divided by max arena diagonal distance, y by wall height, each in [-1, 1]",
    ),
    ObservationField(
        94,
        1,
        "object_2_distance_norm",
        "Distance to that object",
        "divided by max arena diagonal distance, in [0, 1]; 1 if the slot is empty",
    ),
    ObservationField(
        95,
        1,
        "object_2_bearing_norm",
        "Signed horizontal offset to that object",
        "angle / 180 degrees, in [-1, 1]; 0 if the slot is empty",
    ),
    ObservationField(
        96,
        1,
        "object_2_kind_norm",
        "Object kind as a normalized ordinal",
        "kind / (OBJECT_KIND_COUNT - 1), in [0, 1]; 0 if the slot is empty",
    ),
    ObservationField(
        97,
        1,
        "object_2_visible",
        "1 when this slot holds a real sighting, 0 when it is empty",
        "0/1",
    ),
    ObservationField(
        98,
        3,
        "object_3_relative_position_norm",
        "Third-nearest visible object's closest surface point, relative to the agent",
        "x/z divided by max arena diagonal distance, y by wall height, each in [-1, 1]",
    ),
    ObservationField(
        101,
        1,
        "object_3_distance_norm",
        "Distance to that object",
        "divided by max arena diagonal distance, in [0, 1]; 1 if the slot is empty",
    ),
    ObservationField(
        102,
        1,
        "object_3_bearing_norm",
        "Signed horizontal offset to that object",
        "angle / 180 degrees, in [-1, 1]; 0 if the slot is empty",
    ),
    ObservationField(
        103,
        1,
        "object_3_kind_norm",
        "Object kind as a normalized ordinal",
        "kind / (OBJECT_KIND_COUNT - 1), in [0, 1]; 0 if the slot is empty",
    ),
    ObservationField(
        104,
        1,
        "object_3_visible",
        "1 when this slot holds a real sighting, 0 when it is empty",
        "0/1",
    ),
    ObservationField(
        105,
        1,
        "visible_object_count_norm",
        "How many objects are currently visible (capped like every count field)",
        "count / COUNT_NORMALIZER, saturated at 1.0",
    ),
)

## The engine build this contract is implemented and tested against.
##
## Single source of truth for the version string: project.godot, the CI
## workflows, the CLI, the README and every doc used to repeat it literally,
## and `python/tests/test_docs_consistency.py` now fails if any of them
## disagree with this constant. Bumping the engine is therefore one edit
## here plus whatever the drift test reports.
GODOT_VERSION: str = "4.7.2"

OBSERVATION_FIELD_COUNT: int = sum(field.width for field in OBSERVATION_SPEC)
OBSERVATION_LOW: float = -1.0
OBSERVATION_HIGH: float = 1.0
## Enemies beyond this rank still exist and affect reward/simulation, but are
## not individually reported in the observation vector.
OBSERVATION_MAX_TRACKED_ENEMIES: int = 3
## Length of the immutable v1 prefix of OBSERVATION_SPEC.
OBSERVATION_LEGACY_FIELD_COUNT: int = 33
## Length of the v2 prefix (everything before the conditions/exploration block).
OBSERVATION_V2_FIELD_COUNT: int = 65
## Length of the v3 prefix (everything before the object block of contract v4).
OBSERVATION_V3_FIELD_COUNT: int = 84
## Objects reported individually, nearest visible first (contract v4). Mirrors
## SandboxConfig.OBSERVATION_MAX_TRACKED_OBJECTS.
OBSERVATION_MAX_TRACKED_OBJECTS: int = 3
## Count fields (visible enemies, corpses, audible events, visible objects)
## are divided by this and clamped, so the vector never carries a raw count.
## Mirrors ``Observation.COUNT_NORMALIZER``; a drift test parses the GDScript
## constant, and the Control Center's Stats page converts back with it.
OBSERVATION_COUNT_NORMALIZER: int = 8
## ``Obstacle.Kind`` (``scripts/world/obstacle.gd``) in declaration order.
## ``object_k_kind_norm`` is this ordinal divided by ``len(...) - 1``; the order
## is a wire detail, so ``test_contract.py`` parses the GDScript enum and fails
## if the two drift apart. The fence (``BOUNDARY``) is never reported by the
## visibility query, but stays in the list because the ordinal must not shift.
OBJECT_KIND_NAMES: tuple[str, ...] = (
    "wall",
    "crate",
    "pillar",
    "low_cover",
    "high_cover",
    "platform",
    "boundary",
)


# Semantic channels of the local observation vector. Each channel has a
# defined neutral "no information" encoding (zeros, with the corresponding
# `*_visible` / `*_confidence` flags at 0), which keeps downstream consumers
# from treating unavailable perception as privileged information.
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
        "contact_uncertainty_norm",
    ),
    "objects": (
        "object_1_relative_position_norm",
        "object_1_distance_norm",
        "object_1_bearing_norm",
        "object_1_kind_norm",
        "object_1_visible",
        "object_2_relative_position_norm",
        "object_2_distance_norm",
        "object_2_bearing_norm",
        "object_2_kind_norm",
        "object_2_visible",
        "object_3_relative_position_norm",
        "object_3_distance_norm",
        "object_3_bearing_norm",
        "object_3_kind_norm",
        "object_3_visible",
        "visible_object_count_norm",
    ),
    "sound": (
        "second_sound_bearing_norm",
        "second_sound_loudness",
        "sound_direction_error_norm",
        "distinct_sound_source_count_norm",
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
    # Environmental conditions the agent can feel, never the label that
    # produced them.
    "conditions": ("local_illumination",),
    # Contacts beyond the individually tracked slots, described
    # statistically so the observation shape is independent of the enemy
    # count.
    "contacts": (
        "overflow_contact_count_norm",
        "overflow_visible_count_norm",
        "overflow_mean_distance_norm",
        "overflow_min_distance_norm",
    ),
    # Which contact the agent is treating as its primary, and how recently
    # that changed.
    "target": (
        "target_priority_norm",
        "target_switch_recent",
    ),
    # Map knowledge built by looking. Zeros mean "nothing known", which is
    # what an adapter without a spatial memory must emit.
    "exploration": (
        "explored_fraction",
        "current_area_known",
        "time_since_area_visited_norm",
        "remembered_cover_distance_norm",
        "remembered_cover_bearing_norm",
        "remembered_danger_distance_norm",
        "remembered_danger_bearing_norm",
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


## Name -> (start index, width). Built once from OBSERVATION_SPEC so that
## nothing downstream has to hardcode an integer offset into the vector.
## Hardcoded indices are how an "append-only" contract silently breaks:
## every consumer must look a field up by NAME.
OBSERVATION_INDEX: dict[str, tuple[int, int]] = {
    field.name: (field.index, field.width) for field in OBSERVATION_SPEC
}


def observation_index(name: str) -> int:
    """Start index of an observation field, by name."""
    try:
        return OBSERVATION_INDEX[name][0]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown observation field: {name!r}") from exc


def observation_value(observation: Sequence[float], name: str) -> float:
    """Reads one scalar observation field by name.

    Raises for a multi-value field (use ``observation_slice``) rather than
    silently returning only its first component.
    """
    index, width = OBSERVATION_INDEX[name]
    if width != 1:
        raise ValueError(f"{name!r} has width {width}; use observation_slice()")
    return float(observation[index])


def observation_slice(observation: Sequence[float], name: str) -> list[float]:
    """Reads a (possibly multi-value) observation field by name."""
    index, width = OBSERVATION_INDEX[name]
    return [float(value) for value in observation[index : index + width]]


def validate_observation_spec() -> None:
    """Raises ValueError if OBSERVATION_SPEC is internally inconsistent.

    This deliberately raises rather than asserting. ``python -O`` strips
    ``assert`` statements, and a consistency check that silently becomes
    a no-op under an interpreter flag is worse than no check at all - it
    still reads like a guarantee.
    """
    expected_index = 0
    for field in OBSERVATION_SPEC:
        if field.index != expected_index:
            raise ValueError(
                f"observation field {field.name!r} starts at {field.index}, "
                f"expected {expected_index}"
            )
        if field.width <= 0:
            raise ValueError(f"observation field {field.name!r} has width {field.width}")
        expected_index += field.width
    if expected_index != OBSERVATION_FIELD_COUNT:
        raise ValueError(
            f"OBSERVATION_SPEC covers {expected_index} values, "
            f"OBSERVATION_FIELD_COUNT says {OBSERVATION_FIELD_COUNT}"
        )

    # Every field belongs to exactly one observation group. This is what makes
    # OBSERVATION_GROUPS a usable implementation checklist rather than
    # decorative documentation.
    grouped: list[str] = [name for names in OBSERVATION_GROUPS.values() for name in names]
    if len(grouped) != len(set(grouped)):
        raise ValueError("a field appears in more than one observation group")
    declared = {field.name for field in OBSERVATION_SPEC}
    missing = declared - set(grouped)
    unknown = set(grouped) - declared
    if missing:
        raise ValueError(f"observation fields not assigned to a group: {sorted(missing)}")
    if unknown:
        raise ValueError(f"OBSERVATION_GROUPS references unknown fields: {sorted(unknown)}")

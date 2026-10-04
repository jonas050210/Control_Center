# gdlint:disable=max-file-lines
# Over the 1000-line budget on purpose. This file is the contract: one flat
# FIELD_SPEC table, the matching variables, the matching arr[i] assignments
# and the matching to_dict() keys have to be readable side by side, and the
# drift tests in python/tests/test_contract.py compare exactly those four
# lists in order. Splitting the v4 object slots (or any other version's
# fields) into a helper script would keep the checks passing while making
# the one thing a reviewer must verify - "does the vector still mean what
# the table says" - impossible to see without jumping between files.
## Observation
##
## Builds the structured, numeric, (mostly) normalized observation vector
## consumed by an RL policy. This is the ONLY observation mode implemented
## in milestone 1 (SandboxConfig.ObservationMode.STRUCTURED). The class is
## written so that a future RGB/screen mode or a human-input-derived mode
## can be added alongside it (see docs/ARCHITECTURE.md) without changing
## this struct's meaning for existing consumers.
##
## Multi-enemy contract (see docs/OBSERVATION_ACTION_CONTRACT.md for the
## full field-by-field table): indices [0-16] are the original single-enemy
## contract, unchanged, where "the enemy" means the PRIMARY target (nearest
## alive enemy, dead-target fallback if none are alive). Indices [17-32] are
## additive: the primary enemy's aim bearing, how many enemies are alive,
## and up to two more tracked enemies (2nd/3rd nearest alive) so a policy
## can react to multiple simultaneous threats instead of only ever seeing
## one. Only information a player standing in the agent's position could
## reasonably perceive is exposed (relative positions/directions/health/
## aliveness) — no privileged simulator-only state.
class_name Observation
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const AgentState = preload("res://scripts/agent/agent_state.gd")
const VectorMath = preload("res://scripts/core/vector_math.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

## Contract v4 = 84 (v3, unchanged) + 22 fields for the world OBJECTS the
## agent can see: three ranked slots (relative position, distance, bearing,
## kind, visible) plus how many objects are visible in total. Objects are
## the cover boxes, crates, pillars and platforms of the arena - the fence
## (Obstacle.Kind.BOUNDARY) is excluded, and a slot is only filled by an
## object the agent could actually see (inside the FOV cone, within vision
## range, not hidden behind other geometry). Nothing here is a hidden
## object, a whole map or a box the agent has never looked at; the policy
## gets the same three nearest visible pieces of cover a player would
## notice, which is what makes "duck behind this crate" a decision it can
## learn instead of guess.
##
## Contract v3 = 65 (v2, unchanged) + 19 fields for conditions, contact
## overflow, target selection, richer hearing and map knowledge. Indices
## [0-32] keep their exact v1 meaning and [0-64] their exact v2 meaning;
## everything new is strictly appended, so older weights still line up with
## the same semantics for the prefix they were trained on.
##
## Every v3 field is something the agent could work out for itself from
## what it perceived. There is no map id, no lighting mode, no enemy count,
## no hidden geometry: "it is dark HERE", "two more contacts I am not
## tracking individually", "I have seen 40% of this place".
const FIELD_COUNT: int = 126
## The v4 prefix length, for the same reason as the older prefixes: contract
## v5 appended the vision block (106-125) and changed nothing below 106.
const V4_FIELD_COUNT: int = 106
## The v2 prefix length, for the same reason as LEGACY_FIELD_COUNT.
const V2_FIELD_COUNT: int = 65
## The v1 prefix length, kept as a named constant because several tests and
## the Roblox adapter boundary assert the prefix is never reordered.
const LEGACY_FIELD_COUNT: int = 33
## Total number of enemies individually reported (primary + tracked extras).
const MAX_TRACKED_ENEMIES: int = SandboxConfig.OBSERVATION_MAX_TRACKED_ENEMIES
## Objects reported individually, nearest visible first (contract v4).
const MAX_TRACKED_OBJECTS: int = SandboxConfig.OBSERVATION_MAX_TRACKED_OBJECTS
## Saturation point for the count-style fields (visible/remembered enemies,
## corpses, audible events). Counts above this clamp to 1.0.
const COUNT_NORMALIZER: int = 8
## Mirrors SoundBus.Category cardinality. Duplicated as a plain int rather
## than preloading SoundBus here, because Observation must stay loadable
## without the perception layer (the Roblox adapter boundary depends on it).
const SOUND_CATEGORY_COUNT: int = 7

## Machine-readable layout of `to_array()` and the single Godot-side source
## of truth for observation field names/indices/groups.
##
## It exists so tooling (the Control Center's Observation Inspector, any
## future export/logging code) can label the raw vector WITHOUT hard-coding
## a second field list that silently rots when the contract changes. Adding
## or reordering a field means editing `to_array()`, this list, the field
## table in docs/OBSERVATION_ACTION_CONTRACT.md and
## python/sandboxai/contract.py together.
##
## Each entry is {"index": int, "width": int, "name": String, "group": String}.
## `name` matches python/sandboxai/contract.py OBSERVATION_SPEC exactly, and
## python/tests/test_contract.py statically compares the two lists so drift
## fails a test instead of producing mislabeled UI.
##
## This is a constant: building it costs nothing at runtime and nothing on
## the training hot path ever reads it (`to_array()` does not touch it).
const FIELD_SPEC: Array = [
	{"index": 0, "width": 3, "name": "agent_position_norm", "group": "agent"},
	{"index": 3, "width": 3, "name": "agent_velocity_norm", "group": "agent"},
	{"index": 6, "width": 3, "name": "agent_forward", "group": "agent"},
	{"index": 9, "width": 1, "name": "agent_health_norm", "group": "agent"},
	{
		"index": 10,
		"width": 3,
		"name": "primary_enemy_relative_position_norm",
		"group": "primary_enemy"
	},
	{"index": 13, "width": 1, "name": "primary_enemy_distance_norm", "group": "primary_enemy"},
	{"index": 14, "width": 1, "name": "primary_enemy_health_norm", "group": "primary_enemy"},
	{"index": 15, "width": 1, "name": "weapon_ready", "group": "weapon"},
	{"index": 16, "width": 1, "name": "in_combat", "group": "weapon"},
	{"index": 17, "width": 1, "name": "primary_enemy_bearing_norm", "group": "primary_enemy"},
	{"index": 18, "width": 1, "name": "alive_enemy_count_norm", "group": "world"},
	{
		"index": 19,
		"width": 3,
		"name": "secondary_enemy_relative_position_norm",
		"group": "secondary_enemy"
	},
	{"index": 22, "width": 1, "name": "secondary_enemy_distance_norm", "group": "secondary_enemy"},
	{"index": 23, "width": 1, "name": "secondary_enemy_bearing_norm", "group": "secondary_enemy"},
	{"index": 24, "width": 1, "name": "secondary_enemy_health_norm", "group": "secondary_enemy"},
	{"index": 25, "width": 1, "name": "secondary_enemy_alive", "group": "secondary_enemy"},
	{
		"index": 26,
		"width": 3,
		"name": "tertiary_enemy_relative_position_norm",
		"group": "tertiary_enemy"
	},
	{"index": 29, "width": 1, "name": "tertiary_enemy_distance_norm", "group": "tertiary_enemy"},
	{"index": 30, "width": 1, "name": "tertiary_enemy_bearing_norm", "group": "tertiary_enemy"},
	{"index": 31, "width": 1, "name": "tertiary_enemy_health_norm", "group": "tertiary_enemy"},
	{"index": 32, "width": 1, "name": "tertiary_enemy_alive", "group": "tertiary_enemy"},
	{"index": 33, "width": 1, "name": "agent_on_ground", "group": "agent"},
	{"index": 34, "width": 1, "name": "agent_vertical_velocity_norm", "group": "agent"},
	{"index": 35, "width": 1, "name": "agent_in_cover", "group": "agent"},
	{"index": 36, "width": 1, "name": "agent_forward_clearance_norm", "group": "agent"},
	{"index": 37, "width": 1, "name": "primary_enemy_visible", "group": "primary_enemy"},
	{"index": 38, "width": 1, "name": "primary_enemy_in_fov", "group": "primary_enemy"},
	{"index": 39, "width": 1, "name": "primary_enemy_los_clear", "group": "primary_enemy"},
	{"index": 40, "width": 1, "name": "primary_enemy_elevation_norm", "group": "primary_enemy"},
	{"index": 41, "width": 1, "name": "primary_enemy_info_age_norm", "group": "memory"},
	{"index": 42, "width": 1, "name": "primary_enemy_confidence", "group": "memory"},
	{"index": 43, "width": 1, "name": "primary_enemy_source_visual", "group": "memory"},
	{"index": 44, "width": 1, "name": "primary_enemy_source_sound", "group": "memory"},
	{"index": 45, "width": 1, "name": "secondary_enemy_visible", "group": "secondary_enemy"},
	{"index": 46, "width": 1, "name": "secondary_enemy_info_age_norm", "group": "memory"},
	{"index": 47, "width": 1, "name": "secondary_enemy_elevation_norm", "group": "secondary_enemy"},
	{"index": 48, "width": 1, "name": "tertiary_enemy_visible", "group": "tertiary_enemy"},
	{"index": 49, "width": 1, "name": "tertiary_enemy_info_age_norm", "group": "memory"},
	{"index": 50, "width": 1, "name": "tertiary_enemy_elevation_norm", "group": "tertiary_enemy"},
	{"index": 51, "width": 3, "name": "last_sound_direction", "group": "sound"},
	{"index": 54, "width": 1, "name": "last_sound_distance_norm", "group": "sound"},
	{"index": 55, "width": 1, "name": "last_sound_bearing_norm", "group": "sound"},
	{"index": 56, "width": 1, "name": "last_sound_age_norm", "group": "sound"},
	{"index": 57, "width": 1, "name": "last_sound_loudness", "group": "sound"},
	{"index": 58, "width": 1, "name": "last_sound_category_norm", "group": "sound"},
	{"index": 59, "width": 1, "name": "audible_event_count_norm", "group": "sound"},
	{"index": 60, "width": 1, "name": "nearest_obstacle_distance_norm", "group": "world"},
	{"index": 61, "width": 1, "name": "nearest_obstacle_bearing_norm", "group": "world"},
	{"index": 62, "width": 1, "name": "visible_enemy_count_norm", "group": "world"},
	{"index": 63, "width": 1, "name": "remembered_enemy_count_norm", "group": "memory"},
	{"index": 64, "width": 1, "name": "corpse_count_norm", "group": "world"},
	{"index": 65, "width": 1, "name": "local_illumination", "group": "conditions"},
	{"index": 66, "width": 1, "name": "overflow_contact_count_norm", "group": "contacts"},
	{"index": 67, "width": 1, "name": "overflow_visible_count_norm", "group": "contacts"},
	{"index": 68, "width": 1, "name": "overflow_mean_distance_norm", "group": "contacts"},
	{"index": 69, "width": 1, "name": "overflow_min_distance_norm", "group": "contacts"},
	{"index": 70, "width": 1, "name": "target_priority_norm", "group": "target"},
	{"index": 71, "width": 1, "name": "target_switch_recent", "group": "target"},
	{"index": 72, "width": 1, "name": "second_sound_bearing_norm", "group": "sound"},
	{"index": 73, "width": 1, "name": "second_sound_loudness", "group": "sound"},
	{"index": 74, "width": 1, "name": "sound_direction_error_norm", "group": "sound"},
	{"index": 75, "width": 1, "name": "distinct_sound_source_count_norm", "group": "sound"},
	{"index": 76, "width": 1, "name": "explored_fraction", "group": "exploration"},
	{"index": 77, "width": 1, "name": "current_area_known", "group": "exploration"},
	{"index": 78, "width": 1, "name": "time_since_area_visited_norm", "group": "exploration"},
	{"index": 79, "width": 1, "name": "remembered_cover_distance_norm", "group": "exploration"},
	{"index": 80, "width": 1, "name": "remembered_cover_bearing_norm", "group": "exploration"},
	{"index": 81, "width": 1, "name": "remembered_danger_distance_norm", "group": "exploration"},
	{"index": 82, "width": 1, "name": "remembered_danger_bearing_norm", "group": "exploration"},
	{"index": 83, "width": 1, "name": "contact_uncertainty_norm", "group": "memory"},
	{"index": 84, "width": 3, "name": "object_1_relative_position_norm", "group": "objects"},
	{"index": 87, "width": 1, "name": "object_1_distance_norm", "group": "objects"},
	{"index": 88, "width": 1, "name": "object_1_bearing_norm", "group": "objects"},
	{"index": 89, "width": 1, "name": "object_1_kind_norm", "group": "objects"},
	{"index": 90, "width": 1, "name": "object_1_visible", "group": "objects"},
	{"index": 91, "width": 3, "name": "object_2_relative_position_norm", "group": "objects"},
	{"index": 94, "width": 1, "name": "object_2_distance_norm", "group": "objects"},
	{"index": 95, "width": 1, "name": "object_2_bearing_norm", "group": "objects"},
	{"index": 96, "width": 1, "name": "object_2_kind_norm", "group": "objects"},
	{"index": 97, "width": 1, "name": "object_2_visible", "group": "objects"},
	{"index": 98, "width": 3, "name": "object_3_relative_position_norm", "group": "objects"},
	{"index": 101, "width": 1, "name": "object_3_distance_norm", "group": "objects"},
	{"index": 102, "width": 1, "name": "object_3_bearing_norm", "group": "objects"},
	{"index": 103, "width": 1, "name": "object_3_kind_norm", "group": "objects"},
	{"index": 104, "width": 1, "name": "object_3_visible", "group": "objects"},
	{"index": 105, "width": 1, "name": "visible_object_count_norm", "group": "objects"},
	{
		"index": 106,
		"width": 1,
		"name": "primary_enemy_screen_x",
		"group": "vision",
	},
	{
		"index": 107,
		"width": 1,
		"name": "primary_enemy_screen_y",
		"group": "vision",
	},
	{
		"index": 108,
		"width": 1,
		"name": "primary_enemy_screen_half_width",
		"group": "vision",
	},
	{
		"index": 109,
		"width": 1,
		"name": "primary_enemy_screen_half_height",
		"group": "vision",
	},
	{
		"index": 110,
		"width": 1,
		"name": "primary_enemy_exposure_fraction",
		"group": "vision",
	},
	{
		"index": 111,
		"width": 1,
		"name": "primary_enemy_illumination",
		"group": "vision",
	},
	{
		"index": 112,
		"width": 1,
		"name": "secondary_enemy_screen_x",
		"group": "vision",
	},
	{
		"index": 113,
		"width": 1,
		"name": "secondary_enemy_screen_y",
		"group": "vision",
	},
	{
		"index": 114,
		"width": 1,
		"name": "secondary_enemy_screen_half_width",
		"group": "vision",
	},
	{
		"index": 115,
		"width": 1,
		"name": "secondary_enemy_screen_half_height",
		"group": "vision",
	},
	{
		"index": 116,
		"width": 1,
		"name": "secondary_enemy_exposure_fraction",
		"group": "vision",
	},
	{
		"index": 117,
		"width": 1,
		"name": "secondary_enemy_illumination",
		"group": "vision",
	},
	{
		"index": 118,
		"width": 1,
		"name": "tertiary_enemy_screen_x",
		"group": "vision",
	},
	{
		"index": 119,
		"width": 1,
		"name": "tertiary_enemy_screen_y",
		"group": "vision",
	},
	{
		"index": 120,
		"width": 1,
		"name": "tertiary_enemy_screen_half_width",
		"group": "vision",
	},
	{
		"index": 121,
		"width": 1,
		"name": "tertiary_enemy_screen_half_height",
		"group": "vision",
	},
	{
		"index": 122,
		"width": 1,
		"name": "tertiary_enemy_exposure_fraction",
		"group": "vision",
	},
	{
		"index": 123,
		"width": 1,
		"name": "tertiary_enemy_illumination",
		"group": "vision",
	},
	{"index": 124, "width": 1, "name": "reticle_on_primary", "group": "vision"},
	{"index": 125, "width": 1, "name": "primary_contact_clarity", "group": "vision"},
]

## Per-component suffixes appended to a multi-value field's name (a width-3
## field covers three consecutive vector indices).
const FIELD_COMPONENT_SUFFIXES: Array = [".x", ".y", ".z", ".w"]

## Absolute path to this very script. `build()` constructs a new instance via
## `load(SELF_PATH).new()` rather than `Observation.new()`: referencing the
## script's own `class_name` in a value context needs the editor global-class
## cache, which is absent during standalone `godot --headless --script ...`
## runs, and `preload(self)` would be a compile-time cyclic reference. `load()`
## resolves at runtime against the already-compiled, cached script.
const SELF_PATH: String = "res://scripts/core/observation.gd"

var agent_position_norm: Vector3 = Vector3.ZERO
var agent_velocity_norm: Vector3 = Vector3.ZERO
var agent_forward: Vector3 = Vector3.FORWARD
var agent_health_norm: float = 1.0

## Primary enemy (nearest alive, or dead-fallback): legacy single-enemy
## fields, unchanged in meaning since milestone 1.
var enemy_relative_position_norm: Vector3 = Vector3.ZERO
var enemy_relative_direction: Vector3 = Vector3.ZERO
var enemy_distance_norm: float = 0.0
var enemy_health_norm: float = 0.0
var enemy_alive: bool = false

var weapon_ready: bool = true
var in_combat: bool = false

## Signed horizontal aim offset from the agent's forward direction to the
## primary enemy, normalized by 180 degrees (-1..1). 0 means the enemy is
## dead-center; +1/-1 means directly behind. This is far more directly
## useful for learning "turn toward the target" than raw relative xyz alone.
var enemy_bearing_norm: float = 0.0
## Alive enemy count normalized by the total number of enemies configured
## for this environment (0 if there are none).
var alive_enemy_count_norm: float = 0.0

## Second-nearest alive enemy. Zeroed/not-alive if fewer than 2 are alive.
var secondary_enemy_relative_position_norm: Vector3 = Vector3.ZERO
var secondary_enemy_distance_norm: float = 1.0
var secondary_enemy_bearing_norm: float = 0.0
var secondary_enemy_health_norm: float = 0.0
var secondary_enemy_alive: bool = false

## Third-nearest alive enemy. Zeroed/not-alive if fewer than 3 are alive.
var tertiary_enemy_relative_position_norm: Vector3 = Vector3.ZERO
var tertiary_enemy_distance_norm: float = 1.0
var tertiary_enemy_bearing_norm: float = 0.0
var tertiary_enemy_health_norm: float = 0.0
var tertiary_enemy_alive: bool = false

# ---------------------------------------------------------------------------
# Contract v2: vertical state, perception, memory, sound and world context.
#
# These exist because the simulation now genuinely produces the underlying
# information (FOV/LOS gating, decaying memory, sound events, jumping). The
# rule from docs/OBSERVATION_ACTION_CONTRACT.md still holds: nothing here is
# privileged. `*_visible` tells the policy whether the corresponding
# position block is a live sighting or a decaying memory, which is exactly
# what a human has and what prevents the belief fields from being a lie.
# ---------------------------------------------------------------------------
var agent_on_ground: bool = true
var agent_vertical_velocity_norm: float = 0.0
var agent_in_cover: bool = false
var agent_forward_clearance_norm: float = 1.0

var primary_enemy_visible: bool = false
var primary_enemy_in_fov: bool = false
var primary_enemy_los_clear: bool = false
var primary_enemy_elevation_norm: float = 0.0
var primary_enemy_info_age_norm: float = 0.0
var primary_enemy_confidence: float = 0.0
var primary_enemy_source_visual: bool = false
var primary_enemy_source_sound: bool = false

var secondary_enemy_visible: bool = false
var secondary_enemy_info_age_norm: float = 0.0
var secondary_enemy_elevation_norm: float = 0.0

var tertiary_enemy_visible: bool = false
var tertiary_enemy_info_age_norm: float = 0.0
var tertiary_enemy_elevation_norm: float = 0.0

var last_sound_direction: Vector3 = Vector3.ZERO
var last_sound_distance_norm: float = 0.0
var last_sound_bearing_norm: float = 0.0
var last_sound_age_norm: float = 0.0
var last_sound_loudness: float = 0.0
var last_sound_category_norm: float = 0.0
var audible_event_count_norm: float = 0.0

var nearest_obstacle_distance_norm: float = 1.0
var nearest_obstacle_bearing_norm: float = 0.0
var visible_enemy_count_norm: float = 0.0
var remembered_enemy_count_norm: float = 0.0
var corpse_count_norm: float = 0.0

# ---------------------------------------------------------------------------
# Contract v4: the world OBJECTS the agent can see.
#
# Three ranked slots (nearest visible first) plus the total visible count.
# A slot is only filled by geometry the agent could actually see from where
# it stands — inside its FOV cone, within its vision range and not hidden
# behind another box — so this block never leaks a hidden object, an unseen
# part of the layout, or the arena fence (Obstacle.Kind.BOUNDARY is skipped
# by the world query).
#
# An empty slot keeps the neutral "nothing there" encoding: zero position,
# distance 1.0 (the same value the absent enemy slots use), bearing 0.0 and
# kind 0.0. `object_k_visible` is the authoritative flag for "this slot
# holds a real sighting"; the count is how many objects are visible in
# total, capped by COUNT_NORMALIZER like every other count field.
# ---------------------------------------------------------------------------
var object_1_relative_position_norm: Vector3 = Vector3.ZERO
var object_1_distance_norm: float = 1.0
var object_1_bearing_norm: float = 0.0
var object_1_kind_norm: float = 0.0
var object_1_visible: float = 0.0
var object_2_relative_position_norm: Vector3 = Vector3.ZERO
var object_2_distance_norm: float = 1.0
var object_2_bearing_norm: float = 0.0
var object_2_kind_norm: float = 0.0
var object_2_visible: float = 0.0
var object_3_relative_position_norm: Vector3 = Vector3.ZERO
var object_3_distance_norm: float = 1.0
var object_3_bearing_norm: float = 0.0
var object_3_kind_norm: float = 0.0
var object_3_visible: float = 0.0
var visible_object_count_norm: float = 0.0

# ---------------------------------------------------------------------------
# Contract v5: the vision block.
#
# Everything below here describes what a contact LOOKS LIKE from where the
# agent is standing, in the coordinates of the agent's own screen: the view
# spans [-1, 1] on both axes, +x is right, +y is up.
#
# Before v5 the vector could place a contact (distance, bearing, elevation)
# but not size it: "an enemy at 12 m" and "an enemy at 12 m whose head is the
# only thing over the crate, in a shadow" produced the same numbers. Six
# values per contact close that gap and are all things a player sees:
#
#   screen_x / screen_y          where the box sits on the screen
#   screen_half_width / _height  how much of the screen it covers
#   exposure_fraction            how much of the body is NOT behind cover
#   illumination                 how much light it is standing in
#
# `reticle_on_primary` answers "is the crosshair inside that box" - one
# comparison in the engine instead of a computation the policy has to learn -
# and `primary_contact_clarity` folds the contact's illumination and the
# medium between them into "how well can I make this shot out at all", which
# is the question the old vector could not express at all.
#
# A contact the agent cannot see has NO box: these fields are zero for it,
# which reads as "not on my screen" rather than "on my screen at the origin".
# ---------------------------------------------------------------------------
var primary_enemy_screen_x: float = 0.0
var primary_enemy_screen_y: float = 0.0
var primary_enemy_screen_half_width: float = 0.0
var primary_enemy_screen_half_height: float = 0.0
var primary_enemy_exposure_fraction: float = 0.0
var primary_enemy_illumination: float = 0.0
var secondary_enemy_screen_x: float = 0.0
var secondary_enemy_screen_y: float = 0.0
var secondary_enemy_screen_half_width: float = 0.0
var secondary_enemy_screen_half_height: float = 0.0
var secondary_enemy_exposure_fraction: float = 0.0
var secondary_enemy_illumination: float = 0.0
var tertiary_enemy_screen_x: float = 0.0
var tertiary_enemy_screen_y: float = 0.0
var tertiary_enemy_screen_half_width: float = 0.0
var tertiary_enemy_screen_half_height: float = 0.0
var tertiary_enemy_exposure_fraction: float = 0.0
var tertiary_enemy_illumination: float = 0.0
var reticle_on_primary: bool = false
var primary_contact_clarity: float = 0.0
# ---------------------------------------------------------------------------
# Contract v3: conditions, contact overflow, target selection, hearing
# detail and map knowledge.
#
# Same rule as v2, applied to harder cases:
#   * `local_illumination` is how bright it is WHERE THE AGENT STANDS. The
#     lighting MODE is never exposed; a human in a dark room knows it is
#     dark, but not that the level designer called it "night".
#   * the overflow fields describe contacts beyond the individually
#     tracked slots STATISTICALLY, so eight enemies and one enemy produce
#     the same vector shape and the policy still knows it is outnumbered.
#   * the exploration fields come from SpatialMemory, i.e. only from places
#     the agent has actually looked at. Zeros mean "I know nothing", which
#     is the honest answer when map tracking is off.
# ---------------------------------------------------------------------------
var local_illumination: float = 1.0

var overflow_contact_count_norm: float = 0.0
var overflow_visible_count_norm: float = 0.0
var overflow_mean_distance_norm: float = 0.0
var overflow_min_distance_norm: float = 0.0

var target_priority_norm: float = 0.0
var target_switch_recent: bool = false

var second_sound_bearing_norm: float = 0.0
var second_sound_loudness: float = 0.0
var sound_direction_error_norm: float = 0.0
var distinct_sound_source_count_norm: float = 0.0

var explored_fraction: float = 0.0
var current_area_known: bool = false
var time_since_area_visited_norm: float = 1.0
var remembered_cover_distance_norm: float = 0.0
var remembered_cover_bearing_norm: float = 0.0
var remembered_danger_distance_norm: float = 0.0
var remembered_danger_bearing_norm: float = 0.0

var contact_uncertainty_norm: float = 0.0


## Builds an Observation from the agent, the full enemy list, and arena
## configuration used for normalization. Up to MAX_TRACKED_ENEMIES nearest
## alive enemies (primary, secondary, tertiary) are individually reported;
## any remaining enemies still exist in the simulation but are not
## individually observed.
## Builds the observation.
##
## `context` is optional and carries the perception layer's output. When it
## is empty the function behaves exactly like contract v1 — ground-truth
## enemy blocks, neutral v2 fields — which is what curriculum levels 1-4 and
## every existing test rely on. When it is supplied, the enemy blocks are
## rebuilt from the agent's BELIEF (fresh sighting, else decaying memory)
## and the v2 flags describe which one it is.
##
## Recognized context keys (all optional):
##   beliefs        Array   AgentPerception belief entries, best target first
##   sounds         Array   AgentPerception.sound_snapshot()
##   world          ArenaWorld or null
##   forward_clearance float
##   in_cover       bool
##   corpse_count   int
##   enemy_slots    int     total enemies the episode started with
##   local_illumination float  perceived brightness at the agent (v3)
##   sound_summary  Dictionary SoundBus.summarize() of `sounds` (v3)
##   contact_summary Dictionary AgentPerception.summarize_contacts() (v3)
##   target_priority_norm float  TargetSelector priority of the primary
##   target_switch_recent bool   the primary slot changed recently
##   exploration    Dictionary map-knowledge snapshot (v3, see
##                             _apply_exploration_context)
static func build(
	agent: AgentState, enemies: Array, arena_half_extent: float, context: Dictionary = {}
) -> Observation:
	var obs: Observation = (load(SELF_PATH) as GDScript).new()

	obs.agent_position_norm = Vector3(
		agent.position.x / arena_half_extent,
		agent.position.y / SandboxConfig.ARENA_WALL_HEIGHT,
		agent.position.z / arena_half_extent
	)
	# Clamped per component: with gravity enabled the vertical velocity can
	# exceed the horizontal move speed (jump velocity is 6.0 m/s against a
	# 4.5 m/s run), which would push this field outside the declared
	# [-1, 1] observation bounds. The horizontal components are unaffected
	# because they are already capped by move_speed.
	var raw_velocity: Vector3 = agent.velocity / maxf(agent.move_speed, 0.0001)
	obs.agent_velocity_norm = Vector3(
		clampf(raw_velocity.x, -1.0, 1.0),
		clampf(raw_velocity.y, -1.0, 1.0),
		clampf(raw_velocity.z, -1.0, 1.0)
	)
	obs.agent_forward = agent.get_forward_vector()
	obs.agent_health_norm = agent.health / maxf(agent.max_health, 0.0001)
	obs.weapon_ready = agent.weapon.is_ready()

	var ranked_alive: Array = rank_alive_enemies(enemies, agent.position)
	obs.alive_enemy_count_norm = (
		float(ranked_alive.size()) / float(maxi(1, enemies.size())) if enemies.size() > 0 else 0.0
	)

	var primary: EnemyState = (
		ranked_alive[0] if ranked_alive.size() > 0 else _fallback_enemy(enemies)
	)
	if primary != null:
		var to_enemy: Vector3 = primary.position - agent.position
		var distance: float = to_enemy.length()
		obs.enemy_relative_position_norm = to_enemy / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		obs.enemy_relative_direction = (
			to_enemy.normalized() if distance > 0.0001 else Vector3.ZERO
		)
		obs.enemy_distance_norm = distance / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		obs.enemy_health_norm = primary.health / maxf(primary.max_health, 0.0001)
		obs.enemy_alive = primary.alive
		obs.enemy_bearing_norm = _horizontal_bearing_norm(agent, primary.position)
		obs.in_combat = primary.alive and distance <= agent.weapon.range_m

	if ranked_alive.size() > 1:
		var secondary: EnemyState = ranked_alive[1]
		var to_secondary: Vector3 = secondary.position - agent.position
		var secondary_distance: float = to_secondary.length()
		obs.secondary_enemy_relative_position_norm = (
			to_secondary / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		)
		obs.secondary_enemy_distance_norm = (
			secondary_distance / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		)
		obs.secondary_enemy_health_norm = secondary.health / maxf(secondary.max_health, 0.0001)
		obs.secondary_enemy_alive = true
		obs.secondary_enemy_bearing_norm = _horizontal_bearing_norm(agent, secondary.position)

	if ranked_alive.size() > 2:
		var tertiary: EnemyState = ranked_alive[2]
		var to_tertiary: Vector3 = tertiary.position - agent.position
		var tertiary_distance: float = to_tertiary.length()
		obs.tertiary_enemy_relative_position_norm = (
			to_tertiary / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		)
		obs.tertiary_enemy_distance_norm = (
			tertiary_distance / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		)
		obs.tertiary_enemy_health_norm = tertiary.health / maxf(tertiary.max_health, 0.0001)
		obs.tertiary_enemy_alive = true
		obs.tertiary_enemy_bearing_norm = _horizontal_bearing_norm(agent, tertiary.position)

	_apply_vertical_state(obs, agent)
	_apply_default_visibility(obs, agent, ranked_alive)
	if not context.is_empty():
		_apply_context(obs, agent, enemies, context)
	return obs


## Vertical/locomotion fields. Always available — they describe the agent's
## own body, which it can never be wrong about.
static func _apply_vertical_state(obs: Observation, agent: AgentState) -> void:
	obs.agent_on_ground = agent.on_ground
	obs.agent_vertical_velocity_norm = clampf(
		agent.velocity.y / maxf(SandboxConfig.JUMP_VELOCITY, 0.0001), -1.0, 1.0
	)


## Neutral v2 defaults for the no-perception path: with gating disabled the
## agent really does see every living enemy, so reporting visible = true is
## accurate rather than fabricated.
static func _apply_default_visibility(
	obs: Observation, agent: AgentState, ranked_alive: Array
) -> void:
	var eye: Vector3 = agent.get_eye_position()
	if ranked_alive.size() > 0:
		var primary: EnemyState = ranked_alive[0]
		obs.primary_enemy_visible = true
		obs.primary_enemy_in_fov = true
		obs.primary_enemy_los_clear = true
		obs.primary_enemy_confidence = 1.0
		obs.primary_enemy_source_visual = true
		obs.primary_enemy_elevation_norm = _elevation_norm(eye, primary.get_chest_position())
	if ranked_alive.size() > 1:
		obs.secondary_enemy_visible = true
		obs.secondary_enemy_elevation_norm = _elevation_norm(
			eye, (ranked_alive[1] as EnemyState).get_chest_position()
		)
	if ranked_alive.size() > 2:
		obs.tertiary_enemy_visible = true
		obs.tertiary_enemy_elevation_norm = _elevation_norm(
			eye, (ranked_alive[2] as EnemyState).get_chest_position()
		)
	obs.visible_enemy_count_norm = _count_norm(ranked_alive.size())


## Rewrites the enemy blocks from perception beliefs and fills the sound,
## memory and world-context fields.
static func _apply_context(
	obs: Observation, agent: AgentState, enemies: Array, context: Dictionary
) -> void:
	obs.agent_in_cover = bool(context.get("in_cover", false))
	var clearance: float = float(context.get("forward_clearance", SandboxConfig.VISION_RANGE))
	obs.agent_forward_clearance_norm = clampf(
		clearance / maxf(SandboxConfig.VISION_RANGE, 0.0001), 0.0, 1.0
	)
	obs.corpse_count_norm = _count_norm(int(context.get("corpse_count", 0)))
	_apply_world_context(obs, agent, context)
	_apply_sound_context(obs, context)
	_apply_condition_context(obs, context)
	_apply_exploration_context(obs, context)

	if not context.has("beliefs"):
		return
	var beliefs: Array = context["beliefs"]
	var slots: int = maxi(1, int(context.get("enemy_slots", enemies.size())))
	obs.alive_enemy_count_norm = clampf(float(beliefs.size()) / float(slots), 0.0, 1.0)

	_clear_enemy_blocks(obs)
	var visible_count: int = 0
	var remembered_count: int = 0
	for rank in range(mini(beliefs.size(), MAX_TRACKED_ENEMIES)):
		var belief: Dictionary = beliefs[rank]
		if bool(belief.get("visible", false)):
			visible_count += 1
		else:
			remembered_count += 1
		_apply_belief(obs, agent, belief, rank)
	for rank in range(MAX_TRACKED_ENEMIES, beliefs.size()):
		if bool((beliefs[rank] as Dictionary).get("visible", false)):
			visible_count += 1
		else:
			remembered_count += 1
	obs.visible_enemy_count_norm = _count_norm(visible_count)
	obs.remembered_enemy_count_norm = _count_norm(remembered_count)


static func _apply_world_context(obs: Observation, agent: AgentState, context: Dictionary) -> void:
	var world = context.get("world")
	if world == null:
		obs.nearest_obstacle_distance_norm = 1.0
		obs.nearest_obstacle_bearing_norm = 0.0
		return
	var info: Dictionary = world.nearest_obstacle_info(
		agent.position, agent.get_forward_horizontal(), SandboxConfig.ARENA_MAX_DISTANCE
	)
	obs.nearest_obstacle_distance_norm = clampf(
		float(info["distance"]) / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001), 0.0, 1.0
	)
	obs.nearest_obstacle_bearing_norm = clampf(float(info["bearing_deg"]) / 180.0, -1.0, 1.0)
	_apply_object_context(obs, agent, world, context)


## Fills the contract-v4 object slots from the world's own visibility query.
##
## The query is asked for COUNT_NORMALIZER entries rather than
## MAX_TRACKED_OBJECTS: the visible total is a real field, so the list must
## not be truncated at the slot budget before it is counted. Only the first
## MAX_TRACKED_OBJECTS entries become slots.
static func _apply_object_context(
	obs: Observation, agent: AgentState, world, context: Dictionary
) -> void:
	var objects: Array = world.visible_object_infos(
		agent.get_eye_position(),
		agent.get_forward_horizontal(),
		float(context.get("fov_deg", SandboxConfig.AGENT_FOV_DEG)),
		float(context.get("vision_range", SandboxConfig.VISION_RANGE)),
		COUNT_NORMALIZER
	)
	obs.visible_object_count_norm = _count_norm(objects.size())
	for offset in range(mini(objects.size(), MAX_TRACKED_OBJECTS)):
		_apply_object_entry(obs, objects[offset], offset)


## Writes one visible object into its ranked slot.
static func _apply_object_entry(obs: Observation, entry: Dictionary, offset: int) -> void:
	var delta: Vector3 = entry.get("relative_position", Vector3.ZERO)
	var position_norm := Vector3(
		clampf(delta.x / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001), -1.0, 1.0),
		clampf(delta.y / maxf(SandboxConfig.ARENA_WALL_HEIGHT, 0.0001), -1.0, 1.0),
		clampf(delta.z / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001), -1.0, 1.0)
	)
	var distance_norm: float = clampf(
		float(entry.get("distance", 0.0)) / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001), 0.0, 1.0
	)
	var bearing_norm: float = clampf(float(entry.get("bearing_deg", 0.0)) / 180.0, -1.0, 1.0)
	# Kind is an ordinal, normalized by its cardinality exactly like the
	# sound category - never a raw enum value leaking into the vector.
	var kind_norm: float = clampf(
		float(int(entry.get("kind", 0))) / float(maxi(1, SandboxConfig.OBJECT_KIND_COUNT - 1)),
		0.0,
		1.0
	)
	if offset == 0:
		obs.object_1_relative_position_norm = position_norm
		obs.object_1_distance_norm = distance_norm
		obs.object_1_bearing_norm = bearing_norm
		obs.object_1_kind_norm = kind_norm
		obs.object_1_visible = 1.0
	elif offset == 1:
		obs.object_2_relative_position_norm = position_norm
		obs.object_2_distance_norm = distance_norm
		obs.object_2_bearing_norm = bearing_norm
		obs.object_2_kind_norm = kind_norm
		obs.object_2_visible = 1.0
	elif offset == 2:
		obs.object_3_relative_position_norm = position_norm
		obs.object_3_distance_norm = distance_norm
		obs.object_3_bearing_norm = bearing_norm
		obs.object_3_kind_norm = kind_norm
		obs.object_3_visible = 1.0


static func _apply_sound_context(obs: Observation, context: Dictionary) -> void:
	var sounds: Array = context.get("sounds", [])
	obs.audible_event_count_norm = _count_norm(sounds.size())
	if sounds.is_empty():
		return
	var loudest: Dictionary = sounds[0]
	obs.last_sound_direction = loudest.get("direction", Vector3.ZERO)
	obs.last_sound_distance_norm = clampf(
		float(loudest.get("distance", 0.0)) / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001),
		0.0,
		1.0
	)
	obs.last_sound_bearing_norm = clampf(float(loudest.get("bearing_deg", 0.0)) / 180.0, -1.0, 1.0)
	obs.last_sound_age_norm = clampf(
		float(loudest.get("age", 0.0)) / maxf(SandboxConfig.SOUND_EVENT_LIFETIME, 0.0001), 0.0, 1.0
	)
	obs.last_sound_loudness = clampf(float(loudest.get("loudness", 0.0)), 0.0, 1.0)
	# Category is an ordinal, normalized onto [0, 1] by the category count.
	obs.last_sound_category_norm = clampf(
		float(int(loudest.get("category", 0))) / float(maxi(1, SOUND_CATEGORY_COUNT - 1)), 0.0, 1.0
	)
	# v3: how trustworthy that bearing is, and whether noise is coming from
	# more than one place at once.
	obs.sound_direction_error_norm = clampf(
		float(loudest.get("direction_error_deg", 0.0)) / 90.0, 0.0, 1.0
	)
	var summary: Dictionary = context.get("sound_summary", {})
	if summary.is_empty():
		return
	obs.second_sound_bearing_norm = clampf(
		float(summary.get("second_bearing_deg", 0.0)) / 180.0, -1.0, 1.0
	)
	obs.second_sound_loudness = clampf(float(summary.get("second_loudness", 0.0)), 0.0, 1.0)
	obs.distinct_sound_source_count_norm = _count_norm(int(summary.get("distinct_sources", 0)))


## v3 conditions, contact overflow and target-selection fields.
static func _apply_condition_context(obs: Observation, context: Dictionary) -> void:
	obs.local_illumination = clampf(float(context.get("local_illumination", 1.0)), 0.0, 1.0)
	obs.target_priority_norm = clampf(float(context.get("target_priority_norm", 0.0)), 0.0, 1.0)
	obs.target_switch_recent = bool(context.get("target_switch_recent", false))

	var contacts: Dictionary = context.get("contact_summary", {})
	if contacts.is_empty():
		return
	obs.overflow_contact_count_norm = _count_norm(int(contacts.get("overflow_count", 0)))
	obs.overflow_visible_count_norm = _count_norm(int(contacts.get("overflow_visible", 0)))
	obs.overflow_mean_distance_norm = clampf(
		(
			float(contacts.get("overflow_mean_distance", 0.0))
			/ maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		),
		0.0,
		1.0
	)
	obs.overflow_min_distance_norm = clampf(
		(
			float(contacts.get("overflow_min_distance", 0.0))
			/ maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
		),
		0.0,
		1.0
	)
	obs.contact_uncertainty_norm = clampf(float(contacts.get("memory_uncertainty", 0.0)), 0.0, 1.0)


## v3 map-knowledge fields.
##
## `exploration` is the dictionary produced by EnvironmentCore from the
## agent's own SpatialMemory:
##   {explored_fraction, area_known, time_since_visit, cover_distance,
##    cover_bearing_deg, danger_distance, danger_bearing_deg}
## A missing entry means "not known", which is why every default here is
## the value that says "nothing remembered" rather than a plausible guess.
static func _apply_exploration_context(obs: Observation, context: Dictionary) -> void:
	var exploration: Dictionary = context.get("exploration", {})
	if exploration.is_empty():
		return
	obs.explored_fraction = clampf(float(exploration.get("explored_fraction", 0.0)), 0.0, 1.0)
	obs.current_area_known = bool(exploration.get("area_known", false))
	var since: float = float(exploration.get("time_since_visit", -1.0))
	obs.time_since_area_visited_norm = (
		1.0
		if since < 0.0
		else clampf(since / maxf(SandboxConfig.EXPLORATION_MAX_RECALL_AGE, 0.0001), 0.0, 1.0)
	)
	obs.remembered_cover_distance_norm = _distance_norm(
		float(exploration.get("cover_distance", -1.0))
	)
	obs.remembered_cover_bearing_norm = clampf(
		float(exploration.get("cover_bearing_deg", 0.0)) / 180.0, -1.0, 1.0
	)
	obs.remembered_danger_distance_norm = _distance_norm(
		float(exploration.get("danger_distance", -1.0))
	)
	obs.remembered_danger_bearing_norm = clampf(
		float(exploration.get("danger_bearing_deg", 0.0)) / 180.0, -1.0, 1.0
	)


## Normalized distance where a negative input means "nothing remembered"
## and maps to 0.0, i.e. the same value the field has at reset.
static func _distance_norm(distance: float) -> float:
	if distance < 0.0:
		return 0.0
	return clampf(distance / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001), 0.0, 1.0)


static func _clear_enemy_blocks(obs: Observation) -> void:
	obs.enemy_relative_position_norm = Vector3.ZERO
	obs.enemy_relative_direction = Vector3.ZERO
	obs.enemy_distance_norm = 0.0
	obs.enemy_health_norm = 0.0
	obs.enemy_alive = false
	obs.enemy_bearing_norm = 0.0
	obs.in_combat = false
	obs.primary_enemy_visible = false
	obs.primary_enemy_in_fov = false
	obs.primary_enemy_los_clear = false
	obs.primary_enemy_elevation_norm = 0.0
	obs.primary_enemy_confidence = 0.0
	obs.primary_enemy_source_visual = false
	obs.primary_enemy_source_sound = false
	obs.secondary_enemy_relative_position_norm = Vector3.ZERO
	obs.secondary_enemy_distance_norm = 0.0
	obs.secondary_enemy_bearing_norm = 0.0
	obs.secondary_enemy_health_norm = 0.0
	obs.secondary_enemy_alive = false
	obs.secondary_enemy_visible = false
	obs.secondary_enemy_elevation_norm = 0.0
	obs.tertiary_enemy_relative_position_norm = Vector3.ZERO
	obs.tertiary_enemy_distance_norm = 0.0
	obs.tertiary_enemy_bearing_norm = 0.0
	obs.tertiary_enemy_health_norm = 0.0
	obs.tertiary_enemy_alive = false
	obs.tertiary_enemy_visible = false
	obs.tertiary_enemy_elevation_norm = 0.0
	obs.primary_enemy_screen_x = 0.0
	obs.primary_enemy_screen_y = 0.0
	obs.primary_enemy_screen_half_width = 0.0
	obs.primary_enemy_screen_half_height = 0.0
	obs.primary_enemy_exposure_fraction = 0.0
	obs.primary_enemy_illumination = 0.0
	obs.secondary_enemy_screen_x = 0.0
	obs.secondary_enemy_screen_y = 0.0
	obs.secondary_enemy_screen_half_width = 0.0
	obs.secondary_enemy_screen_half_height = 0.0
	obs.secondary_enemy_exposure_fraction = 0.0
	obs.secondary_enemy_illumination = 0.0
	obs.tertiary_enemy_screen_x = 0.0
	obs.tertiary_enemy_screen_y = 0.0
	obs.tertiary_enemy_screen_half_width = 0.0
	obs.tertiary_enemy_screen_half_height = 0.0
	obs.tertiary_enemy_exposure_fraction = 0.0
	obs.tertiary_enemy_illumination = 0.0
	obs.reticle_on_primary = false
	obs.primary_contact_clarity = 0.0


## Writes the contract-v5 vision block of one enemy slot: where the contact's
## box sits on the agent's screen, how much of the screen it covers, how much
## of the body is unoccluded, and how much light it is standing in.
##
## Split out of `_apply_belief` because beliefs are not the only thing that
## fills it. The levels that run without a perception layer have no belief to
## read - yet the agent still has the enemy on its screen there, so they
## measure the same block directly (see `EnvironmentCore._build_observation`).
## One writer for both paths, so they cannot drift into two dialects.
##
## A reading without a box (a target at or behind the eye, a replay recorded
## before contract v5, an adapter that does not model geometry) leaves the
## block at zero, which reads as "no box" - never as a box at the centre of
## the screen.
static func apply_vision_reading(obs: Observation, rank: int, reading: Dictionary) -> void:
	var box: Dictionary = reading.get("screen_box", {})
	var in_front: bool = bool(box.get("in_front", false))
	var screen_x: float = float(box.get("center_x", 0.0)) if in_front else 0.0
	var screen_y: float = float(box.get("center_y", 0.0)) if in_front else 0.0
	var half_width: float = float(box.get("half_width", 0.0)) if in_front else 0.0
	var half_height: float = float(box.get("half_height", 0.0)) if in_front else 0.0
	var exposure: float = clampf(float(reading.get("exposure_fraction", 0.0)), 0.0, 1.0)
	var illumination: float = clampf(float(reading.get("illumination", 0.0)), 0.0, 1.0)
	var clarity: float = clampf(float(reading.get("clarity", 0.0)), 0.0, 1.0)
	var on_reticle: bool = bool(reading.get("reticle_on_target", false))

	match rank:
		0:
			obs.primary_enemy_screen_x = screen_x
			obs.primary_enemy_screen_y = screen_y
			obs.primary_enemy_screen_half_width = half_width
			obs.primary_enemy_screen_half_height = half_height
			obs.primary_enemy_exposure_fraction = exposure
			obs.primary_enemy_illumination = illumination
			obs.reticle_on_primary = on_reticle
			obs.primary_contact_clarity = clarity
		1:
			obs.secondary_enemy_screen_x = screen_x
			obs.secondary_enemy_screen_y = screen_y
			obs.secondary_enemy_screen_half_width = half_width
			obs.secondary_enemy_screen_half_height = half_height
			obs.secondary_enemy_exposure_fraction = exposure
			obs.secondary_enemy_illumination = illumination
		2:
			obs.tertiary_enemy_screen_x = screen_x
			obs.tertiary_enemy_screen_y = screen_y
			obs.tertiary_enemy_screen_half_width = half_width
			obs.tertiary_enemy_screen_half_height = half_height
			obs.tertiary_enemy_exposure_fraction = exposure
			obs.tertiary_enemy_illumination = illumination


static func _apply_belief(
	obs: Observation, agent: AgentState, belief: Dictionary, rank: int
) -> void:
	var believed: Vector3 = belief.get("position", agent.position)
	var to_target: Vector3 = believed - agent.position
	var distance: float = to_target.length()
	var relative: Vector3 = to_target / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
	var distance_norm: float = distance / maxf(SandboxConfig.ARENA_MAX_DISTANCE, 0.0001)
	var bearing: float = _horizontal_bearing_norm(agent, believed)
	var health: float = clampf(float(belief.get("health_norm", 0.0)), 0.0, 1.0)
	var visible: bool = bool(belief.get("visible", false))
	var age_norm: float = clampf(
		float(belief.get("age", 0.0)) / maxf(SandboxConfig.MEMORY_MAX_AGE, 0.0001), 0.0, 1.0
	)
	var elevation: float = clampf(float(belief.get("elevation_deg", 0.0)) / 90.0, -1.0, 1.0)
	apply_vision_reading(
		obs,
		rank,
		{
			"screen_box": belief.get("screen_box", {}),
			"exposure_fraction": belief.get("exposure_fraction", 0.0),
			"illumination": belief.get("illumination", 0.0),
			"clarity": belief.get("clarity", 0.0),
			"reticle_on_target": belief.get("reticle_on_target", false),
		}
	)

	match rank:
		0:
			obs.enemy_relative_position_norm = relative
			obs.enemy_relative_direction = (
				to_target.normalized() if distance > 0.0001 else Vector3.ZERO
			)
			obs.enemy_distance_norm = distance_norm
			obs.enemy_health_norm = health
			obs.enemy_alive = bool(belief.get("alive", true))
			obs.enemy_bearing_norm = bearing
			obs.in_combat = visible and distance <= agent.weapon.range_m
			obs.primary_enemy_visible = visible
			obs.primary_enemy_in_fov = bool(belief.get("in_fov", false))
			obs.primary_enemy_los_clear = bool(belief.get("los_clear", false))
			obs.primary_enemy_elevation_norm = elevation
			obs.primary_enemy_info_age_norm = age_norm
			obs.primary_enemy_confidence = clampf(float(belief.get("confidence", 0.0)), 0.0, 1.0)
			obs.primary_enemy_source_visual = int(belief.get("source", 0)) == 1
			obs.primary_enemy_source_sound = int(belief.get("source", 0)) == 2
		1:
			obs.secondary_enemy_relative_position_norm = relative
			obs.secondary_enemy_distance_norm = distance_norm
			obs.secondary_enemy_health_norm = health
			obs.secondary_enemy_alive = bool(belief.get("alive", true))
			obs.secondary_enemy_bearing_norm = bearing
			obs.secondary_enemy_visible = visible
			obs.secondary_enemy_info_age_norm = age_norm
			obs.secondary_enemy_elevation_norm = elevation
		2:
			obs.tertiary_enemy_relative_position_norm = relative
			obs.tertiary_enemy_distance_norm = distance_norm
			obs.tertiary_enemy_health_norm = health
			obs.tertiary_enemy_alive = bool(belief.get("alive", true))
			obs.tertiary_enemy_bearing_norm = bearing
			obs.tertiary_enemy_visible = visible
			obs.tertiary_enemy_info_age_norm = age_norm
			obs.tertiary_enemy_elevation_norm = elevation


## Normalizes a small count onto [0, 1] using a fixed saturation point, so
## the field stays inside the declared observation bounds no matter how
## many enemies/corpses/sounds an experiment configures.
static func _count_norm(value: int) -> float:
	return clampf(float(value) / float(COUNT_NORMALIZER), 0.0, 1.0)


## Signed vertical angle from an eye position to a point, normalized by 90
## degrees into [-1, 1].
## Vertical offset normalized onto [-1, 1]: positive above the eye. The
## angle comes from `VectorMath.elevation_deg`, the one home of the
## convention, so this field cannot drift away from the perception system's
## `elevation_deg` again.
static func _elevation_norm(from_eye: Vector3, to_position: Vector3) -> float:
	return clampf(VectorMath.elevation_deg(from_eye, to_position) / 90.0, -1.0, 1.0)


## Returns every alive enemy, nearest-to-agent first. Deterministic given a
## deterministic enemy list: entries are sorted by squared distance with the
## original enemy index as an explicit tie-breaker, so two enemies at the
## exact same distance always resolve in list order regardless of the
## engine's sort-stability guarantees (which Godot does not document).
##
## Public because this ranking IS the contract's definition of
## primary/secondary/tertiary enemy. Tooling that has to explain which
## enemies ended up in the observation (the Control Center's perception
## view) calls this instead of re-implementing the rule and drifting from
## it.
static func rank_alive_enemies(enemies: Array, agent_position: Vector3) -> Array:
	var alive_list: Array = []
	for i in range(enemies.size()):
		var enemy: EnemyState = enemies[i]
		if enemy.alive:
			alive_list.append(
				{
					"enemy": enemy,
					"index": i,
					"dist_sq": enemy.position.distance_squared_to(agent_position)
				}
			)
	alive_list.sort_custom(
		func(a, b):
			if (a as Dictionary).dist_sq != (b as Dictionary).dist_sq:
				return (a as Dictionary).dist_sq < (b as Dictionary).dist_sq
			return (a as Dictionary).index < (b as Dictionary).index
	)
	var ranked: Array = []
	for entry in alive_list:
		ranked.append((entry as Dictionary).enemy)
	return ranked


## Stable fallback used only when no enemy is alive, so a fully "dead"
## episode still reports a stable, if zeroed-out, primary-enemy observation.
static func _fallback_enemy(enemies: Array) -> EnemyState:
	return enemies[0] if enemies.size() > 0 else null


## Bearing to a target, normalized onto [-1, 1]: positive is to the agent's
## right, the way a positive `look_yaw_axis` turns (contract index 17), and
## measured against `AgentState.get_forward_horizontal()` (yaw = 0, i.e.
## forward = (0, 0, -1)).
##
## The sign convention lives in `VectorMath`, which is the one implementation
## of it - the world, sound and memory queries used to compute their own with
## the opposite sign, which put a contact and a crate on the same side of the
## agent on opposite sides of the vector.
static func _horizontal_bearing_norm(agent: AgentState, target_position: Vector3) -> float:
	var bearing_deg: float = VectorMath.signed_bearing_deg(
		agent.get_forward_horizontal(), agent.position, target_position
	)
	return clampf(bearing_deg / 180.0, -1.0, 1.0)


## Flat float array in a fixed, documented order — the shape an RL policy
## network would consume. See docs/OBSERVATION_ACTION_CONTRACT.md for the
## full table. Order:
## [0-2]   agent_position_norm (x,y,z)
## [3-5]   agent_velocity_norm (x,y,z)
## [6-8]   agent_forward (x,y,z)
## [9]     agent_health_norm
## [10-12] enemy_relative_position_norm (x,y,z)         -- primary enemy
## [13]    enemy_distance_norm                          -- primary enemy
## [14]    enemy_health_norm                             -- primary enemy
## [15]    weapon_ready (0/1)
## [16]    in_combat (0/1)
## [17]    enemy_bearing_norm                            -- primary enemy
## [18]    alive_enemy_count_norm
## [19-21] secondary_enemy_relative_position_norm (x,y,z)
## [22]    secondary_enemy_distance_norm
## [23]    secondary_enemy_bearing_norm
## [24]    secondary_enemy_health_norm
## [25]    secondary_enemy_alive (0/1)
## [26-28] tertiary_enemy_relative_position_norm (x,y,z)
## [29]    tertiary_enemy_distance_norm
## [30]    tertiary_enemy_bearing_norm
## [31]    tertiary_enemy_health_norm
## [32]    tertiary_enemy_alive (0/1)
##   ... (see docs/OBSERVATION_ACTION_CONTRACT.md for 33-105)
## [106-111] primary_enemy_screen_x / _y / _half_width / _half_height /
##           _exposure_fraction / _illumination   -- the primary contact's box
## [112-117] the same six values for the secondary contact
## [118-123] the same six values for the tertiary contact
## [124]    reticle_on_primary (0/1)   -- crosshair inside the primary's box
## [125]    primary_contact_clarity    -- how well it can be made out at all
func to_array() -> PackedFloat32Array:
	# Pre-allocated once and assigned by index: this runs once per
	# environment per simulation step on the training hot path, and
	# sequential append() calls would otherwise reallocate the packed array
	# as it grows. The assignment order below is the contract — see the
	# field table in the comment block above.
	var arr := PackedFloat32Array()
	arr.resize(FIELD_COUNT)
	arr[0] = agent_position_norm.x
	arr[1] = agent_position_norm.y
	arr[2] = agent_position_norm.z
	arr[3] = agent_velocity_norm.x
	arr[4] = agent_velocity_norm.y
	arr[5] = agent_velocity_norm.z
	arr[6] = agent_forward.x
	arr[7] = agent_forward.y
	arr[8] = agent_forward.z
	arr[9] = agent_health_norm
	arr[10] = enemy_relative_position_norm.x
	arr[11] = enemy_relative_position_norm.y
	arr[12] = enemy_relative_position_norm.z
	arr[13] = enemy_distance_norm
	arr[14] = enemy_health_norm
	arr[15] = 1.0 if weapon_ready else 0.0
	arr[16] = 1.0 if in_combat else 0.0
	arr[17] = enemy_bearing_norm
	arr[18] = alive_enemy_count_norm
	arr[19] = secondary_enemy_relative_position_norm.x
	arr[20] = secondary_enemy_relative_position_norm.y
	arr[21] = secondary_enemy_relative_position_norm.z
	arr[22] = secondary_enemy_distance_norm
	arr[23] = secondary_enemy_bearing_norm
	arr[24] = secondary_enemy_health_norm
	arr[25] = 1.0 if secondary_enemy_alive else 0.0
	arr[26] = tertiary_enemy_relative_position_norm.x
	arr[27] = tertiary_enemy_relative_position_norm.y
	arr[28] = tertiary_enemy_relative_position_norm.z
	arr[29] = tertiary_enemy_distance_norm
	arr[30] = tertiary_enemy_bearing_norm
	arr[31] = tertiary_enemy_health_norm
	arr[32] = 1.0 if tertiary_enemy_alive else 0.0
	arr[33] = 1.0 if agent_on_ground else 0.0
	arr[34] = agent_vertical_velocity_norm
	arr[35] = 1.0 if agent_in_cover else 0.0
	arr[36] = agent_forward_clearance_norm
	arr[37] = 1.0 if primary_enemy_visible else 0.0
	arr[38] = 1.0 if primary_enemy_in_fov else 0.0
	arr[39] = 1.0 if primary_enemy_los_clear else 0.0
	arr[40] = primary_enemy_elevation_norm
	arr[41] = primary_enemy_info_age_norm
	arr[42] = primary_enemy_confidence
	arr[43] = 1.0 if primary_enemy_source_visual else 0.0
	arr[44] = 1.0 if primary_enemy_source_sound else 0.0
	arr[45] = 1.0 if secondary_enemy_visible else 0.0
	arr[46] = secondary_enemy_info_age_norm
	arr[47] = secondary_enemy_elevation_norm
	arr[48] = 1.0 if tertiary_enemy_visible else 0.0
	arr[49] = tertiary_enemy_info_age_norm
	arr[50] = tertiary_enemy_elevation_norm
	arr[51] = last_sound_direction.x
	arr[52] = last_sound_direction.y
	arr[53] = last_sound_direction.z
	arr[54] = last_sound_distance_norm
	arr[55] = last_sound_bearing_norm
	arr[56] = last_sound_age_norm
	arr[57] = last_sound_loudness
	arr[58] = last_sound_category_norm
	arr[59] = audible_event_count_norm
	arr[60] = nearest_obstacle_distance_norm
	arr[61] = nearest_obstacle_bearing_norm
	arr[62] = visible_enemy_count_norm
	arr[63] = remembered_enemy_count_norm
	arr[64] = corpse_count_norm
	arr[65] = local_illumination
	arr[66] = overflow_contact_count_norm
	arr[67] = overflow_visible_count_norm
	arr[68] = overflow_mean_distance_norm
	arr[69] = overflow_min_distance_norm
	arr[70] = target_priority_norm
	arr[71] = 1.0 if target_switch_recent else 0.0
	arr[72] = second_sound_bearing_norm
	arr[73] = second_sound_loudness
	arr[74] = sound_direction_error_norm
	arr[75] = distinct_sound_source_count_norm
	arr[76] = explored_fraction
	arr[77] = 1.0 if current_area_known else 0.0
	arr[78] = time_since_area_visited_norm
	arr[79] = remembered_cover_distance_norm
	arr[80] = remembered_cover_bearing_norm
	arr[81] = remembered_danger_distance_norm
	arr[82] = remembered_danger_bearing_norm
	arr[83] = contact_uncertainty_norm
	arr[84] = object_1_relative_position_norm.x
	arr[85] = object_1_relative_position_norm.y
	arr[86] = object_1_relative_position_norm.z
	arr[87] = object_1_distance_norm
	arr[88] = object_1_bearing_norm
	arr[89] = object_1_kind_norm
	arr[90] = object_1_visible
	arr[91] = object_2_relative_position_norm.x
	arr[92] = object_2_relative_position_norm.y
	arr[93] = object_2_relative_position_norm.z
	arr[94] = object_2_distance_norm
	arr[95] = object_2_bearing_norm
	arr[96] = object_2_kind_norm
	arr[97] = object_2_visible
	arr[98] = object_3_relative_position_norm.x
	arr[99] = object_3_relative_position_norm.y
	arr[100] = object_3_relative_position_norm.z
	arr[101] = object_3_distance_norm
	arr[102] = object_3_bearing_norm
	arr[103] = object_3_kind_norm
	arr[104] = object_3_visible
	arr[105] = visible_object_count_norm
	arr[106] = primary_enemy_screen_x
	arr[107] = primary_enemy_screen_y
	arr[108] = primary_enemy_screen_half_width
	arr[109] = primary_enemy_screen_half_height
	arr[110] = primary_enemy_exposure_fraction
	arr[111] = primary_enemy_illumination
	arr[112] = secondary_enemy_screen_x
	arr[113] = secondary_enemy_screen_y
	arr[114] = secondary_enemy_screen_half_width
	arr[115] = secondary_enemy_screen_half_height
	arr[116] = secondary_enemy_exposure_fraction
	arr[117] = secondary_enemy_illumination
	arr[118] = tertiary_enemy_screen_x
	arr[119] = tertiary_enemy_screen_y
	arr[120] = tertiary_enemy_screen_half_width
	arr[121] = tertiary_enemy_screen_half_height
	arr[122] = tertiary_enemy_exposure_fraction
	arr[123] = tertiary_enemy_illumination
	arr[124] = 1.0 if reticle_on_primary else 0.0
	arr[125] = primary_contact_clarity
	return arr


func to_dict() -> Dictionary:
	return {
		"agent_position_norm": agent_position_norm,
		"agent_velocity_norm": agent_velocity_norm,
		"agent_forward": agent_forward,
		"agent_health_norm": agent_health_norm,
		"enemy_relative_position_norm": enemy_relative_position_norm,
		"enemy_relative_direction": enemy_relative_direction,
		"enemy_distance_norm": enemy_distance_norm,
		"enemy_health_norm": enemy_health_norm,
		"enemy_alive": enemy_alive,
		"enemy_bearing_norm": enemy_bearing_norm,
		"alive_enemy_count_norm": alive_enemy_count_norm,
		"weapon_ready": weapon_ready,
		"in_combat": in_combat,
		"secondary_enemy_relative_position_norm": secondary_enemy_relative_position_norm,
		"secondary_enemy_distance_norm": secondary_enemy_distance_norm,
		"secondary_enemy_bearing_norm": secondary_enemy_bearing_norm,
		"secondary_enemy_health_norm": secondary_enemy_health_norm,
		"secondary_enemy_alive": secondary_enemy_alive,
		"tertiary_enemy_relative_position_norm": tertiary_enemy_relative_position_norm,
		"tertiary_enemy_distance_norm": tertiary_enemy_distance_norm,
		"tertiary_enemy_bearing_norm": tertiary_enemy_bearing_norm,
		"tertiary_enemy_health_norm": tertiary_enemy_health_norm,
		"tertiary_enemy_alive": tertiary_enemy_alive,
		"agent_on_ground": agent_on_ground,
		"agent_vertical_velocity_norm": agent_vertical_velocity_norm,
		"agent_in_cover": agent_in_cover,
		"agent_forward_clearance_norm": agent_forward_clearance_norm,
		"primary_enemy_visible": primary_enemy_visible,
		"primary_enemy_in_fov": primary_enemy_in_fov,
		"primary_enemy_los_clear": primary_enemy_los_clear,
		"primary_enemy_elevation_norm": primary_enemy_elevation_norm,
		"primary_enemy_info_age_norm": primary_enemy_info_age_norm,
		"primary_enemy_confidence": primary_enemy_confidence,
		"primary_enemy_source_visual": primary_enemy_source_visual,
		"primary_enemy_source_sound": primary_enemy_source_sound,
		"secondary_enemy_visible": secondary_enemy_visible,
		"secondary_enemy_info_age_norm": secondary_enemy_info_age_norm,
		"secondary_enemy_elevation_norm": secondary_enemy_elevation_norm,
		"tertiary_enemy_visible": tertiary_enemy_visible,
		"tertiary_enemy_info_age_norm": tertiary_enemy_info_age_norm,
		"tertiary_enemy_elevation_norm": tertiary_enemy_elevation_norm,
		"last_sound_direction": last_sound_direction,
		"last_sound_distance_norm": last_sound_distance_norm,
		"last_sound_bearing_norm": last_sound_bearing_norm,
		"last_sound_age_norm": last_sound_age_norm,
		"last_sound_loudness": last_sound_loudness,
		"last_sound_category_norm": last_sound_category_norm,
		"audible_event_count_norm": audible_event_count_norm,
		"nearest_obstacle_distance_norm": nearest_obstacle_distance_norm,
		"nearest_obstacle_bearing_norm": nearest_obstacle_bearing_norm,
		"visible_enemy_count_norm": visible_enemy_count_norm,
		"remembered_enemy_count_norm": remembered_enemy_count_norm,
		"corpse_count_norm": corpse_count_norm,
		"local_illumination": local_illumination,
		"overflow_contact_count_norm": overflow_contact_count_norm,
		"overflow_visible_count_norm": overflow_visible_count_norm,
		"overflow_mean_distance_norm": overflow_mean_distance_norm,
		"overflow_min_distance_norm": overflow_min_distance_norm,
		"target_priority_norm": target_priority_norm,
		"target_switch_recent": target_switch_recent,
		"second_sound_bearing_norm": second_sound_bearing_norm,
		"second_sound_loudness": second_sound_loudness,
		"sound_direction_error_norm": sound_direction_error_norm,
		"distinct_sound_source_count_norm": distinct_sound_source_count_norm,
		"explored_fraction": explored_fraction,
		"current_area_known": current_area_known,
		"time_since_area_visited_norm": time_since_area_visited_norm,
		"remembered_cover_distance_norm": remembered_cover_distance_norm,
		"remembered_cover_bearing_norm": remembered_cover_bearing_norm,
		"remembered_danger_distance_norm": remembered_danger_distance_norm,
		"remembered_danger_bearing_norm": remembered_danger_bearing_norm,
		"contact_uncertainty_norm": contact_uncertainty_norm,
		"object_1_relative_position_norm": object_1_relative_position_norm,
		"object_1_distance_norm": object_1_distance_norm,
		"object_1_bearing_norm": object_1_bearing_norm,
		"object_1_kind_norm": object_1_kind_norm,
		"object_1_visible": object_1_visible,
		"object_2_relative_position_norm": object_2_relative_position_norm,
		"object_2_distance_norm": object_2_distance_norm,
		"object_2_bearing_norm": object_2_bearing_norm,
		"object_2_kind_norm": object_2_kind_norm,
		"object_2_visible": object_2_visible,
		"object_3_relative_position_norm": object_3_relative_position_norm,
		"object_3_distance_norm": object_3_distance_norm,
		"object_3_bearing_norm": object_3_bearing_norm,
		"object_3_kind_norm": object_3_kind_norm,
		"object_3_visible": object_3_visible,
		"visible_object_count_norm": visible_object_count_norm,
		"primary_enemy_screen_x": primary_enemy_screen_x,
		"primary_enemy_screen_y": primary_enemy_screen_y,
		"primary_enemy_screen_half_width": primary_enemy_screen_half_width,
		"primary_enemy_screen_half_height": primary_enemy_screen_half_height,
		"primary_enemy_exposure_fraction": primary_enemy_exposure_fraction,
		"primary_enemy_illumination": primary_enemy_illumination,
		"secondary_enemy_screen_x": secondary_enemy_screen_x,
		"secondary_enemy_screen_y": secondary_enemy_screen_y,
		"secondary_enemy_screen_half_width": secondary_enemy_screen_half_width,
		"secondary_enemy_screen_half_height": secondary_enemy_screen_half_height,
		"secondary_enemy_exposure_fraction": secondary_enemy_exposure_fraction,
		"secondary_enemy_illumination": secondary_enemy_illumination,
		"tertiary_enemy_screen_x": tertiary_enemy_screen_x,
		"tertiary_enemy_screen_y": tertiary_enemy_screen_y,
		"tertiary_enemy_screen_half_width": tertiary_enemy_screen_half_width,
		"tertiary_enemy_screen_half_height": tertiary_enemy_screen_half_height,
		"tertiary_enemy_exposure_fraction": tertiary_enemy_exposure_fraction,
		"tertiary_enemy_illumination": tertiary_enemy_illumination,
		"reticle_on_primary": reticle_on_primary,
		"primary_contact_clarity": primary_contact_clarity,
	}


## Deep copy of FIELD_SPEC, safe for callers that want to annotate entries.
## Tooling only (Control Center inspector, exporters, tests); neither the
## simulation nor the RL bridge ever calls it.
static func field_layout() -> Array:
	return FIELD_SPEC.duplicate(true)


## One label per observation-vector index (length == FIELD_COUNT), derived
## from FIELD_SPEC. Width-3 fields expand to `name.x`, `name.y`, `name.z`.
static func field_names() -> PackedStringArray:
	var names := PackedStringArray()
	names.resize(FIELD_COUNT)
	for entry_value in FIELD_SPEC:
		var entry: Dictionary = entry_value
		var start: int = int(entry["index"])
		var width: int = int(entry["width"])
		var base_name: String = str(entry["name"])
		for offset in range(width):
			var index: int = start + offset
			if index < 0 or index >= FIELD_COUNT:
				continue
			if width == 1:
				names[index] = base_name
			else:
				names[index] = base_name + str(FIELD_COMPONENT_SUFFIXES[offset])
	return names


## Group label for one observation-vector index ("agent", "primary_enemy",
## ...), derived from FIELD_SPEC. Returns "" for an out-of-range index.
static func field_group(index: int) -> String:
	for entry_value in FIELD_SPEC:
		var entry: Dictionary = entry_value
		var start: int = int(entry["index"])
		if index >= start and index < start + int(entry["width"]):
			return str(entry["group"])
	return ""

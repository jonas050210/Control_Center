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
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


## Contract v2 = 33 (v1, unchanged) + 32 perception/memory/sound/vertical
## fields. Indices [0-32] keep their exact v1 meaning and normalization;
## everything new is strictly appended, so a v1 policy's weights still line
## up with the same semantics for the first 33 inputs.
const FIELD_COUNT: int = 65
## The v1 prefix length, kept as a named constant because several tests and
## the Roblox adapter boundary assert the prefix is never reordered.
const LEGACY_FIELD_COUNT: int = 33
## Total number of enemies individually reported (primary + tracked extras).
const MAX_TRACKED_ENEMIES: int = SandboxConfig.OBSERVATION_MAX_TRACKED_ENEMIES
## Saturation point for the count-style fields (visible/remembered enemies,
## corpses, audible events). Counts above this clamp to 1.0.
const COUNT_NORMALIZER: int = 8
## Mirrors SoundBus.Category cardinality. Duplicated as a plain int rather
## than preloading SoundBus here, because Observation must stay loadable
## without the perception layer (the Roblox adapter boundary depends on it).
const SOUND_CATEGORY_COUNT: int = 6

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
	{
		"index": 47,
		"width": 1,
		"name": "secondary_enemy_elevation_norm",
		"group": "secondary_enemy"
	},
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

	var primary: EnemyState = ranked_alive[0] if ranked_alive.size() > 0 else _fallback_enemy(enemies)
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
		obs.in_combat = primary.alive and distance <= SandboxConfig.WEAPON_RANGE

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


static func _apply_world_context(
	obs: Observation, agent: AgentState, context: Dictionary
) -> void:
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
	obs.nearest_obstacle_bearing_norm = clampf(
		float(info["bearing_deg"]) / 180.0, -1.0, 1.0
	)


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
			obs.in_combat = visible and distance <= SandboxConfig.WEAPON_RANGE
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
static func _elevation_norm(from_eye: Vector3, to_position: Vector3) -> float:
	var delta: Vector3 = to_position - from_eye
	var horizontal: float = Vector2(delta.x, delta.z).length()
	if horizontal < 0.000001:
		return 1.0 if delta.y >= 0.0 else -1.0
	return clampf(rad_to_deg(atan2(delta.y, horizontal)) / 90.0, -1.0, 1.0)


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
				{"enemy": enemy, "index": i, "dist_sq": enemy.position.distance_squared_to(agent_position)}
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


## Signed horizontal angle from the agent's forward direction to
## `target_position`, normalized to [-1, 1] by dividing by 180 degrees.
## Matches the atan2(x, -z) convention already used by AIStubController and
## AgentState.get_forward_horizontal() (yaw=0 -> forward=(0,0,-1)).
static func _horizontal_bearing_norm(agent: AgentState, target_position: Vector3) -> float:
	var to_target: Vector3 = target_position - agent.position
	to_target.y = 0.0
	if to_target.length_squared() < 0.000001:
		return 0.0
	var desired_yaw_rad: float = atan2(to_target.x, -to_target.z)
	var current_yaw_rad: float = deg_to_rad(agent.yaw_deg)
	var diff_rad: float = wrapf(desired_yaw_rad - current_yaw_rad, -PI, PI)
	return clampf(diff_rad / PI, -1.0, 1.0)


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

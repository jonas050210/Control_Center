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


## 17 legacy single-enemy fields + 1 primary bearing + 1 alive-count +
## 2 extra tracked enemies * 7 fields each = 33.
const FIELD_COUNT: int = 33
## Total number of enemies individually reported (primary + tracked extras).
const MAX_TRACKED_ENEMIES: int = SandboxConfig.OBSERVATION_MAX_TRACKED_ENEMIES

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


## Builds an Observation from the agent, the full enemy list, and arena
## configuration used for normalization. Up to MAX_TRACKED_ENEMIES nearest
## alive enemies (primary, secondary, tertiary) are individually reported;
## any remaining enemies still exist in the simulation but are not
## individually observed.
static func build(agent: AgentState, enemies: Array, arena_half_extent: float) -> Observation:
	var obs: Observation = (load(SELF_PATH) as GDScript).new()

	obs.agent_position_norm = Vector3(
		agent.position.x / arena_half_extent,
		agent.position.y / SandboxConfig.ARENA_WALL_HEIGHT,
		agent.position.z / arena_half_extent
	)
	obs.agent_velocity_norm = agent.velocity / maxf(agent.move_speed, 0.0001)
	obs.agent_forward = agent.get_forward_vector()
	obs.agent_health_norm = agent.health / maxf(agent.max_health, 0.0001)
	obs.weapon_ready = agent.weapon.is_ready()

	var ranked_alive: Array = _rank_alive_enemies(enemies, agent.position)
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

	return obs


## Returns every alive enemy, nearest-to-agent first. Deterministic given a
## deterministic enemy list (stable sort by squared distance).
static func _rank_alive_enemies(enemies: Array, agent_position: Vector3) -> Array:
	var alive_list: Array = []
	for e in enemies:
		var enemy: EnemyState = e
		if enemy.alive:
			alive_list.append(enemy)
	alive_list.sort_custom(
		func(a, b):
			return (
				(a as EnemyState).position.distance_squared_to(agent_position)
				< (b as EnemyState).position.distance_squared_to(agent_position)
			)
	)
	return alive_list


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
	var arr := PackedFloat32Array()
	arr.append(agent_position_norm.x)
	arr.append(agent_position_norm.y)
	arr.append(agent_position_norm.z)
	arr.append(agent_velocity_norm.x)
	arr.append(agent_velocity_norm.y)
	arr.append(agent_velocity_norm.z)
	arr.append(agent_forward.x)
	arr.append(agent_forward.y)
	arr.append(agent_forward.z)
	arr.append(agent_health_norm)
	arr.append(enemy_relative_position_norm.x)
	arr.append(enemy_relative_position_norm.y)
	arr.append(enemy_relative_position_norm.z)
	arr.append(enemy_distance_norm)
	arr.append(enemy_health_norm)
	arr.append(1.0 if weapon_ready else 0.0)
	arr.append(1.0 if in_combat else 0.0)
	arr.append(enemy_bearing_norm)
	arr.append(alive_enemy_count_norm)
	arr.append(secondary_enemy_relative_position_norm.x)
	arr.append(secondary_enemy_relative_position_norm.y)
	arr.append(secondary_enemy_relative_position_norm.z)
	arr.append(secondary_enemy_distance_norm)
	arr.append(secondary_enemy_bearing_norm)
	arr.append(secondary_enemy_health_norm)
	arr.append(1.0 if secondary_enemy_alive else 0.0)
	arr.append(tertiary_enemy_relative_position_norm.x)
	arr.append(tertiary_enemy_relative_position_norm.y)
	arr.append(tertiary_enemy_relative_position_norm.z)
	arr.append(tertiary_enemy_distance_norm)
	arr.append(tertiary_enemy_bearing_norm)
	arr.append(tertiary_enemy_health_norm)
	arr.append(1.0 if tertiary_enemy_alive else 0.0)
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
	}

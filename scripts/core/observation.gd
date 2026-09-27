## Observation
##
## Builds the structured, numeric, (mostly) normalized observation vector
## consumed by an RL policy. This is the ONLY observation mode implemented
## in milestone 1 (SandboxConfig.ObservationMode.STRUCTURED). The class is
## written so that a future RGB/screen mode or a human-input-derived mode
## can be added alongside it (see docs/ARCHITECTURE.md) without changing
## this struct's meaning for existing consumers.
class_name Observation
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const AgentState = preload("res://scripts/agent/agent_state.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


const FIELD_COUNT: int = 17

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
var enemy_relative_position_norm: Vector3 = Vector3.ZERO
var enemy_relative_direction: Vector3 = Vector3.ZERO
var enemy_distance_norm: float = 0.0
var enemy_health_norm: float = 0.0
var enemy_alive: bool = false
var weapon_ready: bool = true
var in_combat: bool = false


## Builds an Observation from the agent and the single "primary" enemy
## (nearest alive enemy, or the first enemy if none are alive) plus arena
## configuration used for normalization.
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

	var primary: EnemyState = _pick_primary_enemy(enemies, agent.position)
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
		obs.in_combat = primary.alive and distance <= SandboxConfig.WEAPON_RANGE

	return obs


## Picks the nearest alive enemy to `agent_position`. If no enemy is alive,
## falls back to the first enemy in the array (so a fully "dead" episode
## still reports a stable, if zeroed-out, observation) or null if empty.
static func _pick_primary_enemy(enemies: Array, agent_position: Vector3) -> EnemyState:
	var best: EnemyState = null
	var best_dist: float = INF
	for e in enemies:
		var enemy: EnemyState = e
		if enemy.alive:
			var d: float = enemy.position.distance_squared_to(agent_position)
			if d < best_dist:
				best_dist = d
				best = enemy
	if best == null and enemies.size() > 0:
		best = enemies[0]
	return best


## Flat float array in a fixed, documented order — the shape an RL policy
## network would consume. Order:
## [0-2] agent_position_norm (x,y,z)
## [3-5] agent_velocity_norm (x,y,z)
## [6-8] agent_forward (x,y,z)
## [9]   agent_health_norm
## [10-12] enemy_relative_position_norm (x,y,z)
## [13-15] enemy_relative_direction is folded into distance+direction below
## [13] enemy_distance_norm
## [14] enemy_health_norm
## [15] weapon_ready (0/1)
## [16] in_combat (0/1)
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
		"weapon_ready": weapon_ready,
		"in_combat": in_combat,
	}

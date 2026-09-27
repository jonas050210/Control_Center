## EnemyState
##
## Pure simulation state + deterministic AI for a single enemy target.
## Supports configurable difficulty parameters, idle/chase/attack/dead states,
## and deterministic hit resolution.
class_name EnemyState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


enum AIState { IDLE = 0, CHASE = 1, ATTACK = 2, DEAD = 3 }

var position: Vector3 = SandboxConfig.ENEMY_SPAWN_POSITION
var max_health: float = SandboxConfig.ENEMY_MAX_HEALTH
var health: float = SandboxConfig.ENEMY_MAX_HEALTH
var alive: bool = true
var ai_state: int = AIState.IDLE

var move_speed: float = SandboxConfig.ENEMY_MOVE_SPEED
var radius: float = SandboxConfig.ENEMY_RADIUS
var chest_height: float = SandboxConfig.ENEMY_CHEST_HEIGHT
var detection_range: float = SandboxConfig.ENEMY_DETECTION_RANGE
var attack_range: float = SandboxConfig.ENEMY_ATTACK_RANGE
var attack_damage: float = SandboxConfig.ENEMY_ATTACK_DAMAGE
var attack_cooldown_time: float = SandboxConfig.ENEMY_ATTACK_COOLDOWN
var attack_cooldown_remaining: float = 0.0

## Deterministic strafing state. `strafe_direction` (+1/-1) and
## `strafe_phase` (radians) are normally assigned once per episode from the
## environment's seeded RNG (see EnvironmentCore.reset()); `time_alive`
## accumulates simulation dt so the lateral oscillation is a pure function
## of elapsed time, never further RNG calls, keeping replay deterministic.
var strafe_direction: float = 1.0
var strafe_phase: float = 0.0
var time_alive: float = 0.0


func _init(
	p_max_health: float = SandboxConfig.ENEMY_MAX_HEALTH,
	p_move_speed: float = SandboxConfig.ENEMY_MOVE_SPEED,
	p_attack_damage: float = SandboxConfig.ENEMY_ATTACK_DAMAGE,
	p_attack_cooldown: float = SandboxConfig.ENEMY_ATTACK_COOLDOWN,
	p_attack_range: float = SandboxConfig.ENEMY_ATTACK_RANGE
) -> void:
	max_health = maxf(1.0, p_max_health)
	health = max_health
	move_speed = maxf(0.0, p_move_speed)
	attack_damage = maxf(0.0, p_attack_damage)
	attack_cooldown_time = maxf(0.01, p_attack_cooldown)
	attack_range = maxf(0.1, p_attack_range)
	alive = true
	ai_state = AIState.IDLE
	attack_cooldown_remaining = 0.0


func reset(spawn_position: Vector3 = SandboxConfig.ENEMY_SPAWN_POSITION) -> void:
	position = spawn_position
	health = max_health
	alive = true
	ai_state = AIState.IDLE
	attack_cooldown_remaining = 0.0
	time_alive = 0.0


func configure_difficulty(
	speed_scale: float = 1.0,
	damage_scale: float = 1.0,
	health_scale: float = 1.0,
	cooldown_scale: float = 1.0
) -> void:
	max_health = SandboxConfig.ENEMY_MAX_HEALTH * maxf(0.1, health_scale)
	health = max_health
	move_speed = SandboxConfig.ENEMY_MOVE_SPEED * maxf(0.0, speed_scale)
	attack_damage = SandboxConfig.ENEMY_ATTACK_DAMAGE * maxf(0.0, damage_scale)
	attack_cooldown_time = SandboxConfig.ENEMY_ATTACK_COOLDOWN * maxf(0.1, cooldown_scale)


func get_chest_position() -> Vector3:
	return position + Vector3(0.0, chest_height, 0.0)


## Advances the enemy AI by one tick. Returns the amount of damage the
## enemy dealt to the agent this tick (0 if it did not attack).
##
## `enable_strafe` blends a deterministic sinusoidal lateral component into
## the chase movement (see strafe_direction/strafe_phase/time_alive above),
## making the enemy weave and, at closer range, circle around the agent
## instead of walking in a straight line. It is purely a function of
## elapsed time so two runs with the same seed stay identical.
func update_ai(
	dt: float,
	agent_position: Vector3,
	arena_half_extent: float,
	enable_movement: bool = true,
	enable_attack: bool = true,
	enable_strafe: bool = false
) -> float:
	if attack_cooldown_remaining > 0.0:
		attack_cooldown_remaining = maxf(0.0, attack_cooldown_remaining - dt)

	if not alive:
		ai_state = AIState.DEAD
		return 0.0

	time_alive += dt

	var to_agent: Vector3 = agent_position - position
	to_agent.y = 0.0
	var distance: float = to_agent.length()

	if distance > detection_range:
		ai_state = AIState.IDLE
		return 0.0

	if enable_attack and distance <= attack_range:
		ai_state = AIState.ATTACK
		if attack_cooldown_remaining <= 0.0:
			attack_cooldown_remaining = attack_cooldown_time
			return attack_damage
		return 0.0

	if not enable_movement:
		ai_state = AIState.IDLE
		return 0.0

	ai_state = AIState.CHASE
	if distance > 0.0001:
		var dir: Vector3 = to_agent / distance
		var move_dir: Vector3 = dir
		if enable_strafe:
			move_dir = _blend_strafe_direction(dir, distance)
		position += move_dir * move_speed * dt
	var limit: float = arena_half_extent - radius
	position.x = clampf(position.x, -limit, limit)
	position.z = clampf(position.z, -limit, limit)
	position.y = 0.0
	return 0.0


## Blends the direct chase direction with a perpendicular, sinusoidally
## oscillating lateral component. Enemies far away mostly walk straight in;
## enemies near attack_range weight the lateral component more heavily,
## producing circle-strafing behavior around the agent instead of a trivial
## straight-line approach.
func _blend_strafe_direction(chase_dir: Vector3, distance: float) -> Vector3:
	var perpendicular := Vector3(-chase_dir.z, 0.0, chase_dir.x)
	var strafe_signal: float = sin(
		time_alive * SandboxConfig.ENEMY_STRAFE_ANGULAR_SPEED + strafe_phase
	)
	strafe_signal *= strafe_direction
	# proximity in [0, 0.85]: 0 far away (pure chase), up to 0.85 near attack range.
	var proximity: float = clampf(1.0 - (distance / maxf(attack_range * 4.0, 0.001)), 0.0, 0.85)
	var forward_weight: float = 1.0 - proximity
	var lateral_weight: float = (0.4 + proximity) * SandboxConfig.ENEMY_STRAFE_SPEED_SCALE
	var blended: Vector3 = chase_dir * forward_weight + perpendicular * strafe_signal * lateral_weight
	if blended.length_squared() < 0.0001:
		return chase_dir
	return blended.normalized()


func take_damage(amount: float) -> float:
	if not alive or amount <= 0.0:
		return 0.0
	var applied: float = minf(amount, health)
	health -= applied
	if health <= 0.0:
		health = 0.0
		alive = false
		ai_state = AIState.DEAD
	return applied


func to_dict() -> Dictionary:
	return {
		"position": position,
		"health": health,
		"max_health": max_health,
		"alive": alive,
		"ai_state": ai_state,
		"move_speed": move_speed,
		"attack_damage": attack_damage,
		"attack_cooldown_remaining": attack_cooldown_remaining,
		"strafe_direction": strafe_direction,
	}


static func ai_state_name(state: int) -> String:
	match state:
		AIState.IDLE:
			return "idle"
		AIState.CHASE:
			return "chase"
		AIState.ATTACK:
			return "attack"
		AIState.DEAD:
			return "dead"
		_:
			return "unknown"

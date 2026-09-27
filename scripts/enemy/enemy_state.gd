## EnemyState
##
## Pure simulation state + deterministic AI for a single enemy target.
## Supports configurable difficulty parameters, idle/chase/attack/dead states,
## and deterministic hit resolution.
class_name EnemyState
extends RefCounted

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
func update_ai(
	dt: float,
	agent_position: Vector3,
	arena_half_extent: float,
	enable_movement: bool = true,
	enable_attack: bool = true
) -> float:
	if attack_cooldown_remaining > 0.0:
		attack_cooldown_remaining = maxf(0.0, attack_cooldown_remaining - dt)

	if not alive:
		ai_state = AIState.DEAD
		return 0.0

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
		position += dir * move_speed * dt
	var limit: float = arena_half_extent - radius
	position.x = clampf(position.x, -limit, limit)
	position.z = clampf(position.z, -limit, limit)
	position.y = 0.0
	return 0.0


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
	}

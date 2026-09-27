## EnemyState
##
## Pure simulation state + minimal deterministic AI for a single enemy
## target. No pathfinding, no perception cones, no complex behavior tree —
## just idle / chase / attack, which is enough to give the agent a
## meaningful combat target for milestone 1.
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


func reset(spawn_position: Vector3 = SandboxConfig.ENEMY_SPAWN_POSITION) -> void:
	position = spawn_position
	health = max_health
	alive = true
	ai_state = AIState.IDLE
	attack_cooldown_remaining = 0.0


func get_chest_position() -> Vector3:
	return position + Vector3(0.0, chest_height, 0.0)


## Advances the enemy AI by one tick. Returns the amount of damage the
## enemy dealt to the agent this tick (0 if it did not attack).
func update_ai(dt: float, agent_position: Vector3, arena_half_extent: float) -> float:
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

	if distance <= attack_range:
		ai_state = AIState.ATTACK
		if attack_cooldown_remaining <= 0.0:
			attack_cooldown_remaining = attack_cooldown_time
			return attack_damage
		return 0.0

	ai_state = AIState.CHASE
	var dir: Vector3 = to_agent.normalized()
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

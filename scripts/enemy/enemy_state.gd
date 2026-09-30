## EnemyState
##
## Pure simulation state + deterministic AI for a single enemy target.
## Supports configurable difficulty parameters, idle/chase/attack/dead states,
## and deterministic hit resolution.
##
## Two behavior paths share this state object:
##   * `update_ai()` — the original analytic chase/strafe/melee behavior,
##     preserved unchanged for curriculum levels 1-4 so previously trained
##     policies and existing tests keep their exact dynamics.
##   * `EnemyBrain` — the tactical layer (perception, memory, cover, peek,
##     search, ranged fire, jumping) used from the obstacles/cover level
##     upwards. It reads and writes the same fields declared here.
##
## Death is permanent for the episode: `mark_dead()` flips `alive` to false
## and `corpse` to true. A corpse is never targetable, never perceived as an
## enemy and never emits sound; it only remains as environmental
## information (see `to_corpse_dict()`).
class_name EnemyState
extends RefCounted

enum AIState {
	IDLE = 0,
	CHASE = 1,
	ATTACK = 2,
	DEAD = 3,
	ALERT = 4,
	ENGAGE = 5,
	TAKE_COVER = 6,
	PEEK = 7,
	SEARCH = 8,
	## Moving toward a noise the enemy heard but has never seen. Kept
	## distinct from SEARCH, which walks to a position it actually saw.
	INVESTIGATE = 9,
	## Critically hurt with no cover available: breaking away from the
	## believed threat instead of trading.
	RETREAT = 10,
}

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const CharacterMotor = preload("res://scripts/world/character_motor.gd")
const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const NavigationAgent = preload("res://scripts/world/navigation_agent.gd")
const ReactionProfile = preload("res://scripts/perception/reaction_profile.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const WeaponState = preload("res://scripts/weapon/weapon_state.gd")

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

# ---------------------------------------------------------------------------
# World / vertical movement state (Phase 2)
# ---------------------------------------------------------------------------
var velocity: Vector3 = Vector3.ZERO
var height: float = SandboxConfig.AGENT_HEIGHT
var eye_height: float = SandboxConfig.AGENT_EYE_HEIGHT
var on_ground: bool = true
var yaw_deg: float = 0.0
var footstep_timer: float = 0.0

# ---------------------------------------------------------------------------
# Identity, corpse and tactical state (Phases 4-8)
# ---------------------------------------------------------------------------
## Stable index inside the owning EnvironmentCore's enemy list. Used as the
## memory/sound source key; -1 means "not yet assigned".
var enemy_id: int = -1
## Permanent dead-body flag. Once true it never reverts during an episode.
var corpse: bool = false
var death_time: float = -1.0
var death_position: Vector3 = Vector3.ZERO

## Ranged weapon, used only from the curriculum levels that enable it.
var weapon: WeaponState = WeaponState.new()
## What this enemy believes about the agent (Phase 4).
var memory: EnemyMemory = EnemyMemory.create()
## Reaction latency archetype (Phase 7).
var reaction: ReactionProfile = ReactionProfile.create()

## Continuous seconds the agent has been geometrically visible. Compared
## against the reaction profile's detection delay.
var visual_contact_time: float = 0.0
## Seconds since visual contact was lost (reset while in contact).
var time_since_visual: float = 0.0
## Seconds the current tactical state has been active.
var state_time: float = 0.0
## Seconds spent searching for a lost target.
var search_time: float = 0.0
## Continuous seconds spent standing in a mutually visible position while
## engaging. Line of sight is symmetric, so an enemy that can see the agent
## can also work out that it is itself standing in the open.
var exposure_time: float = 0.0
## Whether this enemy has an active, reaction-confirmed target.
var target_confirmed: bool = false
## Destination the tactical layer is currently moving toward.
var tactical_destination: Vector3 = Vector3.ZERO
var has_tactical_destination: bool = false
## Human-readable reason the brain chose its current state. Debug/UI only.
var tactical_reason: String = "idle"
## Path-following / stuck-recovery state. Only consulted once direct
## steering demonstrably fails, so open layouts pay nothing for it.
var navigation: NavigationAgent = NavigationAgent.create()


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
	corpse = false
	death_time = -1.0
	death_position = Vector3.ZERO
	ai_state = AIState.IDLE
	attack_cooldown_remaining = 0.0
	time_alive = 0.0
	velocity = Vector3.ZERO
	on_ground = true
	footstep_timer = 0.0
	yaw_deg = 0.0
	visual_contact_time = 0.0
	time_since_visual = 0.0
	state_time = 0.0
	search_time = 0.0
	exposure_time = 0.0
	target_confirmed = false
	has_tactical_destination = false
	tactical_destination = Vector3.ZERO
	tactical_reason = "idle"
	navigation.reset(spawn_position)
	weapon.reset()
	memory.clear()


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


## Centre of the head hit-sphere. Deliberately independent of
## `eye_height` (which is a PERCEPTION origin, not a hitbox) so tuning one
## never silently moves the other.
func get_head_position() -> Vector3:
	return position + Vector3(0.0, SandboxConfig.ENEMY_HEAD_HEIGHT, 0.0)


func get_eye_position() -> Vector3:
	return position + Vector3(0.0, eye_height, 0.0)


func get_forward_horizontal() -> Vector3:
	var yaw_rad: float = deg_to_rad(yaw_deg)
	return Vector3(sin(yaw_rad), 0.0, -cos(yaw_rad))


## Faces a world point. Used by the tactical layer so an enemy's FOV cone
## actually points where it is going/looking.
func face_towards(target_position: Vector3, dt: float, turn_speed: float) -> void:
	var delta := Vector3(target_position.x - position.x, 0.0, target_position.z - position.z)
	if delta.is_zero_approx():
		return
	var desired: float = rad_to_deg(atan2(delta.x, -delta.z))
	var difference: float = wrapf(desired - yaw_deg, -180.0, 180.0)
	var step: float = clampf(difference, -turn_speed * dt, turn_speed * dt)
	yaw_deg = wrapf(yaw_deg + step, 0.0, 360.0)


## True when this enemy can be targeted, perceived or heard. A corpse is
## excluded by every one of those systems through this single predicate.
func is_targetable() -> bool:
	return alive and not corpse


## Moves the enemy one tick with gravity, jumping and obstacle collision,
## using the same motor as the agent. Returns the motion events the caller
## turns into sound.
func move_towards(
	destination: Vector3,
	dt: float,
	arena_half_extent: float,
	world = null,
	speed_scale: float = 1.0,
	jump_requested: bool = false
) -> Dictionary:
	var delta := Vector3(destination.x - position.x, 0.0, destination.z - position.z)
	var wish: Vector3 = Vector3.ZERO
	if delta.length() > 0.05:
		wish = delta.normalized()
	var motion: Dictionary = CharacterMotor.step(
		world,
		position,
		velocity,
		wish,
		move_speed * maxf(0.0, speed_scale),
		dt,
		radius,
		height,
		jump_requested,
		on_ground,
		arena_half_extent
	)
	position = motion["position"]
	velocity = motion["velocity"]
	on_ground = bool(motion["on_ground"])

	var footstep: bool = false
	var horizontal_speed: float = Vector2(velocity.x, velocity.z).length()
	if on_ground and horizontal_speed > 0.05:
		footstep_timer -= dt
		if footstep_timer <= 0.0:
			footstep_timer = SandboxConfig.FOOTSTEP_INTERVAL
			footstep = true
	return {
		"jumped": bool(motion["jumped"]),
		"landed": bool(motion["landed"]),
		"footstep": footstep,
		"moving": horizontal_speed > 0.05,
	}


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
	var limit: float = maxf(0.0, arena_half_extent - radius - SandboxConfig.ARENA_BOUNDS_EPSILON)
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
	var blended: Vector3 = (
		chase_dir * forward_weight + perpendicular * strafe_signal * lateral_weight
	)
	if blended.length_squared() < 0.0001:
		return chase_dir
	return blended.normalized()


func take_damage(amount: float) -> float:
	if not alive or amount <= 0.0:
		return 0.0
	var applied: float = minf(amount, health)
	health -= applied
	if health <= 0.0:
		mark_dead()
	return applied


## Permanently kills this enemy for the rest of the episode and turns it
## into an inert corpse. Idempotent.
func mark_dead(at_time: float = -1.0) -> void:
	if corpse:
		return
	health = 0.0
	alive = false
	corpse = true
	ai_state = AIState.DEAD
	death_time = at_time
	death_position = position
	velocity = Vector3.ZERO
	target_confirmed = false
	has_tactical_destination = false
	tactical_reason = "dead"
	memory.clear()


## Environmental information about a dead body. Never includes anything a
## live enemy would expose (no ai_state, no target, no memory).
func to_corpse_dict() -> Dictionary:
	return {
		"id": enemy_id,
		"position": death_position,
		"death_time": death_time,
		"targetable": false,
	}


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
		"corpse": corpse,
		"enemy_id": enemy_id,
		"on_ground": on_ground,
		"yaw_deg": yaw_deg,
		"tactical_reason": tactical_reason,
		"target_confirmed": target_confirmed,
		"time_since_visual": time_since_visual,
		"navigation": navigation.to_dict(),
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
		AIState.ALERT:
			return "alert"
		AIState.ENGAGE:
			return "engage"
		AIState.TAKE_COVER:
			return "take_cover"
		AIState.PEEK:
			return "peek"
		AIState.SEARCH:
			return "search"
		AIState.INVESTIGATE:
			return "investigate"
		AIState.RETREAT:
			return "retreat"
		_:
			return "unknown"

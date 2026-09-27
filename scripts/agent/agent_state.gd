## AgentState
##
## Pure simulation/RL state for the player-controlled combat agent. This
## class has NO dependency on the scene tree, rendering, or input devices —
## it can be created and stepped with `AgentState.new()` inside a unit test
## with zero Godot nodes involved. A separate `AgentView` (Node3D) is
## responsible for turning this state into something visible on screen.
##
## Movement/aiming is resolved analytically (no PhysicsServer stepping) so
## that many environments can be simulated far faster than real time and
## with fully deterministic results.
class_name AgentState
extends RefCounted

var position: Vector3 = SandboxConfig.AGENT_SPAWN_POSITION
var velocity: Vector3 = Vector3.ZERO
var yaw_deg: float = SandboxConfig.AGENT_SPAWN_YAW_DEG
var pitch_deg: float = 0.0

var max_health: float = SandboxConfig.AGENT_MAX_HEALTH
var health: float = SandboxConfig.AGENT_MAX_HEALTH
var alive: bool = true

var move_speed: float = SandboxConfig.AGENT_MOVE_SPEED
var turn_speed_deg: float = SandboxConfig.AGENT_TURN_SPEED_DEG
var pitch_limit_deg: float = SandboxConfig.AGENT_PITCH_LIMIT_DEG
var eye_height: float = SandboxConfig.AGENT_EYE_HEIGHT
var radius: float = SandboxConfig.AGENT_RADIUS

var weapon: WeaponState = WeaponState.new()


func _init(
	p_max_health: float = SandboxConfig.AGENT_MAX_HEALTH,
	p_move_speed: float = SandboxConfig.AGENT_MOVE_SPEED,
	p_turn_speed: float = SandboxConfig.AGENT_TURN_SPEED_DEG,
	p_pitch_limit: float = SandboxConfig.AGENT_PITCH_LIMIT_DEG,
	p_eye_height: float = SandboxConfig.AGENT_EYE_HEIGHT,
	p_radius: float = SandboxConfig.AGENT_RADIUS
) -> void:
	max_health = maxf(1.0, p_max_health)
	health = max_health
	move_speed = maxf(0.0, p_move_speed)
	turn_speed_deg = maxf(0.0, p_turn_speed)
	pitch_limit_deg = maxf(0.0, p_pitch_limit)
	eye_height = maxf(0.1, p_eye_height)
	radius = maxf(0.01, p_radius)
	alive = true
	weapon = WeaponState.new()


## Resets the agent to a fresh episode-start state.
func reset(
	spawn_position: Vector3 = SandboxConfig.AGENT_SPAWN_POSITION,
	spawn_yaw_deg: float = SandboxConfig.AGENT_SPAWN_YAW_DEG
) -> void:
	position = spawn_position
	velocity = Vector3.ZERO
	yaw_deg = spawn_yaw_deg
	pitch_deg = 0.0
	health = max_health
	alive = true
	weapon.reset()


func get_forward_vector() -> Vector3:
	var yaw_rad: float = deg_to_rad(yaw_deg)
	var pitch_rad: float = deg_to_rad(pitch_deg)
	var forward := Vector3(
		sin(yaw_rad) * cos(pitch_rad), sin(pitch_rad), -cos(yaw_rad) * cos(pitch_rad)
	)
	return forward.normalized() if not forward.is_zero_approx() else Vector3.FORWARD


func get_forward_horizontal() -> Vector3:
	var yaw_rad: float = deg_to_rad(yaw_deg)
	return Vector3(sin(yaw_rad), 0.0, -cos(yaw_rad))


func get_right_horizontal() -> Vector3:
	var yaw_rad: float = deg_to_rad(yaw_deg)
	return Vector3(cos(yaw_rad), 0.0, sin(yaw_rad))


func get_eye_position() -> Vector3:
	return position + Vector3(0.0, eye_height, 0.0)


## Applies one tick of a structured Action to movement, aim and weapon
## cooldown. Does NOT resolve combat (hit-testing/damage) — that is the
## responsibility of EnvironmentCore, which needs the enemy list.
func apply_action(action: Action, dt: float, arena_half_extent: float) -> void:
	weapon.tick(dt)
	if not alive:
		velocity = Vector3.ZERO
		return

	# --- Aim ---
	yaw_deg += action.look_yaw_axis * turn_speed_deg * dt
	yaw_deg += action.look_delta.x
	pitch_deg += action.look_pitch_axis * turn_speed_deg * dt
	pitch_deg += action.look_delta.y
	pitch_deg = clampf(pitch_deg, -pitch_limit_deg, pitch_limit_deg)
	yaw_deg = wrapf(yaw_deg, 0.0, 360.0)

	# --- Movement (horizontal only, no jumping/gravity in milestone 1) ---
	var forward: Vector3 = get_forward_horizontal()
	var right: Vector3 = get_right_horizontal()
	var move_dir: Vector3 = forward * float(action.move_axis) + right * float(action.strafe_axis)
	if move_dir.length_squared() > 1.0:
		move_dir = move_dir.normalized()
	velocity = move_dir * move_speed
	position += velocity * dt

	# --- Arena bounds clamp (walls) ---
	var limit: float = arena_half_extent - radius
	position.x = clampf(position.x, -limit, limit)
	position.z = clampf(position.z, -limit, limit)
	position.y = 0.0


func take_damage(amount: float) -> float:
	if not alive or amount <= 0.0:
		return 0.0
	var applied: float = minf(amount, health)
	health -= applied
	if health <= 0.0:
		health = 0.0
		alive = false
	return applied


func to_dict() -> Dictionary:
	return {
		"position": position,
		"velocity": velocity,
		"yaw_deg": yaw_deg,
		"pitch_deg": pitch_deg,
		"health": health,
		"max_health": max_health,
		"alive": alive,
		"weapon": weapon.to_dict(),
	}

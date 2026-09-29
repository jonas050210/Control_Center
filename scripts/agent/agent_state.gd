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
## with fully deterministic results. Since the world layer landed, movement
## also supports gravity, jumping, landing and axis-separated collision
## against `ArenaWorld` obstacles — all through the shared `CharacterMotor`
## so the agent and the scripted enemies obey identical physics. Passing
## `world = null` reproduces the original flat, obstacle-free behavior
## exactly, which is what the pre-world curriculum levels still use.
class_name AgentState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CharacterMotor = preload("res://scripts/world/character_motor.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const WeaponState = preload("res://scripts/weapon/weapon_state.gd")


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
var height: float = SandboxConfig.AGENT_HEIGHT

## Vertical movement state (Phase 2). `on_ground` is true while standing on
## the floor or on a standable box; jumping is only possible from there.
var on_ground: bool = true
## Seconds until the next footstep sound is emitted while moving. Counted
## down by apply_action() so footstep cadence is a function of simulation
## time rather than of frame rate.
var footstep_timer: float = 0.0

var weapon: WeaponState = WeaponState.new()

## Horizontal speed as a fraction of `move_speed`, sampled at the END of
## the previous tick. Weapon bloom is a function of how fast the shooter
## was actually moving when the trigger went down, so it has to be the
## resolved speed (after collisions) rather than the requested direction.
var speed_fraction: float = 0.0
## Recoil recovery applied to the aim on the last tick, in degrees. Purely
## diagnostic (Control Center / metrics); the aim itself already moved.
var last_recoil_recovery: Vector2 = Vector2.ZERO
## True on the tick a reload completed. Diagnostics only.
var reload_finished: bool = false
## Recoil the view has ACTUALLY absorbed, in degrees (x = yaw, y = pitch).
##
## The weapon accumulates the full kick it generated, but the view clamps
## pitch at `pitch_limit_deg`. Firing while already aimed near vertical
## therefore produces a kick the view only partly takes, and recovering the
## weapon's full figure would subtract more than was ever added and walk
## the aim away from where the agent left it. Recovery is clamped to this.
var _recoil_absorbed: Vector2 = Vector2.ZERO


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
	height = maxf(eye_height + 0.05, SandboxConfig.AGENT_HEIGHT)
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
	on_ground = true
	footstep_timer = 0.0
	speed_fraction = 0.0
	last_recoil_recovery = Vector2.ZERO
	reload_finished = false
	_recoil_absorbed = Vector2.ZERO
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


## Inverse of `get_forward_horizontal()`: aims the agent along `direction`
## on the horizontal plane. Aiming is stored as yaw/pitch, so there is no
## writable `forward` property; this is the supported way to point the
## agent at something from a test or a scripted setup.
func set_forward_horizontal(direction: Vector3) -> void:
	var flat := Vector3(direction.x, 0.0, direction.z)
	if flat.is_zero_approx():
		return
	yaw_deg = rad_to_deg(atan2(flat.x, -flat.z))


func get_right_horizontal() -> Vector3:
	var yaw_rad: float = deg_to_rad(yaw_deg)
	return Vector3(cos(yaw_rad), 0.0, sin(yaw_rad))


func get_eye_position() -> Vector3:
	return position + Vector3(0.0, eye_height, 0.0)


## Applies one tick of a structured Action to movement, aim and weapon
## cooldown. Does NOT resolve combat (hit-testing/damage) — that is the
## responsibility of EnvironmentCore, which needs the enemy list.
##
## Returns the motion events the caller turns into sound:
##   {"jumped": bool, "landed": bool, "footstep": bool, "moving": bool}
func apply_action(
	action: Action, dt: float, arena_half_extent: float, world = null
) -> Dictionary:
	# Cooldown, reload, recoil recovery and bloom recovery. With handling
	# disabled `tick_handling` is exactly the old `weapon.tick(dt)`.
	var recovery: Dictionary = weapon.tick_handling(dt)
	reload_finished = bool(recovery.get("reload_finished", false))
	var events: Dictionary = {
		"jumped": false, "landed": false, "footstep": false, "moving": false
	}
	if not alive:
		velocity = Vector3.ZERO
		speed_fraction = 0.0
		return events

	# --- Aim ---
	# Recoil recovery is applied BEFORE the policy's own look input so the
	# agent's correction always acts on the settled view of this tick. The
	# sign convention matches the kick applied in `apply_recoil`.
	var recovery_pitch: float = float(recovery.get("pitch_deg", 0.0))
	var recovery_yaw: float = float(recovery.get("yaw_deg", 0.0))
	# Never hand back more than the view took (see `_recoil_absorbed`).
	recovery_pitch = clampf(
		recovery_pitch, minf(0.0, _recoil_absorbed.y), maxf(0.0, _recoil_absorbed.y)
	)
	recovery_yaw = clampf(
		recovery_yaw, minf(0.0, _recoil_absorbed.x), maxf(0.0, _recoil_absorbed.x)
	)
	_recoil_absorbed -= Vector2(recovery_yaw, recovery_pitch)
	last_recoil_recovery = Vector2(recovery_yaw, recovery_pitch)
	pitch_deg -= recovery_pitch
	yaw_deg -= recovery_yaw
	yaw_deg += action.look_yaw_axis * turn_speed_deg * dt
	yaw_deg += action.look_delta.x
	pitch_deg += action.look_pitch_axis * turn_speed_deg * dt
	pitch_deg += action.look_delta.y
	pitch_deg = clampf(pitch_deg, -pitch_limit_deg, pitch_limit_deg)
	yaw_deg = wrapf(yaw_deg, 0.0, 360.0)

	# --- Movement ---
	var forward: Vector3 = get_forward_horizontal()
	var right: Vector3 = get_right_horizontal()
	var move_dir: Vector3 = forward * float(action.move_axis) + right * float(action.strafe_axis)
	if move_dir.length_squared() > 1.0:
		move_dir = move_dir.normalized()

	# "Shooting slows you down": a shot plants the shooter for a fraction of
	# a second. Returns exactly 1.0 with handling disabled.
	var effective_speed: float = move_speed * weapon.movement_speed_scale()
	var motion: Dictionary = CharacterMotor.step(
		world,
		position,
		velocity,
		move_dir,
		effective_speed,
		dt,
		radius,
		height,
		action.jump,
		on_ground,
		arena_half_extent
	)
	position = motion["position"]
	velocity = motion["velocity"]
	on_ground = bool(motion["on_ground"])
	events["jumped"] = bool(motion["jumped"])
	events["landed"] = bool(motion["landed"])

	# --- Footsteps ---
	var horizontal_speed: float = Vector2(velocity.x, velocity.z).length()
	speed_fraction = clampf(horizontal_speed / maxf(move_speed, 0.0001), 0.0, 1.0)
	events["moving"] = horizontal_speed > 0.05
	if on_ground and events["moving"]:
		footstep_timer -= dt
		if footstep_timer <= 0.0:
			footstep_timer = SandboxConfig.FOOTSTEP_INTERVAL
			events["footstep"] = true
	else:
		footstep_timer = minf(footstep_timer, SandboxConfig.FOOTSTEP_INTERVAL * 0.5)
	return events


## Applies a weapon recoil kick to the agent's own view.
##
## Recoil deliberately moves the REAL aim (and therefore `agent_forward`
## in the observation) rather than a cosmetic offset: that is what makes it
## something the policy can perceive and counter with the look axes without
## adding a single field to the 84-float contract.
func apply_recoil(pitch_kick_deg: float, yaw_kick_deg: float) -> void:
	if pitch_kick_deg == 0.0 and yaw_kick_deg == 0.0:
		return
	var previous_pitch: float = pitch_deg
	pitch_deg = clampf(pitch_deg + pitch_kick_deg, -pitch_limit_deg, pitch_limit_deg)
	yaw_deg = wrapf(yaw_deg + yaw_kick_deg, 0.0, 360.0)
	# Yaw wraps instead of clamping, so it always absorbs the whole kick;
	# pitch only absorbs what the limit allowed.
	_recoil_absorbed += Vector2(yaw_kick_deg, pitch_deg - previous_pitch)


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
		"on_ground": on_ground,
		"height": height,
		"speed_fraction": speed_fraction,
		"weapon": weapon.to_dict(),
	}

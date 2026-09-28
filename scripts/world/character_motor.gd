## CharacterMotor
##
## The one place horizontal movement, gravity, jumping, landing and
## obstacle collision are integrated. Both AgentState and EnemyState route
## their per-tick motion through `step()`, so the agent and its opponents
## obey exactly the same physics — an RL environment in which the policy
## and the scripted enemies used different movement rules would teach the
## wrong thing.
##
## It is a stateless static helper operating on plain values (no Nodes, no
## PhysicsServer, no allocation beyond the returned Dictionary), which keeps
## it usable from hundreds of parallel headless environments.
class_name CharacterMotor
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


## Integrates one tick of motion.
##
## `wish_direction` is a horizontal unit-ish vector in world space (its
## length is treated as the throttle and clamped to 1). `world` may be null,
## in which case only the square arena bounds constrain movement and the
## floor is flat at y = 0 — this is the legacy analytic behavior used by the
## pre-world curriculum levels, preserved bit-for-bit so existing training
## results stay reproducible.
##
## Returns {position, velocity, on_ground, landed, jumped}.
static func step(
	world,
	position: Vector3,
	velocity: Vector3,
	wish_direction: Vector3,
	move_speed: float,
	dt: float,
	radius: float,
	height: float,
	jump_requested: bool,
	on_ground: bool,
	arena_half_extent: float
) -> Dictionary:
	var wish := Vector3(wish_direction.x, 0.0, wish_direction.z)
	if wish.length_squared() > 1.0:
		wish = wish.normalized()

	# Airborne characters keep most of their horizontal momentum and only
	# steer with AIR_CONTROL, which is what makes a jump commit the player
	# to a trajectory instead of being a free dodge.
	var horizontal := Vector3(velocity.x, 0.0, velocity.z)
	var target: Vector3 = wish * move_speed
	if on_ground:
		horizontal = target
	else:
		horizontal = horizontal.lerp(target, clampf(SandboxConfig.AIR_CONTROL, 0.0, 1.0))

	var vertical: float = velocity.y
	var jumped: bool = false
	if jump_requested and on_ground:
		vertical = SandboxConfig.JUMP_VELOCITY
		jumped = true
		on_ground = false
	if not on_ground or vertical > 0.0:
		vertical += SandboxConfig.GRAVITY * dt

	var desired: Vector3 = position + Vector3(horizontal.x, vertical, horizontal.z) * dt

	var resolved: Vector3
	var floor_y: float = 0.0
	if world == null:
		resolved = desired
		var limit: float = maxf(0.0, arena_half_extent - radius - ArenaWorld.BOUNDS_EPSILON)
		resolved.x = clampf(resolved.x, -limit, limit)
		resolved.z = clampf(resolved.z, -limit, limit)
	else:
		var arena: ArenaWorld = world as ArenaWorld
		# Horizontal first, at the CURRENT height: resolve_move() is a
		# horizontal collision solver, and asking it to also place the feet
		# vertically made a descending character hover forever above a
		# standable box (the downward step ends "inside" the box, so it was
		# refused, and the landing test below then never fired).
		resolved = arena.resolve_move(
			position, Vector3(desired.x, position.y, desired.z), radius, height
		)
		floor_y = arena.ground_height(resolved, radius, position.y)
		# Vertical is resolved here: upward motion is stopped by a ceiling
		# (head bump), downward motion is caught by the landing test below.
		var target_y: float = desired.y
		if (
			vertical > 0.0
			and arena.is_blocked(Vector3(resolved.x, target_y, resolved.z), radius, height)
		):
			target_y = position.y
			vertical = 0.0
		resolved.y = target_y

	var landed: bool = false
	if resolved.y <= floor_y + SandboxConfig.LANDING_EPSILON and vertical <= 0.0:
		if not on_ground:
			landed = true
		resolved.y = floor_y
		vertical = 0.0
		on_ground = true
	else:
		on_ground = false

	# Blocked horizontally? Zero that component so the character stops
	# against the wall instead of accumulating phantom speed.
	var moved := Vector3(resolved.x - position.x, 0.0, resolved.z - position.z)
	if dt > 0.0:
		if absf(moved.x) < 0.000001:
			horizontal.x = 0.0
		if absf(moved.z) < 0.000001:
			horizontal.z = 0.0

	return {
		"position": resolved,
		"velocity": Vector3(horizontal.x, vertical, horizontal.z),
		"on_ground": on_ground,
		"landed": landed,
		"jumped": jumped,
	}

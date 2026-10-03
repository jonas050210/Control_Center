## PerceptionSystem
##
## Stateless visual-perception queries: field of view, line of sight and
## the combined "can this observer actually see that target right now?"
## test, plus the bearing/elevation helpers the observation vector needs.
##
## Everything is a pure function of (observer eye, facing, target, world),
## so both the learning agent and the scripted enemies use the exact same
## visibility rules. Nothing in here consults privileged state: a target
## behind a wall genuinely returns `visible = false`, and the caller is
## responsible for deciding whether to fall back on memory or sound.
class_name PerceptionSystem
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const VectorMath = preload("res://scripts/core/vector_math.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

## Extra sample points on the target used for the line-of-sight test, as
## fractions of the target's height. Testing chest AND head means a target
## crouching behind low cover is occluded while its head is not — which is
## the whole point of low cover.
const LOS_SAMPLE_HEIGHTS: Array = [0.55, 0.95]

## Sample heights used for the *exposure* measurement, as fractions of the
## target's height. This is a separate, finer set on purpose: `has_line_of_
## sight` is a boolean the enemy AI also consumes, and widening it would
## change how cover works for everyone. Exposure only ever describes what
## fraction of a body the agent can see, so it can afford five rungs from
## the shins to the head.
const EXPOSURE_SAMPLE_HEIGHTS: Array = [0.15, 0.4, 0.65, 0.85, 1.0]


## Signed horizontal angle (degrees) from `forward` to the direction
## observer -> target, positive to the observer's right, 0 dead ahead.
##
## Delegates to `VectorMath`: this used to be a second implementation, and its
## sign was the opposite of the one `Observation` uses for the same quantity.
static func bearing_deg(forward: Vector3, from_position: Vector3, to_position: Vector3) -> float:
	return VectorMath.signed_bearing_deg(forward, from_position, to_position)


## Vertical angle (degrees) from the observer's eye to a point, positive
## above the eye. Delegates to `VectorMath`, which owns the convention (this
## used to be a second copy of the same formula, like the bearing below).
static func elevation_deg(from_eye: Vector3, to_position: Vector3) -> float:
	return VectorMath.elevation_deg(from_eye, to_position)


## Whether a point lies inside the horizontal FOV cone. `fov_deg` is the
## TOTAL cone angle, so the half-angle test uses fov_deg * 0.5.
static func in_field_of_view(
	forward: Vector3, from_position: Vector3, to_position: Vector3, fov_deg: float
) -> bool:
	return absf(bearing_deg(forward, from_position, to_position)) <= maxf(0.0, fov_deg) * 0.5


## Whether geometry leaves at least one unobstructed line between the
## observer's eye and the target body. `world` may be null (empty arena),
## in which case sight is always clear.
static func has_line_of_sight(
	world, from_eye: Vector3, target_feet: Vector3, target_height: float
) -> bool:
	if world == null:
		return true
	var arena: ArenaWorld = world
	for fraction_value in LOS_SAMPLE_HEIGHTS:
		var sample: Vector3 = target_feet + Vector3(0.0, target_height * float(fraction_value), 0.0)
		if not arena.segment_blocked(from_eye, sample):
			return true
	return false


## How much of a target's body the observer can actually see, in [0, 1].
##
## `has_line_of_sight` answers "is any part of it visible" and returns `true`
## for a silhouette that is nine tenths behind a crate, because a boolean has
## nowhere to put the difference between exposed and almost covered. This is
## that missing number: the share of the sampled body that is unoccluded.
## Zero means "it is behind something"; it never means "it is not there".
static func exposure_fraction(
	world, from_eye: Vector3, target_feet: Vector3, target_height: float
) -> float:
	if world == null:
		return 1.0
	var arena: ArenaWorld = world
	var clear_count: int = 0
	for fraction_value in EXPOSURE_SAMPLE_HEIGHTS:
		var sample: Vector3 = target_feet + Vector3(0.0, target_height * float(fraction_value), 0.0)
		if not arena.segment_blocked(from_eye, sample):
			clear_count += 1
	return float(clear_count) / float(EXPOSURE_SAMPLE_HEIGHTS.size())


## A target's silhouette as a box on the observer's own screen.
##
## This is the quantity a detection model would hand a consumer: not "there is
## something at 12 m, 20 degrees left" but "it covers this rectangle of what I
## am looking at". Every returned value is in normalised device coordinates:
## the view spans [-1, 1] on both axes, +x is the observer's RIGHT (the sign
## convention `VectorMath.signed_bearing_deg` owns) and +y is UP.
##
## `half_width` / `half_height` are the box's half-extents in the same space,
## so the box is `centre +/- half_extent` and the test "is my crosshair on it"
## is `abs(centre) <= half_extent` on both axes - the one computation a policy
## should not have to spend capacity on.
##
## A target at or behind the eye has no box: `in_front` goes false and every
## value is zero, which is the honest answer ("it is not on my screen")
## rather than a clamp at the screen edge that would read as "on my left".
static func target_screen_box(
	eye: Vector3,
	forward: Vector3,
	target_feet: Vector3,
	target_height: float,
	target_radius: float,
	fov_deg: float = SandboxConfig.AGENT_FOV_DEG,
	aspect: float = SandboxConfig.AGENT_VIEW_ASPECT,
) -> Dictionary:
	var empty: Dictionary = {
		"in_front": false,
		"center_x": 0.0,
		"center_y": 0.0,
		"half_width": 0.0,
		"half_height": 0.0,
		"depth": 0.0,
	}
	var flat_forward := Vector3(forward.x, 0.0, forward.z)
	if flat_forward.is_zero_approx():
		return empty
	flat_forward = flat_forward.normalized()
	# `right` is the +x of the agent's screen. Godot's frame is right-handed
	# with Y up, so forward.cross(UP) is the direction a positive bearing
	# (and a positive look_yaw_axis) turns towards - the same side the
	# bearing fields report as positive.
	var right := flat_forward.cross(Vector3.UP).normalized()
	var up_view := right.cross(flat_forward).normalized()
	var center: Vector3 = target_feet + Vector3(0.0, target_height * 0.5, 0.0)
	var delta: Vector3 = center - eye
	var depth: float = delta.dot(flat_forward)
	if depth <= 0.0001:
		return empty
	var tan_half_horizontal: float = tan(deg_to_rad(maxf(fov_deg, 1.0)) * 0.5)
	var tan_half_vertical: float = tan_half_horizontal / maxf(aspect, 0.0001)
	return {
		"in_front": true,
		"center_x": clampf(delta.dot(right) / (depth * tan_half_horizontal), -1.0, 1.0),
		"center_y": clampf(delta.dot(up_view) / (depth * tan_half_vertical), -1.0, 1.0),
		"half_width": clampf(target_radius / (depth * tan_half_horizontal), 0.0, 1.0),
		"half_height": clampf(target_height * 0.5 / (depth * tan_half_vertical), 0.0, 1.0),
		"depth": depth,
	}


## Full visibility evaluation for one target.
##
## Returns a Dictionary with the raw geometric facts. It deliberately does
## NOT apply reaction latency — that is `ReactionProfile`/the caller's
## tracker job, because latency is per-observer state while this is a
## stateless query.
##
##   in_range     -- within `max_range`
##   in_fov       -- inside the horizontal FOV cone
##   los_clear    -- geometry does not occlude the body
##   visible      -- all three of the above
##   distance     -- meters, observer feet to target feet
##   bearing_deg  -- signed horizontal aim offset
##   elevation_deg-- signed vertical angle from the eye
static func evaluate_target(
	world,
	observer_eye: Vector3,
	observer_feet: Vector3,
	forward: Vector3,
	target_feet: Vector3,
	target_height: float,
	fov_deg: float = SandboxConfig.AGENT_FOV_DEG,
	max_range: float = SandboxConfig.VISION_RANGE
) -> Dictionary:
	var distance: float = observer_feet.distance_to(target_feet)
	var in_range: bool = distance <= max_range
	var in_fov: bool = in_field_of_view(forward, observer_feet, target_feet, fov_deg)
	var los_clear: bool = has_line_of_sight(world, observer_eye, target_feet, target_height)
	return {
		"in_range": in_range,
		"in_fov": in_fov,
		"los_clear": los_clear,
		"visible": in_range and in_fov and los_clear,
		"distance": distance,
		"bearing_deg": bearing_deg(forward, observer_feet, target_feet),
		"elevation_deg": elevation_deg(observer_eye, target_feet + Vector3(0.0, 1.2, 0.0)),
	}


## Distance to the first sight-blocking surface straight ahead, clamped to
## `max_distance`. Exposed to the policy as "how far can I see / how close
## is the corner I am about to round".
static func forward_clearance(
	world, from_eye: Vector3, forward: Vector3, max_distance: float
) -> float:
	if world == null:
		return max_distance
	var hit: float = (world as ArenaWorld).ray_hit_distance(from_eye, forward, max_distance)
	return max_distance if hit < 0.0 else hit

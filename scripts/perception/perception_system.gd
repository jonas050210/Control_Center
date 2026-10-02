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

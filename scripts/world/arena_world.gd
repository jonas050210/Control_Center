## ArenaWorld
##
## The world/arena representation every scenario generates into: a square
## floor of `half_extent`, a wall height, and a flat list of axis-aligned
## `Obstacle` boxes. Nothing here is a Godot Node — an ArenaWorld is pure
## data plus queries, so a headless training process can own hundreds of
## them without touching the scene tree or the renderer.
##
## Queries provided to the rest of the simulation:
##   * line-of-sight / weapon-ray occlusion (`segment_blocked`,
##     `segment_hit_fraction`, `ray_hit_distance`)
##   * character movement with collision (`resolve_move`)
##   * ground/standing height for gravity and jumping (`ground_height`)
##   * spawn and cover-point sampling (`sample_free_position`,
##     `find_cover_position`)
##
## Cost model: every query is O(number of obstacles) with no allocation in
## the common path. Generated layouts stay in the 4-30 box range, which
## keeps a full perception update at a few hundred float comparisons.
class_name ArenaWorld
extends RefCounted

const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SELF_PATH: String = "res://scripts/world/arena_world.gd"

## Vertical tolerance for "this surface is close enough below my feet to
## stand on / step up onto" (meters).
const STEP_UP_TOLERANCE: float = 0.35

## Safety margin subtracted from the arena bound clamp (meters).
##
## `half_extent - radius` is evaluated in 32-bit floats inside a Vector3, so
## the clamped coordinate can round to a value a few 1e-7 ABOVE the nominal
## limit and put the character fractionally outside the arena. The margin is
## three orders of magnitude smaller than the character radius, so it changes
## no gameplay, but it makes "the clamp never leaves the arena" exactly true
## instead of true-up-to-rounding.
const BOUNDS_EPSILON: float = SandboxConfig.ARENA_BOUNDS_EPSILON

## Maximum horizontal distance resolved in one collision substep (meters).
##
## `resolve_move()` used to test only the END position of a move, so any
## displacement larger than a box could tunnel straight through it (a single
## `resolve_move(-3, +3)` across a 2 m crate reported "no collision"). The
## move is now swept in substeps no longer than this, which is well under the
## smallest half-extent a generator emits. A normal 60 Hz tick moves ~0.08 m
## and therefore still costs exactly one substep.
const MAX_MOVE_SUBSTEP: float = 0.2
## Upper bound on substeps so a teleport-sized request stays O(1)-ish.
const MAX_MOVE_SUBSTEPS: int = 64

var half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var wall_height: float = SandboxConfig.ARENA_WALL_HEIGHT
var obstacles: Array = []  # Array[Obstacle]
## Identifier of the layout generator that produced this world, e.g.
## "open_arena" or "corner". Telemetry/debug only.
var layout_id: String = "open_arena"
## Seed the layout was generated from. Two ArenaWorlds built with the same
## generator and the same seed are identical box-for-box.
var layout_seed: int = 0


static func create(
	p_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT,
	p_wall_height: float = SandboxConfig.ARENA_WALL_HEIGHT
) -> ArenaWorld:
	var world: ArenaWorld = (load(SELF_PATH) as GDScript).new() as ArenaWorld
	world.half_extent = maxf(2.0, p_half_extent)
	world.wall_height = maxf(1.0, p_wall_height)
	return world


func clear() -> void:
	obstacles.clear()


func add_obstacle(obstacle: Obstacle) -> Obstacle:
	obstacle.obstacle_id = obstacles.size()
	obstacles.append(obstacle)
	return obstacle


## Convenience box builder used by the layout generators.
func add_box(
	p_center: Vector3, p_half_extents: Vector3, p_kind: int = Obstacle.Kind.WALL
) -> Obstacle:
	return add_obstacle(Obstacle.make(p_center, p_half_extents, p_kind))


func obstacle_count() -> int:
	return obstacles.size()


# ---------------------------------------------------------------------------
# Line of sight / ray casting
# ---------------------------------------------------------------------------


## Fraction along `from -> to` at which sight is first blocked, or -1.0 when
## the segment is clear. Only obstacles with `blocks_sight` participate.
func segment_hit_fraction(from: Vector3, to: Vector3) -> float:
	var best: float = -1.0
	for obstacle_value in obstacles:
		var obstacle: Obstacle = obstacle_value
		if not obstacle.blocks_sight:
			continue
		var t: float = obstacle.segment_intersection(from, to)
		if t >= 0.0 and (best < 0.0 or t < best):
			best = t
	return best


## True when geometry occludes the straight line between two points.
func segment_blocked(from: Vector3, to: Vector3) -> bool:
	return segment_hit_fraction(from, to) >= 0.0


## Number of sight-blocking boxes the segment passes through. Used by the
## sound model, where each wall attenuates but does not silence a sound.
func occluder_count(from: Vector3, to: Vector3) -> int:
	var count: int = 0
	for obstacle_value in obstacles:
		var obstacle: Obstacle = obstacle_value
		if obstacle.blocks_sight and obstacle.segment_intersection(from, to) >= 0.0:
			count += 1
	return count


## Distance from `origin` along unit `direction` to the first sight-blocking
## surface within `max_distance`, or -1.0 if nothing is hit. Weapon rays use
## this so a shot cannot pass through a wall to reach an enemy behind it.
func ray_hit_distance(origin: Vector3, direction: Vector3, max_distance: float) -> float:
	if direction.is_zero_approx() or max_distance <= 0.0:
		return -1.0
	var to: Vector3 = origin + direction.normalized() * max_distance
	var fraction: float = segment_hit_fraction(origin, to)
	if fraction < 0.0:
		return -1.0
	return fraction * max_distance


# ---------------------------------------------------------------------------
# Movement / collision / gravity
# ---------------------------------------------------------------------------


## True when an upright character of `radius`/`height` standing with its
## feet at `feet_position` would intersect solid geometry.
func is_blocked(feet_position: Vector3, radius: float, height: float) -> bool:
	for obstacle_value in obstacles:
		var obstacle: Obstacle = obstacle_value
		if not obstacle.blocks_movement:
			continue
		if obstacle.overlaps_character(feet_position, radius, height):
			return true
	return false


## Highest standable surface at or below `from_y + STEP_UP_TOLERANCE` under
## the character's footprint. Returns 0.0 (the arena floor) when nothing
## else supports it.
##
## The footprint is inflated by the FULL character radius, exactly like the
## collision test in `is_blocked()`. With a smaller support footprint there
## was a ring around every standable box that was solid for movement but not
## standable, so a character descending onto the edge of a platform fell
## into the box's collision volume and got stuck against its side.
func ground_height(feet_position: Vector3, radius: float, from_y: float) -> float:
	var best: float = 0.0
	var ceiling: float = from_y + STEP_UP_TOLERANCE
	for obstacle_value in obstacles:
		var obstacle: Obstacle = obstacle_value
		if not obstacle.standable:
			continue
		if not obstacle.contains_xz(feet_position, radius):
			continue
		var top: float = obstacle.top_y()
		if top <= ceiling and top > best:
			best = top
	return best


## Largest coordinate a character of `radius` may occupy on X/Z.
func movement_limit(radius: float) -> float:
	return maxf(0.0, half_extent - radius - BOUNDS_EPSILON)


## Axis-separated SWEPT horizontal move with wall sliding, then an
## arena-bounds clamp. Returns the resolved feet position.
##
## Axis separation (advance X, then Z) is what produces natural "slide along
## the wall" behavior for a strafing agent instead of sticking on contact.
## The sweep is what keeps a fast or teleport-sized move from passing
## through geometry: each axis advances in substeps of at most
## MAX_MOVE_SUBSTEP and stops at the last free one.
func resolve_move(
	from_position: Vector3, desired: Vector3, radius: float, height: float
) -> Vector3:
	var resolved: Vector3 = from_position
	var delta_x: float = desired.x - from_position.x
	var delta_z: float = desired.z - from_position.z
	var distance: float = sqrt(delta_x * delta_x + delta_z * delta_z)
	var substeps: int = clampi(
		int(ceil(distance / MAX_MOVE_SUBSTEP)), 1, MAX_MOVE_SUBSTEPS
	)
	var blocked_x: bool = false
	var blocked_z: bool = false
	for index in range(1, substeps + 1):
		var fraction: float = float(index) / float(substeps)
		if not blocked_x:
			var step_x := Vector3(
				from_position.x + delta_x * fraction, resolved.y, resolved.z
			)
			if is_blocked(step_x, radius, height):
				blocked_x = true
			else:
				resolved = step_x
		if not blocked_z:
			var step_z := Vector3(
				resolved.x, resolved.y, from_position.z + delta_z * fraction
			)
			if is_blocked(step_z, radius, height):
				blocked_z = true
			else:
				resolved = step_z
		if blocked_x and blocked_z:
			break

	# Vertical motion is resolved by the caller (gravity/jump); here we only
	# carry the requested height through and refuse to end up inside a box.
	var step_y := Vector3(resolved.x, desired.y, resolved.z)
	if not is_blocked(step_y, radius, height):
		resolved = step_y

	var limit: float = movement_limit(radius)
	resolved.x = clampf(resolved.x, -limit, limit)
	resolved.z = clampf(resolved.z, -limit, limit)
	return resolved


## True when an upright character fits at `feet_position` and stays inside
## the arena bounds. Uses the same limit `resolve_move()` clamps to, so a
## clamped position is never reported as out of bounds.
func is_position_free(feet_position: Vector3, radius: float, height: float) -> bool:
	var limit: float = movement_limit(radius)
	if absf(feet_position.x) > limit or absf(feet_position.z) > limit:
		return false
	return not is_blocked(feet_position, radius, height)


## Rejection-samples a free floor position using the caller's seeded RNG, so
## the same seed always yields the same spawns. Falls back to the arena
## center-ish best effort after `attempts` tries rather than looping forever.
func sample_free_position(
	rng: RandomNumberGenerator, radius: float, height: float, attempts: int = 24
) -> Vector3:
	var limit: float = half_extent - radius - 0.2
	var fallback := Vector3.ZERO
	for _index in range(maxi(1, attempts)):
		var candidate := Vector3(
			rng.randf_range(-limit, limit), 0.0, rng.randf_range(-limit, limit)
		)
		candidate.y = ground_height(candidate, radius, 0.0)
		if is_position_free(candidate, radius, height):
			return candidate
		fallback = candidate
	fallback.y = 0.0
	return fallback


## Samples a free position near `around` within `[min_radius, max_radius]`.
## Used by scenarios that want an enemy "just around that corner" rather
## than anywhere in the arena.
func sample_free_position_near(
	rng: RandomNumberGenerator,
	around: Vector3,
	min_radius: float,
	max_radius: float,
	radius: float,
	height: float,
	attempts: int = 24
) -> Vector3:
	for _index in range(maxi(1, attempts)):
		var angle: float = rng.randf_range(0.0, TAU)
		var distance: float = rng.randf_range(min_radius, maxf(min_radius, max_radius))
		var candidate := Vector3(
			around.x + cos(angle) * distance, 0.0, around.z + sin(angle) * distance
		)
		var limit: float = half_extent - radius - 0.2
		candidate.x = clampf(candidate.x, -limit, limit)
		candidate.z = clampf(candidate.z, -limit, limit)
		candidate.y = ground_height(candidate, radius, 0.0)
		if is_position_free(candidate, radius, height):
			return candidate
	return sample_free_position(rng, radius, height, attempts)


# ---------------------------------------------------------------------------
# Tactical queries
# ---------------------------------------------------------------------------


## Finds a nearby standing position from which `threat_eye` cannot see
## `eye_height` above the feet. Returns the searcher's own position when no
## better spot exists, so callers never have to handle a null.
##
## The candidate ring is deterministic (fixed angular steps, no RNG) which
## keeps enemy cover-seeking reproducible for a given world state.
func find_cover_position(
	from_position: Vector3,
	threat_eye: Vector3,
	search_radius: float,
	radius: float,
	height: float,
	eye_height: float
) -> Vector3:
	var best: Vector3 = from_position
	var best_distance: float = INF
	var rings: Array = [search_radius * 0.5, search_radius, search_radius * 1.5]
	for ring_value in rings:
		var ring: float = ring_value
		for step in range(12):
			var angle: float = float(step) * TAU / 12.0
			var candidate := Vector3(
				from_position.x + cos(angle) * ring, 0.0, from_position.z + sin(angle) * ring
			)
			var limit: float = half_extent - radius - 0.2
			candidate.x = clampf(candidate.x, -limit, limit)
			candidate.z = clampf(candidate.z, -limit, limit)
			candidate.y = ground_height(candidate, radius, 0.0)
			if not is_position_free(candidate, radius, height):
				continue
			var candidate_eye: Vector3 = candidate + Vector3(0.0, eye_height, 0.0)
			if not segment_blocked(candidate_eye, threat_eye):
				continue
			var distance: float = from_position.distance_squared_to(candidate)
			if distance < best_distance:
				best_distance = distance
				best = candidate
	return best


## Finds a nearby position that DOES see `threat_eye` — the "peek" spot an
## enemy steps out to after hiding. Returns `from_position` if none exists.
func find_peek_position(
	from_position: Vector3,
	threat_eye: Vector3,
	search_radius: float,
	radius: float,
	height: float,
	eye_height: float
) -> Vector3:
	var best: Vector3 = from_position
	var best_distance: float = INF
	for step in range(12):
		var angle: float = float(step) * TAU / 12.0
		var candidate := Vector3(
			from_position.x + cos(angle) * search_radius,
			0.0,
			from_position.z + sin(angle) * search_radius
		)
		var limit: float = half_extent - radius - 0.2
		candidate.x = clampf(candidate.x, -limit, limit)
		candidate.z = clampf(candidate.z, -limit, limit)
		candidate.y = ground_height(candidate, radius, 0.0)
		if not is_position_free(candidate, radius, height):
			continue
		var candidate_eye: Vector3 = candidate + Vector3(0.0, eye_height, 0.0)
		if segment_blocked(candidate_eye, threat_eye):
			continue
		var distance: float = from_position.distance_squared_to(candidate)
		if distance < best_distance:
			best_distance = distance
			best = candidate
	return best


## Distance to the nearest obstacle surface from a point, and its bearing
## relative to `forward`. Returned as {"distance": float, "bearing_deg":
## float, "kind": int}; distance is `max_distance` when nothing is near.
func nearest_obstacle_info(
	point: Vector3, forward: Vector3, max_distance: float
) -> Dictionary:
	var best_distance: float = max_distance
	var best_delta: Vector3 = Vector3.ZERO
	var best_kind: int = -1
	for obstacle_value in obstacles:
		var obstacle: Obstacle = obstacle_value
		if not obstacle.blocks_movement:
			continue
		var closest := Vector3(
			clampf(point.x, obstacle.min_corner().x, obstacle.max_corner().x),
			point.y,
			clampf(point.z, obstacle.min_corner().z, obstacle.max_corner().z)
		)
		var delta: Vector3 = closest - point
		delta.y = 0.0
		var distance: float = delta.length()
		if distance < best_distance:
			best_distance = distance
			best_delta = delta
			best_kind = obstacle.kind
	var bearing: float = 0.0
	if best_kind >= 0 and best_delta.length_squared() > 0.000001:
		var flat_forward := Vector3(forward.x, 0.0, forward.z)
		if not flat_forward.is_zero_approx():
			flat_forward = flat_forward.normalized()
			var direction: Vector3 = best_delta.normalized()
			var dot: float = clampf(flat_forward.dot(direction), -1.0, 1.0)
			var sign_value: float = signf(flat_forward.cross(direction).y)
			if sign_value == 0.0:
				sign_value = 1.0
			bearing = rad_to_deg(acos(dot)) * sign_value
	return {"distance": best_distance, "bearing_deg": bearing, "kind": best_kind}


func to_dict() -> Dictionary:
	var boxes: Array = []
	for obstacle_value in obstacles:
		boxes.append((obstacle_value as Obstacle).to_dict())
	return {
		"layout_id": layout_id,
		"layout_seed": layout_seed,
		"half_extent": half_extent,
		"wall_height": wall_height,
		"obstacle_count": obstacles.size(),
		"obstacles": boxes,
	}

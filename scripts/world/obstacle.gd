## Obstacle
##
## One axis-aligned box of world geometry. Everything the world layer knows
## about — walls, crates, pillars, low/high cover, corridor segments and
## standable platforms — is expressed as an AABB so that line-of-sight,
## collision and cover queries are a handful of float comparisons. That is
## deliberate: the simulation must stay cheap enough to run dozens of
## environments in parallel on a CPU with no PhysicsServer involved.
##
## Two independent flags describe what a box actually does:
##   * `blocks_sight`    -- occludes line-of-sight rays and weapon rays
##   * `blocks_movement` -- solid for horizontal character movement
##
## They are separate because low cover (crouch-height crates) blocks
## movement but only blocks sight for targets low to the ground, and a
## purely visual smoke/curtain (future) would block sight but not movement.
## The vertical extent of the box is what resolves "shoot over low cover":
## a ray that passes above `top_y()` is not occluded at all.
class_name Obstacle
extends RefCounted

enum Kind {
	WALL = 0,
	CRATE = 1,
	PILLAR = 2,
	LOW_COVER = 3,
	HIGH_COVER = 4,
	PLATFORM = 5,
	BOUNDARY = 6,
}

const SELF_PATH: String = "res://scripts/world/obstacle.gd"

## Center of the box in world space. `center.y` is the vertical MIDDLE of
## the box, so a 2 m tall wall sitting on the floor has center.y == 1.0.
var center: Vector3 = Vector3.ZERO
## Positive half-sizes along each axis.
var half_extents: Vector3 = Vector3.ONE
var kind: int = Kind.WALL
var blocks_sight: bool = true
var blocks_movement: bool = true
## Whether a character may stand on this box's top face. Used for the
## vertical/jump scenarios; walls and boundaries are not standable so a
## policy cannot cheese its way on top of the arena fence.
var standable: bool = false
## Stable identifier inside one generated layout, for telemetry/debug only.
var obstacle_id: int = -1


func _init(
	p_center: Vector3 = Vector3.ZERO, p_half_extents: Vector3 = Vector3.ONE, p_kind: int = Kind.WALL
) -> void:
	center = p_center
	half_extents = Vector3(
		maxf(0.01, absf(p_half_extents.x)),
		maxf(0.01, absf(p_half_extents.y)),
		maxf(0.01, absf(p_half_extents.z))
	)
	kind = p_kind
	blocks_sight = true
	blocks_movement = true
	standable = p_kind == Kind.CRATE or p_kind == Kind.PLATFORM or p_kind == Kind.LOW_COVER


## Factory used everywhere instead of `Obstacle.new(...)`: referencing the
## script's own `class_name` in a value context needs the editor global
## class cache, which does not exist during standalone
## `godot --headless --script ...` runs (same rationale as Action/Observation).
static func make(p_center: Vector3, p_half_extents: Vector3, p_kind: int = Kind.WALL) -> Obstacle:
	var script := load(SELF_PATH) as GDScript
	return script.new(p_center, p_half_extents, p_kind) as Obstacle


func min_corner() -> Vector3:
	return center - half_extents


func max_corner() -> Vector3:
	return center + half_extents


func top_y() -> float:
	return center.y + half_extents.y


func bottom_y() -> float:
	return center.y - half_extents.y


## Horizontal footprint test, optionally inflated by a character radius.
func contains_xz(point: Vector3, margin: float = 0.0) -> bool:
	return (
		absf(point.x - center.x) <= half_extents.x + margin
		and absf(point.z - center.z) <= half_extents.z + margin
	)


## Full 3D overlap test against an upright cylinder approximated by its
## bounding box (cheap and slightly conservative, which is the safe side
## for "do not let a character stand inside geometry").
func overlaps_character(feet_position: Vector3, radius: float, height: float) -> bool:
	if not contains_xz(feet_position, radius):
		return false
	var feet: float = feet_position.y
	var head: float = feet_position.y + height
	return head > bottom_y() and feet < top_y()


## Slab-method segment/AABB intersection. Returns the normalized parameter
## t in [0,1] along `from -> to` of the first intersection, or -1.0 when
## the segment misses the box entirely.
func segment_intersection(from: Vector3, to: Vector3) -> float:
	var delta: Vector3 = to - from
	var t_min: float = 0.0
	var t_max: float = 1.0
	var lo: Vector3 = min_corner()
	var hi: Vector3 = max_corner()

	for axis in range(3):
		var origin: float = from[axis]
		var direction: float = delta[axis]
		var low: float = lo[axis]
		var high: float = hi[axis]
		if absf(direction) < 0.000001:
			if origin < low or origin > high:
				return -1.0
			continue
		var t1: float = (low - origin) / direction
		var t2: float = (high - origin) / direction
		if t1 > t2:
			var swap: float = t1
			t1 = t2
			t2 = swap
		t_min = maxf(t_min, t1)
		t_max = minf(t_max, t2)
		if t_min > t_max:
			return -1.0
	return t_min


func to_dict() -> Dictionary:
	return {
		"id": obstacle_id,
		"kind": kind,
		"kind_name": kind_name(kind),
		"center": center,
		"half_extents": half_extents,
		"blocks_sight": blocks_sight,
		"blocks_movement": blocks_movement,
		"standable": standable,
	}


static func kind_name(value: int) -> String:
	match value:
		Kind.WALL:
			return "wall"
		Kind.CRATE:
			return "crate"
		Kind.PILLAR:
			return "pillar"
		Kind.LOW_COVER:
			return "low_cover"
		Kind.HIGH_COVER:
			return "high_cover"
		Kind.PLATFORM:
			return "platform"
		Kind.BOUNDARY:
			return "boundary"
		_:
			return "unknown"

## WorldGenerator
##
## Seeded, modular arena layouts. There is deliberately NO single hard-coded
## map: every layout is a small generator function that places boxes using a
## locally seeded RandomNumberGenerator, so `build(layout_id, seed)` is a
## pure function of its two arguments and the same seed always reproduces
## the same world box-for-box.
##
## Layout ids are stable strings (not enum ordinals) because they are
## persisted in scenario definitions, telemetry and the Control Center UI.
## Adding a layout means adding one `_build_*` function and one match arm.
class_name WorldGenerator
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

## Every layout this generator can produce, in curriculum-ish order.
const LAYOUT_IDS: Array = [
	"open_arena",
	"scattered_cover",
	"corner",
	"cover_field",
	"corridor",
	"rooms",
	"pillars",
	"vertical",
	"randomized",
]

## Thickness of a generated wall segment (half-extent along its thin axis).
const WALL_HALF_THICKNESS: float = 0.35
const LOW_COVER_HEIGHT: float = 1.0
const HIGH_COVER_HEIGHT: float = 2.4


## Builds one layout. Unknown ids fall back to the empty open arena rather
## than failing, so a stale scenario definition degrades instead of
## crashing a training run.
static func build(
	layout_id: String,
	seed_value: int = 0,
	half_extent: float = SandboxConfig.ARENA_HALF_EXTENT,
	wall_height: float = SandboxConfig.ARENA_WALL_HEIGHT
) -> ArenaWorld:
	var world: ArenaWorld = ArenaWorld.create(half_extent, wall_height)
	world.layout_id = layout_id
	world.layout_seed = seed_value
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value

	match layout_id:
		"open_arena":
			pass
		"scattered_cover":
			_build_scattered_cover(world, rng)
		"corner":
			_build_corner(world, rng)
		"cover_field":
			_build_cover_field(world, rng)
		"corridor":
			_build_corridor(world, rng)
		"rooms":
			_build_rooms(world, rng)
		"pillars":
			_build_pillars(world, rng)
		"vertical":
			_build_vertical(world, rng)
		"randomized":
			_build_randomized(world, rng)
		_:
			world.layout_id = "open_arena"
	return world


static func layout_ids() -> Array:
	return LAYOUT_IDS.duplicate()


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------


## A handful of crates and one short wall, randomly placed. The gentlest
## introduction to obstacles: plenty of open firing lanes remain.
static func _build_scattered_cover(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var count: int = rng.randi_range(3, 5)
	for _index in range(count):
		var size: float = rng.randf_range(0.7, 1.3)
		var height: float = rng.randf_range(0.8, 1.6)
		var position := Vector3(
			rng.randf_range(-world.half_extent * 0.7, world.half_extent * 0.7),
			height * 0.5,
			rng.randf_range(-world.half_extent * 0.7, world.half_extent * 0.7)
		)
		world.add_box(position, Vector3(size, height * 0.5, size), Obstacle.Kind.CRATE)


## An L-shaped wall junction: the canonical "fight around the corner"
## geometry. The corner's orientation is seeded so an agent cannot memorize
## one side.
static func _build_corner(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var flip_x: float = 1.0 if rng.randf() > 0.5 else -1.0
	var flip_z: float = 1.0 if rng.randf() > 0.5 else -1.0
	var arm: float = rng.randf_range(world.half_extent * 0.35, world.half_extent * 0.6)
	var height: float = HIGH_COVER_HEIGHT
	var offset_x: float = flip_x * rng.randf_range(0.0, world.half_extent * 0.2)
	var offset_z: float = flip_z * rng.randf_range(0.0, world.half_extent * 0.2)

	world.add_box(
		Vector3(offset_x, height * 0.5, offset_z + flip_z * arm * 0.5),
		Vector3(WALL_HALF_THICKNESS, height * 0.5, arm * 0.5),
		Obstacle.Kind.WALL
	)
	world.add_box(
		Vector3(offset_x + flip_x * arm * 0.5, height * 0.5, offset_z),
		Vector3(arm * 0.5, height * 0.5, WALL_HALF_THICKNESS),
		Obstacle.Kind.WALL
	)
	# One low crate near the corner mouth gives the defender a peek spot.
	world.add_box(
		Vector3(
			offset_x - flip_x * rng.randf_range(1.5, 3.0),
			LOW_COVER_HEIGHT * 0.5,
			offset_z - flip_z * rng.randf_range(1.5, 3.0)
		),
		Vector3(0.8, LOW_COVER_HEIGHT * 0.5, 0.8),
		Obstacle.Kind.LOW_COVER
	)


## Alternating low and high cover in two staggered rows: teaches crouch-line
## vs full occlusion and gives both sides usable cover.
static func _build_cover_field(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var rows: int = 2
	var columns: int = rng.randi_range(3, 4)
	var spacing_x: float = (world.half_extent * 1.4) / float(columns)
	for row in range(rows):
		var z: float = lerpf(-world.half_extent * 0.45, world.half_extent * 0.45, float(row))
		for column in range(columns):
			var x: float = -world.half_extent * 0.7 + spacing_x * (float(column) + 0.5)
			x += rng.randf_range(-0.6, 0.6)
			var low: bool = ((row + column) % 2) == 0
			var height: float = LOW_COVER_HEIGHT if low else HIGH_COVER_HEIGHT
			var kind: int = Obstacle.Kind.LOW_COVER if low else Obstacle.Kind.HIGH_COVER
			world.add_box(
				Vector3(x, height * 0.5, z + rng.randf_range(-0.5, 0.5)),
				Vector3(1.0, height * 0.5, 0.7),
				kind
			)


## Two long parallel walls forming a lane with a single side opening.
static func _build_corridor(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var lane_half_width: float = rng.randf_range(1.6, 2.6)
	var length: float = world.half_extent * 1.6
	var height: float = HIGH_COVER_HEIGHT
	var gap_center: float = rng.randf_range(-length * 0.2, length * 0.2)
	var gap_half: float = rng.randf_range(1.0, 1.8)

	world.add_box(
		Vector3(-lane_half_width, height * 0.5, 0.0),
		Vector3(WALL_HALF_THICKNESS, height * 0.5, length * 0.5),
		Obstacle.Kind.WALL
	)
	# The opposite wall is split in two so the lane has exactly one opening.
	var upper_start: float = gap_center + gap_half
	var lower_end: float = gap_center - gap_half
	var upper_length: float = maxf(0.2, (length * 0.5) - upper_start)
	var lower_length: float = maxf(0.2, lower_end + (length * 0.5))
	world.add_box(
		Vector3(lane_half_width, height * 0.5, upper_start + upper_length * 0.5),
		Vector3(WALL_HALF_THICKNESS, height * 0.5, upper_length * 0.5),
		Obstacle.Kind.WALL
	)
	world.add_box(
		Vector3(lane_half_width, height * 0.5, -(length * 0.5) + lower_length * 0.5),
		Vector3(WALL_HALF_THICKNESS, height * 0.5, lower_length * 0.5),
		Obstacle.Kind.WALL
	)


## Two rooms joined by a doorway plus an ambush alcove: the layout used by
## the ambush and lost-target scenarios.
static func _build_rooms(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var height: float = HIGH_COVER_HEIGHT
	var divider_z: float = rng.randf_range(-2.0, 2.0)
	var door_center: float = rng.randf_range(-world.half_extent * 0.4, world.half_extent * 0.4)
	var door_half: float = rng.randf_range(1.0, 1.6)
	var span: float = world.half_extent

	var left_length: float = maxf(0.2, (door_center - door_half) + span)
	var right_length: float = maxf(0.2, span - (door_center + door_half))
	world.add_box(
		Vector3(-span + left_length * 0.5, height * 0.5, divider_z),
		Vector3(left_length * 0.5, height * 0.5, WALL_HALF_THICKNESS),
		Obstacle.Kind.WALL
	)
	world.add_box(
		Vector3(door_center + door_half + right_length * 0.5, height * 0.5, divider_z),
		Vector3(right_length * 0.5, height * 0.5, WALL_HALF_THICKNESS),
		Obstacle.Kind.WALL
	)
	# Alcove wall inside the far room, perfect for waiting out of sight.
	var alcove_x: float = rng.randf_range(-span * 0.6, span * 0.6)
	world.add_box(
		Vector3(alcove_x, height * 0.5, divider_z - rng.randf_range(3.0, 5.0)),
		Vector3(1.8, height * 0.5, WALL_HALF_THICKNESS),
		Obstacle.Kind.WALL
	)


## Regular grid of thin full-height pillars: many sight lines, all narrow.
static func _build_pillars(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var per_axis: int = rng.randi_range(2, 3)
	var spacing: float = (world.half_extent * 1.3) / float(per_axis)
	for ix in range(per_axis):
		for iz in range(per_axis):
			var x: float = -world.half_extent * 0.65 + spacing * (float(ix) + 0.5)
			var z: float = -world.half_extent * 0.65 + spacing * (float(iz) + 0.5)
			world.add_box(
				Vector3(
					x + rng.randf_range(-0.3, 0.3),
					world.wall_height * 0.5,
					z + rng.randf_range(-0.3, 0.3)
				),
				Vector3(0.45, world.wall_height * 0.5, 0.45),
				Obstacle.Kind.PILLAR
			)


## Standable platforms at two heights plus one high wall. Elevation changes
## which sight lines exist, which is exactly what the jump scenarios test.
static func _build_vertical(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var low_top: float = rng.randf_range(0.9, 1.2)
	var high_top: float = low_top + rng.randf_range(0.8, 1.1)
	var side: float = 1.0 if rng.randf() > 0.5 else -1.0

	world.add_box(
		Vector3(side * rng.randf_range(2.0, 4.0), low_top * 0.5, rng.randf_range(-3.0, 3.0)),
		Vector3(1.8, low_top * 0.5, 1.8),
		Obstacle.Kind.PLATFORM
	)
	world.add_box(
		Vector3(
			side * rng.randf_range(4.5, 6.5), high_top * 0.5, rng.randf_range(-4.0, 4.0)
		),
		Vector3(1.6, high_top * 0.5, 1.6),
		Obstacle.Kind.PLATFORM
	)
	world.add_box(
		Vector3(-side * rng.randf_range(2.0, 5.0), HIGH_COVER_HEIGHT * 0.5, 0.0),
		Vector3(WALL_HALF_THICKNESS, HIGH_COVER_HEIGHT * 0.5, rng.randf_range(2.0, 4.0)),
		Obstacle.Kind.WALL
	)


## Picks one of the concrete layouts at random and then perturbs it with a
## couple of extra crates. Still fully deterministic for a given seed.
static func _build_randomized(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var candidates: Array = [
		"scattered_cover", "corner", "cover_field", "corridor", "rooms", "pillars", "vertical"
	]
	var pick: String = str(candidates[rng.randi_range(0, candidates.size() - 1)])
	match pick:
		"scattered_cover":
			_build_scattered_cover(world, rng)
		"corner":
			_build_corner(world, rng)
		"cover_field":
			_build_cover_field(world, rng)
		"corridor":
			_build_corridor(world, rng)
		"rooms":
			_build_rooms(world, rng)
		"pillars":
			_build_pillars(world, rng)
		"vertical":
			_build_vertical(world, rng)
	var extras: int = rng.randi_range(1, 3)
	for _index in range(extras):
		var height: float = rng.randf_range(0.8, 1.8)
		world.add_box(
			Vector3(
				rng.randf_range(-world.half_extent * 0.8, world.half_extent * 0.8),
				height * 0.5,
				rng.randf_range(-world.half_extent * 0.8, world.half_extent * 0.8)
			),
			Vector3(rng.randf_range(0.6, 1.1), height * 0.5, rng.randf_range(0.6, 1.1)),
			Obstacle.Kind.CRATE
		)

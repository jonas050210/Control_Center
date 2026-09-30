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
	"multi_room",
	"ambush",
	"sound_maze",
	"combat_complex",
	"crossfire_complex",
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
		"multi_room":
			_build_multi_room(world, rng)
		"ambush":
			_build_ambush(world, rng)
		"sound_maze":
			_build_sound_maze(world, rng)
		"combat_complex":
			_build_combat_complex(world, rng)
		"crossfire_complex":
			_build_crossfire_complex(world, rng)
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
		Vector3(side * rng.randf_range(4.5, 6.5), high_top * 0.5, rng.randf_range(-4.0, 4.0)),
		Vector3(1.6, high_top * 0.5, 1.6),
		Obstacle.Kind.PLATFORM
	)
	world.add_box(
		Vector3(-side * rng.randf_range(2.0, 5.0), HIGH_COVER_HEIGHT * 0.5, 0.0),
		Vector3(WALL_HALF_THICKNESS, HIGH_COVER_HEIGHT * 0.5, rng.randf_range(2.0, 4.0)),
		Obstacle.Kind.WALL
	)


## A full RL-combat sandbox: four distinct areas connected by doorways,
## an open center, a long lane, side rooms and enough cover to create both
## short peeks and long sightlines. Authored spawn points are deliberately
## numerous and symmetric-ish; the scenario sampler chooses from them with
## the episode seed, reducing map memorization while keeping determinism.
static func _build_combat_complex(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var height: float = HIGH_COVER_HEIGHT
	var span: float = world.half_extent
	var door_half: float = rng.randf_range(1.1, 1.6)
	var center_shift_x: float = rng.randf_range(-0.8, 0.8)
	var center_shift_z: float = rng.randf_range(-0.8, 0.8)

	# North/south divider walls with seeded doorways. These create rooms but
	# leave the center as an open contested courtyard.
	_add_split_wall(world, center_shift_z + 3.0, -span, span, -4.4, door_half, height, false)
	_add_split_wall(world, center_shift_z - 3.2, -span, span, 4.2, door_half, height, false)
	# A vertical divider is offset and heavily gapped so it creates varied
	# sightlines without disconnecting the floor.
	_add_split_wall(world, center_shift_x - 3.4, -span, span, 0.0, door_half * 1.35, height, true)
	_add_split_wall(world, center_shift_x + 4.2, -span, span, -1.8, door_half, height, true)

	# East long lane / corridor: long-range shots through the lane, but the
	# side opening and low crates make it contestable rather than a tunnel.
	var lane_x: float = span * 0.68
	world.add_box(
		Vector3(lane_x - 1.6, height * 0.5, 0.0),
		Vector3(WALL_HALF_THICKNESS, height * 0.5, span * 0.72),
		Obstacle.Kind.WALL
	)
	_add_split_wall(
		world,
		lane_x + 1.6,
		-span * 0.75,
		span * 0.75,
		rng.randf_range(-2.0, 2.0),
		1.4,
		height,
		true
	)
	for z_value in [-5.2, 0.0, 5.2]:
		world.add_box(
			Vector3(lane_x, LOW_COVER_HEIGHT * 0.5, z_value + rng.randf_range(-0.4, 0.4)),
			Vector3(0.75, LOW_COVER_HEIGHT * 0.5, 0.75),
			Obstacle.Kind.LOW_COVER
		)

	# West broken cover maze: short sightlines and blind corner checks.
	for i in range(4):
		var z: float = -span * 0.55 + float(i) * span * 0.35
		var x: float = -span * 0.58 + rng.randf_range(-0.4, 0.4)
		world.add_box(
			Vector3(x, height * 0.5, z),
			Vector3(rng.randf_range(1.4, 2.4), height * 0.5, WALL_HALF_THICKNESS),
			Obstacle.Kind.WALL
		)
		world.add_box(
			Vector3(
				x + rng.randf_range(1.0, 2.2), LOW_COVER_HEIGHT * 0.5, z + rng.randf_range(0.8, 1.6)
			),
			Vector3(0.7, LOW_COVER_HEIGHT * 0.5, 0.7),
			Obstacle.Kind.LOW_COVER
		)

	# Central courtyard cover: open movement remains possible, but standing in
	# the middle is punishable from several angles.
	var central_cover: Array = [
		Vector3(-1.4, LOW_COVER_HEIGHT * 0.5, 0.2),
		Vector3(1.5, LOW_COVER_HEIGHT * 0.5, -0.4),
		Vector3(0.0, HIGH_COVER_HEIGHT * 0.5, 2.0),
	]
	for cover_value in central_cover:
		var cover: Vector3 = cover_value
		var cover_height: float = LOW_COVER_HEIGHT if cover.y < 0.8 else HIGH_COVER_HEIGHT
		var kind: int = Obstacle.Kind.HIGH_COVER
		if cover_height == LOW_COVER_HEIGHT:
			kind = Obstacle.Kind.LOW_COVER
		world.add_box(
			cover + Vector3(rng.randf_range(-0.35, 0.35), 0.0, rng.randf_range(-0.35, 0.35)),
			Vector3(0.9, cover_height * 0.5, 0.7),
			kind
		)

	_add_combat_complex_spawns(world, span)


## A second high-variety combat layout: a central crossfire plaza, two safe
## back rooms and staggered lane blockers. It shares the same tactical ideas
## as combat_complex but different geometry, so curriculum/evaluation can
## ask for "complex cover fight" without one memorized floor plan.
static func _build_crossfire_complex(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var height: float = HIGH_COVER_HEIGHT
	var span: float = world.half_extent
	var door_half: float = rng.randf_range(1.0, 1.5)

	# Two room shells in opposite corners, each with a doorway facing the
	# center. Walls are axis-aligned but their seeded offsets vary sightlines.
	var room_offset: float = span * 0.48
	_add_split_wall(world, -room_offset, -span, -1.0, -room_offset, door_half, height, true)
	_add_split_wall(world, -room_offset, 1.0, span, room_offset, door_half, height, false)
	_add_split_wall(world, room_offset, 1.0, span, room_offset, door_half, height, true)
	_add_split_wall(world, room_offset, -span, -1.0, -room_offset, door_half, height, false)

	# Staggered high walls force pathing decisions and make flank routes
	# meaningful; the low cover in between creates peek fights.
	for i in range(5):
		var z: float = -span * 0.6 + float(i) * span * 0.3
		var sign_value: float = 1.0 if i % 2 == 0 else -1.0
		world.add_box(
			Vector3(sign_value * rng.randf_range(1.6, 3.0), height * 0.5, z),
			Vector3(1.7, height * 0.5, WALL_HALF_THICKNESS),
			Obstacle.Kind.WALL
		)
		world.add_box(
			Vector3(
				-sign_value * rng.randf_range(1.0, 2.5),
				LOW_COVER_HEIGHT * 0.5,
				z + rng.randf_range(-0.6, 0.6)
			),
			Vector3(0.85, LOW_COVER_HEIGHT * 0.5, 0.85),
			Obstacle.Kind.LOW_COVER
		)

	# Long north/south sightline with a break in the middle.
	world.add_box(
		Vector3(0.0, height * 0.5, -span * 0.82),
		Vector3(span * 0.55, height * 0.5, WALL_HALF_THICKNESS),
		Obstacle.Kind.WALL
	)
	world.add_box(
		Vector3(0.0, height * 0.5, span * 0.82),
		Vector3(span * 0.55, height * 0.5, WALL_HALF_THICKNESS),
		Obstacle.Kind.WALL
	)

	# A few standable boxes turn jump/height into a tactical option at higher
	# curriculum levels without requiring a separate vertical map.
	for side in [-1.0, 1.0]:
		var top: float = rng.randf_range(0.9, 1.15)
		world.add_box(
			Vector3(side * span * 0.25, top * 0.5, rng.randf_range(-span * 0.25, span * 0.25)),
			Vector3(1.25, top * 0.5, 1.25),
			Obstacle.Kind.PLATFORM
		)

	_add_combat_complex_spawns(world, span)


static func _add_combat_complex_spawns(world: ArenaWorld, span: float) -> void:
	var agent_points: Array = [
		Vector3(-span * 0.70, 0.0, span * 0.70),
		Vector3(span * 0.68, 0.0, span * 0.64),
		Vector3(-span * 0.72, 0.0, -span * 0.68),
		Vector3(span * 0.66, 0.0, -span * 0.70),
		Vector3(0.0, 0.0, span * 0.78),
		Vector3(0.0, 0.0, -span * 0.78),
	]
	for index in range(agent_points.size()):
		var point: Vector3 = agent_points[index]
		world.add_spawn_point("agent", point, _yaw_towards(point, Vector3.ZERO), "agent_%d" % index)

	var enemy_points: Array = [
		Vector3(-span * 0.72, 0.0, 0.0),
		Vector3(span * 0.72, 0.0, 0.0),
		Vector3(-span * 0.35, 0.0, -span * 0.35),
		Vector3(span * 0.35, 0.0, span * 0.35),
		Vector3(-span * 0.20, 0.0, span * 0.58),
		Vector3(span * 0.20, 0.0, -span * 0.58),
		Vector3(0.0, 0.0, 0.0),
		Vector3(span * 0.62, 0.0, -span * 0.62),
		Vector3(-span * 0.62, 0.0, span * 0.62),
	]
	for index in range(enemy_points.size()):
		var point: Vector3 = enemy_points[index]
		world.add_spawn_point("enemy", point, _yaw_towards(point, Vector3.ZERO), "enemy_%d" % index)


static func _yaw_towards(from_position: Vector3, target: Vector3) -> float:
	var delta := Vector3(target.x - from_position.x, 0.0, target.z - from_position.z)
	if delta.is_zero_approx():
		return 0.0
	return rad_to_deg(atan2(delta.x, -delta.z))


## Picks one of the concrete layouts at random and then perturbs it with a
## couple of extra crates. Still fully deterministic for a given seed.
static func _build_randomized(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var candidates: Array = [
		"scattered_cover",
		"corner",
		"cover_field",
		"corridor",
		"rooms",
		"pillars",
		"vertical",
		"multi_room",
		"ambush",
		"sound_maze",
		"combat_complex",
		"crossfire_complex",
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
		"multi_room":
			_build_multi_room(world, rng)
		"ambush":
			_build_ambush(world, rng)
		"sound_maze":
			_build_sound_maze(world, rng)
		"combat_complex":
			_build_combat_complex(world, rng)
		"crossfire_complex":
			_build_crossfire_complex(world, rng)
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


## Four rooms around a cross-shaped pair of dividing walls, each divider
## pierced by one seeded doorway. This is the canonical multi-room combat
## layout: there is no sight line longer than one room, every engagement
## starts at a door, and a navigation graph is genuinely required to move
## between rooms.
static func _build_multi_room(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var height: float = HIGH_COVER_HEIGHT
	var span: float = world.half_extent
	var center_x: float = rng.randf_range(-span * 0.2, span * 0.2)
	var center_z: float = rng.randf_range(-span * 0.2, span * 0.2)
	var door_half: float = rng.randf_range(1.0, 1.5)

	# Vertical divider (constant x) split by two doorways, one per half.
	var north_door: float = rng.randf_range(center_z + 1.5, span - 1.5)
	var south_door: float = rng.randf_range(-span + 1.5, center_z - 1.5)
	_add_split_wall(world, center_x, -span, span, north_door, door_half, height, true)
	_add_split_wall(
		world, center_x, -span, north_door - door_half, south_door, door_half, height, true
	)
	# Horizontal divider (constant z) with one doorway.
	var east_door: float = rng.randf_range(center_x + 1.5, span - 1.5)
	_add_split_wall(world, center_z, -span, span, east_door, door_half, height, false)

	# One crate per quadrant so every room has usable cover.
	for quadrant in range(4):
		var sign_x: float = 1.0 if (quadrant % 2) == 0 else -1.0
		var sign_z: float = 1.0 if quadrant < 2 else -1.0
		world.add_box(
			Vector3(
				center_x + sign_x * rng.randf_range(2.0, span * 0.6),
				LOW_COVER_HEIGHT * 0.5,
				center_z + sign_z * rng.randf_range(2.0, span * 0.6)
			),
			Vector3(0.9, LOW_COVER_HEIGHT * 0.5, 0.9),
			Obstacle.Kind.LOW_COVER
		)


## Adds a wall along one axis, split by a doorway. `vertical` selects a wall
## of constant x (true) or constant z (false). Shared by the multi-room and
## ambush layouts so a doorway is built exactly one way in the project.
static func _add_split_wall(
	world: ArenaWorld,
	fixed_coordinate: float,
	from_value: float,
	to_value: float,
	door_center: float,
	door_half: float,
	height: float,
	vertical: bool
) -> void:
	if to_value - from_value <= 0.2:
		return
	var segments: Array = [
		Vector2(from_value, clampf(door_center - door_half, from_value, to_value)),
		Vector2(clampf(door_center + door_half, from_value, to_value), to_value),
	]
	for segment_value in segments:
		var segment: Vector2 = segment_value
		var length: float = segment.y - segment.x
		if length <= 0.2:
			continue
		var middle: float = segment.x + length * 0.5
		if vertical:
			world.add_box(
				Vector3(fixed_coordinate, height * 0.5, middle),
				Vector3(WALL_HALF_THICKNESS, height * 0.5, length * 0.5),
				Obstacle.Kind.WALL
			)
		else:
			world.add_box(
				Vector3(middle, height * 0.5, fixed_coordinate),
				Vector3(length * 0.5, height * 0.5, WALL_HALF_THICKNESS),
				Obstacle.Kind.WALL
			)


## A single approach lane flanked by two blind alcoves. Whoever walks the
## lane is exposed from the side the moment they pass an alcove mouth, which
## is what makes "check your corners" a learnable behavior rather than a
## scripted one.
static func _build_ambush(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var height: float = HIGH_COVER_HEIGHT
	var span: float = world.half_extent
	var lane_half: float = rng.randf_range(1.8, 2.6)
	_add_split_wall(world, -lane_half, -span, span, rng.randf_range(-2.0, 2.0), 1.2, height, true)
	_add_split_wall(world, lane_half, -span, span, rng.randf_range(-2.0, 2.0), 1.2, height, true)

	# Alcove back walls, one on each side, at staggered depths.
	for side in range(2):
		var sign_x: float = 1.0 if side == 0 else -1.0
		var depth: float = rng.randf_range(2.5, 4.0)
		var alcove_z: float = rng.randf_range(-span * 0.5, span * 0.5)
		world.add_box(
			Vector3(sign_x * (lane_half + depth), height * 0.5, alcove_z),
			Vector3(WALL_HALF_THICKNESS, height * 0.5, rng.randf_range(1.6, 2.6)),
			Obstacle.Kind.WALL
		)
		world.add_box(
			Vector3(sign_x * (lane_half + depth * 0.5), LOW_COVER_HEIGHT * 0.5, alcove_z),
			Vector3(0.7, LOW_COVER_HEIGHT * 0.5, 0.7),
			Obstacle.Kind.LOW_COVER
		)


## Dense staggered wall stubs: almost nothing is ever visible, but sound
## travels around the stubs with only one or two occluders. The intended
## lesson is "navigate by hearing", so the geometry deliberately maximizes
## occlusion while keeping the whole layout one connected region.
static func _build_sound_maze(world: ArenaWorld, rng: RandomNumberGenerator) -> void:
	var height: float = HIGH_COVER_HEIGHT
	var span: float = world.half_extent
	var rows: int = rng.randi_range(3, 4)
	var spacing: float = (span * 2.0) / float(rows + 1)
	for row in range(rows):
		var z: float = -span + spacing * float(row + 1)
		var stub_count: int = rng.randi_range(2, 3)
		for stub in range(stub_count):
			var x: float = -span * 0.75 + (span * 1.5 / float(stub_count)) * (float(stub) + 0.5)
			var length: float = rng.randf_range(1.6, 3.0)
			world.add_box(
				Vector3(x + rng.randf_range(-0.6, 0.6), height * 0.5, z),
				Vector3(length * 0.5, height * 0.5, WALL_HALF_THICKNESS),
				Obstacle.Kind.WALL
			)

## Tests for the world layer: obstacle geometry, seeded layout generation,
## collision resolution, ground height and the cover/peek queries.
class_name TestWorld
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")


func test_obstacle_bounds_and_containment() -> SandboxTest:
	var t := SandboxTest.new("obstacle_bounds_and_containment")
	var box: Obstacle = Obstacle.make(
		Vector3(2.0, 0.5, -1.0), Vector3(1.0, 0.5, 2.0), Obstacle.Kind.CRATE
	)
	t.assert_vec_almost_eq(box.min_corner(), Vector3(1.0, 0.0, -3.0))
	t.assert_vec_almost_eq(box.max_corner(), Vector3(3.0, 1.0, 1.0))
	t.assert_almost_eq(box.top_y(), 1.0)
	t.assert_almost_eq(box.bottom_y(), 0.0)
	t.assert_true(box.contains_xz(Vector3(2.0, 0.0, 0.0)))
	t.assert_false(box.contains_xz(Vector3(9.0, 0.0, 0.0)))
	t.assert_eq(Obstacle.kind_name(box.kind), "crate")
	return t


func test_segment_intersection_detects_and_misses() -> SandboxTest:
	var t := SandboxTest.new("segment_intersection_detects_and_misses")
	var wall: Obstacle = Obstacle.make(
		Vector3(0.0, 1.5, 0.0), Vector3(0.25, 1.5, 4.0), Obstacle.Kind.WALL
	)
	# A segment crossing the wall from -X to +X must be blocked near the middle.
	var hit: float = wall.segment_intersection(Vector3(-5.0, 1.0, 0.0), Vector3(5.0, 1.0, 0.0))
	t.assert_gte(hit, 0.0, "a segment through the wall must intersect it")
	t.assert_almost_eq(hit, 0.475, 0.02)
	# A segment passing above the wall must not be blocked.
	var over: float = wall.segment_intersection(Vector3(-5.0, 4.0, 0.0), Vector3(5.0, 4.0, 0.0))
	t.assert_lt(over, 0.0, "a segment above the wall must not intersect it")
	# A segment entirely to one side must not be blocked.
	var beside: float = wall.segment_intersection(Vector3(-5.0, 1.0, 9.0), Vector3(5.0, 1.0, 9.0))
	t.assert_lt(beside, 0.0)
	return t


func test_layout_generation_is_deterministic_for_a_seed() -> SandboxTest:
	var t := SandboxTest.new("layout_generation_is_deterministic_for_a_seed")
	for layout_value in WorldGenerator.LAYOUT_IDS:
		var layout_id: String = str(layout_value)
		var a: ArenaWorld = WorldGenerator.build(layout_id, 4242)
		var b: ArenaWorld = WorldGenerator.build(layout_id, 4242)
		t.assert_eq(
			a.obstacle_count(),
			b.obstacle_count(),
			"%s must produce the same obstacle count for the same seed" % layout_id
		)
		for index in range(a.obstacle_count()):
			var left: Obstacle = a.obstacles[index]
			var right: Obstacle = b.obstacles[index]
			t.assert_vec_almost_eq(
				left.center, right.center, 0.0001, "%s obstacle %d moved" % [layout_id, index]
			)
			t.assert_vec_almost_eq(left.half_extents, right.half_extents, 0.0001)
			t.assert_eq(left.kind, right.kind)
	return t


func test_different_seeds_produce_different_randomized_layouts() -> SandboxTest:
	var t := SandboxTest.new("different_seeds_produce_different_randomized_layouts")
	var a: ArenaWorld = WorldGenerator.build("randomized", 1)
	var b: ArenaWorld = WorldGenerator.build("randomized", 2)
	var identical: bool = a.obstacle_count() == b.obstacle_count()
	if identical:
		for index in range(a.obstacle_count()):
			if (
				(a.obstacles[index] as Obstacle).center.distance_to(
					(b.obstacles[index] as Obstacle).center
				)
				> 0.001
			):
				identical = false
				break
	t.assert_false(identical, "two different seeds must not generate the same random arena")
	return t


func test_unknown_layout_falls_back_to_open_arena() -> SandboxTest:
	var t := SandboxTest.new("unknown_layout_falls_back_to_open_arena")
	var fallback: ArenaWorld = WorldGenerator.build("does_not_exist", 7)
	var open_arena: ArenaWorld = WorldGenerator.build("open_arena", 7)
	t.assert_eq(fallback.obstacle_count(), open_arena.obstacle_count())
	t.assert_eq(fallback.layout_id, "open_arena")
	return t


func test_collision_blocks_movement_through_a_box() -> SandboxTest:
	var t := SandboxTest.new("collision_blocks_movement_through_a_box")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 1.2, 0.0), Vector3(1.0, 1.2, 1.0), Obstacle.Kind.HIGH_COVER)
	var from := Vector3(-3.0, 0.0, 0.0)
	var to := Vector3(3.0, 0.0, 0.0)
	var resolved: Vector3 = world.resolve_move(from, to, 0.4, 1.8)
	t.assert_lt(resolved.x, -1.0, "the agent must be stopped on the near side of the box")
	t.assert_true(world.is_blocked(Vector3(0.0, 0.0, 0.0), 0.4, 1.8))
	t.assert_false(world.is_blocked(Vector3(5.0, 0.0, 5.0), 0.4, 1.8))
	return t


func test_movement_slides_along_a_wall_instead_of_stopping_dead() -> SandboxTest:
	var t := SandboxTest.new("movement_slides_along_a_wall_instead_of_stopping_dead")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 1.5, 0.0), Vector3(0.25, 1.5, 5.0), Obstacle.Kind.WALL)
	var from := Vector3(-1.0, 0.0, 0.0)
	# Pushing diagonally into the wall must still produce Z movement.
	var to := Vector3(-0.2, 0.0, 1.0)
	var resolved: Vector3 = world.resolve_move(from, to, 0.4, 1.8)
	t.assert_lt(resolved.x, -0.6, "X must be blocked by the wall")
	t.assert_gt(resolved.z, 0.5, "Z must still slide along the wall")
	return t


func test_arena_bounds_are_always_enforced() -> SandboxTest:
	var t := SandboxTest.new("arena_bounds_are_always_enforced")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	var resolved: Vector3 = world.resolve_move(
		Vector3(9.0, 0.0, 9.0), Vector3(50.0, 0.0, 50.0), 0.4, 1.8
	)
	t.assert_lte(resolved.x, 9.6)
	t.assert_lte(resolved.z, 9.6)
	t.assert_gte(resolved.x, -9.6)
	# A clamped position must also be reported as a legal standing position;
	# float rounding used to push it a hair past the limit, so the arena
	# clamp and the free-position test disagreed about the same point.
	t.assert_true(world.is_position_free(resolved, 0.4, 1.8), "the clamped position must be inside")
	var negative: Vector3 = world.resolve_move(
		Vector3(-9.0, 0.0, -9.0), Vector3(-50.0, 0.0, -50.0), 0.4, 1.8
	)
	t.assert_gte(negative.x, -9.6)
	t.assert_gte(negative.z, -9.6)
	t.assert_true(world.is_position_free(negative, 0.4, 1.8))
	return t


## Regression: `resolve_move()` only tested the END position of a move, so
## a displacement wider than a box passed straight through it. A swept move
## must stop on the near side no matter how large the step is.
func test_fast_movement_cannot_tunnel_through_geometry() -> SandboxTest:
	var t := SandboxTest.new("fast_movement_cannot_tunnel_through_geometry")
	var world: ArenaWorld = ArenaWorld.create(20.0)
	world.add_box(Vector3(0.0, 1.5, 0.0), Vector3(0.35, 1.5, 6.0), Obstacle.Kind.WALL)
	for distance in [1.0, 2.0, 5.0, 12.0, 30.0]:
		var from := Vector3(-float(distance), 0.0, 0.0)
		var to := Vector3(float(distance), 0.0, 0.0)
		var resolved: Vector3 = world.resolve_move(from, to, 0.4, 1.8)
		t.assert_lt(resolved.x, -0.7, "a %d m step must not cross the wall" % int(distance))
		t.assert_false(world.is_blocked(resolved, 0.4, 1.8))
	return t


func test_ground_height_returns_box_top_for_standable_platforms() -> SandboxTest:
	var t := SandboxTest.new("ground_height_returns_box_top_for_standable_platforms")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 0.5, 0.0), Vector3(2.0, 0.5, 2.0), Obstacle.Kind.PLATFORM)
	# Falling from above the platform lands on top of it.
	t.assert_almost_eq(world.ground_height(Vector3(0.0, 3.0, 0.0), 0.4, 3.0), 1.0)
	# Standing away from the platform lands on the floor.
	t.assert_almost_eq(world.ground_height(Vector3(8.0, 3.0, 8.0), 0.4, 3.0), 0.0)
	return t


func test_sampled_free_positions_are_actually_free_and_seeded() -> SandboxTest:
	var t := SandboxTest.new("sampled_free_positions_are_actually_free_and_seeded")
	var world: ArenaWorld = WorldGenerator.build("cover_field", 99)
	var rng_a := RandomNumberGenerator.new()
	rng_a.seed = 1234
	var rng_b := RandomNumberGenerator.new()
	rng_b.seed = 1234
	for _i in range(10):
		var position_a: Vector3 = world.sample_free_position(rng_a, 0.4, 1.8)
		var position_b: Vector3 = world.sample_free_position(rng_b, 0.4, 1.8)
		t.assert_vec_almost_eq(position_a, position_b, 0.0001, "sampling must be seed-reproducible")
		t.assert_true(
			world.is_position_free(position_a, 0.4, 1.8),
			"sampled position must not be inside a box"
		)
	return t


func test_cover_position_breaks_line_of_sight_from_the_threat() -> SandboxTest:
	var t := SandboxTest.new("cover_position_breaks_line_of_sight_from_the_threat")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 1.5, 0.0), Vector3(2.0, 1.5, 0.5), Obstacle.Kind.HIGH_COVER)
	var threat_eye := Vector3(0.0, 1.6, -6.0)
	var cover: Vector3 = world.find_cover_position(
		Vector3(0.0, 0.0, 2.0), threat_eye, 4.0, 0.4, 1.8, 1.6
	)
	t.assert_true(
		world.segment_blocked(threat_eye, cover + Vector3(0.0, 1.6, 0.0)),
		"the returned cover position must be hidden from the threat"
	)
	return t

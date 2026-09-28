## Tests for the agent's long-term spatial memory.
##
## The properties that matter here are epistemic, not geometric. A memory
## system is only honest if:
##   * information the agent never perceived is ABSENT (not zero, not
##     defaulted to the truth) -- `test_unseen_geometry_is_never_known`;
##   * beliefs decay, so age turns knowledge into uncertainty;
##   * a fresh observation replaces a stale one;
##   * reset wipes everything, so nothing leaks between episodes;
##   * replaying the same observation stream reproduces the same memory.
class_name TestSpatialMemory
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SpatialMemory = preload("res://scripts/exploration/spatial_memory.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")

const HALF_EXTENT: float = 10.0


func _memory() -> SpatialMemory:
	return SpatialMemory.create(HALF_EXTENT, 2.0)


func _look(memory: SpatialMemory, at: Vector3, forward: Vector3, world) -> int:
	return memory.observe_cells(
		at + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0),
		at,
		forward,
		world,
		null,
		SandboxConfig.AGENT_FOV_DEG,
		SandboxConfig.VISION_RANGE
	)


func test_fresh_memory_knows_nothing() -> SandboxTest:
	var t := SandboxTest.new("fresh_memory_knows_nothing")
	var memory: SpatialMemory = _memory()
	t.assert_gt(float(memory.cell_count), 0.0)
	t.assert_eq(memory.known_cell_count(), 0)
	t.assert_eq(memory.visited_cell_count(), 0)
	t.assert_almost_eq(memory.coverage_fraction(), 0.0, 0.0001)
	t.assert_false(memory.is_known(Vector3.ZERO), "an unobserved origin must be unknown")
	t.assert_almost_eq(memory.time_since_visit(Vector3.ZERO), SpatialMemory.NEVER, 0.0001)
	return t


func test_observation_marks_only_what_was_looked_at() -> SandboxTest:
	var t := SandboxTest.new("observation_marks_only_what_was_looked_at")
	var memory: SpatialMemory = _memory()
	# The agent stands one full cell away from the south edge. With a 2 m
	# grid over a 10 m half extent, the cells are [..., 6..8), [8..10): at
	# z = 8.0 the agent occupied the LAST row, so the "behind" probe at
	# z = 9.5 fell in the agent's OWN cell, which observe_self() marks known
	# by definition. z = 6.0 puts the agent in row [6..8) so that z = 9.5 is
	# a genuinely different, genuinely unobserved cell.
	var origin := Vector3(0.0, 0.0, 6.0)
	memory.observe_self(origin, 1.0)
	# Look north (-Z) from the south edge: cells to the south (behind) must
	# stay unknown because they are outside the field of view.
	_look(memory, origin, Vector3.FORWARD, null)
	t.assert_true(memory.is_known(Vector3(0.0, 0.0, 0.0)), "cell straight ahead must be known")
	t.assert_ne(
		memory.cell_index(Vector3(0.0, 0.0, 9.5)),
		memory.cell_index(origin),
		"the probe must not land in the agent's own cell"
	)
	t.assert_false(
		memory.is_known(Vector3(0.0, 0.0, 9.5)), "cell behind the agent must stay unknown"
	)
	t.assert_gt(memory.coverage_fraction(), 0.0)
	t.assert_lt(memory.coverage_fraction(), 1.0)
	return t


func test_unseen_geometry_is_never_known() -> SandboxTest:
	var t := SandboxTest.new("unseen_geometry_is_never_known")
	# A two-room map: standing in one room must not reveal the other, even
	# though the simulation obviously knows what is in there.
	var world: ArenaWorld = WorldGenerator.build("rooms", 4242, HALF_EXTENT, 3.0)
	var memory: SpatialMemory = _memory()
	var hidden_unknown: int = 0
	var checked: int = 0
	var origin := Vector3(-8.0, 0.0, 8.0)
	memory.observe_self(origin, 1.0)
	_look(memory, origin, Vector3.FORWARD, world)
	for index in range(memory.cell_count):
		var center: Vector3 = memory.cell_center(index)
		if not world.segment_blocked(
			origin + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0),
			center + Vector3(0.0, 0.5, 0.0)
		):
			continue
		checked += 1
		if not memory.is_known(center):
			hidden_unknown += 1
	t.assert_gt(float(checked), 0.0, "the layout produced no occluded cells to test")
	t.assert_eq(hidden_unknown, checked, "a wall-occluded cell was marked known")
	return t


func test_walking_records_visits_and_a_route() -> SandboxTest:
	var t := SandboxTest.new("walking_records_visits_and_a_route")
	var memory: SpatialMemory = _memory()
	for step in range(6):
		memory.observe_self(Vector3(-6.0 + float(step) * 2.0, 0.0, 0.0), 1.0)
		memory.tick(0.1)
	t.assert_eq(memory.visited_cell_count(), 6)
	t.assert_eq(memory.route.size(), 6)
	t.assert_true(memory.is_visited(Vector3(-6.0, 0.0, 0.0)))
	t.assert_false(memory.is_visited(Vector3(6.0, 0.0, 6.0)))
	# Standing still must not append duplicate route entries.
	for _i in range(5):
		memory.observe_self(Vector3(4.0, 0.0, 0.0), 1.0)
	t.assert_eq(memory.route.size(), 6)
	return t


func test_confidence_decays_and_a_new_look_restores_it() -> SandboxTest:
	var t := SandboxTest.new("confidence_decays_and_a_new_look_restores_it")
	var memory: SpatialMemory = _memory()
	var spot := Vector3(2.0, 0.0, 2.0)
	memory.observe_self(spot, 1.0)
	var index: int = memory.cell_index(spot)
	t.assert_almost_eq(memory.confidence[index], 1.0, 0.0001)
	t.assert_almost_eq(memory.mean_uncertainty(), 0.0, 0.0001)

	memory.tick(SandboxConfig.EXPLORATION_HALF_LIFE)
	t.assert_almost_eq(memory.confidence[index], 0.5, 0.01)
	t.assert_gt(memory.mean_uncertainty(), 0.4)
	# Still known: the agent does not forget that a place exists, it only
	# becomes unsure whether its picture of it is current.
	t.assert_true(memory.is_known(spot))

	memory.observe_self(spot, 1.0)
	t.assert_almost_eq(memory.confidence[index], 1.0, 0.0001)
	t.assert_almost_eq(memory.mean_uncertainty(), 0.0, 0.0001)
	return t


func test_confidence_never_falls_below_the_floor() -> SandboxTest:
	var t := SandboxTest.new("confidence_never_falls_below_the_floor")
	var memory: SpatialMemory = _memory()
	memory.observe_self(Vector3.ZERO, 1.0)
	for _i in range(200):
		memory.tick(10.0)
	var index: int = memory.cell_index(Vector3.ZERO)
	t.assert_gte(memory.confidence[index], SandboxConfig.EXPLORATION_MIN_CONFIDENCE)
	t.assert_lte(memory.confidence[index], SandboxConfig.EXPLORATION_MIN_CONFIDENCE + 0.0001)
	return t


func test_stale_observation_is_replaced_by_the_new_one() -> SandboxTest:
	var t := SandboxTest.new("stale_observation_is_replaced_by_the_new_one")
	var memory: SpatialMemory = _memory()
	var spot := Vector3(-4.0, 0.0, 4.0)
	var index: int = memory.cell_index(spot)
	memory.observe_self(spot, 0.2)
	t.assert_almost_eq(memory.illumination[index], 0.2, 0.0001)
	memory.tick(5.0)
	memory.observe_self(spot, 0.9)
	t.assert_almost_eq(memory.illumination[index], 0.9, 0.0001)
	t.assert_almost_eq(memory.time_since_visit(spot), 0.0, 0.0001)
	return t


func test_danger_is_remembered_and_locatable() -> SandboxTest:
	var t := SandboxTest.new("danger_is_remembered_and_locatable")
	var memory: SpatialMemory = _memory()
	t.assert_true(
		memory.nearest_remembered_danger(Vector3.ZERO).is_empty(), "nothing dangerous known yet"
	)
	memory.mark_danger(Vector3(6.0, 0.0, 6.0), 1.0)
	var found: Dictionary = memory.nearest_remembered_danger(Vector3.ZERO)
	t.assert_false(found.is_empty())
	t.assert_almost_eq(float(found["distance"]), Vector3(7.0, 0.0, 7.0).length(), 3.0)
	return t


func test_reset_clears_every_belief() -> SandboxTest:
	var t := SandboxTest.new("reset_clears_every_belief")
	var memory: SpatialMemory = _memory()
	memory.observe_self(Vector3(1.0, 0.0, 1.0), 1.0)
	memory.mark_danger(Vector3(-3.0, 0.0, -3.0), 1.0)
	memory.tick(1.0)
	t.assert_gt(float(memory.known_cell_count()), 0.0)

	memory.clear()
	t.assert_eq(memory.known_cell_count(), 0)
	t.assert_eq(memory.visited_cell_count(), 0)
	t.assert_eq(memory.route.size(), 0)
	t.assert_almost_eq(memory.time_seconds, 0.0, 0.0001)
	t.assert_true(memory.nearest_remembered_danger(Vector3.ZERO).is_empty())
	return t


func test_identical_observation_streams_produce_identical_memory() -> SandboxTest:
	var t := SandboxTest.new("identical_observation_streams_produce_identical_memory")
	var world: ArenaWorld = WorldGenerator.build("cover_field", 99, HALF_EXTENT, 3.0)
	var lighting: LightingProfile = LightingProfile.from_id("high_contrast", 5)
	var results: Array = []
	for _run in range(2):
		var memory: SpatialMemory = _memory()
		for step in range(10):
			var position := Vector3(-7.0 + float(step) * 1.5, 0.0, 3.0)
			memory.observe_self(position, lighting.illumination_at(position))
			memory.observe_cells(
				position + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0),
				position,
				Vector3.FORWARD,
				world,
				lighting,
				SandboxConfig.AGENT_FOV_DEG,
				SandboxConfig.VISION_RANGE
			)
			memory.tick(0.2)
		results.append(memory)
	var first: SpatialMemory = results[0]
	var second: SpatialMemory = results[1]
	t.assert_eq(first.known_cell_count(), second.known_cell_count())
	t.assert_eq(first.route, second.route)
	for index in range(first.cell_count):
		t.assert_almost_eq(first.observed_time[index], second.observed_time[index], 0.0001)
		t.assert_almost_eq(first.confidence[index], second.confidence[index], 0.0001)
		t.assert_almost_eq(first.illumination[index], second.illumination[index], 0.0001)
		t.assert_almost_eq(first.openness[index], second.openness[index], 0.0001)
	return t


func test_darkness_shrinks_what_a_single_look_reveals() -> SandboxTest:
	var t := SandboxTest.new("darkness_shrinks_what_a_single_look_reveals")
	var origin := Vector3(0.0, 0.0, 9.0)
	var bright: SpatialMemory = _memory()
	bright.observe_cells(
		origin + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0),
		origin,
		Vector3.FORWARD,
		null,
		LightingProfile.from_id("normal", 1),
		SandboxConfig.AGENT_FOV_DEG,
		SandboxConfig.VISION_RANGE
	)
	var dark: SpatialMemory = _memory()
	dark.observe_cells(
		origin + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0),
		origin,
		Vector3.FORWARD,
		null,
		LightingProfile.from_id("night", 1),
		SandboxConfig.AGENT_FOV_DEG,
		SandboxConfig.VISION_RANGE
	)
	t.assert_lt(
		float(dark.known_cell_count()),
		float(bright.known_cell_count()),
		"night must reveal fewer cells than daylight"
	)
	t.assert_gt(float(dark.known_cell_count()), 0.0, "night is dark, not blind")
	return t


func test_nearest_unknown_points_at_the_frontier() -> SandboxTest:
	var t := SandboxTest.new("nearest_unknown_points_at_the_frontier")
	var memory: SpatialMemory = _memory()
	var origin := Vector3(0.0, 0.0, 9.0)
	memory.observe_self(origin, 1.0)
	_look(memory, origin, Vector3.FORWARD, null)
	var frontier: Dictionary = memory.nearest_unknown(origin)
	t.assert_false(frontier.is_empty(), "a partially explored map must have a frontier")
	t.assert_false(
		memory.is_known(frontier["position"]), "the frontier cell must itself be unknown"
	)
	return t

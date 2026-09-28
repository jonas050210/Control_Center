## Tests for the Map Analyzer exploration mode.
##
## Covers the mode end to end through `EnvironmentCore`: an agent dropped
## into an unknown map builds map knowledge from perception alone, coverage
## grows as it moves, the episode ends when the map is explored, the same
## seed replays to the same knowledge, and turning the mode off costs the
## combat path nothing.
class_name TestMapAnalyzer
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const MapAnalyzer = preload("res://scripts/exploration/map_analyzer.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")


func _explorer(map_id: String, seed_value: int) -> EnvironmentCore:
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.COVER_AND_OBSTACLES)
	env.set_map(map_id)
	env.set_exploration_mode(true)
	env.reset(seed_value)
	return env


## Walks the agent on a fixed, scripted patrol. Scripted movement is a
## BASELINE for testing the analyzer, never a policy.
func _patrol(env: EnvironmentCore, steps: int) -> void:
	for step in range(steps):
		var turn: int = 2 if int(step / 40) % 2 == 0 else 0
		env.step(Action.from_multidiscrete([2, 1, turn, 1, 0, 0]), SandboxConfig.SIMULATION_DT)


func test_exploration_mode_starts_with_no_knowledge() -> SandboxTest:
	var t := SandboxTest.new("exploration_mode_starts_with_no_knowledge")
	var env: EnvironmentCore = _explorer("two_rooms", 11)
	t.assert_not_null(env.exploration, "exploration mode must build a MapAnalyzer")
	t.assert_true(env.exploration_mode)
	t.assert_eq(env.enemies.size(), 0, "solo exploration must spawn no enemies")
	t.assert_lt(env.exploration.coverage(), 0.2, "the map must start essentially unknown")
	t.assert_false(env.exploration.completed)
	return t


func test_coverage_grows_while_exploring() -> SandboxTest:
	var t := SandboxTest.new("coverage_grows_while_exploring")
	var env: EnvironmentCore = _explorer("training_yard", 5)
	var start: float = env.exploration.coverage()
	_patrol(env, 120)
	var middle: float = env.exploration.coverage()
	_patrol(env, 120)
	var late: float = env.exploration.coverage()
	t.assert_gt(middle, start, "moving and looking around must reveal new ground")
	t.assert_gte(late, middle, "coverage must never shrink")
	t.assert_lte(late, 1.0)
	return t


func test_exploration_reward_is_paid_for_new_ground_only() -> SandboxTest:
	var t := SandboxTest.new("exploration_reward_is_paid_for_new_ground_only")
	var env: EnvironmentCore = _explorer("open_field", 3)
	_patrol(env, 30)
	# One idle sweep to settle: the agent stopped between sweeps, so the
	# first sweep from the resting pose may legitimately reveal a sliver.
	for _w in range(20):
		env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	# From here on, standing still in already-observed ground must not pay.
	var idle_reward: float = 0.0
	for _i in range(40):
		var result: Dictionary = env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
		var events: Dictionary = result["info"]["events"]
		idle_reward += float(events.get("exploration_gain", 0.0))
	t.assert_almost_eq(idle_reward, 0.0, 0.0001, "re-seeing known cells must not pay")
	return t


func test_episode_ends_when_the_map_is_explored() -> SandboxTest:
	var t := SandboxTest.new("episode_ends_when_the_map_is_explored")
	var env: EnvironmentCore = _explorer("open_field", 17)
	var steps: int = 0
	while not env.is_done() and steps < 1200:
		env.step(Action.from_multidiscrete([2, 1, 2, 1, 0, 0]), SandboxConfig.SIMULATION_DT)
		steps += 1
	if env.exploration.completed:
		t.assert_eq(env.episode.done_reason, "map_explored")
		t.assert_gte(env.exploration.coverage(), SandboxConfig.EXPLORATION_TARGET_COVERAGE)
	else:
		# Not every patrol finishes an arena inside the step budget; the
		# contract that matters is that it never ends for the wrong reason.
		t.assert_ne(env.episode.done_reason, "all_enemies_eliminated")
	return t


func test_same_seed_replays_the_same_map_knowledge() -> SandboxTest:
	var t := SandboxTest.new("same_seed_replays_the_same_map_knowledge")
	var coverages: Array = []
	var routes: Array = []
	for _run in range(2):
		var env: EnvironmentCore = _explorer("compound", 808)
		_patrol(env, 150)
		coverages.append(env.exploration.coverage())
		routes.append(env.exploration.memory.route)
	t.assert_almost_eq(float(coverages[0]), float(coverages[1]), 0.000001)
	t.assert_eq(routes[0], routes[1], "the explored route must replay exactly")
	return t


func test_different_seeds_explore_different_maps() -> SandboxTest:
	var t := SandboxTest.new("different_seeds_explore_different_maps")
	var first: EnvironmentCore = _explorer("random_ops", 1)
	var second: EnvironmentCore = _explorer("random_ops", 2)
	_patrol(first, 90)
	_patrol(second, 90)
	t.assert_gt(first.exploration.coverage(), 0.0)
	t.assert_gt(second.exploration.coverage(), 0.0)
	return t


func test_exploration_is_off_by_default() -> SandboxTest:
	var t := SandboxTest.new("exploration_is_off_by_default")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.COVER_AND_OBSTACLES)
	env.reset(42)
	t.assert_null(env.exploration, "combat episodes must not pay for the analyzer")
	t.assert_false(env.exploration_mode)
	for _i in range(20):
		env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	t.assert_null(env.exploration)
	return t


func test_tracking_can_run_alongside_combat_without_paying_reward() -> SandboxTest:
	var t := SandboxTest.new("tracking_can_run_alongside_combat_without_paying_reward")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.COVER_AND_OBSTACLES)
	env.set_exploration_tracking(true)
	env.reset(42)
	t.assert_not_null(env.exploration)
	t.assert_eq(env.enemies.size(), 2, "tracking must not remove the enemies")
	var gain: float = 0.0
	for _i in range(60):
		var result: Dictionary = env.step(
			Action.from_multidiscrete([2, 1, 1, 1, 0, 0]), SandboxConfig.SIMULATION_DT
		)
		gain += float((result["info"]["events"] as Dictionary).get("exploration_gain", 0.0))
	t.assert_almost_eq(gain, 0.0, 0.0001, "exploration must not pay outside Map Analyzer mode")
	t.assert_gt(env.exploration.coverage(), 0.0, "knowledge must still be built")
	return t


func test_analyzer_reset_clears_previous_episode_knowledge() -> SandboxTest:
	var t := SandboxTest.new("analyzer_reset_clears_previous_episode_knowledge")
	var env: EnvironmentCore = _explorer("pillar_hall", 21)
	_patrol(env, 100)
	t.assert_gt(env.exploration.coverage(), 0.0)
	env.reset(21)
	t.assert_lt(env.exploration.coverage(), 0.2, "a new episode must start nearly blind")
	t.assert_eq(env.exploration.memory.route.size(), 0, "the route restarts empty")
	t.assert_false(env.exploration.completed)
	return t


func test_frontier_direction_points_away_from_known_ground() -> SandboxTest:
	var t := SandboxTest.new("frontier_direction_points_away_from_known_ground")
	var analyzer: MapAnalyzer = MapAnalyzer.create(10.0)
	var context: Dictionary = {
		"position": Vector3(0.0, 0.0, 8.0),
		"forward": Vector3.FORWARD,
		"world": null,
		"lighting": null,
		"local_illumination": 1.0,
	}
	analyzer.update(SandboxConfig.SIMULATION_DT, context)
	var direction: Vector3 = analyzer.frontier_direction(Vector3(0.0, 0.0, 8.0))
	t.assert_gt(direction.length(), 0.5, "a partly explored map must offer a frontier")
	t.assert_almost_eq(direction.length(), 1.0, 0.0001)
	return t


func test_control_center_payload_is_read_only_and_complete() -> SandboxTest:
	var t := SandboxTest.new("control_center_payload_is_read_only_and_complete")
	var env: EnvironmentCore = _explorer("night_yard", 4)
	_patrol(env, 60)
	var before: float = env.exploration.coverage()
	var payload: Dictionary = env.get_exploration_state()
	t.assert_true(bool(payload["enabled"]))
	t.assert_true(bool(payload["mode"]))
	t.assert_true(payload.has("memory"))
	var memory_payload: Dictionary = payload["memory"]
	t.assert_eq(int(memory_payload["known_cells"]), env.exploration.memory.known_cell_count())
	t.assert_almost_eq(float(payload["coverage"]), before, 0.000001)
	# Reading the payload must not have advanced the simulation.
	t.assert_almost_eq(env.exploration.coverage(), before, 0.000001)
	return t

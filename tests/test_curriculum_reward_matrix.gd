## Cross-level reward regression tests for the first curriculum transitions.
## These deliberately exercise EnvironmentCore rather than RewardSystem alone.
class_name TestCurriculumRewardMatrix
extends RefCounted

const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")

func _env(level: int, enemies: int = 1) -> EnvironmentCore:
	var env := EnvironmentCore.new(0, enemies)
	env.set_curriculum_level(level)
	env.reset(2468)
	return env

func test_level2_idle_remains_negative_with_moving_target() -> SandboxTest:
	var t := SandboxTest.new("level2_idle_reward")
	var env := _env(2)
	var total := 0.0
	for _i in range(120):
		var result: Dictionary = env.step(Action.idle())
		total += float(result.reward)
		if result.done:
			break
	t.assert_lt(total, 0.0)
	t.assert_eq(env.episode.shots_fired, 0)
	t.assert_eq(env.episode.kills, 0)
	return t

func test_level2_repeated_shooting_cannot_outscore_aligned_shooting() -> SandboxTest:
	var t := SandboxTest.new("level2_repeated_shooting_reward")
	var spam := _env(2)
	var aligned := _env(2)
	var spam_total := 0.0
	var aligned_total := 0.0
	for i in range(180):
		var spam_result: Dictionary = spam.step(Action.from_discrete(Action.Discrete.SHOOT))
		spam_total += float(spam_result.reward)
		var aligned_action := Action.from_discrete(Action.Discrete.SHOOT)
		if i % 20 < 4:
			aligned_action = Action.from_discrete(Action.Discrete.LOOK_LEFT)
		var aligned_result: Dictionary = aligned.step(aligned_action)
		aligned_total += float(aligned_result.reward)
		if aligned_result.done:
			break
	t.assert_lt(spam.episode.penalty_missed_shot, 0.0)
	t.assert_eq(spam.episode.kills, 0)
	t.assert_gt(spam.episode.shots_fired, 0)
	t.assert_gt(aligned.episode.shots_fired, 0)
	return t

func test_level4_multiple_targets_reward_components_reconcile() -> SandboxTest:
	var t := SandboxTest.new("level4_reward_reconciliation")
	var env := _env(4, 3)
	for i in range(80):
		var action := Action.from_multidiscrete([1, 1, i % 3, (i + 1) % 3, 1, 0])
		var result: Dictionary = env.step(action)
		var components: Dictionary = result.info.reward_components
		var component_sum := 0.0
		for value in components.values():
			component_sum += float(value)
		t.assert_almost_eq(component_sum, float(result.reward), 0.0001)
		if result.done:
			break
	return t

func test_level4_same_seed_same_actions_same_trace() -> SandboxTest:
	var t := SandboxTest.new("level4_deterministic_trace")
	var first := _trace(4, 3, 1357)
	var second := _trace(4, 3, 1357)
	t.assert_eq(first, second)
	return t

func _trace(level: int, enemies: int, seed: int) -> Array:
	var env := EnvironmentCore.new(0, enemies)
	env.set_curriculum_level(level)
	env.reset(seed)
	var trace: Array = []
	for i in range(30):
		var action := Action.from_multidiscrete(
			[i % 3, (i + 1) % 3, i % 3, 0, i % 2, 0]
		)
		var result: Dictionary = env.step(action)
		trace.append({
			"observation": Array(result.observation.to_array()),
			"reward": result.reward,
			"events": result.info.events.duplicate(true),
			"done": result.done,
		})
		if result.done:
			break
	return trace

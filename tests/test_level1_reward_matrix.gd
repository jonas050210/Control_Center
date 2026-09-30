## End-to-end Level 1 reward regression matrix.
## These tests deliberately use EnvironmentCore.step(), not RewardSystem alone.
class_name TestLevel1RewardMatrix
extends RefCounted

const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")


func _level1() -> EnvironmentCore:
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(1)
	env.reset(123)
	return env


func _trace(seed: int, actions: Array) -> Array:
	var env := _level1()
	env.reset(seed)
	var trace: Array = []
	for action in actions:
		var result: Dictionary = env.step(action)
		(
			trace
			. append(
				{
					"observation": Array(result.observation.to_array()),
					"reward": result.reward,
					"events": result.info.events.duplicate(true),
					"done": result.done,
					"reason": result.info.done_reason,
				}
			)
		)
		if result.done:
			break
	return trace


func test_end_to_end_passive_is_negative() -> SandboxTest:
	var t := SandboxTest.new("level1_passive_end_to_end")
	var env := _level1()
	var total := 0.0
	for _i in range(30):
		var result: Dictionary = env.step(Action.idle())
		total += float(result.reward)
	t.assert_lt(total, 0.0)
	t.assert_eq(env.episode.shots_fired, 0)
	t.assert_eq(env.episode.kills, 0)
	t.assert_lt(env.episode.get_reward_breakdown().penalty_passivity, 0.0)
	return t


func test_aim_only_and_oscillation_do_not_farm_survival() -> SandboxTest:
	var t := SandboxTest.new("level1_aim_oscillation_end_to_end")
	var aim := _level1()
	var oscillation := _level1()
	var aim_total := 0.0
	var oscillation_total := 0.0
	for i in range(60):
		var a: Dictionary = aim.step(Action.from_discrete(Action.Discrete.LOOK_RIGHT))
		var oscillate: int = Action.Discrete.LOOK_RIGHT if i % 2 == 0 else Action.Discrete.LOOK_LEFT
		var b: Dictionary = oscillation.step(Action.from_discrete(oscillate))
		aim_total += float(a.reward)
		oscillation_total += float(b.reward)
	t.assert_lte(aim_total, 0.05)
	t.assert_lte(oscillation_total, 0.05)
	t.assert_eq(aim.episode.shots_fired, 0)
	t.assert_eq(oscillation.episode.shots_fired, 0)
	return t


func test_movement_cycle_does_not_repeat_distance_reward() -> SandboxTest:
	var t := SandboxTest.new("level1_movement_cycle_reward")
	var env := _level1()
	var total := 0.0
	for _i in range(120):
		var cycle: int = (
			Action.Discrete.MOVE_FORWARD if _i % 2 == 0 else Action.Discrete.MOVE_BACKWARD
		)
		var action := Action.from_discrete(cycle)
		total += float(env.step(action).reward)
	t.assert_lt(total, 1.0, "movement cycling must not dominate the kill reward")
	t.assert_eq(env.episode.kills, 0)
	return t


func test_repeated_shooting_is_worse_than_aligned_shooting() -> SandboxTest:
	var t := SandboxTest.new("level1_repeated_shooting_vs_useful")
	var spam := _level1()
	var useful := _level1()
	# The aligned arm uses the same explicit aim setup as
	# test_aligned_shooting_kills_target_and_records_events and runs until
	# the episode ends: killing the 100 HP target takes four 25-damage
	# hits, which on the 0.5 s weapon cooldown cannot happen before tick 90
	# (60 Hz). A 30-tick cap with no setup difference made this arm run the
	# exact same trace as the spam arm, so the totals could only be equal.
	useful.agent.position = Vector3(0.0, 0.0, 5.0)
	useful.agent.yaw_deg = 0.0
	useful.agent.pitch_deg = 0.0
	useful.enemies[0].position = Vector3(0.0, 0.0, 0.0)
	var spam_total := 0.0
	var useful_total := 0.0
	for _i in range(30):
		spam_total += float(spam.step(Action.from_discrete(Action.Discrete.SHOOT)).reward)
	for _i in range(100):
		var useful_result: Dictionary = useful.step(Action.from_discrete(Action.Discrete.SHOOT))
		useful_total += float(useful_result.reward)
		if useful_result.done:
			break
	t.assert_lt(spam_total, useful_total)
	t.assert_gt(useful.episode.shots_hit, 0)
	t.assert_gt(useful.episode.kills, 0)
	t.assert_lt(spam.episode.penalty_useless_shot, 0.0)
	return t


func test_timeout_is_not_success() -> SandboxTest:
	var t := SandboxTest.new("level1_timeout_terminal_state")
	var env := _level1()
	var last: Dictionary = {}
	for _i in range(SandboxConfig.MAX_EPISODE_STEPS):
		last = env.step(Action.idle())
		if last.done:
			break
	t.assert_true(last.done)
	t.assert_eq(last.info.done_reason, "timeout")
	t.assert_true(last.info["TimeLimit.truncated"])
	t.assert_eq(env.episode.kills, 0)
	t.assert_false(bool(last.info.metrics.win))
	return t


func test_aligned_shooting_kills_target_and_records_events() -> SandboxTest:
	var t := SandboxTest.new("level1_aligned_shoot_kill")
	var env := _level1()
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	env.agent.pitch_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)
	var saw_hit := false
	var last: Dictionary = {}
	for _i in range(100):
		last = env.step(Action.from_discrete(Action.Discrete.SHOOT))
		saw_hit = saw_hit or bool(last.info.events.hit)
		if last.done:
			break
	t.assert_true(saw_hit)
	t.assert_eq(env.episode.kills, 1)
	t.assert_true(last.done)
	t.assert_eq(last.info.done_reason, "all_enemies_eliminated")
	t.assert_gt(env.episode.reward_hits, 0.0)
	t.assert_gt(env.episode.reward_kills, 0.0)
	return t


func test_full_trace_is_deterministic_and_seed_sensitive() -> SandboxTest:
	var t := SandboxTest.new("level1_full_trace_determinism")
	var actions: Array = []
	for i in range(40):
		actions.append(Action.from_multidiscrete([i % 3, (i + 1) % 3, i % 3, 1, 0, 0]))
	var first := _trace(77, actions)
	var second := _trace(77, actions)
	t.assert_eq(first, second, "same seed and actions must produce identical trace")
	var different := _trace(78, actions)
	t.assert_ne(first, different, "changing seed must change the randomized trace")
	return t


func test_dead_target_cannot_generate_reward() -> SandboxTest:
	var t := SandboxTest.new("level1_dead_target_no_reward")
	var env := _level1()
	env.enemies[0].alive = false
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))
	t.assert_false(result.info.events.hit)
	t.assert_false(result.info.events.kill)
	t.assert_true(result.info.events.useless_shot)
	t.assert_eq(env.episode.kills, 0)
	t.assert_eq(env.episode.reward_hits, 0.0)
	t.assert_eq(env.episode.reward_kills, 0.0)
	return t


func test_reward_breakdown_reconciles_each_step() -> SandboxTest:
	var t := SandboxTest.new("level1_reward_breakdown_reconciliation")
	var env := _level1()
	for i in range(20):
		var discrete: int = Action.Discrete.SHOOT if i % 3 == 0 else Action.Discrete.IDLE
		var result: Dictionary = env.step(Action.from_discrete(discrete))
		var direct_components: Dictionary = result.info.reward_components
		var direct_sum := 0.0
		for value in direct_components.values():
			direct_sum += float(value)
		t.assert_almost_eq(direct_sum, float(result.reward), 0.0001)
		var breakdown: Dictionary = env.episode.get_reward_breakdown()
		var component_sum := 0.0
		for key in breakdown.keys():
			if key != "total":
				component_sum += float(breakdown[key])
		t.assert_almost_eq(component_sum, float(breakdown.total), 0.0001)
		t.assert_almost_eq(float(result.reward), env.episode.last_reward, 0.0001)
		if result.done:
			break
	return t

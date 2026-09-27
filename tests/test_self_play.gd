## Self-play foundation tests: two independent policy/action slots and
## per-agent match metrics.
class_name TestSelfPlay
extends RefCounted

# Explicit dependency: headless --script runs do not populate the editor class cache.
const TestResult = preload("res://tests/sandbox_test.gd")


func test_self_play_reset_has_two_observations() -> TestResult:
	var t := TestResult.new("self_play_reset_has_two_observations")
	var match_env := SelfPlayEnvironmentCore.new()
	var observations: Array = match_env.reset(10, 20)
	t.assert_eq(observations.size(), 2)
	t.assert_eq(observations[0].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(observations[1].to_array().size(), Observation.FIELD_COUNT)
	return t


func test_self_play_step_returns_per_agent_rewards_and_infos() -> TestResult:
	var t := TestResult.new("self_play_step_returns_per_agent_metrics")
	var match_env := SelfPlayEnvironmentCore.new()
	match_env.reset(1, 2)
	var result: Dictionary = match_env.step([Action.idle(), Action.idle()])
	t.assert_eq(result.rewards.size(), 2)
	t.assert_eq(result.infos.size(), 2)
	t.assert_true(result.infos[0].metrics.has("kills"))
	t.assert_true(result.infos[1].metrics.has("kills"))
	return t

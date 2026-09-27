## Tests for EpisodeState bookkeeping.
class_name TestEpisodeState
extends RefCounted

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_start_new_episode_resets_counters() -> SandboxTest:
	var t := SandboxTest.new("episode_start_resets_counters")
	var ep := EpisodeState.new()
	ep.record_step(1.0)
	ep.record_step(2.0)
	ep.mark_done("agent_died")
	ep.start_new_episode()
	t.assert_eq(ep.step_count, 0)
	t.assert_almost_eq(ep.cumulative_reward, 0.0, 0.0001)
	t.assert_false(ep.done)
	t.assert_eq(ep.episode_count, 1)
	return t


func test_record_step_accumulates_reward() -> SandboxTest:
	var t := SandboxTest.new("episode_record_step_accumulates_reward")
	var ep := EpisodeState.new()
	ep.record_step(1.0)
	ep.record_step(-0.5)
	t.assert_eq(ep.step_count, 2)
	t.assert_almost_eq(ep.cumulative_reward, 0.5, 0.0001)
	t.assert_almost_eq(ep.last_reward, -0.5, 0.0001)
	return t


func test_is_timeout_threshold() -> SandboxTest:
	var t := SandboxTest.new("episode_is_timeout_threshold")
	var ep := EpisodeState.new()
	for _i in range(10):
		ep.record_step(0.0)
	t.assert_true(ep.is_timeout(10))
	t.assert_false(ep.is_timeout(11))
	return t

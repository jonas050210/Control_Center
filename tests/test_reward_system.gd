## Tests for the centralized RewardSystem calculation.
class_name TestRewardSystem
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const RewardSystem = preload("res://scripts/reward/reward_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_hit_gives_configured_reward() -> SandboxTest:
	var t := SandboxTest.new("reward_hit")
	var reward := RewardSystem.compute({"hit": true, "alive": true})
	t.assert_almost_eq(reward, SandboxConfig.REWARD_HIT + SandboxConfig.REWARD_SURVIVE_TICK, 0.0001)
	return t


func test_kill_gives_configured_reward() -> SandboxTest:
	var t := SandboxTest.new("reward_kill")
	var reward := RewardSystem.compute({"hit": true, "kill": true, "alive": true})
	var expected: float = (
		SandboxConfig.REWARD_HIT + SandboxConfig.REWARD_KILL + SandboxConfig.REWARD_SURVIVE_TICK
	)
	t.assert_almost_eq(reward, expected, 0.0001)
	return t


func test_damage_taken_is_penalized_proportionally() -> SandboxTest:
	var t := SandboxTest.new("reward_damage_taken_penalty")
	var reward := RewardSystem.compute({"damage_taken": 20.0, "alive": true})
	var expected: float = (
		20.0 * SandboxConfig.PENALTY_DAMAGE_TAKEN_PER_HP + SandboxConfig.REWARD_SURVIVE_TICK
	)
	t.assert_almost_eq(reward, expected, 0.0001)
	t.assert_lt(
		reward,
		SandboxConfig.REWARD_SURVIVE_TICK,
		"taking damage should net a worse reward than doing nothing"
	)
	return t


func test_death_applies_large_penalty_and_no_survive_bonus() -> SandboxTest:
	var t := SandboxTest.new("reward_death_penalty")
	var reward := RewardSystem.compute({"died": true, "damage_taken": 100.0, "alive": false})
	var expected: float = (
		SandboxConfig.PENALTY_DEATH + 100.0 * SandboxConfig.PENALTY_DAMAGE_TAKEN_PER_HP
	)
	t.assert_almost_eq(reward, expected, 0.0001)
	t.assert_lt(reward, 0.0, "dying should be net negative")
	return t


func test_useless_shot_is_penalized() -> SandboxTest:
	var t := SandboxTest.new("reward_useless_shot_penalty")
	var reward := RewardSystem.compute({"useless_shot": true, "alive": true})
	t.assert_almost_eq(
		reward, SandboxConfig.PENALTY_USELESS_SHOT + SandboxConfig.REWARD_SURVIVE_TICK, 0.0001
	)
	return t


## Regression: genuine misses must stay much cheaper than impossible trigger
## pulls, otherwise the expected value of shooting turns negative while aim is
## being learned and PPO collapses to a never-shoot policy.
func test_missed_shot_is_cheaper_than_useless_shot() -> SandboxTest:
	var t := SandboxTest.new("reward_missed_shot_cheaper_than_useless_shot")
	var miss_reward := RewardSystem.compute({"missed_shot": true, "alive": true})
	t.assert_almost_eq(
		miss_reward,
		SandboxConfig.PENALTY_MISSED_SHOT + SandboxConfig.REWARD_SURVIVE_TICK,
		0.0001
	)
	t.assert_gt(SandboxConfig.PENALTY_MISSED_SHOT, SandboxConfig.PENALTY_USELESS_SHOT,
		"a genuine miss must be cheaper than an impossible pull")
	# Sanity: an aimed hit (+1.0) must outweigh roughly fifty real misses.
	t.assert_gt(SandboxConfig.REWARD_HIT, 50.0 * -SandboxConfig.PENALTY_MISSED_SHOT,
		"hitting must dominate missing often enough for shooting to stay learnable")
	return t


func test_survive_tick_alone_is_small_and_positive() -> SandboxTest:
	var t := SandboxTest.new("reward_survive_tick")
	var reward := RewardSystem.compute({"alive": true})
	t.assert_almost_eq(reward, SandboxConfig.REWARD_SURVIVE_TICK, 0.0001)
	t.assert_gt(reward, 0.0)
	return t


func test_positioning_reward_is_clamped_and_small() -> SandboxTest:
	var t := SandboxTest.new("reward_positioning_clamped")
	var reward := RewardSystem.compute({"positioning_delta": 1000.0, "alive": true})
	var max_possible: float = (
		SandboxConfig.REWARD_POSITIONING_MAX + SandboxConfig.REWARD_SURVIVE_TICK
	)
	t.assert_true(
		reward <= max_possible + 0.0001, "positioning reward must be clamped, not unbounded"
	)
	return t

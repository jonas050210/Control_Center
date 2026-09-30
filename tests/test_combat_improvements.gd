## Tests for closest hit detection, reward breakdowns, and health check.
class_name TestCombatImprovements
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_closest_enemy_hit_first() -> SandboxTest:
	var t := SandboxTest.new("closest_enemy_hit_first")
	var env := EnvironmentCore.new(0, 2)
	env.reset(42)

	# Position agent facing -Z
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	env.agent.pitch_deg = 0.0

	# Near enemy at z=0, Far enemy at z=-5
	env.enemies[0].position = Vector3(0.0, 0.0, -5.0)  # far enemy in array slot 0
	env.enemies[1].position = Vector3(0.0, 0.0, 0.0)  # near enemy in array slot 1

	env.agent.weapon.cooldown_remaining = 0.0
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))

	t.assert_true(result.info.events.hit, "shot should connect")
	t.assert_eq(result.info.events.shot_result, "hit")
	t.assert_almost_eq(
		env.enemies[1].health,
		env.enemies[1].max_health - env.agent.weapon.damage,
		0.001,
		"nearer enemy should take damage first"
	)
	t.assert_almost_eq(
		env.enemies[0].health,
		env.enemies[0].max_health,
		0.001,
		"farther occluded enemy should not take damage"
	)
	return t


func test_reward_breakdown_tracking() -> SandboxTest:
	var t := SandboxTest.new("reward_breakdown_tracking")
	var env := EnvironmentCore.new(0, 1)
	env.reset(10)
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	env.agent.pitch_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)

	env.agent.weapon.cooldown_remaining = 0.0
	env.step(Action.from_discrete(Action.Discrete.SHOOT))

	var breakdown: Dictionary = env.episode.get_reward_breakdown()
	t.assert_gt(breakdown.reward_hits, 0.0, "reward_hits should be recorded in breakdown")
	t.assert_gt(breakdown.reward_damage, 0.0, "damage shaping should be recorded in breakdown")
	t.assert_eq(breakdown.reward_survive, 0.0, "combat must not pay passive survival")
	return t


## Regression: a near miss at a live, clear target is a genuine aiming
## attempt (missed_shot, cheap), NOT impossible spam. A shot nowhere near a
## target is useless_shot and receives the harsher penalty.
func test_near_miss_is_missed_shot_not_useless() -> SandboxTest:
	var t := SandboxTest.new("near_miss_is_missed_shot_not_useless")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(3)
	env.reset(10)
	# Enemy straight ahead at 10m. A 6 degree yaw is inside the near-miss cone
	# but outside the 0.8m hit radius, so this is exactly a plausible miss.
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 6.0
	env.agent.pitch_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, -5.0)
	env.agent.weapon.cooldown_remaining = 0.0
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))
	t.assert_false(result.info.events.hit, "shot should not connect")
	t.assert_true(result.info.events.shot_fired, "weapon was ready and must have fired")
	t.assert_true(result.info.events.missed_shot, "near miss must count as missed_shot")
	t.assert_false(result.info.events.useless_shot, "near miss must NOT count as useless_shot")
	t.assert_eq(result.info.events.shot_result, "near_miss")
	t.assert_eq(env.episode.near_miss_shots, 1)
	var breakdown: Dictionary = env.episode.get_reward_breakdown()
	t.assert_almost_eq(breakdown.penalty_missed_shot, -0.01, 0.002)
	t.assert_almost_eq(breakdown.penalty_useless_shot, 0.0, 0.0001)
	return t


func test_random_spray_is_useless_not_missed() -> SandboxTest:
	var t := SandboxTest.new("random_spray_is_useless_not_missed")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(3)
	env.reset(10)
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 90.0
	env.agent.pitch_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, -5.0)
	env.agent.weapon.cooldown_remaining = 0.0
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))
	t.assert_false(result.info.events.hit)
	t.assert_true(result.info.events.shot_fired)
	t.assert_true(result.info.events.useless_shot, "spraying far away from target is useless")
	t.assert_false(result.info.events.missed_shot)
	t.assert_eq(result.info.events.shot_result, "useless_spam")
	t.assert_eq(env.episode.useless_shots, 1)
	return t


func test_cooldown_pull_is_useless_shot_and_does_not_fire() -> SandboxTest:
	var t := SandboxTest.new("cooldown_pull_is_useless_shot_and_does_not_fire")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(3)
	env.reset(10)
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)
	env.agent.weapon.cooldown_remaining = 0.0
	env.step(Action.from_discrete(Action.Discrete.SHOOT))  # fires, starts cooldown
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))  # still on cooldown
	t.assert_false(result.info.events.shot_fired, "second pull cannot fire during cooldown")
	t.assert_true(result.info.events.useless_shot, "cooldown pull is an impossible shot")
	t.assert_false(result.info.events.missed_shot, "nothing fired, so it cannot be a miss")
	t.assert_eq(result.info.events.shot_result, "cooldown")
	t.assert_eq(env.episode.cooldown_shots, 1)
	var breakdown: Dictionary = env.episode.get_reward_breakdown()
	t.assert_almost_eq(breakdown.penalty_useless_shot, -0.1, 0.001)
	t.assert_eq(env.episode.shots_fired, 1, "only the first pull actually fired")
	return t


func test_shoot_with_no_alive_target_is_useless_not_missed() -> SandboxTest:
	var t := SandboxTest.new("shoot_with_no_alive_target_is_useless_not_missed")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(1)  # enemy never moves/attacks: controlled conditions
	env.reset(10)
	env.enemies[0].position = Vector3(0.0, 0.0, -5.0)
	env.enemies[0].alive = false
	env.agent.weapon.cooldown_remaining = 0.0
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))
	t.assert_true(result.info.events.shot_fired)
	t.assert_true(result.info.events.useless_shot, "firing at nothing is an impossible shot")
	t.assert_false(result.info.events.missed_shot, "no live target means no genuine miss")
	t.assert_eq(result.info.events.shot_result, "useless_no_target")
	return t


## Regression: positioning reward must reflect the agent's own motion only.
## A standing-still agent must not collect reward just because an enemy is
## walking toward it (enemy movement is resolved AFTER the agent moves).
func test_stationary_agent_gains_no_positioning_from_enemy_approach() -> SandboxTest:
	var t := SandboxTest.new("stationary_agent_gains_no_positioning_from_enemy_approach")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(2)  # enemy moves, does not attack
	env.reset(3)
	var result: Dictionary = env.step(Action.idle())
	t.assert_almost_eq(
		result.info.events.positioning_delta,
		0.0,
		0.0001,
		"idle agent must not be paid for the enemy's own approach"
	)
	var breakdown: Dictionary = env.episode.get_reward_breakdown()
	t.assert_almost_eq(breakdown.reward_positioning, 0.0, 0.0001)
	return t


func test_health_check_reports_healthy() -> SandboxTest:
	var t := SandboxTest.new("health_check_reports_healthy")
	var env := EnvironmentCore.new(0, 2)
	env.reset(1)
	var report: Dictionary = env.health_check()
	t.assert_true(report.healthy, "fresh environment should pass health check")
	t.assert_eq(report.issues.size(), 0)
	return t

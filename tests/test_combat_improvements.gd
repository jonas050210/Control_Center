## Tests for closest hit detection, reward breakdowns, and health check.
class_name TestCombatImprovements
extends RefCounted

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
	env.enemies[1].position = Vector3(0.0, 0.0, 0.0)   # near enemy in array slot 1

	env.agent.weapon.cooldown_remaining = 0.0
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))

	t.assert_true(result.info.events.hit, "shot should connect")
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
	t.assert_gt(breakdown.reward_survive, 0.0, "reward_survive should be recorded in breakdown")
	return t


func test_health_check_reports_healthy() -> SandboxTest:
	var t := SandboxTest.new("health_check_reports_healthy")
	var env := EnvironmentCore.new(0, 2)
	env.reset(1)
	var report: Dictionary = env.health_check()
	t.assert_true(report.healthy, "fresh environment should pass health check")
	t.assert_eq(report.issues.size(), 0)
	return t

## Tests for EnvironmentCore: the RL reset/step/observation/reward/done
## interface, deterministic seeding, combat resolution and episode
## termination.
class_name TestEnvironmentCore
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_environment_creation_has_agent_and_enemies() -> SandboxTest:
	var t := SandboxTest.new("environment_creation_has_agent_and_enemies")
	var env := EnvironmentCore.new(0, 2)
	t.assert_not_null(env.agent, "environment should own an agent")
	t.assert_eq(env.enemies.size(), 2, "environment should own the requested number of enemies")
	return t


func test_reset_returns_full_health_observation() -> SandboxTest:
	var t := SandboxTest.new("reset_returns_full_health_observation")
	var env := EnvironmentCore.new(0, 1)
	var obs: Observation = env.reset(1)
	t.assert_almost_eq(obs.agent_health_norm, 1.0, 0.0001, "agent should start at full health")
	t.assert_almost_eq(obs.enemy_health_norm, 1.0, 0.0001, "enemy should start at full health")
	t.assert_true(obs.enemy_alive, "enemy should start alive")
	t.assert_eq(env.episode.step_count, 0, "step count should reset to zero")
	t.assert_eq(env.episode.episode_count, 1, "first reset should start episode 1")
	return t


func test_reset_after_steps_returns_to_spawn_state() -> SandboxTest:
	var t := SandboxTest.new("reset_after_steps_returns_to_spawn_state")
	var env := EnvironmentCore.new(0, 1)
	env.reset(7)
	var spawn_position: Vector3 = env.agent.position
	for _i in range(20):
		env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))
	t.assert_true(env.agent.position.distance_to(spawn_position) > 0.01, "agent should have moved")
	env.reset(7)
	t.assert_vec_almost_eq(
		env.agent.position, spawn_position, 0.001, "reset should restore spawn position"
	)
	t.assert_almost_eq(
		env.agent.health, env.agent.max_health, 0.001, "reset should restore full health"
	)
	t.assert_eq(env.episode.step_count, 0, "reset should zero the step counter")
	t.assert_eq(env.episode.episode_count, 2, "second reset should start episode 2")
	return t


func test_deterministic_reset_same_seed_gives_identical_observation() -> SandboxTest:
	var t := SandboxTest.new("deterministic_reset_same_seed")
	var env_a := EnvironmentCore.new(0, 3)
	var env_b := EnvironmentCore.new(1, 3)
	var obs_a: Observation = env_a.reset(999)
	var obs_b: Observation = env_b.reset(999)
	t.assert_eq(
		obs_a.to_array(),
		obs_b.to_array(),
		"same seed should produce identical initial observation arrays"
	)
	for i in range(env_a.enemies.size()):
		t.assert_vec_almost_eq(
			env_a.enemies[i].position,
			env_b.enemies[i].position,
			0.0001,
			"enemy spawn positions should match"
		)
	return t


func test_negative_seed_reset_continues_seeded_stream() -> SandboxTest:
	var t := SandboxTest.new("negative_seed_reset_continues_seeded_stream")
	var env_a := EnvironmentCore.new(0, 1)
	var env_b := EnvironmentCore.new(1, 1)
	env_a.reset(4242)
	env_b.reset(4242)
	# A negative seed (used by vector auto-reset between episodes) must keep
	# the seeded RNG stream, so two identically seeded environments stay in
	# lockstep across episode boundaries instead of being re-randomized.
	env_a.reset(-1)
	env_b.reset(-1)
	t.assert_vec_almost_eq(
		env_a.enemies[0].position,
		env_b.enemies[0].position,
		0.000001,
		"auto-reset (seed -1) should continue the deterministic seeded stream"
	)
	return t


func test_different_seeds_generally_differ() -> SandboxTest:
	var t := SandboxTest.new("different_seeds_generally_differ")
	var env_a := EnvironmentCore.new(0, 1)
	var env_b := EnvironmentCore.new(1, 1)
	env_a.reset(1)
	env_b.reset(2)
	var same_spawn: bool = env_a.enemies[0].position.is_equal_approx(env_b.enemies[0].position)
	t.assert_false(
		same_spawn,
		"different seeds should (with overwhelming probability) produce different enemy spawns"
	)
	return t


func test_step_applies_movement_action() -> SandboxTest:
	var t := SandboxTest.new("step_applies_movement_action")
	var env := EnvironmentCore.new(0, 1)
	env.reset(3)
	var start_pos: Vector3 = env.agent.position
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))
	t.assert_true(
		env.agent.position.distance_to(start_pos) > 0.0, "agent should move after a movement action"
	)
	t.assert_true(
		result.has("observation") and result.has("reward") and result.has("done"),
		"step() must return observation/reward/done"
	)
	return t


func test_shooting_hits_and_kills_enemy() -> SandboxTest:
	var t := SandboxTest.new("shooting_hits_and_kills_enemy")
	var env := EnvironmentCore.new(0, 1)
	env.reset(5)
	# Force a clean, guaranteed line-of-sight shot regardless of spawn jitter.
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	env.agent.pitch_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, -5.0)

	var hits_needed: int = ceili(env.enemies[0].max_health / env.agent.weapon.damage)
	var got_hit: bool = false
	var got_kill: bool = false
	for _i in range(hits_needed):
		if env.is_done():
			break
		# Force the weapon ready so this test exercises the hit-test/damage
		# path deterministically without stepping through real cooldown time.
		env.agent.weapon.cooldown_remaining = 0.0
		var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))
		if result.info.events.hit:
			got_hit = true
		if result.info.events.kill:
			got_kill = true

	t.assert_true(got_hit, "at least one shot should register as a hit")
	t.assert_true(got_kill, "enough hits should eventually kill the enemy")
	t.assert_false(env.enemies[0].alive, "enemy should be dead")
	t.assert_gt(float(env.episode.total_kills), 0.0, "kill counter should increment")
	return t


func test_agent_death_ends_episode_and_counts_death() -> SandboxTest:
	var t := SandboxTest.new("agent_death_ends_episode_and_counts_death")
	var env := EnvironmentCore.new(0, 1)
	env.reset(11)
	env.agent.position = Vector3(0.0, 0.0, 0.5)
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)  # inside attack range immediately

	var steps: int = 0
	while not env.is_done() and steps < 800:
		env.step(Action.idle())
		steps += 1

	t.assert_true(env.is_done(), "episode should end once the agent dies")
	t.assert_false(env.agent.alive, "agent should be dead")
	t.assert_eq(env.episode.done_reason, "agent_died")
	t.assert_eq(env.episode.total_deaths, 1)
	return t


func test_episode_times_out_when_nothing_happens_too_long() -> SandboxTest:
	var t := SandboxTest.new("episode_times_out")
	var env := EnvironmentCore.new(0, 1)
	env.max_steps = 5
	env.reset(13)
	# Keep the enemy far away and passive-safe distance so it never reaches
	# the agent within the tiny step budget, isolating the timeout path.
	env.enemies[0].position = Vector3(0.0, 0.0, 9.0)
	env.agent.position = Vector3(0.0, 0.0, -9.0)
	var result: Dictionary = {}
	for _i in range(5):
		result = env.step(Action.idle())
	t.assert_true(result.done, "episode should be done after max_steps")
	t.assert_eq(env.episode.done_reason, "timeout")
	return t


func test_step_after_done_is_a_safe_noop() -> SandboxTest:
	var t := SandboxTest.new("step_after_done_is_safe_noop")
	var env := EnvironmentCore.new(0, 1)
	env.max_steps = 2
	env.reset(1)
	env.step(Action.idle())
	env.step(Action.idle())
	t.assert_true(env.is_done())
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))
	t.assert_true(
		result.info.get("already_done", false), "stepping a done environment should be a no-op"
	)
	return t


func test_get_observations_rewards_done_accessors() -> SandboxTest:
	var t := SandboxTest.new("accessors_return_current_state")
	var env := EnvironmentCore.new(0, 1)
	env.reset(2)
	env.step(Action.idle())
	t.assert_not_null(env.get_observations())
	t.assert_almost_eq(env.get_rewards(), env.episode.last_reward, 0.0001)
	t.assert_eq(env.is_done(), env.episode.done)
	return t

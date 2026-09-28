## Self-play foundation tests: two independent policy/action slots,
## world geometry parity, perception gating, sound events, and match metrics.
class_name TestSelfPlay
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SelfPlayEnvironmentCore = preload("res://scripts/self_play/self_play_environment.gd")


func test_self_play_reset_has_two_observations() -> SandboxTest:
	var t := SandboxTest.new("self_play_reset_has_two_observations")
	var match_env := SelfPlayEnvironmentCore.new()
	var observations: Array = match_env.reset(10, 20)
	t.assert_eq(observations.size(), 2)
	t.assert_eq(observations[0].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(observations[1].to_array().size(), Observation.FIELD_COUNT)
	return t


func test_self_play_step_returns_per_agent_rewards_and_infos() -> SandboxTest:
	var t := SandboxTest.new("self_play_step_returns_per_agent_metrics")
	var match_env := SelfPlayEnvironmentCore.new()
	match_env.reset(1, 2)
	var result: Dictionary = match_env.step([Action.idle(), Action.idle()])
	t.assert_eq(result.rewards.size(), 2)
	t.assert_eq(result.infos.size(), 2)
	t.assert_true(result.infos[0].metrics.has("kills"))
	t.assert_true(result.infos[1].metrics.has("kills"))
	return t


func test_self_play_with_world_blocks_weapon_ray_occlusion() -> SandboxTest:
	var t := SandboxTest.new("self_play_with_world_blocks_weapon_ray_occlusion")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("corner")
	env.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	env.reset(42, 42)

	# Position agent A on one side of corner and agent B on the other side
	env.agent_a.reset(Vector3(-4.0, 0.0, -4.0), 0.0)
	env.agent_b.reset(Vector3(4.0, 0.0, 4.0), 180.0)
	env._sync_proxies()

	# Agent A aims towards B and shoots
	var aim_dir: Vector3 = (env.agent_b.position - env.agent_a.position).normalized()
	env.agent_a.forward = aim_dir

	var shoot_action := Action.new(0, 0, 0, 0, true, Vector2.ZERO, false)
	var res: Dictionary = env.step([shoot_action, Action.idle()])

	# Ray is occluded by corner wall, so agent B should take 0 damage
	t.assert_false(res.infos[0].events.hit)
	t.assert_eq(res.infos[0].events.damage_dealt, 0.0)
	t.assert_eq(env.agent_b.health, env.agent_b.max_health)
	return t


func test_self_play_perception_gating_detects_visible_and_hidden() -> SandboxTest:
	var t := SandboxTest.new("self_play_perception_gating_detects_visible_and_hidden")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("corner")
	env.set_curriculum_level(CurriculumConfig.Level.FOV_LOS)
	env.reset(100, 100)

	# Hidden around corner
	env.agent_a.reset(Vector3(-4.0, 0.0, -4.0), 0.0)
	env.agent_b.reset(Vector3(4.0, 0.0, 4.0), 180.0)
	env._sync_proxies()

	var obs: Array = env.get_observations()
	# Agent A cannot see Agent B
	t.assert_false(obs[0].primary_enemy_visible)
	t.assert_false(obs[0].primary_enemy_los_clear)

	# Move into direct line of sight
	env.agent_b.reset(Vector3(-4.0, 0.0, 4.0), 180.0)
	env.agent_a.forward = Vector3(0.0, 0.0, 1.0)
	env._sync_proxies()
	env.step([Action.idle(), Action.idle()])

	var obs_los: Array = env.get_observations()
	t.assert_true(obs_los[0].primary_enemy_los_clear)
	return t


func test_self_play_sound_emission_and_hearing() -> SandboxTest:
	var t := SandboxTest.new("self_play_sound_emission_and_hearing")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("open_arena")
	env.set_curriculum_level(CurriculumConfig.Level.SOUND)
	env.reset(1, 2)

	# Place agents near each other
	env.agent_a.reset(Vector3(0.0, 0.0, -2.0), 0.0)
	env.agent_b.reset(Vector3(0.0, 0.0, 2.0), 180.0)
	env._sync_proxies()

	# Agent A fires a shot
	var shoot_action := Action.new(0, 0, 0, 0, true, Vector2.ZERO, false)
	env.step([shoot_action, Action.idle()])

	# Agent B's perception should hear Agent A's shot
	var heard_b: Array = env.perception_b.heard
	t.assert_false(heard_b.is_empty())
	if not heard_b.is_empty():
		t.assert_eq(heard_b[0].category, 3)  # SHOT category
	return t


func test_self_play_map_and_lighting_configuration() -> SandboxTest:
	var t := SandboxTest.new("self_play_map_and_lighting_configuration")
	var env := SelfPlayEnvironmentCore.new()
	t.assert_true(env.set_map("two_rooms"))
	t.assert_true(env.set_lighting_mode("low_light"))
	t.assert_false(env.set_map("non_existent_map_id_12345"))
	t.assert_false(env.set_lighting_mode("invalid_lighting_mode_xyz"))

	var cond: Dictionary = env.get_episode_condition()
	t.assert_eq(cond.map_id, "two_rooms")
	t.assert_eq(cond.lighting, "low_light")
	return t


func test_self_play_independent_rng_and_observations() -> SandboxTest:
	var t := SandboxTest.new("self_play_independent_rng_and_observations")
	var env := SelfPlayEnvironmentCore.new()
	env.reset(12345, 67890)
	t.assert_ne(env.rng_a.seed, env.rng_b.seed)
	var obs: Array = env.get_observations()
	t.assert_eq(obs.size(), 2)
	t.assert_ne(obs[0].to_array(), obs[1].to_array())
	return t


func test_self_play_zero_information_leakage_when_occluded() -> SandboxTest:
	var t := SandboxTest.new("self_play_zero_information_leakage_when_occluded")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("corner")
	env.set_curriculum_level(CurriculumConfig.Level.FOV_LOS)
	env.reset(555, 777)

	# Place A and B on opposite sides of the corner
	env.agent_a.reset(Vector3(-5.0, 0.0, -5.0), 0.0)
	env.agent_b.reset(Vector3(5.0, 0.0, 5.0), 180.0)
	env._sync_proxies()

	# Perception update with no sounds
	env.perception_a.reset()
	env.perception_b.reset()
	var beliefs_a: Array = env.perception_a.update(
		env.agent_a, [env.proxy_b], env.world, env.sound_bus, 0.0, 0
	)
	var beliefs_b: Array = env.perception_b.update(
		env.agent_b, [env.proxy_a], env.world, env.sound_bus, 0.0, 1
	)
	env._beliefs_a = beliefs_a
	env._beliefs_b = beliefs_b

	var obs: Array = env.get_observations()
	var obs_a: Observation = obs[0]
	var obs_b: Observation = obs[1]

	# Both agents must have 0 visibility / 0 clear line of sight
	t.assert_false(obs_a.primary_enemy_visible)
	t.assert_false(obs_a.primary_enemy_los_clear)
	t.assert_false(obs_b.primary_enemy_visible)
	t.assert_false(obs_b.primary_enemy_los_clear)

	# Enemy health / speed / distance must not leak ground truth
	t.assert_eq(obs_a.primary_enemy_health_norm, 0.0)
	t.assert_eq(obs_b.primary_enemy_health_norm, 0.0)
	return t


func test_self_play_deterministic_replay_trajectory() -> SandboxTest:
	var t := SandboxTest.new("self_play_deterministic_replay_trajectory")
	var env1 := SelfPlayEnvironmentCore.new()
	env1.set_layout("open_arena")
	env1.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	var obs1: Array = env1.reset(42, 42 + 1000003)

	var env2 := SelfPlayEnvironmentCore.new()
	env2.set_layout("open_arena")
	env2.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	var obs2: Array = env2.reset(42, 42 + 1000003)

	t.assert_eq(obs1[0].to_array(), obs2[0].to_array())
	t.assert_eq(obs1[1].to_array(), obs2[1].to_array())

	var step_actions := [
		[Action.new(1, 0, 0, 0, false), Action.new(-1, 0, 0, 0, false)],
		[Action.new(1, 0, 0, 0, true), Action.new(0, 1, 0, 0, false)],
		[Action.new(0, 0, 1, 0, false), Action.new(0, -1, 0, 0, true)],
	]

	for act_pair in step_actions:
		var r1: Dictionary = env1.step(act_pair)
		var r2: Dictionary = env2.step(act_pair)
		t.assert_eq(r1.rewards, r2.rewards)
		t.assert_eq(r1.done, r2.done)
		t.assert_eq(r1.observations[0].to_array(), r2.observations[0].to_array())
		t.assert_eq(r1.observations[1].to_array(), r2.observations[1].to_array())

	return t

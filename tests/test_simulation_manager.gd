## Tests for SimulationManager: multiple independent environments, batch
## RL-style API, and the fast/headless stepping path.
class_name TestSimulationManager
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const EnvironmentView = preload("res://scripts/env/environment_view.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_build_creates_requested_environment_count() -> SandboxTest:
	var t := SandboxTest.new("sim_manager_builds_requested_environment_count")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(5, 1)
	t.assert_eq(sim.environments.size(), 5)
	t.assert_eq(sim.controllers.size(), 5)
	sim.free()
	return t


func test_headless_mode_creates_no_views() -> SandboxTest:
	var t := SandboxTest.new("sim_manager_headless_creates_no_views")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(3, 1)
	for v in sim.views:
		t.assert_null(v, "no EnvironmentView should be created in headless mode")
	sim.free()
	return t


func test_environments_are_independent_objects_with_no_shared_state() -> SandboxTest:
	var t := SandboxTest.new("environments_are_independent")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(3, 1)

	var env0: EnvironmentCore = sim.environments[0]
	var env1: EnvironmentCore = sim.environments[1]
	t.assert_true(env0 != env1, "environments must be distinct object instances")
	t.assert_true(env0.agent != env1.agent, "agents must be distinct object instances")
	t.assert_true(env0.episode != env1.episode, "episode state must be distinct object instances")

	# Drive env0 hard and confirm env1/env2 are completely unaffected.
	var env1_pos_before: Vector3 = env1.agent.position
	var env1_health_before: float = env1.agent.health
	for _i in range(30):
		sim.step_all(
			[Action.from_discrete(Action.Discrete.MOVE_FORWARD), Action.idle(), Action.idle()]
		)

	t.assert_true(
		env0.agent.position.distance_to(SandboxConfig.AGENT_SPAWN_POSITION) > 0.01,
		"env0 agent should have moved"
	)
	t.assert_vec_almost_eq(
		env1.agent.position, env1_pos_before, 0.0001, "env1 should be unaffected by env0's actions"
	)
	t.assert_almost_eq(
		env1.agent.health,
		env1_health_before,
		0.0001,
		"env1 health should be unaffected by env0's combat"
	)
	sim.free()
	return t


func test_resetting_one_environment_does_not_affect_others() -> SandboxTest:
	var t := SandboxTest.new("resetting_one_env_does_not_affect_others")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(2, 1)

	for _i in range(10):
		sim.step_all(
			[
				Action.from_discrete(Action.Discrete.MOVE_FORWARD),
				Action.from_discrete(Action.Discrete.MOVE_FORWARD)
			]
		)

	var env1_pos_before_reset: Vector3 = sim.environments[1].agent.position
	sim.environments[0].reset(42)

	t.assert_vec_almost_eq(
		sim.environments[0].agent.position,
		SandboxConfig.AGENT_SPAWN_POSITION,
		0.001,
		"env0 should be back at spawn"
	)
	t.assert_vec_almost_eq(
		sim.environments[1].agent.position,
		env1_pos_before_reset,
		0.0001,
		"env1 must be untouched by env0's reset"
	)
	sim.free()
	return t


func test_batch_get_observations_rewards_done_match_environment_count() -> SandboxTest:
	var t := SandboxTest.new("batch_accessors_match_environment_count")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(4, 1)
	sim.step_all([Action.idle(), Action.idle(), Action.idle(), Action.idle()])
	t.assert_eq(sim.get_observations().size(), 4)
	t.assert_eq(sim.get_rewards().size(), 4)
	t.assert_eq(sim.is_done_all().size(), 4)
	sim.free()
	return t


func test_run_headless_steps_advances_without_a_provided_action() -> SandboxTest:
	var t := SandboxTest.new("run_headless_steps_advances_simulation")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(2, 1)
	var provider := func(_i: int) -> Action:
		return Action.from_discrete(Action.Discrete.MOVE_FORWARD)
	sim.run_headless_steps(15, provider)
	for env in sim.environments:
		t.assert_true(
			(
				(env as EnvironmentCore).agent.position.distance_to(
					SandboxConfig.AGENT_SPAWN_POSITION
				)
				> 0.01
			),
			"headless stepping should move every environment's agent"
		)
	sim.free()
	return t


func test_deterministic_base_seed_gives_reproducible_environment_set() -> SandboxTest:
	var t := SandboxTest.new("deterministic_base_seed_reproducible")
	var sim_a := SimulationManager.new()
	sim_a.create_visuals = false
	sim_a.base_seed = 500
	sim_a.build(3, 1)

	var sim_b := SimulationManager.new()
	sim_b.create_visuals = false
	sim_b.base_seed = 500
	sim_b.build(3, 1)

	for i in range(3):
		var a: EnvironmentCore = sim_a.environments[i]
		var b: EnvironmentCore = sim_b.environments[i]
		t.assert_vec_almost_eq(
			a.enemies[0].position,
			b.enemies[0].position,
			0.0001,
			"same base_seed should reproduce the same environment set"
		)

	sim_a.free()
	sim_b.free()
	return t

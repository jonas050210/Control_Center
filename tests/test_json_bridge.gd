## Tests for RLAdapter and request handling abstractions.
class_name TestJsonBridge
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_adapter_spaces_and_reset() -> SandboxTest:
	var t := SandboxTest.new("adapter_spaces_and_reset")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(2, 1)
	var adapter := RLAdapter.new(sim)

	var action_space: Dictionary = RLAdapter.action_space_info()
	var obs_space: Dictionary = RLAdapter.observation_space_info()
	t.assert_eq(action_space.dimension, Action.MULTI_DISCRETE_SIZE)
	t.assert_eq(obs_space.size, Observation.FIELD_COUNT)

	var obs: Array = adapter.reset(123)
	t.assert_eq(obs.size(), 2)
	t.assert_eq(obs[0].size(), Observation.FIELD_COUNT)

	var step_res: Dictionary = adapter.step([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
	t.assert_eq(step_res.observations.size(), 2)
	t.assert_eq(step_res.rewards.size(), 2)
	t.assert_eq(step_res.dones.size(), 2)
	t.assert_eq(step_res.infos.size(), 2)

	var health_reports: Array = adapter.health_check()
	t.assert_eq(health_reports.size(), 2)
	t.assert_true(health_reports[0].healthy)

	sim.free()
	return t


## Regression: the exact MultiDiscrete format SB3 emits ([0..2, 0..2, 0..2,
## 0..2, 0..1, 0..1], shoot at index 4) must reach the weapon intact. If the shoot
## bit were dropped or misread anywhere in the bridge/adapters, this test
## reports zero shots fired even though index 4 is set.
func test_adapter_multidiscrete_shoot_reaches_simulation() -> SandboxTest:
	var t := SandboxTest.new("adapter_multidiscrete_shoot_reaches_simulation")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(1, 1)
	var adapter := RLAdapter.new(sim)
	adapter.reset(7)

	# Deterministic alignment: agent faces the enemy straight down -Z.
	var env: EnvironmentCore = sim.environments[0]
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	env.agent.pitch_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)
	env.agent.weapon.cooldown_remaining = 0.0

	var result: Dictionary = adapter.step([[1, 1, 1, 1, 1, 0]])
	t.assert_eq(result.infos[0].metrics.shots_fired, 1, "shoot bit at index 4 must fire the weapon")
	t.assert_eq(result.infos[0].metrics.shots_hit, 1, "an aligned shot must hit")
	t.assert_almost_eq(env.enemies[0].health, env.enemies[0].max_health - 25.0, 0.001)
	t.assert_gt(float(result.rewards[0]), 0.9, "a hit must be net positive for the step")

	var result_idle: Dictionary = adapter.step([[1, 1, 1, 1, 0, 0]])
	t.assert_eq(
		result_idle.infos[0].metrics.shots_fired, 1,
		"shoot=0 must not fire; the episode total must stay at one shot"
	)

	sim.free()
	return t


func test_adapter_reset_indices() -> SandboxTest:
	var t := SandboxTest.new("adapter_reset_indices")
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(3, 1)
	var adapter := RLAdapter.new(sim)

	adapter.step([Action.from_discrete(Action.Discrete.MOVE_FORWARD), Action.idle(), Action.idle()])
	var res: Array = adapter.reset_indices([0], 999)
	t.assert_eq(res.size(), 1)
	t.assert_eq(res[0].index, 0)

	sim.free()
	return t

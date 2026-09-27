## Tests for RLAdapter and request handling abstractions.
class_name TestJsonBridge
extends RefCounted

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

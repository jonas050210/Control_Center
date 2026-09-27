## Tests for RLAdapter: the framework-agnostic reset/step/observations/
## rewards/done façade over SimulationManager.
class_name TestRLAdapter
extends RefCounted


func _make_adapter(env_count: int = 2) -> RLAdapter:
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(env_count, 1)
	return RLAdapter.new(sim)


func test_reset_returns_one_observation_per_environment() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_reset_returns_one_observation_per_environment")
	var adapter := _make_adapter(3)
	var observations: Array = adapter.reset(1)
	t.assert_eq(observations.size(), 3)
	adapter.simulation_manager.free()
	return t


func test_step_accepts_discrete_int_actions() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_step_accepts_discrete_int_actions")
	var adapter := _make_adapter(2)
	var result: Dictionary = adapter.step([Action.Discrete.MOVE_FORWARD, Action.Discrete.IDLE])
	t.assert_true(result.has("observations"))
	t.assert_true(result.has("rewards"))
	t.assert_true(result.has("dones"))
	t.assert_true(result.has("infos"))
	t.assert_eq(result.observations.size(), 2)
	t.assert_eq(result.rewards.size(), 2)
	adapter.simulation_manager.free()
	return t


func test_batch_accessors_delegate_to_simulation_manager() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_batch_accessors_delegate")
	var adapter := _make_adapter(2)
	adapter.step([Action.Discrete.IDLE, Action.Discrete.IDLE])
	t.assert_eq(adapter.get_observations().size(), 2)
	t.assert_eq(adapter.get_rewards().size(), 2)
	t.assert_eq(adapter.is_done().size(), 2)
	adapter.simulation_manager.free()
	return t


func test_action_and_observation_space_info_are_well_formed() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_space_info_well_formed")
	var action_info: Dictionary = RLAdapter.action_space_info()
	var obs_info: Dictionary = RLAdapter.observation_space_info()
	t.assert_eq(action_info.discrete_choices, Action.DISCRETE_COUNT)
	t.assert_eq(obs_info.get("size", -1), Observation.FIELD_COUNT)
	return t

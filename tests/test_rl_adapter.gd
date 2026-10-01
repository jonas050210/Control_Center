## Tests for RLAdapter: the framework-agnostic reset/step/observations/
## rewards/done façade over SimulationManager.
class_name TestRLAdapter
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


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


func test_reset_returns_flat_observation_vectors() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_reset_returns_flat_observation_vectors")
	var adapter := _make_adapter(2)
	var observations: Array = adapter.reset(7)
	t.assert_true(observations[0] is Array)
	t.assert_eq(observations[0].size(), Observation.FIELD_COUNT)
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


func test_compact_info_omits_only_redundant_nonterminal_diagnostics() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_compact_info_preserves_semantics")
	var adapter := _make_adapter(1)
	var full_adapter := _make_adapter(1)
	var env: EnvironmentCore = adapter.simulation_manager.environments[0]
	var full_env: EnvironmentCore = full_adapter.simulation_manager.environments[0]
	env.max_steps = 2
	full_env.max_steps = 2
	var action: Array = [[1, 1, 1, 1, 0, 0]]
	var first: Dictionary = adapter.step(action, true)
	var full_first: Dictionary = full_adapter.step(action)
	t.assert_eq(
		first.observations, full_first.observations, "compact mode must not change observations"
	)
	t.assert_eq(first.rewards, full_first.rewards, "compact mode must not change rewards")
	t.assert_eq(first.dones, full_first.dones, "compact mode must not change done flags")
	t.assert_false(first.dones[0])
	t.assert_true(first.infos[0].has("events"))
	t.assert_false(first.infos[0].has("metrics"))
	t.assert_false(first.infos[0].has("reward_components"))
	t.assert_true(full_first.infos[0].has("metrics"))
	t.assert_true(full_first.infos[0].has("reward_components"))
	var terminal: Dictionary = adapter.step([[1, 1, 1, 1, 0, 0]], true)
	t.assert_true(terminal.dones[0])
	t.assert_true(terminal.infos[0].has("metrics"))
	t.assert_true(terminal.infos[0].has("terminal_observation"))
	t.assert_false(terminal.infos[0].has("reward_components"))
	adapter.simulation_manager.free()
	full_adapter.simulation_manager.free()
	return t


func test_done_flag_is_preserved_after_vector_auto_reset() -> SandboxTest:
	var t := SandboxTest.new("rl_adapter_done_flag_preserved_after_auto_reset")
	var adapter := _make_adapter(1)
	(adapter.simulation_manager.environments[0] as EnvironmentCore).max_steps = 1
	var result: Dictionary = adapter.step([Action.Discrete.IDLE])
	t.assert_true(result.dones[0])
	t.assert_true(adapter.is_done()[0])
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
	t.assert_eq(action_info.nvec, Action.MULTI_DISCRETE_NVECS)
	t.assert_eq(action_info.dimension, Action.MULTI_DISCRETE_SIZE)
	t.assert_eq(obs_info.get("size", -1), Observation.FIELD_COUNT)
	return t

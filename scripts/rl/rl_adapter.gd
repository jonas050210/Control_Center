## RLAdapter
##
## Framework-agnostic batch façade. It is intentionally usable from the
## in-engine recorder and from the JSON-lines process bridge; neither side
## needs to know about rendering or Godot Nodes beyond SimulationManager.
class_name RLAdapter
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


var simulation_manager: SimulationManager


func _init(p_simulation_manager: SimulationManager) -> void:
	simulation_manager = p_simulation_manager


static func action_space_info() -> Dictionary:
	return {
		"type": "multi_discrete",
		"nvec": Action.MULTI_DISCRETE_NVECS.duplicate(),
		"dimension": Action.MULTI_DISCRETE_SIZE,
		# Kept for backwards compatibility with single-discrete API.
		"discrete_choices": Action.DISCRETE_COUNT,
		"fields": [
			"move_axis", "strafe_axis", "look_yaw_axis", "look_pitch_axis", "shoot", "jump"
		],
		"continuous_reserved": ["look_delta.x", "look_delta.y"],
	}


static func observation_space_info() -> Dictionary:
	return {
		"type": "structured_float_vector",
		"size": Observation.FIELD_COUNT,
		"shape": [Observation.FIELD_COUNT],
		"low": -1.0,
		"high": 1.0,
		"mode": SandboxConfig.ACTIVE_OBSERVATION_MODE,
		"modalities": SandboxConfig.OBSERVATION_MODALITIES.duplicate(),
		"rgb_enabled": SandboxConfig.RGB_OBSERVATION_ENABLED,
		"frame_stack": SandboxConfig.RGB_FRAME_STACK,
	}


func reset(seed_base: int = -1) -> Array:
	var observations: Array = []
	for observation in simulation_manager.reset_all(seed_base):
		observations.append(_observation_to_array(observation))
	return observations


func reset_indices(indices: Array, seed_base: int = -1) -> Array:
	var observations: Array = []
	for item in simulation_manager.reset_indices(indices, seed_base):
		observations.append(
			{"index": item.index, "observation": _observation_to_array(item.observation)}
		)
	return observations


func step(actions: Array) -> Dictionary:
	var resolved: Array = []
	for action_value in actions:
		resolved.append(_resolve_action(action_value))

	var results: Array = simulation_manager.step_all(resolved)
	var observations: Array = []
	var rewards: Array = []
	var dones: Array = []
	var infos: Array = []
	for result in results:
		observations.append(_observation_to_array(result.observation))
		rewards.append(float(result.reward))
		dones.append(bool(result.done))
		var info: Dictionary = result.info.duplicate(true)
		if result.has("terminal_observation"):
			info["terminal_observation"] = _observation_to_array(result.terminal_observation)
		var done_reason: String = str(info.get("done_reason", ""))
		info["TimeLimit.truncated"] = done_reason == "timeout"
		infos.append(info)

	return {
		"observations": observations,
		"rewards": rewards,
		"dones": dones,
		"infos": infos,
	}


func get_observations() -> Array:
	var observations: Array = []
	for observation in simulation_manager.get_observations():
		observations.append(_observation_to_array(observation))
	return observations


func get_rewards() -> Array:
	return simulation_manager.get_rewards()


func is_done() -> Array:
	return simulation_manager.is_done_all()


func get_metrics() -> Array:
	return simulation_manager.get_metrics()


func get_reward_breakdowns() -> Array:
	return simulation_manager.get_reward_breakdowns()


func health_check() -> Array:
	return simulation_manager.health_check_all()


func _resolve_action(action_value) -> Action:
	if action_value is Action:
		return action_value
	if typeof(action_value) == TYPE_INT or typeof(action_value) == TYPE_FLOAT:
		return Action.from_discrete(int(action_value))
	if action_value is Array:
		return _resolve_array_action(action_value)
	if action_value is Dictionary:
		return Action.new(
			int(action_value.get("move_axis", 0)),
			int(action_value.get("strafe_axis", 0)),
			int(action_value.get("look_yaw_axis", 0)),
			int(action_value.get("look_pitch_axis", 0)),
			bool(action_value.get("shoot", false)),
			Vector2(
				float(action_value.get("look_delta_x", 0.0)),
				float(action_value.get("look_delta_y", 0.0))
			),
			bool(action_value.get("jump", false))
		)
	return Action.idle()


## Decodes the array forms of an action.
func _resolve_array_action(action_value: Array) -> Action:
	# An 8-value array is the contract-v2 canonical log
	# [move, strafe, yaw, pitch, shoot, jump, look_dx, look_dy]; a
	# 7-value array is the v1 log without `jump`. Anything shorter is a
	# MultiDiscrete action (5 = v1, 6 = v2).
	if action_value.size() >= 8:
		return Action.new(
			int(action_value[0]),
			int(action_value[1]),
			int(action_value[2]),
			int(action_value[3]),
			bool(action_value[4]),
			Vector2(float(action_value[6]), float(action_value[7])),
			bool(action_value[5])
		)
	if action_value.size() == 7:
		return Action.new(
			int(action_value[0]),
			int(action_value[1]),
			int(action_value[2]),
			int(action_value[3]),
			bool(action_value[4]),
			Vector2(float(action_value[5]), float(action_value[6]))
		)
	return Action.from_multidiscrete(action_value)


static func _observation_to_array(observation) -> Array:
	if observation is Observation:
		return Array(observation.to_array())
	if observation is PackedFloat32Array:
		return Array(observation)
	if observation is Array:
		return observation
	return []

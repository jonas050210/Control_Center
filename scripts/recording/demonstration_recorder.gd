## DemonstrationRecorder
##
## Records the same Action objects emitted by HumanController around the
## existing SimulationManager.step_all call. JSONL is intentionally used as
## the interchange format: each transition is one compact JSON object and
## Python can stream it without a Godot-specific decoder.
class_name DemonstrationRecorder
extends RefCounted

const SCHEMA_VERSION: int = 1

var recording: bool = false
var dataset_metadata: Dictionary = {}
var transitions: Array = []
var _episode_ids: Dictionary = {}


func start_recording(metadata: Dictionary = {}) -> void:
	recording = true
	transitions.clear()
	_episode_ids.clear()
	dataset_metadata = {
		"schema": "sandboxai.demonstrations",
		"schema_version": SCHEMA_VERSION,
		"observation_dim": Observation.FIELD_COUNT,
		"action_encoding": "[move, strafe, yaw, pitch, shoot, look_delta_x, look_delta_y]",
		"created_ticks_usec": Time.get_ticks_usec(),
	}
	for key in metadata:
		dataset_metadata[key] = metadata[key]


func stop_recording() -> void:
	recording = false


func new_episode(env_index: int) -> void:
	_episode_ids[env_index] = int(_episode_ids.get(env_index, 0)) + 1


func record_transition(
	env_index: int, observation_before: PackedFloat32Array, action: Action, result: Dictionary
) -> void:
	if not recording or action == null:
		return
	var episode_id: int = int(_episode_ids.get(env_index, 0))
	var info: Dictionary = result.get("info", {})
	var metrics: Dictionary = info.get("metrics", {})
	var next_observation = result.get("terminal_observation", result.get("observation", null))
	(
		transitions
		. append(
			{
				"observation": Array(observation_before),
				"action": action.to_array(),
				"next_observation": _observation_to_array(next_observation),
				"reward": float(result.get("reward", 0.0)),
				"done": bool(result.get("done", false)),
				"timestamp": float(Time.get_ticks_usec()) / 1000000.0,
				"step": int(metrics.get("episode_length", transitions.size())),
				"episode_id": episode_id,
				"environment_id": env_index,
				"info": info,
			}
		)
	)
	if bool(result.get("done", false)):
		new_episode(env_index)


func save_dataset(path: String) -> bool:
	var file_path: String = (
		ProjectSettings.globalize_path(path)
		if path.begins_with("user://") or path.begins_with("res://")
		else path
	)
	var parent: String = file_path.get_base_dir()
	if not parent.is_empty():
		DirAccess.make_dir_recursive_absolute(parent)
	var file := FileAccess.open(file_path, FileAccess.WRITE)
	if file == null:
		push_error("Could not open demonstration dataset for writing: %s" % file_path)
		return false
	file.store_line(JSON.stringify(dataset_metadata))
	for transition in transitions:
		file.store_line(JSON.stringify(transition))
	file.flush()
	file.close()
	return true


func load_dataset(path: String) -> Array:
	var file_path: String = (
		ProjectSettings.globalize_path(path)
		if path.begins_with("user://") or path.begins_with("res://")
		else path
	)
	var file := FileAccess.open(file_path, FileAccess.READ)
	if file == null:
		push_error("Could not open demonstration dataset: %s" % path)
		return []
	var loaded: Array = []
	var first: bool = true
	while not file.eof_reached():
		var line: String = file.get_line().strip_edges()
		if line.is_empty():
			continue
		var value = JSON.parse_string(line)
		if first and value is Dictionary and value.get("schema", "") == "sandboxai.demonstrations":
			dataset_metadata = value
			first = false
			continue
		first = false
		if value is Dictionary:
			loaded.append(value)
	file.close()
	transitions = loaded
	return transitions


func get_transition_count() -> int:
	return transitions.size()


func _observation_to_array(value) -> Array:
	if value is Observation:
		return Array(value.to_array())
	if value is PackedFloat32Array:
		return Array(value)
	if value is Array:
		return value
	return []

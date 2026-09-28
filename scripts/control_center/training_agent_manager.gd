## TrainingAgentManager
##
## Registry for the managed training agents shown on the dashboard. Each
## agent wraps ONE existing TrainingRunController — the process boundary
## that launches the real Python backend and mirrors its published status
## and JSONL events. The manager adds only identity (stable id + label),
## wall-clock bookkeeping and multi-agent fan-out of the controller
## signals; it never computes training metrics of its own.
##
## Agent 1 is the session's original primary controller so the existing
## single-run workflow (Training tab -> Start) and every historical test
## keep their exact semantics. Additional agents get their own controller
## and their own run directory, so several backends can run side by side.
class_name TrainingAgentManager
extends Node

signal agents_changed
signal agent_state_changed(agent_id: int, state: int)
signal agent_event(agent_id: int, entry: Dictionary)

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var _entries: Array = []  # Array[Dictionary]; insertion order == display order
var _next_agent_id: int = 1


## Adopts an existing controller (the session's primary training_run) as
## the first agent without changing its ownership or lifecycle.
func register_primary(controller: TrainingRunController) -> int:
	if controller == null:
		return -1
	for entry_value in _entries:
		if (entry_value as Dictionary).get("controller") == controller:
			return int((entry_value as Dictionary)["id"])
	return _register(controller, true)


## Launches a new managed agent from a snapshot of the given configuration.
## The primary controller is reused only while it is completely unused
## (Idle); a finished/error agent keeps its card and a fresh controller is
## created instead. Returns the agent id, or -1 when the launch was
## rejected before a process could start.
func launch(config: ControlCenterConfig) -> int:
	if config == null:
		return -1
	if not launch_validation_error(config).is_empty():
		return -1
	var target_id: int = -1
	var primary: Dictionary = _primary_entry()
	var primary_idle: bool = false
	if not primary.is_empty():
		var primary_controller: TrainingRunController = primary["controller"]
		primary_idle = primary_controller.state == TrainingRunController.State.IDLE
	if primary_idle:
		target_id = int(primary["id"])
	else:
		var controller := TrainingRunController.new()
		controller.name = "TrainingRunController%d" % _next_agent_id
		add_child(controller)
		target_id = _register(controller, false)
	var entry: Dictionary = _entry(target_id)
	var controller_node: TrainingRunController = entry["controller"]
	var started: bool = controller_node.start(config)
	entry["config"] = config.duplicate_config()
	entry["launched_unix"] = Time.get_unix_time_from_system()
	entry["terminal_unix"] = 0.0
	agents_changed.emit()
	if not started and not bool(entry["primary"]):
		# Validation failures never reached a process; the card would only
		# repeat the error the caller already displays.
		remove(target_id)
		return -1
	return target_id


## Uses the controller's own validation so this registry never invents a
## second set of launch rules.
func launch_validation_error(config: ControlCenterConfig) -> String:
	var primary: Dictionary = _primary_entry()
	if not primary.is_empty():
		return (primary["controller"] as TrainingRunController).validation_error(config)
	var validator := TrainingRunController.new()
	var problem: String = validator.validation_error(config)
	validator.free()
	return problem


## Registers an externally created controller as an additional agent
## WITHOUT starting it. Used by focused tests and by alternate transports
## that already own a controller; the normal UI path is `launch()`.
func adopt_agent(controller: TrainingRunController) -> int:
	if controller == null:
		return -1
	for entry_value in _entries:
		if (entry_value as Dictionary).get("controller") == controller:
			return int((entry_value as Dictionary)["id"])
	if controller.get_parent() == null:
		add_child(controller)
	return _register(controller, false)


## Called by the session when the primary controller is (re)started through
## the original single-run path, so this launch is bookkept identically.
func note_primary_started(config: ControlCenterConfig) -> void:
	var entry: Dictionary = _primary_entry()
	if entry.is_empty():
		return
	if config != null:
		entry["config"] = config.duplicate_config()
	entry["launched_unix"] = Time.get_unix_time_from_system()
	entry["terminal_unix"] = 0.0
	agents_changed.emit()


func agent_ids() -> Array:
	var ids: Array = []
	for entry_value in _entries:
		ids.append(int((entry_value as Dictionary)["id"]))
	return ids


func agent_count() -> int:
	return _entries.size()


func active_count() -> int:
	var count: int = 0
	for entry_value in _entries:
		var controller: TrainingRunController = (entry_value as Dictionary)["controller"]
		if TrainingRunController.is_active_state(controller.state):
			count += 1
	return count


func get_controller(agent_id: int) -> TrainingRunController:
	var entry: Dictionary = _entry(agent_id)
	if entry.is_empty():
		return null
	return entry["controller"]


func has_agent(agent_id: int) -> bool:
	return not _entry(agent_id).is_empty()


## One display snapshot per agent: the controller's authoritative backend
## status plus identity and wall-clock runtime. No metric is synthesized —
## keys the backend never published stay absent.
func agent_snapshot(agent_id: int) -> Dictionary:
	var entry: Dictionary = _entry(agent_id)
	if entry.is_empty():
		return {}
	var controller: TrainingRunController = entry["controller"]
	var snapshot: Dictionary = controller.snapshot()
	snapshot["agent_id"] = int(entry["id"])
	snapshot["label"] = str(entry["label"])
	snapshot["is_primary"] = bool(entry["primary"])
	snapshot["algorithm"] = _algorithm_label(entry, snapshot)
	snapshot["runtime_seconds"] = runtime_seconds(agent_id)
	return snapshot


func snapshots() -> Array:
	var out: Array = []
	for entry_value in _entries:
		out.append(agent_snapshot(int((entry_value as Dictionary)["id"])))
	return out


## Measured wall-clock lifetime of the current run of this agent, frozen at
## the moment a terminal state was observed. null before the first launch.
func runtime_seconds(agent_id: int):
	var entry: Dictionary = _entry(agent_id)
	if entry.is_empty():
		return null
	var launched: float = float(entry.get("launched_unix", 0.0))
	if launched <= 0.0:
		return null
	var terminal: float = float(entry.get("terminal_unix", 0.0))
	if terminal > 0.0:
		return maxf(0.0, terminal - launched)
	return maxf(0.0, Time.get_unix_time_from_system() - launched)


func pause(agent_id: int) -> bool:
	var controller: TrainingRunController = get_controller(agent_id)
	return controller != null and controller.pause()


func resume(agent_id: int) -> bool:
	var controller: TrainingRunController = get_controller(agent_id)
	return controller != null and controller.resume()


func stop(agent_id: int) -> bool:
	var controller: TrainingRunController = get_controller(agent_id)
	return controller != null and controller.stop()


## Removes a finished/error agent card. Active agents cannot be removed —
## stop is the only way out, exactly like the underlying controller.
## The primary controller is reset (its card stays) rather than freed
## because the session owns it.
func remove(agent_id: int) -> bool:
	var entry: Dictionary = _entry(agent_id)
	if entry.is_empty():
		return false
	var controller: TrainingRunController = entry["controller"]
	if TrainingRunController.is_active_state(controller.state):
		return false
	if bool(entry["primary"]):
		controller.reset()
		entry["launched_unix"] = 0.0
		entry["terminal_unix"] = 0.0
		agents_changed.emit()
		return true
	_entries.erase(entry)
	if controller.get_parent() == self:
		remove_child(controller)
		controller.queue_free()
	agents_changed.emit()
	return true


func _register(controller: TrainingRunController, primary: bool) -> int:
	var agent_id: int = _next_agent_id
	_next_agent_id += 1
	var entry: Dictionary = {
		"id": agent_id,
		"label": "Agent %d" % agent_id,
		"controller": controller,
		"primary": primary,
		"config": null,
		"launched_unix": 0.0,
		"terminal_unix": 0.0,
	}
	_entries.append(entry)
	controller.state_changed.connect(_on_controller_state_changed.bind(agent_id))
	controller.training_event.connect(_on_controller_event.bind(agent_id))
	agents_changed.emit()
	return agent_id


func _on_controller_state_changed(state: int, agent_id: int) -> void:
	var entry: Dictionary = _entry(agent_id)
	if not entry.is_empty():
		if state in [TrainingRunController.State.FINISHED, TrainingRunController.State.ERROR]:
			if float(entry.get("terminal_unix", 0.0)) <= 0.0:
				entry["terminal_unix"] = Time.get_unix_time_from_system()
		elif TrainingRunController.is_active_state(state):
			entry["terminal_unix"] = 0.0
	agent_state_changed.emit(agent_id, state)


func _on_controller_event(event_entry: Dictionary, agent_id: int) -> void:
	agent_event.emit(agent_id, event_entry)


func _entry(agent_id: int) -> Dictionary:
	for entry_value in _entries:
		if int((entry_value as Dictionary)["id"]) == agent_id:
			return entry_value
	return {}


func _primary_entry() -> Dictionary:
	for entry_value in _entries:
		if bool((entry_value as Dictionary)["primary"]):
			return entry_value
	return {}


static func _algorithm_label(entry: Dictionary, snapshot: Dictionary) -> String:
	var from_status: String = str(snapshot.get("training_type", ""))
	if not from_status.is_empty():
		return from_status
	var config = entry.get("config")
	if config != null:
		return ControlCenterConfig.training_type_name(config.training_type)
	return ""

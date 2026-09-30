## TrainingRunController
##
## Process boundary between the Godot Control Center and the existing Python
## training backends. It does not optimize, calculate rewards or derive
## metrics. It launches the real CLI, sends cooperative commands through a
## JSON file and renders only status published by python/sandboxai/run_control.py.
class_name TrainingRunController
extends Node

signal state_changed(state: int)
signal status_changed(status: Dictionary)
signal training_event(entry: Dictionary)

enum State {
	IDLE = 0,
	STARTING = 1,
	RUNNING = 2,
	PAUSED = 3,
	STOPPING = 4,
	FINISHED = 5,
	ERROR = 6,
}

const POLL_INTERVAL_SECONDS: float = 0.2
## The event ring buffer keeps exactly the dashboard's live-log capacity so
## the headless monitor can always fill its log to the bound (see
## ControlCenterConfig.LIVE_LOG_LINES).
const MAX_RECENT_EVENTS: int = ControlCenterConfig.LIVE_LOG_LINES

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")

var state: int = State.IDLE
var process_id: int = -1
var status: Dictionary = {}
var recent_events: Array = []
## Lifetime count of ingested events. `recent_events` is a bounded ring, so
## consumers that render incrementally (live log panels) diff against this
## counter instead of re-reading the whole ring every frame.
var total_events_ingested: int = 0
var last_error: String = ""
var run_directory: String = ""
var command_file: String = ""
var status_file: String = ""
var event_file: String = ""
var launched_command: PackedStringArray = PackedStringArray()

var _poll_accumulator: float = 0.0
var _event_file_position: int = 0
var _command_sequence: int = 0
var _started_msec: int = 0


static func state_name(value: int) -> String:
	match value:
		State.IDLE:
			return "Idle"
		State.STARTING:
			return "Starting"
		State.RUNNING:
			return "Running"
		State.PAUSED:
			return "Paused"
		State.STOPPING:
			return "Stopping"
		State.FINISHED:
			return "Finished"
		State.ERROR:
			return "Error"
		_:
			return "Unknown"


static func is_active_state(value: int) -> bool:
	return value in [State.STARTING, State.RUNNING, State.PAUSED, State.STOPPING]


func can_start(config: ControlCenterConfig) -> bool:
	return validation_error(config).is_empty() and not is_active_state(state)


func validation_error(config: ControlCenterConfig) -> String:
	var problem: String = ""
	if config == null:
		problem = "training configuration is missing"
	elif not ControlCenterConfig.training_type_available(config.training_type):
		problem = ControlCenterConfig.training_type_unavailable_reason(config.training_type)
	elif config.resume_from_checkpoint and config.checkpoint_path.strip_edges().is_empty():
		problem = "Resume is enabled, but no checkpoint is selected."
	elif (
		config.resume_from_checkpoint
		and not FileAccess.file_exists(_filesystem_path(config.checkpoint_path))
	):
		problem = "Checkpoint does not exist: %s" % config.checkpoint_path
	elif (
		config.resume_from_checkpoint
		and config.training_type == ControlCenterConfig.TrainingType.PPO
		and config.checkpoint_path.get_extension().to_lower() != "zip"
	):
		problem = "PPO resume requires a Stable-Baselines3 .zip checkpoint."
	elif (
		config.resume_from_checkpoint
		and config.training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING
		and config.checkpoint_path.get_extension().to_lower() != "pt"
	):
		problem = "Behavior Cloning resume requires a .pt checkpoint."
	elif (
		config.training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING
		and config.bc_dataset_path.strip_edges().is_empty()
	):
		problem = "Behavior Cloning requires a demonstration dataset."
	elif (
		config.training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING
		and not FileAccess.file_exists(_filesystem_path(config.bc_dataset_path))
	):
		problem = "Dataset does not exist: %s" % config.bc_dataset_path
	return problem


## Full executable + argument vector. Kept public so the UI can display the
## exact command and tests can verify configuration propagation without
## launching a subprocess.
func build_command(config: ControlCenterConfig, paths: Dictionary = {}) -> PackedStringArray:
	var resolved_paths: Dictionary = paths if not paths.is_empty() else _preview_paths()
	var command := PackedStringArray([config.python_executable, "-m", "sandboxai"])
	if config.training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING:
		command.append("bc-train")
		_append_option(command, "--dataset", _filesystem_path(config.bc_dataset_path))
		_append_option(command, "--epochs", str(config.bc_epochs))
		_append_option(command, "--batch-size", str(config.batch_size))
		_append_option(command, "--learning-rate", str(config.learning_rate))
		_append_option(command, "--seed", str(config.seed))
		_append_option(
			command,
			"--device",
			ControlCenterConfig.training_device_argument(config.training_device)
		)
		_append_option(command, "--output-dir", str(resolved_paths["output_dir"]))
		if config.resume_from_checkpoint:
			_append_option(command, "--resume-checkpoint", _filesystem_path(config.checkpoint_path))
	else:
		command.append("resume" if config.resume_from_checkpoint else "train")
		if config.resume_from_checkpoint:
			_append_option(command, "--checkpoint", _filesystem_path(config.checkpoint_path))
		_append_option(command, "--steps", str(config.total_training_steps))
		_append_option(command, "--env-count", str(config.environment_count))
		_append_option(command, "--enemy-count", str(config.enemy_count))
		_append_option(command, "--curriculum-level", str(config.curriculum_level))
		_append_option(command, "--seed", str(config.seed))
		_append_option(
			command,
			"--device",
			ControlCenterConfig.training_device_argument(config.training_device)
		)
		_append_option(command, "--learning-rate", str(config.learning_rate))
		_append_option(command, "--rollout-length", str(config.rollout_length))
		_append_option(command, "--batch-size", str(config.batch_size))
		_append_option(command, "--gamma", str(config.gamma))
		_append_option(command, "--gae-lambda", str(config.gae_lambda))
		_append_option(command, "--entropy-coefficient", str(config.entropy_coefficient))
		_append_option(command, "--clip-range", str(config.clip_range))
		_append_option(command, "--checkpoint-frequency", str(config.checkpoint_frequency))
		_append_option(command, "--evaluation-frequency", str(config.evaluation_frequency))
		_append_option(command, "--godot-executable", config.godot_executable)
		_append_option(command, "--project-path", ProjectSettings.globalize_path("res://"))
		_append_option(command, "--output-root", _project_path("training"))
		_append_option(command, "--run-id", str(resolved_paths["run_id"]))

	_append_option(command, "--control-file", str(resolved_paths["command_file"]))
	_append_option(command, "--status-file", str(resolved_paths["status_file"]))
	_append_option(command, "--event-log-file", str(resolved_paths["event_file"]))
	return command


## Human-readable form of `build_command` for display/copy in the UI.
func command_line(config: ControlCenterConfig) -> String:
	var built: PackedStringArray = build_command(config)
	if built.size() < 4:
		return ""
	var parts := PackedStringArray()
	for value in built:
		parts.append('"%s"' % value if value.contains(" ") else value)
	return " ".join(parts)


func start(config: ControlCenterConfig) -> bool:
	var problem: String = validation_error(config)
	if not problem.is_empty():
		_fail(problem)
		return false
	if is_active_state(state):
		return false

	# Starting after a finished/error run is a fresh lifecycle. No previous
	# metric, event or error may leak into the new dashboard.
	status = {}
	recent_events.clear()
	total_events_ingested = 0
	last_error = ""
	process_id = -1
	_prepare_run_paths(config.training_type)
	var paths: Dictionary = {
		"run_id": run_directory.get_file(),
		"output_dir": _project_path("training/bc_runs").path_join(run_directory.get_file()),
		"command_file": ProjectSettings.globalize_path(command_file),
		"status_file": ProjectSettings.globalize_path(status_file),
		"event_file": ProjectSettings.globalize_path(event_file),
	}
	launched_command = build_command(config, paths)
	if launched_command.size() < 2:
		_fail("could not build the training command")
		return false

	var executable: String = launched_command[0]
	var arguments := PackedStringArray()
	for index in range(1, launched_command.size()):
		arguments.append(launched_command[index])
	_set_state(State.STARTING)
	status = {
		"state": "Starting",
		"training_type": ControlCenterConfig.training_type_name(config.training_type),
		"training_mode": ControlCenterConfig.training_mode_name(config.training_mode),
		"total_training_steps": config.total_training_steps,
		"environment_count": config.environment_count,
		"device_requested": ControlCenterConfig.training_device_argument(config.training_device),
		"command": Array(launched_command),
	}
	_write_command("run")
	process_id = OS.create_process(executable, arguments, false)
	if process_id <= 0:
		_fail(
			(
				"Could not start '%s'. Install the Python package or set the Python executable."
				% executable
			)
		)
		return false
	_started_msec = Time.get_ticks_msec()
	status["process_id"] = process_id
	status_changed.emit(status.duplicate(true))
	return true


func pause() -> bool:
	if state != State.RUNNING:
		return false
	_write_command("pause")
	# The backend remains authoritative; state changes when its status file
	# confirms that it reached a safe pause boundary.
	return true


func resume() -> bool:
	if state != State.PAUSED:
		return false
	_write_command("resume")
	return true


func stop() -> bool:
	if state not in [State.STARTING, State.RUNNING, State.PAUSED]:
		return false
	_write_command("stop")
	_set_state(State.STOPPING)
	return true


func force_stop() -> bool:
	if process_id <= 0 or not OS.is_process_running(process_id):
		return false
	var result: Error = OS.kill(process_id)
	if result != OK:
		return false
	_fail("Training process was force-terminated before graceful checkpoint cleanup.")
	return true


func reset() -> bool:
	if is_active_state(state):
		return false
	state = State.IDLE
	process_id = -1
	status = {}
	recent_events.clear()
	total_events_ingested = 0
	last_error = ""
	run_directory = ""
	command_file = ""
	status_file = ""
	event_file = ""
	launched_command = PackedStringArray()
	_event_file_position = 0
	state_changed.emit(state)
	status_changed.emit({})
	return true


func apply_status(values: Dictionary) -> void:
	## Public for focused tests and for alternate backend transports.
	status = values.duplicate(true)
	var incoming: String = str(status.get("state", ""))
	var mapped: int = _state_from_name(incoming)
	# A locally issued stop remains pending until the backend reaches its safe
	# boundary; a slightly older Starting/Running sample must not undo it.
	if state == State.STOPPING and mapped in [State.STARTING, State.RUNNING, State.PAUSED]:
		mapped = State.STOPPING
	if mapped >= 0:
		_set_state(mapped)
	if state == State.ERROR:
		last_error = str(status.get("error", last_error))
	status_changed.emit(status.duplicate(true))


func snapshot() -> Dictionary:
	var copy: Dictionary = status.duplicate(true)
	copy["state_id"] = state
	copy["state"] = state_name(state)
	copy["active"] = is_active_state(state)
	copy["process_id"] = process_id
	copy["last_error"] = last_error
	copy["run_directory"] = run_directory
	copy["recent_events"] = recent_events.duplicate(true)
	copy["total_events"] = total_events_ingested
	return copy


func _exit_tree() -> void:
	if is_active_state(state) and not command_file.is_empty():
		# Do not orphan a trainer owned by a closing Control Center. The Python
		# side performs its normal graceful checkpoint cleanup independently.
		_write_command("stop")


func _process(delta: float) -> void:
	if not is_active_state(state):
		return
	_poll_accumulator += delta
	if _poll_accumulator < POLL_INTERVAL_SECONDS:
		return
	_poll_accumulator = 0.0
	_poll_status()
	_poll_events()
	if process_id > 0 and not OS.is_process_running(process_id):
		# Give the atomic final status write one more polling interval before
		# declaring an abnormal process exit.
		_poll_status()
		if is_active_state(state) and Time.get_ticks_msec() - _started_msec > 500:
			_fail("Training process exited without publishing a terminal status.")


func _poll_status() -> void:
	if status_file.is_empty() or not FileAccess.file_exists(status_file):
		return
	var file := FileAccess.open(status_file, FileAccess.READ)
	if file == null:
		return
	var parsed = JSON.parse_string(file.get_as_text())
	if parsed is Dictionary:
		apply_status(parsed)


func _poll_events() -> void:
	if event_file.is_empty() or not FileAccess.file_exists(event_file):
		return
	var file := FileAccess.open(event_file, FileAccess.READ)
	if file == null:
		return
	if _event_file_position > file.get_length():
		_event_file_position = 0
	file.seek(_event_file_position)
	while file.get_position() < file.get_length():
		var line: String = file.get_line()
		if line.strip_edges().is_empty():
			continue
		var parsed = JSON.parse_string(line)
		if parsed is Dictionary:
			recent_events.append(parsed)
			total_events_ingested += 1
			while recent_events.size() > MAX_RECENT_EVENTS:
				recent_events.pop_front()
			training_event.emit((parsed as Dictionary).duplicate(true))
	_event_file_position = file.get_position()


func _prepare_run_paths(training_type: int) -> void:
	var prefix: String = (
		"bc" if training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING else "ppo"
	)
	var run_id: String = (
		"control_center_%s_%d_%d"
		% [prefix, int(Time.get_unix_time_from_system()), Time.get_ticks_msec()]
	)
	run_directory = "user://control_center_runs/%s" % run_id
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(run_directory))
	command_file = run_directory.path_join("command.json")
	status_file = run_directory.path_join("status.json")
	event_file = run_directory.path_join("events.jsonl")
	_event_file_position = 0
	for path in [command_file, status_file, event_file]:
		if FileAccess.file_exists(path):
			DirAccess.remove_absolute(ProjectSettings.globalize_path(path))


func _write_command(command: String) -> void:
	if command_file.is_empty():
		return
	_command_sequence += 1
	var file := FileAccess.open(command_file, FileAccess.WRITE)
	if file != null:
		(
			file
			. store_string(
				(
					JSON
					. stringify(
						{
							"command": command,
							"sequence": _command_sequence,
							"issued_at": Time.get_unix_time_from_system(),
						}
					)
				)
			)
		)


func _set_state(value: int) -> void:
	if state == value:
		return
	state = value
	state_changed.emit(state)


func _fail(message: String) -> void:
	last_error = message
	status["state"] = "Error"
	status["error"] = message
	_set_state(State.ERROR)
	status_changed.emit(status.duplicate(true))


static func _state_from_name(value: String) -> int:
	match value.strip_edges().to_lower():
		"idle":
			return State.IDLE
		"starting":
			return State.STARTING
		"running":
			return State.RUNNING
		"paused":
			return State.PAUSED
		"stopping":
			return State.STOPPING
		"finished":
			return State.FINISHED
		"error":
			return State.ERROR
		_:
			return -1


static func _append_option(command: PackedStringArray, key: String, value: String) -> void:
	command.append(key)
	command.append(value)


static func _filesystem_path(path: String) -> String:
	var value: String = path.strip_edges()
	if value.begins_with("res://") or value.begins_with("user://"):
		return ProjectSettings.globalize_path(value)
	if value.is_absolute_path():
		return value
	return ProjectSettings.globalize_path("res://%s" % value.trim_prefix("./"))


static func _project_path(relative: String) -> String:
	return ProjectSettings.globalize_path("res://" + relative)


func _preview_paths() -> Dictionary:
	var base: String = ProjectSettings.globalize_path("user://control_center_runs/preview")
	return {
		"run_id": "control_center_preview",
		"output_dir": _project_path("training/bc_runs/control_center_preview"),
		"command_file": base.path_join("command.json"),
		"status_file": base.path_join("status.json"),
		"event_file": base.path_join("events.jsonl"),
	}

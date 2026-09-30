## Training configuration editor backed directly by ControlCenterConfig.
## Field defaults and command-line meanings mirror the existing Python PPO
## and behavior-cloning backends; unsupported self-play optimization is
## identified explicitly instead of simulated.
class_name ControlCenterTrainingConfigPanel
extends VBoxContainer

signal configuration_changed

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var session
var _type: OptionButton
var _mode: OptionButton
var _device: OptionButton
var _steps: SpinBox
var _epochs: SpinBox
var _environment_count: SpinBox
var _curriculum: OptionButton
var _seed: SpinBox
var _dataset: LineEdit
var _checkpoint: LineEdit
var _resume: CheckButton
var _learning_rate: SpinBox
var _rollout: SpinBox
var _batch: SpinBox
var _gamma: SpinBox
var _gae_lambda: SpinBox
var _entropy: SpinBox
var _clip: SpinBox
var _checkpoint_frequency: SpinBox
var _evaluation_frequency: SpinBox
var _python_executable: LineEdit
var _godot_executable: LineEdit
var _note: Label
var _command: Label
var _ppo_controls: Array = []
var _bc_controls: Array = []
var _all_editors: Array = []
var _updating: bool = false


func setup(p_session) -> void:
	session = p_session
	add_theme_constant_override("separation", 6)
	add_child(
		ControlCenterTheme.make_label(
			"Training configuration",
			ControlCenterTheme.FONT_SIZE_TITLE,
			ControlCenterTheme.COLOR_TITLE
		)
	)
	var description := ControlCenterTheme.make_label(
		"Starts the repository's real Python backend; values below are passed to its existing CLI.",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	description.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(description)

	var grid := ControlCenterTheme.make_grid(2)
	add_child(grid)
	_type = _add_option(
		grid,
		"training type",
		[
			["PPO", ControlCenterConfig.TrainingType.PPO],
			["Behavior Cloning", ControlCenterConfig.TrainingType.BEHAVIOR_CLONING],
			["Self-Play", ControlCenterConfig.TrainingType.SELF_PLAY],
		]
	)
	_type.item_selected.connect(_on_type_changed)
	_mode = _add_option(
		grid,
		"dashboard mode",
		[
			["Visual Mode", ControlCenterConfig.TrainingMode.VISUAL],
			["Headless Mode", ControlCenterConfig.TrainingMode.HEADLESS],
		]
	)
	_mode.item_selected.connect(_on_mode_changed)
	_device = _add_option(
		grid,
		"device",
		[
			["Auto", ControlCenterConfig.TrainingDevice.AUTO],
			["CPU", ControlCenterConfig.TrainingDevice.CPU],
			["GPU (CUDA)", ControlCenterConfig.TrainingDevice.GPU],
		]
	)
	_device.item_selected.connect(_on_device_changed)

	_steps = _add_spin(
		grid, "training steps", 1, ControlCenterConfig.MAX_TRAINING_STEPS, 1000, 1_000_000
	)
	_steps.value_changed.connect(_on_steps_changed)
	_ppo_controls.append(_steps)
	_epochs = _add_spin(grid, "BC epochs", 1, ControlCenterConfig.MAX_BC_EPOCHS, 1, 25)
	_epochs.value_changed.connect(_on_epochs_changed)
	_bc_controls.append(_epochs)
	_environment_count = _add_spin(
		grid, "environments", 1, ControlCenterConfig.MAX_ENVIRONMENT_COUNT, 1, 8
	)
	_environment_count.value_changed.connect(_on_environment_changed)
	_ppo_controls.append(_environment_count)

	_add_key(grid, "curriculum")
	_curriculum = ControlCenterTheme.make_option_button()
	for level in range(1, CurriculumConfig.Level.AGENT_VS_AGENT + 1):
		_curriculum.add_item("%d  %s" % [level, CurriculumConfig.level_name(level)], level)
	_curriculum.item_selected.connect(_on_curriculum_changed)
	grid.add_child(_curriculum)
	_ppo_controls.append(_curriculum)

	_seed = _add_spin(grid, "seed", 0, 2_147_483_647, 1, 1234)
	_seed.value_changed.connect(_on_seed_changed)

	_add_key(grid, "dataset")
	_dataset = _path_row(grid, "Select the demonstration JSONL dataset", _browse_dataset)
	_dataset.text_changed.connect(_on_dataset_changed)
	_bc_controls.append(_dataset.get_parent())

	_add_key(grid, "checkpoint")
	_checkpoint = _path_row(grid, "Select a PPO .zip or BC .pt checkpoint", _browse_checkpoint)
	_checkpoint.text_changed.connect(_on_checkpoint_changed)

	_add_key(grid, "resume")
	_resume = CheckButton.new()
	_resume.text = "Resume from selected checkpoint"
	_resume.toggled.connect(_on_resume_changed)
	grid.add_child(_resume)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"Backend parameters",
			ControlCenterTheme.FONT_SIZE_TITLE,
			ControlCenterTheme.COLOR_TITLE
		)
	)
	var advanced := ControlCenterTheme.make_grid(2)
	add_child(advanced)
	_learning_rate = _add_spin(advanced, "learning rate", 0.000001, 1.0, 0.0001, 0.0003)
	_learning_rate.value_changed.connect(_on_learning_rate_changed)
	_rollout = _add_spin(advanced, "rollout length (0 = auto)", 0, 1_000_000, 1, 0)
	_rollout.value_changed.connect(_on_rollout_changed)
	_ppo_controls.append(_rollout)
	_batch = _add_spin(advanced, "batch size", 1, 1_000_000, 1, 256)
	_batch.value_changed.connect(_on_batch_changed)
	_gamma = _add_spin(advanced, "gamma", 0.000001, 1.0, 0.001, 0.99)
	_gamma.value_changed.connect(_on_gamma_changed)
	_ppo_controls.append(_gamma)
	_gae_lambda = _add_spin(advanced, "GAE lambda", 0.0, 1.0, 0.001, 0.95)
	_gae_lambda.value_changed.connect(_on_gae_changed)
	_ppo_controls.append(_gae_lambda)
	_entropy = _add_spin(advanced, "entropy coefficient", 0.0, 10.0, 0.001, 0.01)
	_entropy.value_changed.connect(_on_entropy_changed)
	_ppo_controls.append(_entropy)
	_clip = _add_spin(advanced, "clip range", 0.000001, 10.0, 0.01, 0.2)
	_clip.value_changed.connect(_on_clip_changed)
	_ppo_controls.append(_clip)
	_checkpoint_frequency = _add_spin(
		advanced, "checkpoint every", 1, ControlCenterConfig.MAX_TRAINING_STEPS, 1000, 100_000
	)
	_checkpoint_frequency.value_changed.connect(_on_checkpoint_frequency_changed)
	_ppo_controls.append(_checkpoint_frequency)
	_evaluation_frequency = _add_spin(
		advanced, "evaluate every", 1, ControlCenterConfig.MAX_TRAINING_STEPS, 1000, 50_000
	)
	_evaluation_frequency.value_changed.connect(_on_evaluation_frequency_changed)
	_ppo_controls.append(_evaluation_frequency)
	_add_key(advanced, "Python executable")
	_python_executable = LineEdit.new()
	_python_executable.tooltip_text = "Python 3.11 executable used for the managed trainer process."
	_python_executable.text_changed.connect(_on_python_executable_changed)
	advanced.add_child(_python_executable)
	_add_key(advanced, "Godot executable")
	_godot_executable = LineEdit.new()
	_godot_executable.tooltip_text = "Godot 4.7.2 executable passed to the Python trainer."
	_godot_executable.text_changed.connect(_on_godot_executable_changed)
	advanced.add_child(_godot_executable)
	_ppo_controls.append(_godot_executable)

	_note = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_WARN
	)
	_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_note)
	_command = ControlCenterTheme.make_value_label("")
	_command.clip_text = false
	_command.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_command)
	var copy := ControlCenterTheme.make_button("Copy command")
	copy.pressed.connect(_copy_command)
	add_child(copy)

	_all_editors = [
		_type, _mode, _device, _steps, _epochs, _environment_count, _curriculum, _seed,
		_dataset, _checkpoint, _resume, _learning_rate, _rollout, _batch, _gamma,
		_gae_lambda, _entropy, _clip, _checkpoint_frequency, _evaluation_frequency,
		_python_executable, _godot_executable,
	]
	_sync()


func refresh(_snapshot: Dictionary = {}) -> void:
	var active: bool = TrainingRunController.is_active_state(session.training_run.state)
	for editor_value in _all_editors:
		var editor: Control = editor_value
		if editor is LineEdit:
			(editor as LineEdit).editable = not active
		else:
			editor.set("disabled", active)
	for path_edit in [_dataset, _checkpoint]:
		for child in path_edit.get_parent().get_children():
			if child is Button:
				(child as Button).disabled = active
	var problem: String = session.training_run.validation_error(session.config)
	_note.text = problem
	_note.visible = not problem.is_empty()
	_command.text = session.training_command_line()
	var focus: Control = get_viewport().gui_get_focus_owner() if is_inside_tree() else null
	if not _updating and (focus == null or not is_ancestor_of(focus)):
		_sync()


func _sync() -> void:
	_updating = true
	var config: ControlCenterConfig = session.config
	_type.selected = config.training_type
	_mode.selected = config.training_mode
	_device.selected = config.training_device
	_steps.value = config.total_training_steps
	_epochs.value = config.bc_epochs
	_environment_count.value = config.environment_count
	_curriculum.selected = config.curriculum_level - 1
	_seed.value = config.seed
	_dataset.text = config.bc_dataset_path
	_checkpoint.text = config.checkpoint_path
	_resume.button_pressed = config.resume_from_checkpoint
	_learning_rate.value = config.learning_rate
	_rollout.value = config.rollout_length
	_batch.value = config.batch_size
	_gamma.value = config.gamma
	_gae_lambda.value = config.gae_lambda
	_entropy.value = config.entropy_coefficient
	_clip.value = config.clip_range
	_checkpoint_frequency.value = config.checkpoint_frequency
	_evaluation_frequency.value = config.evaluation_frequency
	_python_executable.text = config.python_executable
	_godot_executable.text = config.godot_executable
	var ppo: bool = config.training_type == ControlCenterConfig.TrainingType.PPO
	var bc: bool = (
		config.training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING
	)
	for control_value in _ppo_controls:
		(control_value as Control).visible = ppo
	for control_value in _bc_controls:
		(control_value as Control).visible = bc
	_updating = false


func _add_option(grid: GridContainer, key: String, values: Array) -> OptionButton:
	_add_key(grid, key)
	var option := ControlCenterTheme.make_option_button()
	for value in values:
		option.add_item(str(value[0]), int(value[1]))
	grid.add_child(option)
	return option


func _add_key(grid: GridContainer, key: String) -> void:
	grid.add_child(
		ControlCenterTheme.make_label(
			key, ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)


func _add_spin(
	grid: GridContainer, key: String, minimum: float, maximum: float, step: float, value: float
) -> SpinBox:
	_add_key(grid, key)
	var spin := ControlCenterTheme.make_spin_box(minimum, maximum, step, value)
	spin.allow_greater = false
	grid.add_child(spin)
	return spin


func _path_row(grid: GridContainer, tooltip: String, callback: Callable) -> LineEdit:
	var row := ControlCenterTheme.make_row()
	var edit := LineEdit.new()
	edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	edit.tooltip_text = tooltip
	row.add_child(edit)
	var browse := ControlCenterTheme.make_button("Browse…", tooltip)
	browse.pressed.connect(callback)
	row.add_child(browse)
	grid.add_child(row)
	return edit


func _choose_file(title: String, filters: PackedStringArray, callback: Callable) -> void:
	var dialog := FileDialog.new()
	dialog.title = title
	dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	dialog.access = FileDialog.ACCESS_FILESYSTEM
	dialog.filters = filters
	dialog.file_selected.connect(callback)
	dialog.canceled.connect(dialog.queue_free)
	dialog.file_selected.connect(func(_path: String): dialog.queue_free())
	add_child(dialog)
	dialog.popup_centered_ratio(0.7)


func _browse_dataset() -> void:
	_choose_file(
		"Demonstration dataset",
		PackedStringArray(["*.jsonl ; JSONL datasets"]),
		_set_dataset
	)


func _browse_checkpoint() -> void:
	_choose_file(
		"Training checkpoint",
		PackedStringArray(["*.zip ; PPO checkpoints", "*.pt ; Behavior Cloning checkpoints"]),
		_set_checkpoint
	)


func _set_dataset(path: String) -> void:
	_dataset.text = path
	_on_dataset_changed(path)


func _set_checkpoint(path: String) -> void:
	_checkpoint.text = path
	_on_checkpoint_changed(path)


func _changed() -> void:
	if _updating:
		return
	session.config.sanitize()
	session.config.save_preferences()
	configuration_changed.emit()


func _on_type_changed(index: int) -> void:
	if _updating:
		return
	session.config.training_type = _type.get_item_id(index)
	_changed()
	_sync()


func _on_mode_changed(index: int) -> void:
	if _updating:
		return
	session.config.training_mode = _mode.get_item_id(index)
	_changed()


func _on_device_changed(index: int) -> void:
	if _updating:
		return
	session.config.training_device = _device.get_item_id(index)
	_changed()


func _on_steps_changed(value: float) -> void:
	if _updating:
		return
	session.config.total_training_steps = int(value)
	_changed()


func _on_epochs_changed(value: float) -> void:
	if _updating:
		return
	session.config.bc_epochs = int(value)
	_changed()


func _on_environment_changed(value: float) -> void:
	if _updating:
		return
	session.request_setting("environment_count", int(value))
	_changed()


func _on_curriculum_changed(index: int) -> void:
	if _updating:
		return
	session.set_curriculum_level(_curriculum.get_item_id(index))
	_changed()


func _on_seed_changed(value: float) -> void:
	if _updating:
		return
	session.request_setting("seed", int(value))
	_changed()


func _on_dataset_changed(value: String) -> void:
	if _updating:
		return
	session.config.bc_dataset_path = value
	_changed()


func _on_checkpoint_changed(value: String) -> void:
	if _updating:
		return
	session.config.checkpoint_path = value
	_changed()


func _on_resume_changed(value: bool) -> void:
	if _updating:
		return
	session.config.resume_from_checkpoint = value
	_changed()


func _on_learning_rate_changed(value: float) -> void:
	if _updating:
		return
	session.config.learning_rate = value
	_changed()


func _on_rollout_changed(value: float) -> void:
	if _updating:
		return
	session.config.rollout_length = int(value)
	_changed()


func _on_batch_changed(value: float) -> void:
	if _updating:
		return
	session.config.batch_size = int(value)
	_changed()


func _on_gamma_changed(value: float) -> void:
	if _updating:
		return
	session.config.gamma = value
	_changed()


func _on_gae_changed(value: float) -> void:
	if _updating:
		return
	session.config.gae_lambda = value
	_changed()


func _on_entropy_changed(value: float) -> void:
	if _updating:
		return
	session.config.entropy_coefficient = value
	_changed()


func _on_clip_changed(value: float) -> void:
	if _updating:
		return
	session.config.clip_range = value
	_changed()


func _on_checkpoint_frequency_changed(value: float) -> void:
	if _updating:
		return
	session.config.checkpoint_frequency = int(value)
	_changed()


func _on_evaluation_frequency_changed(value: float) -> void:
	if _updating:
		return
	session.config.evaluation_frequency = int(value)
	_changed()


func _on_python_executable_changed(value: String) -> void:
	if _updating:
		return
	session.config.python_executable = value
	_changed()


func _on_godot_executable_changed(value: String) -> void:
	if _updating:
		return
	session.config.godot_executable = value
	_changed()


func _copy_command() -> void:
	DisplayServer.clipboard_set(session.training_command_line())

## Lifecycle controls for the external Python training backend. Button state
## is derived from TrainingRunController, never from optimistic UI toggles.
class_name ControlCenterTrainingControlsPanel
extends PanelContainer

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var session
var _start: Button
var _stop: Button
var _pause: Button
var _resume: Button
var _reset: Button
var _state: Label
var _detail: Label


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var row := ControlCenterTheme.make_row()
	add_child(row)
	row.add_child(
		ControlCenterTheme.make_label(
			"TRAINING", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_start = _button(row, "Start", _on_start, "Launch the configured real Python backend")
	_stop = _button(row, "Stop", _on_stop, "Request graceful stop and final checkpoint")
	_pause = _button(row, "Pause", _on_pause, "Pause at the next safe backend boundary")
	_resume = _button(row, "Resume", _on_resume, "Resume a cooperatively paused backend")
	_reset = _button(row, "Reset", _on_reset, "Clear a finished/error run from the dashboard")
	row.add_child(VSeparator.new())
	_state = ControlCenterTheme.make_label("Idle", ControlCenterTheme.FONT_SIZE_NORMAL)
	_state.custom_minimum_size = Vector2(90.0, 0.0)
	row.add_child(_state)
	_detail = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_detail.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_detail.clip_text = true
	row.add_child(_detail)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	var controller: TrainingRunController = session.training_run
	var state: int = controller.state
	var validation: String = controller.validation_error(session.config)
	_start.disabled = (
		state
		not in [
			TrainingRunController.State.IDLE,
			TrainingRunController.State.FINISHED,
			TrainingRunController.State.ERROR,
		]
		or not validation.is_empty()
	)
	_stop.disabled = state not in [TrainingRunController.State.STARTING,
		TrainingRunController.State.RUNNING, TrainingRunController.State.PAUSED]
	_pause.disabled = state != TrainingRunController.State.RUNNING
	_resume.disabled = state != TrainingRunController.State.PAUSED
	_reset.disabled = state not in [TrainingRunController.State.FINISHED,
		TrainingRunController.State.ERROR]
	_state.text = TrainingRunController.state_name(state)
	var color: Color = ControlCenterTheme.COLOR_MUTED
	if state == TrainingRunController.State.RUNNING:
		color = ControlCenterTheme.COLOR_OK
	elif state in [TrainingRunController.State.STARTING, TrainingRunController.State.PAUSED,
		TrainingRunController.State.STOPPING]:
		color = ControlCenterTheme.COLOR_WARN
	elif state == TrainingRunController.State.ERROR:
		color = ControlCenterTheme.COLOR_BAD
	_state.add_theme_color_override("font_color", color)
	if not validation.is_empty() and state in [TrainingRunController.State.IDLE,
		TrainingRunController.State.FINISHED, TrainingRunController.State.ERROR]:
		_detail.text = validation
	elif state == TrainingRunController.State.ERROR:
		_detail.text = controller.last_error
	else:
		_detail.text = "%s · %s · %s" % [
			ControlCenterConfig.training_type_name(session.config.training_type),
			ControlCenterConfig.training_mode_name(session.config.training_mode),
			ControlCenterConfig.training_device_argument(session.config.training_device),
		]


func _button(parent: Control, text: String, callback: Callable, tooltip: String) -> Button:
	var button := ControlCenterTheme.make_button(text, tooltip)
	button.pressed.connect(callback)
	parent.add_child(button)
	return button


func _on_start() -> void:
	session.start_training()
	refresh()


func _on_stop() -> void:
	session.training_run.stop()
	refresh()


func _on_pause() -> void:
	session.training_run.pause()


func _on_resume() -> void:
	session.training_run.resume()


func _on_reset() -> void:
	session.training_run.reset()
	refresh()

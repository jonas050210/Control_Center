## TRAINING page: hosts the existing full training configuration editor
## and turns it into a launch screen with one primary action. No
## configuration capability is duplicated or removed — the editor child IS
## the original ControlCenterTrainingConfigPanel.
class_name ControlCenterTrainingLaunchPanel
extends PanelContainer

signal training_started(agent_id: int)

const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const ControlCenterTrainingConfigPanel = preload(
	"res://scripts/control_center/ui/training_config_panel.gd"
)
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var session
var config_panel: ControlCenterTrainingConfigPanel

var _start: Button
var _start_additional: Button
var _hint: Label


func setup(p_session, p_config_panel: ControlCenterTrainingConfigPanel) -> void:
	session = p_session
	config_panel = p_config_panel
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	var action_row := ControlCenterTheme.make_row()
	root.add_child(action_row)
	_start = ControlCenterTheme.make_button(
		"START TRAINING", "Launch the configured Python backend as the primary agent"
	)
	_start.custom_minimum_size = Vector2(180.0, 34.0)
	_start.pressed.connect(_on_start)
	action_row.add_child(_start)
	_start_additional = ControlCenterTheme.make_button(
		"Start as additional agent",
		"Launch another managed agent without touching agents that are already running"
	)
	_start_additional.pressed.connect(_on_start_additional)
	action_row.add_child(_start_additional)
	_hint = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_hint.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_hint.clip_text = true
	action_row.add_child(_hint)

	var scroll := ControlCenterTheme.make_scroll()
	root.add_child(scroll)
	config_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(config_panel)
	refresh()


func refresh(snapshot: Dictionary = {}) -> void:
	config_panel.refresh(snapshot)
	var state: int = session.training_run.state
	var problem: String = session.training_run.validation_error(session.config)
	_start.disabled = (
		state
		not in [
			TrainingRunController.State.IDLE,
			TrainingRunController.State.FINISHED,
			TrainingRunController.State.ERROR,
		]
		or not problem.is_empty()
	)
	_start_additional.disabled = not problem.is_empty()
	if not problem.is_empty():
		_hint.text = problem
	elif TrainingRunController.is_active_state(state):
		_hint.text = (
			"Primary agent is %s — use 'additional agent' to launch another run."
			% TrainingRunController.state_name(state)
		)
	else:
		_hint.text = "Launches the real Python backend with the configuration below."


func _on_start() -> void:
	if session.start_training():
		training_started.emit(1)
	refresh()


func _on_start_additional() -> void:
	var agent_id: int = session.launch_agent()
	if agent_id > 0:
		training_started.emit(agent_id)
	refresh()

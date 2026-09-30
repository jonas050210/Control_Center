## Compact dashboard tile for one managed training agent. Everything shown
## comes from the agent snapshot (backend status + registry identity);
## metrics the backend has not published render as "n/a".
class_name ControlCenterAgentCard
extends PanelContainer

signal pause_requested(agent_id: int)
signal resume_requested(agent_id: int)
signal stop_requested(agent_id: int)
signal details_requested(agent_id: int)
signal remove_requested(agent_id: int)

const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var agent_id: int = -1

var _title: Label
var _algorithm: Label
var _state: Label
var _progress: ProgressBar
var _metrics: Label
var _footer: Label
var _pause: Button
var _resume: Button
var _stop: Button
var _details: Button
var _remove: Button


func setup(show_details_button: bool = true) -> void:
	custom_minimum_size = Vector2(280.0, 0.0)
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 4)
	add_child(root)

	var header := ControlCenterTheme.make_row()
	root.add_child(header)
	_title = ControlCenterTheme.make_label(
		"Agent", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TEXT
	)
	header.add_child(_title)
	_algorithm = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_AI
	)
	header.add_child(_algorithm)
	_state = ControlCenterTheme.make_label(
		"Idle", ControlCenterTheme.FONT_SIZE_NORMAL, ControlCenterTheme.COLOR_MUTED
	)
	_state.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_state.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	header.add_child(_state)

	_progress = ProgressBar.new()
	_progress.min_value = 0.0
	_progress.max_value = 100.0
	_progress.show_percentage = false
	_progress.custom_minimum_size = Vector2(0.0, 6.0)
	root.add_child(_progress)

	_metrics = ControlCenterTheme.make_value_label("")
	_metrics.clip_text = false
	root.add_child(_metrics)

	_footer = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_footer.clip_text = true
	root.add_child(_footer)

	var buttons := ControlCenterTheme.make_row()
	root.add_child(buttons)
	_pause = _make_button(buttons, "Pause", _on_pause)
	_resume = _make_button(buttons, "Resume", _on_resume)
	_stop = _make_button(buttons, "Stop", _on_stop)
	_details = _make_button(buttons, "Details", _on_details)
	_details.visible = show_details_button
	_remove = _make_button(buttons, "Clear", _on_remove)


## Renders one agent snapshot (see TrainingAgentManager.agent_snapshot).
func update_from(snapshot: Dictionary) -> void:
	agent_id = int(snapshot.get("agent_id", -1))
	var state_name: String = str(snapshot.get("state", "Idle"))
	var state_id: int = int(snapshot.get("state_id", TrainingRunController.State.IDLE))
	_title.text = str(snapshot.get("label", "Agent"))
	_algorithm.text = str(snapshot.get("algorithm", ""))
	ControlCenterTheme.apply_status_badge(
		_state, state_name, ControlCenterTheme.state_color(state_name)
	)
	var has_progress: bool = snapshot.has("progress")
	_progress.value = clampf(float(snapshot.get("progress", 0.0)), 0.0, 1.0) * 100.0
	_progress.modulate = Color(1, 1, 1, 1.0 if has_progress else 0.25)
	_metrics.text = "\n".join(_metric_rows(snapshot))
	_footer.text = _footer_text(snapshot)
	_pause.disabled = state_id != TrainingRunController.State.RUNNING
	_resume.disabled = state_id != TrainingRunController.State.PAUSED
	_stop.disabled = state_id not in [
		TrainingRunController.State.STARTING,
		TrainingRunController.State.RUNNING,
		TrainingRunController.State.PAUSED,
	]
	_remove.disabled = state_id not in [
		TrainingRunController.State.FINISHED,
		TrainingRunController.State.ERROR,
	]


static func _metric_rows(snapshot: Dictionary) -> Array:
	var rows: Array = []
	if snapshot.has("timesteps") or snapshot.has("total_training_steps"):
		rows.append(
			"steps        %d / %d"
			% [int(snapshot.get("timesteps", 0)), int(snapshot.get("total_training_steps", 0))]
		)
	elif snapshot.has("epoch") or snapshot.has("total_epochs"):
		rows.append(
			"epoch        %d / %d"
			% [int(snapshot.get("epoch", 0)), int(snapshot.get("total_epochs", 0))]
		)
	else:
		rows.append("steps        n/a")
	rows.append("episodes     " + ControlCenterTheme.optional_metric(snapshot, "episodes", 0))
	rows.append(
		"reward       " + ControlCenterTheme.optional_metric(snapshot, "mean_episode_reward", 3)
	)
	rows.append(
		"kills/deaths %s / %s"
		% [
			ControlCenterTheme.optional_metric(snapshot, "mean_kills", 2),
			ControlCenterTheme.optional_metric(snapshot, "mean_deaths", 2),
		]
	)
	rows.append(
		"accuracy     "
		+ ControlCenterTheme.optional_metric(snapshot, "mean_accuracy", 1, 100.0, "%")
	)
	rows.append(
		"throughput   "
		+ ControlCenterTheme.optional_metric(snapshot, "steps_per_second", 1, 1.0, " sps")
	)
	if snapshot.has("train_loss") or snapshot.has("validation_loss"):
		rows.append(
			"loss         %s / %s"
			% [
				ControlCenterTheme.optional_metric(snapshot, "train_loss", 5),
				ControlCenterTheme.optional_metric(snapshot, "validation_loss", 5),
			]
		)
	return rows


static func _footer_text(snapshot: Dictionary) -> String:
	var parts: Array = []
	parts.append(
		"runtime %s" % ControlCenterTheme.format_duration(snapshot.get("runtime_seconds"))
	)
	if snapshot.has("current_checkpoint"):
		parts.append("ckpt %s" % str(snapshot["current_checkpoint"]).get_file())
	var error_text: String = str(snapshot.get("last_error", ""))
	if not error_text.is_empty():
		parts.append(error_text)
	return " · ".join(parts)


func _make_button(parent: Control, text: String, callback: Callable) -> Button:
	var button := ControlCenterTheme.make_button(text)
	button.pressed.connect(callback)
	parent.add_child(button)
	return button


func _on_pause() -> void:
	pause_requested.emit(agent_id)


func _on_resume() -> void:
	resume_requested.emit(agent_id)


func _on_stop() -> void:
	stop_requested.emit(agent_id)


func _on_details() -> void:
	details_requested.emit(agent_id)


func _on_remove() -> void:
	remove_requested.emit(agent_id)

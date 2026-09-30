## Live monitor for ONE managed headless agent: full backend status, the
## real JSONL event stream and the cooperative lifecycle controls.
##
## The log view renders incrementally (only events it has not shown yet)
## and keeps a bounded number of lines so an hours-long run cannot grow
## UI memory. The JSONL file on disk remains the authoritative record.
class_name ControlCenterHeadlessMonitorPanel
extends PanelContainer

signal details_requested(agent_id: int)

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

## Hard cap on log lines kept in the RichTextLabel. Sourced from the shared
## live-log capacity so it can never exceed what the controller's event
## ring actually retains.
const MAX_LOG_LINES: int = ControlCenterConfig.LIVE_LOG_LINES

var session
var agent_id: int = -1

var _title: Label
var _state: Label
var _progress: ProgressBar
var _status_text: Label
var _log: RichTextLabel
var _log_counter: Label
var _pause: Button
var _resume: Button
var _stop: Button
## How many backend events (lifetime counter) are already rendered.
var _consumed_events: int = 0
var _rendered_lines: int = 0


func setup(p_session, p_agent_id: int, compact: bool = false) -> void:
	session = p_session
	agent_id = p_agent_id
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 6)
	add_child(root)

	var header := ControlCenterTheme.make_row()
	root.add_child(header)
	_title = ControlCenterTheme.make_label(
		"Agent %d" % agent_id, ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
	)
	header.add_child(_title)
	_state = ControlCenterTheme.make_status_label("Idle")
	_state.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_state.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	header.add_child(_state)
	_pause = _make_button(header, "Pause", _on_pause, "Pause at the next safe backend boundary")
	_resume = _make_button(header, "Resume", _on_resume, "Resume a cooperatively paused backend")
	_stop = _make_button(header, "Stop", _on_stop, "Graceful stop with final checkpoint")
	if compact:
		var details := _make_button(header, "Details", _on_details, "Open the full monitor")
		details.visible = true

	_progress = ProgressBar.new()
	_progress.min_value = 0.0
	_progress.max_value = 100.0
	_progress.show_percentage = false
	_progress.custom_minimum_size = Vector2(0.0, 6.0)
	root.add_child(_progress)

	var body := HBoxContainer.new()
	body.add_theme_constant_override("separation", 10)
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(body)

	var status_panel := ControlCenterTheme.make_panel("Live status")
	status_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	status_panel.size_flags_stretch_ratio = 0.42
	body.add_child(status_panel)
	var status_scroll := ControlCenterTheme.make_scroll()
	ControlCenterTheme.content_container(status_panel).add_child(status_scroll)
	_status_text = ControlCenterTheme.make_value_label("")
	_status_text.clip_text = false
	_status_text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	status_scroll.add_child(_status_text)

	var log_panel := ControlCenterTheme.make_panel("Live log")
	log_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	log_panel.size_flags_stretch_ratio = 0.58
	body.add_child(log_panel)
	_log = RichTextLabel.new()
	_log.bbcode_enabled = false
	_log.selection_enabled = true
	_log.scroll_following = true
	_log.scroll_active = true
	_log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_log.custom_minimum_size = Vector2(0.0, 96.0)
	_log.add_theme_font_size_override("normal_font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	ControlCenterTheme.content_container(log_panel).add_child(_log)
	_log_counter = ControlCenterTheme.make_label(
		"no events yet", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	ControlCenterTheme.content_container(log_panel).add_child(_log_counter)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	var manager = session.agent_manager
	if manager == null or not manager.has_agent(agent_id):
		return
	var snapshot: Dictionary = manager.agent_snapshot(agent_id)
	var state_id: int = int(snapshot.get("state_id", TrainingRunController.State.IDLE))
	var state_name: String = str(snapshot.get("state", "Idle"))
	_title.text = (
		"%s · %s"
		% [
			str(snapshot.get("label", "Agent %d" % agent_id)),
			str(snapshot.get("algorithm", "")),
		]
	)
	ControlCenterTheme.apply_status_badge(
		_state, state_name, ControlCenterTheme.state_color(state_name)
	)
	_progress.value = clampf(float(snapshot.get("progress", 0.0)), 0.0, 1.0) * 100.0
	_status_text.text = "\n".join(build_status_rows(snapshot))
	_pause.disabled = state_id != TrainingRunController.State.RUNNING
	_resume.disabled = state_id != TrainingRunController.State.PAUSED
	_stop.disabled = (
		state_id
		not in [
			TrainingRunController.State.STARTING,
			TrainingRunController.State.RUNNING,
			TrainingRunController.State.PAUSED,
		]
	)
	_ingest_new_events(manager.get_controller(agent_id))


## Status rows are built exclusively from published backend keys; absent
## values render as n/a. Public + static for focused tests.
static func build_status_rows(snapshot: Dictionary) -> Array:
	var rows: Array = []
	rows.append("state          %s" % str(snapshot.get("state", "Idle")))
	if snapshot.has("epoch") or snapshot.has("total_epochs"):
		rows.append(
			(
				"epoch          %d / %d"
				% [int(snapshot.get("epoch", 0)), int(snapshot.get("total_epochs", 0))]
			)
		)
	else:
		(
			rows
			. append(
				(
					"steps          %s / %s"
					% [
						ControlCenterTheme.optional_metric(snapshot, "timesteps", 0),
						ControlCenterTheme.optional_metric(snapshot, "total_training_steps", 0),
					]
				)
			)
		)
	rows.append("episodes       " + ControlCenterTheme.optional_metric(snapshot, "episodes", 0))
	rows.append(
		"reward         " + ControlCenterTheme.optional_metric(snapshot, "mean_episode_reward", 3)
	)
	rows.append("kills          " + ControlCenterTheme.optional_metric(snapshot, "mean_kills", 2))
	rows.append("deaths         " + ControlCenterTheme.optional_metric(snapshot, "mean_deaths", 2))
	rows.append(
		"shots          " + ControlCenterTheme.optional_metric(snapshot, "mean_shots_fired", 2)
	)
	rows.append(
		"hits           " + ControlCenterTheme.optional_metric(snapshot, "mean_shots_hit", 2)
	)
	(
		rows
		. append(
			(
				"near/useless   %s / %s"
				% [
					ControlCenterTheme.optional_metric(snapshot, "mean_near_miss_shots", 2),
					ControlCenterTheme.optional_metric(snapshot, "mean_useless_shots", 2),
				]
			)
		)
	)
	rows.append(
		"cooldown pulls " + ControlCenterTheme.optional_metric(snapshot, "mean_cooldown_shots", 2)
	)
	rows.append(
		(
			"accuracy       "
			+ ControlCenterTheme.optional_metric(snapshot, "mean_accuracy", 1, 100.0, "%")
		)
	)
	rows.append(
		"damage dealt   " + ControlCenterTheme.optional_metric(snapshot, "mean_damage_dealt", 2)
	)
	rows.append(
		"damage taken   " + ControlCenterTheme.optional_metric(snapshot, "mean_damage_received", 2)
	)
	rows.append(
		(
			"survival       "
			+ ControlCenterTheme.optional_metric(snapshot, "mean_survival_time", 2, 1.0, " s")
		)
	)
	(
		rows
		. append(
			(
				"win/loss       %s / %s"
				% [
					ControlCenterTheme.optional_metric(snapshot, "win_rate", 1, 100.0, "%"),
					ControlCenterTheme.optional_metric(snapshot, "loss_rate", 1, 100.0, "%"),
				]
			)
		)
	)
	if snapshot.has("train_loss") or snapshot.has("validation_loss"):
		(
			rows
			. append(
				(
					"loss           %s / %s"
					% [
						ControlCenterTheme.optional_metric(snapshot, "train_loss", 5),
						ControlCenterTheme.optional_metric(snapshot, "validation_loss", 5),
					]
				)
			)
		)
	if snapshot.has("component_accuracy") or snapshot.has("exact_accuracy"):
		(
			rows
			. append(
				(
					"bc accuracy    %s / %s"
					% [
						ControlCenterTheme.optional_metric(
							snapshot, "component_accuracy", 1, 100.0, "%"
						),
						ControlCenterTheme.optional_metric(
							snapshot, "exact_accuracy", 1, 100.0, "%"
						),
					]
				)
			)
		)
	rows.append(
		(
			"throughput     "
			+ ControlCenterTheme.optional_metric(snapshot, "steps_per_second", 1, 1.0, " sps")
		)
	)
	var eta = snapshot.get("eta_seconds")
	rows.append("ETA            %s" % ControlCenterTheme.format_duration(eta))
	rows.append(
		"runtime        %s" % ControlCenterTheme.format_duration(snapshot.get("runtime_seconds"))
	)
	if snapshot.has("current_checkpoint"):
		rows.append("checkpoint     %s" % str(snapshot["current_checkpoint"]))
	else:
		rows.append("checkpoint     n/a")
	var device_value = snapshot.get("cuda_device", snapshot.get("device", "n/a"))
	rows.append("device         %s" % str(device_value))
	var error_text: String = str(snapshot.get("last_error", ""))
	if not error_text.is_empty():
		rows.append("error          %s" % error_text)
	return rows


## Appends only the events this panel has not rendered yet, trimming the
## oldest lines beyond MAX_LOG_LINES. Never re-renders the whole ring.
func _ingest_new_events(controller: TrainingRunController) -> void:
	if controller == null:
		return
	var total: int = controller.total_events_ingested
	if total < _consumed_events:
		# The controller started a fresh run; its counters restarted.
		_consumed_events = 0
		_rendered_lines = 0
		_log.clear()
	var new_count: int = total - _consumed_events
	if new_count <= 0:
		_update_log_counter(total)
		return
	var ring: Array = controller.recent_events
	# Events older than the ring were dropped by the bounded buffer; they
	# are acknowledged in the counter label instead of being invented.
	var renderable: int = mini(new_count, ring.size())
	for index in range(ring.size() - renderable, ring.size()):
		_append_event(ring[index])
	_consumed_events = total
	_update_log_counter(total)


func _append_event(entry: Dictionary) -> void:
	var category: String = str(entry.get("category", "system")).to_upper()
	var color: Color = ControlCenterTheme.COLOR_MUTED
	if category == "ERROR":
		color = ControlCenterTheme.COLOR_BAD
	elif category == "METRIC":
		color = ControlCenterTheme.COLOR_AI
	var stamp: String = ""
	var wall = entry.get("wall_time")
	if wall != null and str(wall).is_valid_float():
		stamp = ControlCenterTheme.format_clock(float(wall)) + " "
	_log.push_color(ControlCenterTheme.COLOR_MUTED)
	_log.add_text(stamp)
	_log.pop()
	_log.push_color(color)
	_log.add_text("%-7s " % category)
	_log.pop()
	_log.add_text(str(entry.get("message", "")))
	_log.newline()
	_rendered_lines += 1
	while _rendered_lines > MAX_LOG_LINES:
		_log.remove_paragraph(0)
		_rendered_lines -= 1


func _update_log_counter(total: int) -> void:
	if total <= 0:
		_log_counter.text = "no events yet"
	else:
		_log_counter.text = "%d events · showing latest %d" % [total, _rendered_lines]


func rendered_line_count() -> int:
	return _rendered_lines


func _make_button(parent: Control, text: String, callback: Callable, tooltip: String) -> Button:
	var button := ControlCenterTheme.make_button(text, tooltip)
	button.pressed.connect(callback)
	parent.add_child(button)
	return button


func _on_pause() -> void:
	session.agent_manager.pause(agent_id)
	refresh()


func _on_resume() -> void:
	session.agent_manager.resume(agent_id)
	refresh()


func _on_stop() -> void:
	session.agent_manager.stop(agent_id)
	refresh()


func _on_details() -> void:
	details_requested.emit(agent_id)

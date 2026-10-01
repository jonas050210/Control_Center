## ControlCenterStatusBar
##
## Top strip: what is running, which mode, which environment/agent, which
## action source, which camera. Every control here calls an existing
## ControlCenterSession method; nothing is decorative.
class_name ControlCenterStatusBar
extends PanelContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var session
var _mode_buttons: Dictionary = {}
var _state_label: Label
var _rate_label: Label
var _pending_label: Label
var _environment_option: OptionButton
var _agent_option: OptionButton
var _policy_option: OptionButton
var _camera_option: OptionButton
var _updating: bool = false


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)

	# The command strip has several real selectors. Keep them all reachable on
	# narrow windows instead of clipping the right-most state and camera tools.
	var strip_scroll := ScrollContainer.new()
	strip_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	strip_scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	strip_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	add_child(strip_scroll)
	var row := ControlCenterTheme.make_row()
	row.custom_minimum_size = Vector2(1120.0, 0.0)
	strip_scroll.add_child(row)

	row.add_child(
		ControlCenterTheme.make_label(
			"SANDBOXAI // TTK", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	row.add_child(
		ControlCenterTheme.make_label(
			"LOCAL CALIBRATION", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_ACCENT
		)
	)
	row.add_child(VSeparator.new())

	for mode in [
		ControlCenterConfig.Mode.TRAINING,
		ControlCenterConfig.Mode.WATCH,
		ControlCenterConfig.Mode.HUMAN
	]:
		var button := ControlCenterTheme.make_toggle(
			ControlCenterConfig.mode_name(mode), mode == session.config.mode, _mode_tooltip(mode)
		)
		button.pressed.connect(_on_mode_pressed.bind(mode))
		row.add_child(button)
		_mode_buttons[mode] = button

	row.add_child(VSeparator.new())
	_state_label = ControlCenterTheme.make_status_label()
	row.add_child(_state_label)

	row.add_child(VSeparator.new())
	row.add_child(
		ControlCenterTheme.make_label(
			"env", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_environment_option = ControlCenterTheme.make_option_button(
		"Which environment is rendered and inspected. Others keep running headless."
	)
	_environment_option.item_selected.connect(_on_environment_selected)
	row.add_child(_environment_option)

	row.add_child(
		ControlCenterTheme.make_label(
			"agent", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_agent_option = ControlCenterTheme.make_option_button(
		"Agent slot inside the selected environment."
	)
	_agent_option.item_selected.connect(_on_agent_selected)
	row.add_child(_agent_option)

	row.add_child(
		ControlCenterTheme.make_label(
			"policy", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_policy_option = ControlCenterTheme.make_option_button(
		"Which action source drives the AI-controlled environments."
	)
	_policy_option.item_selected.connect(_on_policy_selected)
	row.add_child(_policy_option)

	row.add_child(
		ControlCenterTheme.make_label(
			"camera", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_camera_option = ControlCenterTheme.make_option_button(
		"Presentation-only camera. The policy never receives camera data."
	)
	_camera_option.item_selected.connect(_on_camera_selected)
	row.add_child(_camera_option)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	spacer.mouse_filter = Control.MOUSE_FILTER_IGNORE
	row.add_child(spacer)

	_pending_label = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_WARN
	)
	row.add_child(_pending_label)
	_rate_label = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	row.add_child(_rate_label)

	_populate_options()


func _mode_tooltip(mode: int) -> String:
	match mode:
		ControlCenterConfig.Mode.TRAINING:
			return (
				"Headless-style throughput: no rendering, telemetry or logging. "
				+ "PPO weight updates happen in the external Python trainer."
			)
		ControlCenterConfig.Mode.WATCH:
			return "Render the selected environment while an AI controller drives it."
		_:
			return "Take over the selected environment through the human input pipeline."


func _populate_options() -> void:
	_updating = true
	_environment_option.clear()
	var count: int = session.simulation_manager.environments.size()
	for index in range(count):
		_environment_option.add_item("Environment %d" % index, index)
	_environment_option.selected = clampi(
		session.config.selected_environment, 0, maxi(0, count - 1)
	)

	_agent_option.clear()
	var slots: Array = session.available_agent_slots()
	for slot_value in slots:
		var slot: Dictionary = slot_value
		var label: String = str(slot["label"])
		if not bool(slot["available"]):
			label += "  [unavailable]"
		_agent_option.add_item(label, int(slot["slot"]))
		var item_index: int = _agent_option.item_count - 1
		_agent_option.set_item_disabled(item_index, not bool(slot["available"]))
		_agent_option.set_item_tooltip(item_index, str(slot["note"]))
	_agent_option.selected = clampi(session.config.selected_agent_slot, 0, slots.size() - 1)

	_policy_option.clear()
	for source in [
		ControlCenterConfig.PolicySource.HEURISTIC,
		ControlCenterConfig.PolicySource.IDLE,
		ControlCenterConfig.PolicySource.EXTERNAL_POLICY
	]:
		_policy_option.add_item(ControlCenterConfig.policy_source_name(source), source)
		var policy_index: int = _policy_option.item_count - 1
		var available: bool = ControlCenterConfig.policy_source_available(source)
		_policy_option.set_item_disabled(policy_index, not available)
		if not available:
			_policy_option.set_item_tooltip(
				policy_index,
				(
					"Godot has no neural-network runtime. Trained checkpoints are "
					+ "executed by the Python trainer over the JSON-lines bridge."
				)
			)
	_policy_option.selected = clampi(session.config.policy_source, 0, 2)

	_camera_option.clear()
	for camera_mode in [
		ControlCenterConfig.CameraMode.FIRST_PERSON,
		ControlCenterConfig.CameraMode.THIRD_PERSON,
		ControlCenterConfig.CameraMode.FREE,
		ControlCenterConfig.CameraMode.TOP_DOWN
	]:
		_camera_option.add_item(ControlCenterConfig.camera_mode_name(camera_mode), camera_mode)
	_camera_option.selected = clampi(session.config.camera_mode, 0, 3)
	_updating = false


## Rebuilds the environment list after a rebuild changed the count.
func refresh_options() -> void:
	_populate_options()


func refresh(snapshot: Dictionary) -> void:
	var status: Dictionary = snapshot.get("status", session.get_status())
	for mode in _mode_buttons.keys():
		(_mode_buttons[mode] as Button).button_pressed = int(status["mode"]) == int(mode)

	var running: bool = bool(status["running"])
	var training: Dictionary = session.training_run.snapshot()
	var training_state: String = str(training.get("state", "Idle"))
	var show_training_state: bool = training_state != "Idle"
	var state_text: String = (
		"TRAINING: %s" % training_state.to_upper()
		if show_training_state
		else ("RUNNING" if running else "PAUSED")
	)
	if int(status["mode"]) == ControlCenterConfig.Mode.HUMAN:
		state_text += (
			"  |  input: CAPTURED"
			if bool(status["human_input_enabled"])
			else "  |  input: released"
		)
	var state_color: Color = (
		ControlCenterTheme.COLOR_OK if running else ControlCenterTheme.COLOR_WARN
	)
	if training_state in ["Starting", "Paused", "Stopping"]:
		state_color = ControlCenterTheme.COLOR_WARN
	elif training_state == "Error":
		state_color = ControlCenterTheme.COLOR_BAD
	ControlCenterTheme.apply_status_badge(_state_label, state_text, state_color)

	if training.has("steps_per_second"):
		_rate_label.text = (
			"%.0f train steps/s   %d fps   %s env"
			% [
				float(training["steps_per_second"]),
				int(status["render_fps"]),
				str(training.get("environment_count", "n/a")),
			]
		)
	else:
		_rate_label.text = (
			"%.0f sim steps/s   %d fps   %d env"
			% [
				float(status["steps_per_second"]),
				int(status["render_fps"]),
				int(status["environment_count"]),
			]
		)
	var system_text: String = _compact_system_status(status.get("system", {}))
	if not system_text.is_empty():
		_rate_label.text += "   " + system_text
	var pending: PackedStringArray = status["pending_settings"]
	_pending_label.text = ("" if pending.is_empty() else "pending reset: %s" % ", ".join(pending))

	if not _updating:
		_updating = true
		if _environment_option.item_count > 0:
			_environment_option.selected = clampi(
				int(status["selected_environment"]), 0, _environment_option.item_count - 1
			)
		_updating = false


static func _compact_system_status(system_value) -> String:
	if not (system_value is Dictionary):
		return ""
	var system: Dictionary = system_value
	var cpu: Dictionary = system.get("cpu", {})
	var gpu: Dictionary = system.get("gpu", {})
	var parts: PackedStringArray = PackedStringArray()
	if cpu.get("utilization_percent") != null:
		parts.append("CPU %.0f%%" % float(cpu["utilization_percent"]))
	if gpu.get("utilization_percent") != null:
		parts.append("GPU %.0f%%" % float(gpu["utilization_percent"]))
	elif gpu.get("vram_used_mb") != null and gpu.get("vram_total_mb") != null:
		parts.append("VRAM %.0f/%.0fMB" % [float(gpu["vram_used_mb"]), float(gpu["vram_total_mb"])])
	return "  ".join(parts)


func _on_mode_pressed(mode: int) -> void:
	session.set_mode(mode)
	for other in _mode_buttons.keys():
		(_mode_buttons[other] as Button).button_pressed = int(other) == mode


func _on_environment_selected(index: int) -> void:
	if _updating:
		return
	session.select_environment(index)


func _on_agent_selected(index: int) -> void:
	if _updating:
		return
	if not session.select_agent_slot(index):
		_updating = true
		_agent_option.selected = session.config.selected_agent_slot
		_updating = false


func _on_policy_selected(index: int) -> void:
	if _updating:
		return
	if not session.set_policy_source(index):
		_updating = true
		_policy_option.selected = session.config.policy_source
		_updating = false


func _on_camera_selected(index: int) -> void:
	if _updating:
		return
	session.set_camera_mode(index)

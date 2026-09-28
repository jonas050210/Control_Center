## ControlCenterControlsPanel
##
## Simulation transport controls (Phase 9): play/pause, single step,
## reset selected / reset all, speed presets and a free speed slider,
## plus the HUMAN-mode input arm switch.
##
## Nothing here touches `Engine.time_scale`: speed is implemented by the
## session as "how many fixed simulation steps to run per rendered frame",
## so physics stay deterministic at SandboxConfig.SIMULATION_DT.
class_name ControlCenterControlsPanel
extends PanelContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var session
var _play_button: Button
var _step_button: Button
var _speed_label: Label
var _speed_slider: HSlider
var _speed_buttons: Array = []  # Array[Button], parallel to SPEED_PRESETS
var _human_toggle: Button
var _human_hint: Label
var _progress_label: Label
var _updating: bool = false


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 4)
	add_child(root)

	var transport := ControlCenterTheme.make_row()
	root.add_child(transport)

	_play_button = ControlCenterTheme.make_button("Pause", "Space - pause / resume stepping")
	_play_button.pressed.connect(_on_play_pressed)
	transport.add_child(_play_button)

	_step_button = ControlCenterTheme.make_button("Step", "N - advance exactly one simulation step")
	_step_button.pressed.connect(_on_step_pressed)
	transport.add_child(_step_button)

	var step10 := ControlCenterTheme.make_button("Step x10", "Advance ten simulation steps")
	step10.pressed.connect(_on_step10_pressed)
	transport.add_child(step10)

	transport.add_child(VSeparator.new())

	var reset_button := ControlCenterTheme.make_button(
		"Reset env", "R - reset the selected environment with the configured seed"
	)
	reset_button.pressed.connect(_on_reset_pressed)
	transport.add_child(reset_button)

	var reset_random := ControlCenterTheme.make_button(
		"Reset (random)", "Reset the selected environment with a fresh random seed"
	)
	reset_random.pressed.connect(_on_reset_random_pressed)
	transport.add_child(reset_random)

	var reset_all := ControlCenterTheme.make_button(
		"Reset all", "Reset every environment in this process"
	)
	reset_all.pressed.connect(_on_reset_all_pressed)
	transport.add_child(reset_all)

	_progress_label = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	transport.add_child(_progress_label)

	var speed_row := ControlCenterTheme.make_row()
	root.add_child(speed_row)
	speed_row.add_child(
		ControlCenterTheme.make_label(
			"speed", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	for preset_value in ControlCenterConfig.SPEED_PRESETS:
		var preset: float = float(preset_value)
		var button := ControlCenterTheme.make_toggle(
			"%sx" % ControlCenterTheme.format_number(preset),
			false,
			(
				"Run %s simulation steps per real-time step"
				% ControlCenterTheme.format_number(preset)
			)
		)
		button.pressed.connect(_on_speed_preset.bind(preset))
		speed_row.add_child(button)
		_speed_buttons.append(button)

	_speed_slider = HSlider.new()
	_speed_slider.min_value = ControlCenterConfig.MIN_SPEED
	_speed_slider.max_value = ControlCenterConfig.MAX_SPEED
	_speed_slider.step = 0.05
	_speed_slider.value = 1.0
	_speed_slider.custom_minimum_size = Vector2(140.0, 0.0)
	_speed_slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	_speed_slider.tooltip_text = (
		"Free speed control between %sx and %sx"
		% [
			ControlCenterTheme.format_number(ControlCenterConfig.MIN_SPEED),
			ControlCenterTheme.format_number(ControlCenterConfig.MAX_SPEED),
		]
	)
	_speed_slider.value_changed.connect(_on_speed_slider_changed)
	speed_row.add_child(_speed_slider)

	_speed_label = ControlCenterTheme.make_value_label("1.00x")
	_speed_label.custom_minimum_size = Vector2(56.0, 0.0)
	speed_row.add_child(_speed_label)

	speed_row.add_child(VSeparator.new())

	_human_toggle = ControlCenterTheme.make_toggle(
		"Human input: OFF",
		false,
		"Arms keyboard/mouse control of the selected agent (HUMAN mode only)."
	)
	_human_toggle.toggled.connect(_on_human_toggled)
	speed_row.add_child(_human_toggle)

	_human_hint = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	speed_row.add_child(_human_hint)


func refresh(snapshot: Dictionary) -> void:
	_updating = true
	_play_button.text = "Pause" if session.running else "Play"
	_play_button.add_theme_color_override(
		"font_color",
		ControlCenterTheme.COLOR_TEXT if session.running else ControlCenterTheme.COLOR_WARN
	)
	_step_button.disabled = session.running

	var speed: float = session.config.simulation_speed
	_speed_label.text = "%.2fx" % speed
	if not is_equal_approx(float(_speed_slider.value), speed):
		_speed_slider.value = speed
	for index in range(_speed_buttons.size()):
		var button: Button = _speed_buttons[index]
		button.button_pressed = is_equal_approx(
			float(ControlCenterConfig.SPEED_PRESETS[index]), speed
		)

	var human_mode: bool = session.is_human_mode()
	_human_toggle.disabled = not human_mode
	_human_toggle.button_pressed = session.human_input_enabled
	_human_toggle.text = "Human input: %s" % ("ON" if session.human_input_enabled else "OFF")
	if not human_mode:
		_human_hint.text = "switch to HUMAN mode to drive the agent"
	elif session.human_input_enabled:
		_human_hint.text = "WASD move, mouse/arrows look, LMB or Space shoot, Esc releases mouse"
	else:
		_human_hint.text = "agent idles until input is armed"

	if snapshot.is_empty():
		_progress_label.text = "telemetry off (TRAINING)"
	else:
		var episode: Dictionary = snapshot["episode"]
		_progress_label.text = (
			"step %d / %d   episode %d"
			% [int(episode["step"]), int(episode["max_steps"]), int(episode["episode"])]
		)
	_updating = false


func _on_play_pressed() -> void:
	session.toggle_running()


func _on_step_pressed() -> void:
	session.request_single_step(1)


func _on_step10_pressed() -> void:
	session.request_single_step(10)


func _on_reset_pressed() -> void:
	session.reset_selected_environment(true)


func _on_reset_random_pressed() -> void:
	session.reset_selected_environment_with_random_seed()


func _on_reset_all_pressed() -> void:
	session.reset_all_environments(true)


func _on_speed_preset(value: float) -> void:
	if _updating:
		return
	session.set_speed(value)


func _on_speed_slider_changed(value: float) -> void:
	if _updating:
		return
	session.set_speed(value)


func _on_human_toggled(pressed: bool) -> void:
	if _updating:
		return
	session.set_human_input_enabled(pressed)

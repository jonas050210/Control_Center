## ControlCenterSettingsPanel
##
## Settings tab (Phase 9/10). Only settings the current simulation can
## genuinely change are exposed:
##
##   * curriculum level and scenario preset apply immediately
##     (SimulationManager.set_curriculum_level reconfigures live enemies)
##   * environment count, enemy count and seed are marked "requires reset"
##     and are applied by an explicit rebuild
##   * panel visibility is presentation-only
##
## The TRAINING section shows the exact headless command for real PPO
## training, because gradient updates happen in the Python trainer and not
## in this window.
class_name ControlCenterSettingsPanel
extends VBoxContainer

signal settings_rebuilt
signal tile_layout_changed

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")

var session
var _scenario_option: OptionButton
var _curriculum_option: OptionButton
var _environment_spin: SpinBox
var _enemy_spin: SpinBox
var _seed_spin: SpinBox
var _apply_button: Button
var _pending_label: Label
var _command_label: Label
var _scenario_note: Label
var _updating: bool = false


func setup(p_session) -> void:
	session = p_session
	add_theme_constant_override("separation", 6)

	add_child(
		ControlCenterTheme.make_label(
			"SCENARIO", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_scenario_option = ControlCenterTheme.make_option_button(
		"Presets that only bundle curriculum level + enemy count."
	)
	for entry_value in ControlCenterConfig.SCENARIOS:
		var entry: Dictionary = entry_value
		_scenario_option.add_item(str(entry["label"]))
	_scenario_option.item_selected.connect(_on_scenario_selected)
	add_child(_scenario_option)
	_scenario_note = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_scenario_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_scenario_note)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"LIVE SETTINGS", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	var curriculum_row := ControlCenterTheme.make_row()
	add_child(curriculum_row)
	curriculum_row.add_child(
		ControlCenterTheme.make_label(
			"curriculum", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_curriculum_option = ControlCenterTheme.make_option_button(
		"Applied immediately to the running environments."
	)
	for level in range(
		CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT + 1
	):
		_curriculum_option.add_item(
			"%d  %s" % [level, CurriculumConfig.level_name(level)], level
		)
	_curriculum_option.item_selected.connect(_on_curriculum_selected)
	curriculum_row.add_child(_curriculum_option)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"REQUIRES RESET", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_WARN
		)
	)
	var grid := ControlCenterTheme.make_grid(2)
	add_child(grid)

	grid.add_child(
		ControlCenterTheme.make_label(
			"environments", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_environment_spin = ControlCenterTheme.make_spin_box(
		1.0,
		float(ControlCenterConfig.MAX_ENVIRONMENT_COUNT),
		1.0,
		float(session.config.environment_count),
		"Parallel environments in this process. Only the selected one renders."
	)
	_environment_spin.value_changed.connect(_on_environment_count_changed)
	grid.add_child(_environment_spin)

	grid.add_child(
		ControlCenterTheme.make_label(
			"enemies", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_enemy_spin = ControlCenterTheme.make_spin_box(
		1.0,
		float(ControlCenterConfig.MAX_ENEMY_COUNT),
		1.0,
		float(session.config.enemy_count),
		"Enemies per environment. Curriculum level 4 enforces a minimum of 3."
	)
	_enemy_spin.value_changed.connect(_on_enemy_count_changed)
	grid.add_child(_enemy_spin)

	grid.add_child(
		ControlCenterTheme.make_label(
			"seed", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_seed_spin = ControlCenterTheme.make_spin_box(
		0.0, 1000000.0, 1.0, float(session.config.seed), "Base seed; environment i uses seed + i."
	)
	_seed_spin.value_changed.connect(_on_seed_changed)
	grid.add_child(_seed_spin)

	var button_row := ControlCenterTheme.make_row()
	add_child(button_row)
	_apply_button = ControlCenterTheme.make_button(
		"Apply & reset", "Rebuilds every environment with the pending settings."
	)
	_apply_button.pressed.connect(_on_apply_pressed)
	button_row.add_child(_apply_button)
	var randomize_button := ControlCenterTheme.make_button(
		"Randomize seed", "Picks a new base seed (still requires Apply & reset)."
	)
	randomize_button.pressed.connect(_on_randomize_pressed)
	button_row.add_child(randomize_button)

	_pending_label = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_WARN
	)
	_pending_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_pending_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"PANELS", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	var panel_row := ControlCenterTheme.make_row()
	add_child(panel_row)
	panel_row.add_child(_make_tile_toggle("Simulation", "simulation"))
	panel_row.add_child(_make_tile_toggle("Agent (F1)", "agent"))
	panel_row.add_child(_make_tile_toggle("Inspector (F2)", "inspector"))
	panel_row.add_child(_make_tile_toggle("Training", "training"))
	panel_row.add_child(_make_tile_toggle("Controls", "controls"))
	panel_row.add_child(_make_tile_toggle("Logs (F3)", "logs"))
	var order_row := ControlCenterTheme.make_row()
	add_child(order_row)
	order_row.add_child(
		ControlCenterTheme.make_label(
			"utility order", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	var controls_first := ControlCenterTheme.make_button("Simulation controls ↑")
	controls_first.pressed.connect(_move_tile.bind("controls", -1))
	order_row.add_child(controls_first)
	var logs_first := ControlCenterTheme.make_button("Logs ↑")
	logs_first.pressed.connect(_move_tile.bind("logs", -1))
	order_row.add_child(logs_first)
	var persistence_note := ControlCenterTheme.make_label(
		"Dock sizes, visibility and order persist in user://control_center.cfg.",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	persistence_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(persistence_note)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"HEADLESS TRAINING", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	var training_note := ControlCenterTheme.make_label(
		(
			"TRAINING mode here runs the same environments with rendering, "
			+ "telemetry and logging switched off. PPO weight updates run in the "
			+ "Python trainer, in a separate headless Godot process:"
		),
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	training_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(training_note)
	_command_label = ControlCenterTheme.make_value_label("")
	_command_label.clip_text = false
	_command_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_command_label)
	var copy_button := ControlCenterTheme.make_button(
		"Copy command", "Copies the training command to the clipboard."
	)
	copy_button.pressed.connect(_on_copy_command)
	add_child(copy_button)

	_sync_from_config()


func _make_tile_toggle(text: String, tile_id: String) -> Button:
	var toggle := ControlCenterTheme.make_toggle(
		text, session.config.is_tile_visible(tile_id)
	)
	toggle.toggled.connect(_on_tile_toggled.bind(tile_id))
	return toggle


func refresh(_snapshot: Dictionary) -> void:
	var pending: PackedStringArray = session.pending_setting_keys()
	_pending_label.text = (
		""
		if pending.is_empty()
		else "pending (needs Apply & reset): %s" % ", ".join(pending)
	)
	_apply_button.disabled = pending.is_empty()
	_command_label.text = session.training_command_line()
	# Never overwrite a field the operator is currently typing into.
	var focus_owner: Control = get_viewport().gui_get_focus_owner() if is_inside_tree() else null
	if not _updating and (focus_owner == null or not is_ancestor_of(focus_owner)):
		_sync_from_config()


func _sync_from_config() -> void:
	_updating = true
	var scenario_index: int = 0
	for index in range(ControlCenterConfig.SCENARIOS.size()):
		var entry: Dictionary = ControlCenterConfig.SCENARIOS[index]
		if str(entry["id"]) == session.config.scenario_id:
			scenario_index = index
			break
	_scenario_option.selected = scenario_index
	var scenario_entry: Dictionary = ControlCenterConfig.SCENARIOS[scenario_index]
	_scenario_note.text = str(scenario_entry.get("description", ""))
	_curriculum_option.selected = clampi(
		session.config.curriculum_level - 1, 0, _curriculum_option.item_count - 1
	)
	_environment_spin.value = float(session.config.environment_count)
	_enemy_spin.value = float(session.config.enemy_count)
	_seed_spin.value = float(session.config.seed)
	_updating = false


func _on_scenario_selected(index: int) -> void:
	if _updating:
		return
	var entry: Dictionary = ControlCenterConfig.SCENARIOS[index]
	session.apply_scenario(str(entry["id"]))
	_sync_from_config()


func _on_curriculum_selected(index: int) -> void:
	if _updating:
		return
	session.set_curriculum_level(index + 1)
	_sync_from_config()


func _on_environment_count_changed(value: float) -> void:
	if _updating:
		return
	session.request_setting("environment_count", int(value))


func _on_enemy_count_changed(value: float) -> void:
	if _updating:
		return
	session.request_setting("enemy_count", int(value))


func _on_seed_changed(value: float) -> void:
	if _updating:
		return
	session.request_setting("seed", int(value))


func _on_randomize_pressed() -> void:
	session.randomize_seed()
	_sync_from_config()


func _on_apply_pressed() -> void:
	session.apply_pending_settings()
	_sync_from_config()
	settings_rebuilt.emit()


func _on_tile_toggled(pressed: bool, tile_id: String) -> void:
	session.config.set_tile_visible(tile_id, pressed)
	session.config.save_preferences()
	tile_layout_changed.emit()


func _move_tile(tile_id: String, direction: int) -> void:
	if session.config.move_tile(tile_id, direction):
		session.config.save_preferences()
		tile_layout_changed.emit()


func _on_copy_command() -> void:
	DisplayServer.clipboard_set(session.training_command_line())

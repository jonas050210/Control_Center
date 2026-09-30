## AGENTS page: every known agent as a manageable tile plus the launcher
## for additional agents. Configuration is the SAME ControlCenterConfig the
## Training page edits — this page only selects the algorithm and launches.
class_name ControlCenterAgentsPanel
extends PanelContainer

signal open_agent_requested(agent_id: int)
signal configure_requested

const ControlCenterAgentCard = preload("res://scripts/control_center/ui/agent_card.gd")
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var session

var _algorithm: OptionButton
var _launch_summary: Label
var _validation: Label
var _start_button: Button
var _grid: GridContainer
var _empty: Label
var _cards: Dictionary = {}
var _updating: bool = false


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var scroll := ControlCenterTheme.make_scroll()
	add_child(scroll)
	var root := VBoxContainer.new()
	root.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_theme_constant_override("separation", 10)
	scroll.add_child(root)

	var launcher := ControlCenterTheme.make_panel("Start a new agent")
	root.add_child(launcher)
	var launcher_box := ControlCenterTheme.content_container(launcher)
	var row := ControlCenterTheme.make_row()
	launcher_box.add_child(row)
	row.add_child(
		ControlCenterTheme.make_label(
			"algorithm", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_algorithm = ControlCenterTheme.make_option_button(
		"Training backend for the next launched agent"
	)
	_algorithm.add_item("PPO", ControlCenterConfig.TrainingType.PPO)
	_algorithm.add_item("Behavior Cloning", ControlCenterConfig.TrainingType.BEHAVIOR_CLONING)
	_algorithm.add_item("Self-Play", ControlCenterConfig.TrainingType.SELF_PLAY)
	_algorithm.item_selected.connect(_on_algorithm_selected)
	row.add_child(_algorithm)
	_start_button = ControlCenterTheme.make_primary_button(
		"Start agent", "Launch a managed trainer with the current configuration"
	)
	_start_button.pressed.connect(_on_start_pressed)
	row.add_child(_start_button)
	var configure := ControlCenterTheme.make_button(
		"Configure…", "Open the full training configuration"
	)
	configure.pressed.connect(func(): configure_requested.emit())
	row.add_child(configure)
	_launch_summary = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_launch_summary.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	launcher_box.add_child(_launch_summary)
	_validation = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_WARN
	)
	_validation.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	launcher_box.add_child(_validation)

	root.add_child(
		ControlCenterTheme.make_label(
			"All agents", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_grid = GridContainer.new()
	_grid.columns = 2
	_grid.add_theme_constant_override("h_separation", 10)
	_grid.add_theme_constant_override("v_separation", 10)
	_grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(_grid)
	_empty = ControlCenterTheme.make_empty_label(
		"No agents yet. Configure and start your first training agent above."
	)
	root.add_child(_empty)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	if session == null or session.agent_manager == null:
		return
	_updating = true
	var index: int = _algorithm.get_item_index(session.config.training_type)
	if index >= 0:
		_algorithm.selected = index
	_updating = false
	var problem: String = session.training_run.validation_error(session.config)
	_validation.text = problem
	_validation.visible = not problem.is_empty()
	_start_button.disabled = not problem.is_empty()
	_launch_summary.text = _summary_text()

	var manager = session.agent_manager
	var seen: Array = []
	var count: int = 0
	for id_value in manager.agent_ids():
		var agent_id: int = int(id_value)
		seen.append(agent_id)
		var card: ControlCenterAgentCard = _ensure_card(agent_id)
		card.update_from(manager.agent_snapshot(agent_id))
		count += 1
	_prune_cards(seen)
	_grid.columns = clampi(int(size.x / 320.0), 1, 4)
	_empty.visible = count == 0


func _summary_text() -> String:
	var config: ControlCenterConfig = session.config
	if config.training_type == ControlCenterConfig.TrainingType.BEHAVIOR_CLONING:
		return (
			"%d epochs · batch %d · seed %d · device %s · dataset %s"
			% [
				config.bc_epochs,
				config.batch_size,
				config.seed,
				ControlCenterConfig.training_device_argument(config.training_device),
				config.bc_dataset_path,
			]
		)
	return (
		"%d steps · %d environments · level %d · seed %d · device %s"
		% [
			config.total_training_steps,
			config.environment_count,
			config.curriculum_level,
			config.seed,
			ControlCenterConfig.training_device_argument(config.training_device),
		]
	)


func _ensure_card(agent_id: int) -> ControlCenterAgentCard:
	if _cards.has(agent_id):
		return _cards[agent_id]
	var card := ControlCenterAgentCard.new()
	card.setup(true)
	card.pause_requested.connect(func(id: int): session.agent_manager.pause(id))
	card.resume_requested.connect(func(id: int): session.agent_manager.resume(id))
	card.stop_requested.connect(func(id: int): session.agent_manager.stop(id))
	card.remove_requested.connect(func(id: int): session.agent_manager.remove(id))
	card.details_requested.connect(func(id: int): open_agent_requested.emit(id))
	_grid.add_child(card)
	_cards[agent_id] = card
	return card


func _prune_cards(seen: Array) -> void:
	for agent_id in _cards.keys():
		if seen.has(agent_id):
			continue
		var card: ControlCenterAgentCard = _cards[agent_id]
		if card.get_parent() != null:
			card.get_parent().remove_child(card)
		card.queue_free()
		_cards.erase(agent_id)


func _on_algorithm_selected(index: int) -> void:
	if _updating:
		return
	session.config.training_type = _algorithm.get_item_id(index)
	session.config.sanitize()
	session.config.save_preferences()
	refresh()


func _on_start_pressed() -> void:
	var agent_id: int = session.launch_agent()
	if agent_id > 0:
		open_agent_requested.emit(agent_id)
	refresh()

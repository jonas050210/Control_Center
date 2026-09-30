## HOME page: measured PC status on top, the active agent tiles below and
## a recent area for finished/error agents. Pure presentation over the
## system monitor and the training agent manager.
class_name ControlCenterHomePanel
extends PanelContainer

signal open_agent_requested(agent_id: int)

const ControlCenterAgentCard = preload("res://scripts/control_center/ui/agent_card.gd")
const ControlCenterSystemStatusPanel = preload(
	"res://scripts/control_center/ui/system_status_panel.gd"
)
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var session
var system_status_panel: ControlCenterSystemStatusPanel

var _active_header: Label
var _active_grid: GridContainer
var _active_empty: Label
var _recent_header: Label
var _recent_grid: GridContainer
var _recent_empty: Label
var _cards: Dictionary = {}  # agent_id -> ControlCenterAgentCard


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

	system_status_panel = ControlCenterSystemStatusPanel.new()
	system_status_panel.setup(session)
	root.add_child(system_status_panel)

	_active_header = ControlCenterTheme.make_label(
		"Active agents", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
	)
	root.add_child(_active_header)
	_active_grid = _make_grid()
	root.add_child(_active_grid)
	_active_empty = ControlCenterTheme.make_label(
		"No active agents. Start one from the AGENTS or TRAINING page.",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	root.add_child(_active_empty)

	_recent_header = ControlCenterTheme.make_label(
		"Recent", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
	)
	root.add_child(_recent_header)
	_recent_grid = _make_grid()
	root.add_child(_recent_grid)
	_recent_empty = ControlCenterTheme.make_label(
		"Finished and failed agents appear here until cleared.",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	root.add_child(_recent_empty)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	if session == null or session.agent_manager == null:
		return
	system_status_panel.refresh()
	var manager = session.agent_manager
	var active_count: int = 0
	var recent_count: int = 0
	var seen: Array = []
	for id_value in manager.agent_ids():
		var agent_id: int = int(id_value)
		var snapshot: Dictionary = manager.agent_snapshot(agent_id)
		var state_id: int = int(snapshot.get("state_id", TrainingRunController.State.IDLE))
		if state_id == TrainingRunController.State.IDLE:
			continue
		seen.append(agent_id)
		var card: ControlCenterAgentCard = _ensure_card(agent_id)
		card.update_from(snapshot)
		var terminal: bool = state_id in [
			TrainingRunController.State.FINISHED,
			TrainingRunController.State.ERROR,
		]
		var target: GridContainer = _recent_grid if terminal else _active_grid
		if card.get_parent() != target:
			if card.get_parent() != null:
				card.get_parent().remove_child(card)
			target.add_child(card)
		if terminal:
			recent_count += 1
		else:
			active_count += 1
	_prune_cards(seen)
	_layout_grid(_active_grid)
	_layout_grid(_recent_grid)
	_active_header.text = "Active agents (%d)" % active_count
	_active_empty.visible = active_count == 0
	_recent_empty.visible = recent_count == 0


func card_count() -> int:
	return _cards.size()


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


func _make_grid() -> GridContainer:
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 10)
	grid.add_theme_constant_override("v_separation", 10)
	grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	return grid


func _layout_grid(grid: GridContainer) -> void:
	var width: float = size.x
	grid.columns = clampi(int(width / 320.0), 1, 4)

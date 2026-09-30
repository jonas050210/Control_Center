## HEADLESS page: scalable live monitoring for every managed agent. Each
## agent owns one monitor panel (live status + live log + controls); the
## selector row switches between the all-agents grid and a single focused
## agent.
class_name ControlCenterHeadlessPanel
extends PanelContainer

const ControlCenterHeadlessMonitorPanel = preload(
	"res://scripts/control_center/ui/headless_monitor_panel.gd"
)
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var session
## -1 = show every agent in the grid; otherwise the focused agent id.
var focused_agent_id: int = -1

var _selector_row: HBoxContainer
var _grid: GridContainer
var _empty: Label
var _panels: Dictionary = {}  # agent_id -> monitor panel


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	var header := ControlCenterTheme.make_row()
	root.add_child(header)
	header.add_child(
		ControlCenterTheme.make_label(
			"Headless agents", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_selector_row = ControlCenterTheme.make_row()
	_selector_row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(_selector_row)

	_grid = GridContainer.new()
	_grid.columns = 1
	_grid.add_theme_constant_override("h_separation", 10)
	_grid.add_theme_constant_override("v_separation", 10)
	_grid.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(_grid)

	_empty = ControlCenterTheme.make_empty_label(
		"No launched agents. Start one from the Agents or Training page."
	)
	root.add_child(_empty)
	refresh()


## Focuses one agent's monitor (used by the Details buttons elsewhere).
func focus_agent(agent_id: int) -> void:
	focused_agent_id = agent_id
	refresh()


func show_all() -> void:
	focused_agent_id = -1
	refresh()


func monitor_for(agent_id: int) -> ControlCenterHeadlessMonitorPanel:
	return _panels.get(agent_id)


func refresh(_snapshot: Dictionary = {}) -> void:
	if session == null or session.agent_manager == null:
		return
	var manager = session.agent_manager
	var launched: Array = []
	for id_value in manager.agent_ids():
		var agent_id: int = int(id_value)
		var snapshot: Dictionary = manager.agent_snapshot(agent_id)
		var state_id: int = int(snapshot.get("state_id", TrainingRunController.State.IDLE))
		if state_id != TrainingRunController.State.IDLE:
			launched.append(agent_id)
	# Focus survives the window between "launch issued" and "backend
	# published a non-idle status" — a freshly launched agent may still read
	# Idle here. Only an agent that no longer exists loses focus.
	if focused_agent_id >= 0 and not manager.has_agent(focused_agent_id):
		focused_agent_id = -1
	_sync_selector(launched)
	_sync_panels(launched)
	_empty.visible = launched.is_empty()
	var visible_panels: int = 0
	for agent_id in _panels:
		var panel: ControlCenterHeadlessMonitorPanel = _panels[agent_id]
		var shown: bool = focused_agent_id < 0 or int(agent_id) == focused_agent_id
		panel.visible = shown
		if shown:
			visible_panels += 1
			panel.refresh()
	_grid.columns = _column_count(visible_panels)


func panel_count() -> int:
	return _panels.size()


func _column_count(visible_panels: int) -> int:
	if focused_agent_id >= 0 or visible_panels <= 1:
		return 1
	var by_width: int = clampi(int(size.x / 560.0), 1, 3)
	return mini(by_width, visible_panels)


func _sync_selector(launched: Array) -> void:
	for child in _selector_row.get_children():
		_selector_row.remove_child(child)
		child.queue_free()
	if launched.size() <= 1 and focused_agent_id < 0:
		return
	var all_button := ControlCenterTheme.make_toggle(
		"All", focused_agent_id < 0, "Show every agent in a grid"
	)
	all_button.pressed.connect(show_all)
	_selector_row.add_child(all_button)
	for id_value in launched:
		var agent_id: int = int(id_value)
		var button := ControlCenterTheme.make_toggle(
			"Agent %d" % agent_id, focused_agent_id == agent_id, "Focus this agent"
		)
		button.pressed.connect(focus_agent.bind(agent_id))
		_selector_row.add_child(button)


func _sync_panels(launched: Array) -> void:
	for id_value in launched:
		var agent_id: int = int(id_value)
		if _panels.has(agent_id):
			continue
		var panel := ControlCenterHeadlessMonitorPanel.new()
		panel.setup(session, agent_id, true)
		panel.details_requested.connect(focus_agent)
		_grid.add_child(panel)
		_panels[agent_id] = panel
	for agent_id in _panels.keys():
		if launched.has(agent_id):
			continue
		var panel: ControlCenterHeadlessMonitorPanel = _panels[agent_id]
		if panel.get_parent() != null:
			panel.get_parent().remove_child(panel)
		panel.queue_free()
		_panels.erase(agent_id)

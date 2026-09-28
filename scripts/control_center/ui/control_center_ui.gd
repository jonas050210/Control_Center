## ControlCenterUI
##
## Root CanvasLayer of the Control Center (Phase 2). It owns the layout
## and the refresh loop; every panel is a child that receives the same
## immutable snapshot dictionary.
##
## Performance rules (Phase 15):
##   * the UI is only created when a real display server is present; the
##     entry point never instantiates it in headless mode
##   * panels refresh at REFRESH_HZ (default 10 Hz), not per physics tick
##   * in TRAINING mode the session returns an empty snapshot, so the
##     expensive telemetry build never runs and the panels show
##     "telemetry disabled"
##   * the centre of the screen is a mouse-ignoring overlay so the 3D view
##     keeps receiving gameplay input
class_name ControlCenterUI
extends CanvasLayer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterAgentPanel = preload("res://scripts/control_center/ui/agent_panel.gd")
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterControlsPanel = preload("res://scripts/control_center/ui/controls_panel.gd")
const ControlCenterHud = preload("res://scripts/control_center/ui/hud.gd")
const ControlCenterLogPanel = preload("res://scripts/control_center/ui/log_panel.gd")
const ControlCenterMetricsPanel = preload("res://scripts/control_center/ui/metrics_panel.gd")
const ControlCenterObservationPanel = preload(
	"res://scripts/control_center/ui/observation_panel.gd"
)
const ControlCenterPerceptionPanel = preload(
	"res://scripts/control_center/ui/perception_panel.gd"
)
const ControlCenterReplayPanel = preload("res://scripts/control_center/ui/replay_panel.gd")
const ControlCenterResultsPanel = preload("res://scripts/control_center/ui/results_panel.gd")
const ControlCenterSettingsPanel = preload("res://scripts/control_center/ui/settings_panel.gd")
const ControlCenterStatusBar = preload("res://scripts/control_center/ui/status_bar.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const ControlCenterTrainingConfigPanel = preload(
	"res://scripts/control_center/ui/training_config_panel.gd"
)
const ControlCenterTrainingControlsPanel = preload(
	"res://scripts/control_center/ui/training_controls_panel.gd"
)
const ControlCenterTrainingDashboardPanel = preload(
	"res://scripts/control_center/ui/training_dashboard_panel.gd"
)

const REFRESH_HZ: float = 10.0
const LEFT_PANEL_WIDTH: float = 310.0
const RIGHT_PANEL_WIDTH: float = 380.0

var session
var hud: ControlCenterHud
var status_bar: ControlCenterStatusBar
var agent_panel: ControlCenterAgentPanel
var perception_panel: ControlCenterPerceptionPanel
var observation_panel: ControlCenterObservationPanel
var results_panel: ControlCenterResultsPanel
var metrics_panel: ControlCenterMetricsPanel
var replay_panel: ControlCenterReplayPanel
var settings_panel: ControlCenterSettingsPanel
var controls_panel: ControlCenterControlsPanel
var log_panel: ControlCenterLogPanel
var training_config_panel: ControlCenterTrainingConfigPanel
var training_controls_panel: ControlCenterTrainingControlsPanel
var training_dashboard_panel: ControlCenterTrainingDashboardPanel
var training_monitor_panel: ControlCenterTrainingDashboardPanel

var _left_container: Control
var _right_container: Control
var _bottom_container: Control
var _centre_container: Control
var _tabs: TabContainer
var _body_split: VSplitContainer
var _left_split: HSplitContainer
var _right_split: HSplitContainer
var _refresh_accumulator: float = 0.0
var _last_snapshot: Dictionary = {}


func setup(p_session) -> void:
	session = p_session
	layer = 10
	_build_layout()
	_connect_session()
	refresh_now()


func _build_layout() -> void:
	var root := MarginContainer.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_theme_constant_override("margin_left", 6)
	root.add_theme_constant_override("margin_right", 6)
	root.add_theme_constant_override("margin_top", 6)
	root.add_theme_constant_override("margin_bottom", 6)
	add_child(root)

	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 6)
	column.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(column)

	status_bar = ControlCenterStatusBar.new()
	status_bar.setup(session)
	column.add_child(status_bar)

	training_controls_panel = ControlCenterTrainingControlsPanel.new()
	training_controls_panel.setup(session)
	column.add_child(training_controls_panel)

	_body_split = VSplitContainer.new()
	_body_split.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_body_split.mouse_filter = Control.MOUSE_FILTER_PASS
	column.add_child(_body_split)

	_left_split = HSplitContainer.new()
	_left_split.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_left_split.mouse_filter = Control.MOUSE_FILTER_PASS
	_body_split.add_child(_left_split)

	_left_container = _make_side_column(LEFT_PANEL_WIDTH)
	_left_split.add_child(_left_container)
	agent_panel = ControlCenterAgentPanel.new()
	agent_panel.setup()
	agent_panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_left_container.add_child(agent_panel)

	_right_split = HSplitContainer.new()
	_right_split.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_right_split.mouse_filter = Control.MOUSE_FILTER_PASS
	_left_split.add_child(_right_split)

	_centre_container = Control.new()
	_centre_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_centre_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_centre_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_right_split.add_child(_centre_container)
	hud = ControlCenterHud.new()
	hud.setup()
	_centre_container.add_child(hud)
	training_dashboard_panel = ControlCenterTrainingDashboardPanel.new()
	training_dashboard_panel.setup(session)
	training_dashboard_panel.set_anchors_preset(Control.PRESET_FULL_RECT)
	training_dashboard_panel.visible = false
	_centre_container.add_child(training_dashboard_panel)

	_right_container = _make_side_column(RIGHT_PANEL_WIDTH)
	_right_split.add_child(_right_container)
	_build_tabs(_right_container)

	_bottom_container = VBoxContainer.new()
	_bottom_container.custom_minimum_size = Vector2(0.0, 140.0)
	_bottom_container.add_theme_constant_override("separation", 6)
	_body_split.add_child(_bottom_container)

	controls_panel = ControlCenterControlsPanel.new()
	controls_panel.setup(session)
	_bottom_container.add_child(controls_panel)

	log_panel = ControlCenterLogPanel.new()
	log_panel.setup(session)
	_bottom_container.add_child(log_panel)

	_left_split.split_offset = session.config.left_dock_width
	_right_split.split_offset = -session.config.right_dock_width
	_body_split.split_offset = -session.config.bottom_dock_height
	_left_split.dragged.connect(_on_left_split_dragged)
	_right_split.dragged.connect(_on_right_split_dragged)
	_body_split.dragged.connect(_on_body_split_dragged)
	_apply_tile_order()
	_apply_panel_visibility()


func _make_side_column(width: float) -> Control:
	var column := VBoxContainer.new()
	column.custom_minimum_size = Vector2(width, 0.0)
	column.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_theme_constant_override("separation", 6)
	return column


func _build_tabs(parent: Control) -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_tabs.add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())
	_tabs.add_theme_font_size_override("font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	parent.add_child(_tabs)

	training_config_panel = ControlCenterTrainingConfigPanel.new()
	training_config_panel.setup(session)
	training_config_panel.configuration_changed.connect(_on_training_configuration_changed)
	_tabs.add_child(_wrap_scroll(training_config_panel, "Training"))

	training_monitor_panel = ControlCenterTrainingDashboardPanel.new()
	training_monitor_panel.setup(session)
	_tabs.add_child(_wrap_scroll(training_monitor_panel, "Run"))

	perception_panel = ControlCenterPerceptionPanel.new()
	perception_panel.setup(session)
	_tabs.add_child(_wrap_scroll(perception_panel, "Perception"))

	observation_panel = ControlCenterObservationPanel.new()
	observation_panel.setup()
	_tabs.add_child(_wrap_scroll(observation_panel, "Observation"))

	results_panel = ControlCenterResultsPanel.new()
	results_panel.setup(session)
	_tabs.add_child(_wrap_scroll(results_panel, "Results"))

	metrics_panel = ControlCenterMetricsPanel.new()
	metrics_panel.setup(session)
	_tabs.add_child(_wrap_scroll(metrics_panel, "Metrics"))

	replay_panel = ControlCenterReplayPanel.new()
	replay_panel.setup(session)
	_tabs.add_child(_wrap_scroll(replay_panel, "Replay"))

	settings_panel = ControlCenterSettingsPanel.new()
	settings_panel.setup(session)
	settings_panel.settings_rebuilt.connect(_on_settings_rebuilt)
	settings_panel.tile_layout_changed.connect(_on_tile_layout_changed)
	_tabs.add_child(_wrap_scroll(settings_panel, "Settings"))


func _wrap_scroll(content: Control, title: String) -> ScrollContainer:
	var scroll := ControlCenterTheme.make_scroll()
	scroll.name = title
	content.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(content)
	return scroll


func _connect_session() -> void:
	session.mode_changed.connect(_on_mode_changed)
	session.environments_rebuilt.connect(_on_environments_rebuilt)
	session.selection_changed.connect(_on_selection_changed)
	session.training_state_changed.connect(_on_training_state_changed)


func _process(delta: float) -> void:
	_refresh_accumulator += delta
	var interval: float = 1.0 / REFRESH_HZ
	if _refresh_accumulator < interval:
		return
	_refresh_accumulator = 0.0
	refresh_now()


## Rebuilds the snapshot once and pushes it to every visible panel.
func refresh_now() -> void:
	if session == null:
		return
	_apply_panel_visibility()
	var snapshot: Dictionary = session.build_snapshot(_snapshot_options())
	_last_snapshot = snapshot

	status_bar.refresh(snapshot)
	training_controls_panel.refresh(snapshot)
	controls_panel.refresh(snapshot)
	training_dashboard_panel.refresh(snapshot)
	var headless_dashboard: bool = (
		session.config.training_mode == ControlCenterConfig.TrainingMode.HEADLESS
	)
	training_dashboard_panel.visible = (
		headless_dashboard and session.config.is_tile_visible("training")
	)
	hud.refresh(snapshot, session.config.mode, session.human_input_enabled)
	hud.visible = not headless_dashboard and session.config.is_tile_visible("simulation")

	if _left_container.visible:
		agent_panel.refresh(snapshot)
	if _right_container.visible:
		_refresh_active_tab(snapshot)
	if _bottom_container.visible:
		log_panel.refresh(snapshot)


func _refresh_active_tab(snapshot: Dictionary) -> void:
	# Only the visible tab is refreshed; hidden tabs would burn CPU for
	# text nobody can read.
	match _tabs.current_tab:
		0:
			training_config_panel.refresh(snapshot)
		1:
			training_monitor_panel.refresh(snapshot)
		2:
			perception_panel.refresh(snapshot)
		3:
			observation_panel.refresh(snapshot)
		4:
			results_panel.refresh(snapshot)
		5:
			metrics_panel.refresh(snapshot)
		6:
			# Replay playback advances on wall-clock time, independently of
			# whether the live simulation is running or paused.
			replay_panel.advance(1.0 / REFRESH_HZ)
			replay_panel.refresh(snapshot)
		_:
			settings_panel.refresh(snapshot)


## Only the sections some visible panel will actually read are built.
## The left agent panel needs perception (mini-map + enemy list); the
## full observation table is only built for the Observation tab.
func _snapshot_options() -> Dictionary:
	var needs_perception: bool = _left_container.visible
	var needs_observation: bool = false
	if _right_container.visible:
		if _tabs.current_tab == 2:
			needs_perception = true
		elif _tabs.current_tab == 3:
			needs_observation = true
	return {"observation": needs_observation, "perception": needs_perception}


func _apply_panel_visibility() -> void:
	var headless_dashboard: bool = (
		session.config.training_mode == ControlCenterConfig.TrainingMode.HEADLESS
	)
	var viewport_width: float = get_viewport().get_visible_rect().size.x
	var inspector_visible: bool = session.config.is_tile_visible("inspector")
	_left_container.visible = (
		session.config.is_tile_visible("agent")
		and not session.is_training_mode()
		and not headless_dashboard
		and viewport_width >= 1200.0
	)
	_right_container.visible = (
		inspector_visible and (not headless_dashboard or viewport_width >= 900.0)
	)
	# On a narrow visual dashboard preserve configuration; on a narrow
	# headless dashboard preserve progress/metrics instead.
	_centre_container.visible = (
		headless_dashboard or viewport_width >= 900.0 or not inspector_visible
	)
	var controls_visible: bool = (
		session.config.is_tile_visible("controls") and not headless_dashboard
	)
	var logs_visible: bool = session.config.is_tile_visible("logs")
	_bottom_container.visible = (controls_visible or logs_visible) and not headless_dashboard
	training_controls_panel.visible = session.config.is_tile_visible("training")
	controls_panel.visible = controls_visible
	log_panel.visible = logs_visible


## Reordering is deliberately constrained to the vertical utility tiles.
## The simulation remains central and the inspector remains a dock, while
## Controls and Logs can swap without reparenting the 3D view or losing UI
## state.
func _apply_tile_order() -> void:
	if controls_panel == null or log_panel == null:
		return
	var controls_index: int = session.config.tile_order.find("controls")
	var logs_index: int = session.config.tile_order.find("logs")
	var controls_first: bool = controls_index <= logs_index
	_bottom_container.move_child(controls_panel, 0 if controls_first else 1)
	_bottom_container.move_child(log_panel, 1 if controls_first else 0)


func _on_left_split_dragged(offset: int) -> void:
	session.config.left_dock_width = clampi(offset, 220, 700)
	session.config.save_preferences()


func _on_right_split_dragged(_offset: int) -> void:
	session.config.right_dock_width = clampi(int(_right_container.size.x), 280, 800)
	session.config.save_preferences()


func _on_body_split_dragged(_offset: int) -> void:
	session.config.bottom_dock_height = clampi(int(_bottom_container.size.y), 140, 600)
	session.config.save_preferences()


func _unhandled_key_input(event: InputEvent) -> void:
	var key_event := event as InputEventKey
	if key_event == null or not key_event.pressed or key_event.echo:
		return
	# Gameplay keys belong to the human controller while input is armed.
	var gameplay_active: bool = session.is_human_mode() and session.human_input_enabled
	match key_event.keycode:
		KEY_F1:
			session.config.set_tile_visible(
				"agent", not session.config.is_tile_visible("agent")
			)
			session.config.save_preferences()
			_apply_panel_visibility()
			get_viewport().set_input_as_handled()
		KEY_F2:
			session.config.set_tile_visible(
				"inspector", not session.config.is_tile_visible("inspector")
			)
			session.config.save_preferences()
			_apply_panel_visibility()
			get_viewport().set_input_as_handled()
		KEY_F3:
			session.config.set_tile_visible(
				"logs", not session.config.is_tile_visible("logs")
			)
			session.config.save_preferences()
			_apply_panel_visibility()
			get_viewport().set_input_as_handled()
		KEY_SPACE:
			if not gameplay_active:
				session.toggle_running()
				get_viewport().set_input_as_handled()
		KEY_N:
			if not gameplay_active:
				session.request_single_step(1)
				get_viewport().set_input_as_handled()
		KEY_R:
			if not gameplay_active:
				session.reset_selected_environment(true)
				get_viewport().set_input_as_handled()
		KEY_TAB:
			if not gameplay_active and _tabs.get_tab_count() > 0:
				_tabs.current_tab = (_tabs.current_tab + 1) % _tabs.get_tab_count()
				get_viewport().set_input_as_handled()


func _on_mode_changed(_mode: int) -> void:
	_apply_panel_visibility()
	refresh_now()


func _on_environments_rebuilt() -> void:
	status_bar.refresh_options()
	refresh_now()


func _on_selection_changed(_environment_index: int, _agent_slot: int) -> void:
	refresh_now()


func _on_settings_rebuilt() -> void:
	status_bar.refresh_options()
	refresh_now()


func _on_training_configuration_changed() -> void:
	_apply_tile_order()
	_apply_panel_visibility()
	refresh_now()


func _on_training_state_changed(_state: int) -> void:
	_apply_panel_visibility()
	refresh_now()


func _on_tile_layout_changed() -> void:
	_apply_tile_order()
	_apply_panel_visibility()
	refresh_now()

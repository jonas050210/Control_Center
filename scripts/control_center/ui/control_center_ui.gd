## ControlCenterUI
##
## Root CanvasLayer of the Control Center. It owns the layout and the
## refresh loop; every panel is a child that receives the same immutable
## snapshot dictionary.
##
## The dashboard is organised as PAGES behind one persistent navigation
## rail (HOME / AGENTS / HEADLESS / TRAINING / SIMULATION / ANALYTICS /
## HISTORY / SETTINGS). The simulation page contains the classic 3D view
## with its docks; every other page is an opaque full-area panel. Only the
## active page's content is refreshed.
##
## Performance rules (Phase 15):
##   * the UI is only created when a real display server is present; the
##     entry point never instantiates it in headless mode
##   * panels refresh at REFRESH_HZ (default 10 Hz), not per physics tick
##   * in TRAINING mode the session returns an empty snapshot, so the
##     expensive telemetry build never runs and the panels show
##     "telemetry disabled"
##   * the centre of the simulation page is a mouse-ignoring overlay so
##     the 3D view keeps receiving gameplay input
class_name ControlCenterUI
extends CanvasLayer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterAgentPanel = preload("res://scripts/control_center/ui/agent_panel.gd")
const ControlCenterAgentsPanel = preload("res://scripts/control_center/ui/agents_panel.gd")
const ControlCenterAnalyticsPanel = preload("res://scripts/control_center/ui/analytics_panel.gd")
const ControlCenterAmbientBackdrop = preload("res://scripts/control_center/ui/ambient_backdrop.gd")
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterControlsPanel = preload("res://scripts/control_center/ui/controls_panel.gd")
const ControlCenterHeadlessPanel = preload("res://scripts/control_center/ui/headless_panel.gd")
const ControlCenterHistoryPanel = preload("res://scripts/control_center/ui/history_panel.gd")
const ControlCenterHomePanel = preload("res://scripts/control_center/ui/home_panel.gd")
const ControlCenterHud = preload("res://scripts/control_center/ui/hud.gd")
const ControlCenterLogPanel = preload("res://scripts/control_center/ui/log_panel.gd")
const ControlCenterMetricsPanel = preload("res://scripts/control_center/ui/metrics_panel.gd")
const ControlCenterObservationPanel = preload(
	"res://scripts/control_center/ui/observation_panel.gd"
)
const ControlCenterPerceptionPanel = preload("res://scripts/control_center/ui/perception_panel.gd")
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
const ControlCenterTrainingLaunchPanel = preload(
	"res://scripts/control_center/ui/training_launch_panel.gd"
)
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

const REFRESH_HZ: float = 10.0
const LEFT_PANEL_WIDTH: float = 310.0
const RIGHT_PANEL_WIDTH: float = 380.0

## Navigation entries: page id -> label, in display order.
const PAGES: Array = [
	["home", "Home"],
	["agents", "Agents"],
	["headless", "Headless"],
	["training", "Training"],
	["simulation", "Simulation"],
	["analytics", "Analytics"],
	["history", "History"],
	["settings", "Settings"],
]

var session
var ambient_backdrop: ControlCenterAmbientBackdrop
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
var training_launch_panel: ControlCenterTrainingLaunchPanel
var home_panel: ControlCenterHomePanel
var agents_panel: ControlCenterAgentsPanel
var headless_panel: ControlCenterHeadlessPanel
var analytics_panel: ControlCenterAnalyticsPanel
var history_panel: ControlCenterHistoryPanel

var _left_container: Control
var _right_container: Control
var _bottom_container: Control
var _centre_container: Control
var _tabs: TabContainer
var _body_split: VSplitContainer
var _left_split: HSplitContainer
var _right_split: HSplitContainer
var _page_container: Control
var _pages: Dictionary = {}  # page id -> Control
var _nav_buttons: Dictionary = {}  # page id -> Button
var _refresh_accumulator: float = 0.0
var _last_snapshot: Dictionary = {}


func setup(p_session) -> void:
	session = p_session
	layer = 10
	_build_layout()
	_connect_session()
	set_page(session.config.active_page, false)
	refresh_now()


func _build_layout() -> void:
	# This is intentionally a local Control Center treatment, not a claim
	# about TTK Testing's in-game presentation. It is built only with the
	# interactive UI and therefore stays out of the headless/RL hot path.
	ambient_backdrop = ControlCenterAmbientBackdrop.new()
	ambient_backdrop.motion_enabled = session.config.ui_motion_enabled
	add_child(ambient_backdrop)

	var root := MarginContainer.new()
	root.theme = ControlCenterTheme.build_theme()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_theme_constant_override("margin_left", 12)
	root.add_theme_constant_override("margin_right", 12)
	root.add_theme_constant_override("margin_top", 12)
	root.add_theme_constant_override("margin_bottom", 12)
	add_child(root)

	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 10)
	column.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(column)

	status_bar = ControlCenterStatusBar.new()
	status_bar.setup(session)
	column.add_child(status_bar)

	training_controls_panel = ControlCenterTrainingControlsPanel.new()
	training_controls_panel.setup(session)
	column.add_child(training_controls_panel)

	# Keep navigation visually separate from the working area. A persistent
	# rail is easier to scan than a dense row of eight equally weighted tabs.
	var workspace := HBoxContainer.new()
	workspace.size_flags_vertical = Control.SIZE_EXPAND_FILL
	workspace.add_theme_constant_override("separation", 12)
	workspace.mouse_filter = Control.MOUSE_FILTER_PASS
	column.add_child(workspace)
	workspace.add_child(_build_navigation())

	_page_container = Control.new()
	_page_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_page_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_page_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	workspace.add_child(_page_container)

	_build_simulation_page()
	_build_dashboard_pages()

	_left_split.split_offset = session.config.left_dock_width
	_right_split.split_offset = -session.config.right_dock_width
	_body_split.split_offset = -session.config.bottom_dock_height
	_left_split.dragged.connect(_on_left_split_dragged)
	_right_split.dragged.connect(_on_right_split_dragged)
	_body_split.dragged.connect(_on_body_split_dragged)
	_apply_tile_order()
	_apply_panel_visibility()


func _build_navigation() -> Control:
	var rail := PanelContainer.new()
	rail.custom_minimum_size = Vector2(176.0, 0.0)
	rail.add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var items := VBoxContainer.new()
	items.add_theme_constant_override("separation", 5)
	rail.add_child(items)
	items.add_child(
		ControlCenterTheme.make_label(
			"SANDBOXAI", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	items.add_child(
		ControlCenterTheme.make_label(
			"CONTROL SURFACE", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_ACCENT
		)
	)
	items.add_child(ControlCenterTheme.make_separator())
	items.add_child(
		ControlCenterTheme.make_label(
			"WORKSPACE", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	for index in range(PAGES.size()):
		if index == 5:
			items.add_child(ControlCenterTheme.make_separator())
			items.add_child(
				ControlCenterTheme.make_label(
					"Insights", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
				)
			)
		elif index == 7:
			items.add_child(ControlCenterTheme.make_separator())
		var page_value: Array = PAGES[index]
		var page_id: String = str(page_value[0])
		var button := ControlCenterTheme.make_toggle(
			str(page_value[1]), false, "Open the %s page" % str(page_value[1])
		)
		button.alignment = HORIZONTAL_ALIGNMENT_LEFT
		button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		button.pressed.connect(_on_nav_pressed.bind(page_id))
		items.add_child(button)
		_nav_buttons[page_id] = button
	var spacer := Control.new()
	spacer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	items.add_child(spacer)
	items.add_child(
		ControlCenterTheme.make_label(
			"F1–F3 toggle panels",
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)
	return rail


## The classic simulation operator view (3D view + docks) as one page.
func _build_simulation_page() -> void:
	var page := VBoxContainer.new()
	page.name = "SimulationPage"
	page.set_anchors_preset(Control.PRESET_FULL_RECT)
	page.add_theme_constant_override("separation", 6)
	page.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_page_container.add_child(page)
	_pages["simulation"] = page

	_body_split = VSplitContainer.new()
	_body_split.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_body_split.mouse_filter = Control.MOUSE_FILTER_PASS
	page.add_child(_body_split)

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


## Dashboard pages around the simulation: overview, agent management,
## live headless monitoring, training launch, analytics, history and
## settings.
func _build_dashboard_pages() -> void:
	home_panel = ControlCenterHomePanel.new()
	home_panel.setup(session)
	home_panel.open_agent_requested.connect(_on_open_agent_requested)
	_add_page("home", home_panel)

	agents_panel = ControlCenterAgentsPanel.new()
	agents_panel.setup(session)
	agents_panel.open_agent_requested.connect(_on_open_agent_requested)
	agents_panel.configure_requested.connect(func(): set_page("training"))
	_add_page("agents", agents_panel)

	headless_panel = ControlCenterHeadlessPanel.new()
	headless_panel.setup(session)
	_add_page("headless", headless_panel)

	training_config_panel = ControlCenterTrainingConfigPanel.new()
	training_config_panel.setup(session)
	training_config_panel.configuration_changed.connect(_on_training_configuration_changed)
	training_launch_panel = ControlCenterTrainingLaunchPanel.new()
	training_launch_panel.setup(session, training_config_panel)
	training_launch_panel.training_started.connect(_on_training_started)
	_add_page("training", training_launch_panel)

	analytics_panel = ControlCenterAnalyticsPanel.new()
	analytics_panel.setup(session)
	_add_page("analytics", analytics_panel)

	history_panel = ControlCenterHistoryPanel.new()
	history_panel.setup(session)
	_add_page("history", history_panel)

	settings_panel = ControlCenterSettingsPanel.new()
	settings_panel.setup(session)
	settings_panel.settings_rebuilt.connect(_on_settings_rebuilt)
	settings_panel.tile_layout_changed.connect(_on_tile_layout_changed)
	settings_panel.presentation_changed.connect(_on_presentation_changed)
	var settings_page := PanelContainer.new()
	settings_page.add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var settings_scroll := ControlCenterTheme.make_scroll()
	settings_page.add_child(settings_scroll)
	settings_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	settings_scroll.add_child(settings_panel)
	_add_page("settings", settings_page)


func _add_page(page_id: String, page: Control) -> void:
	page.set_anchors_preset(Control.PRESET_FULL_RECT)
	page.visible = false
	_page_container.add_child(page)
	_pages[page_id] = page


func _make_side_column(width: float) -> Control:
	var column := VBoxContainer.new()
	column.custom_minimum_size = Vector2(width, 0.0)
	column.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_theme_constant_override("separation", 10)
	return column


func _build_tabs(parent: Control) -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_tabs.add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())
	_tabs.add_theme_font_size_override("font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	parent.add_child(_tabs)

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
	session.agent_manager.agents_changed.connect(_on_agents_changed)


func _process(delta: float) -> void:
	_refresh_accumulator += delta
	var interval: float = 1.0 / REFRESH_HZ
	if _refresh_accumulator < interval:
		return
	_refresh_accumulator = 0.0
	refresh_now()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


func active_page() -> String:
	return session.config.active_page if session != null else "home"


func set_page(page_id: String, persist: bool = true) -> void:
	var resolved: String = page_id if _pages.has(page_id) else ControlCenterConfig.DEFAULT_PAGE
	session.config.active_page = resolved
	if persist:
		session.config.save_preferences()
	var selected_page: Control = _pages[resolved] as Control
	for existing_id in _pages:
		(_pages[existing_id] as Control).visible = str(existing_id) == resolved
	# Motion is an operator preference. Disabling it keeps navigation
	# immediate and also stops the ambient backdrop redraw loop.
	if session.config.ui_motion_enabled:
		selected_page.modulate.a = 0.0
		var transition := create_tween()
		transition.set_ease(Tween.EASE_OUT).set_trans(Tween.TRANS_QUAD)
		transition.tween_property(selected_page, "modulate:a", 1.0, 0.14)
	else:
		selected_page.modulate.a = 1.0
	for nav_id in _nav_buttons:
		(_nav_buttons[nav_id] as Button).button_pressed = str(nav_id) == resolved
	if resolved == "history":
		# History is disk state; rescan when the operator opens the page.
		history_panel.reload_runs()
	refresh_now()


func _on_nav_pressed(page_id: String) -> void:
	set_page(page_id)


func _on_open_agent_requested(agent_id: int) -> void:
	headless_panel.focus_agent(agent_id)
	set_page("headless")


func _on_training_started(agent_id: int) -> void:
	# Follow the launch to the live monitor, as a real dashboard would.
	headless_panel.focus_agent(agent_id)
	set_page("headless")


func _on_agents_changed() -> void:
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
	_refresh_active_page(snapshot)


func _refresh_active_page(snapshot: Dictionary) -> void:
	match active_page():
		"home":
			home_panel.refresh(snapshot)
		"agents":
			agents_panel.refresh(snapshot)
		"headless":
			headless_panel.refresh(snapshot)
		"training":
			training_launch_panel.refresh(snapshot)
		"analytics":
			analytics_panel.refresh(snapshot)
		"history":
			history_panel.refresh(snapshot)
		"settings":
			settings_panel.refresh(snapshot)
		_:
			pass


func _refresh_active_tab(snapshot: Dictionary) -> void:
	# Only the visible tab is refreshed; hidden tabs would burn CPU for
	# text nobody can read.
	match _tabs.current_tab:
		0:
			training_monitor_panel.refresh(snapshot)
		1:
			perception_panel.refresh(snapshot)
		2:
			observation_panel.refresh(snapshot)
		3:
			results_panel.refresh(snapshot)
		4:
			metrics_panel.refresh(snapshot)
		5:
			# Replay playback advances on wall-clock time, independently of
			# whether the live simulation is running or paused.
			replay_panel.advance(1.0 / REFRESH_HZ)
			replay_panel.refresh(snapshot)
		_:
			pass


## Only the sections some visible panel will actually read are built.
## The left agent panel needs perception (mini-map + enemy list); the
## full observation table is only built for the Observation tab.
func _snapshot_options() -> Dictionary:
	var needs_perception: bool = _left_container.visible
	var needs_observation: bool = false
	if _right_container.visible:
		if _tabs.current_tab == 1:
			needs_perception = true
		elif _tabs.current_tab == 2:
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
		and viewport_width >= 1400.0
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
	var training_state: int = int(session.training_run.state)
	var training_active: bool = (
		training_state
		in [
			TrainingRunController.State.STARTING,
			TrainingRunController.State.RUNNING,
			TrainingRunController.State.PAUSED,
			TrainingRunController.State.STOPPING,
		]
	)
	training_controls_panel.visible = (
		session.config.is_tile_visible("training")
		and (active_page() == "training" or training_active)
	)
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
			session.config.set_tile_visible("agent", not session.config.is_tile_visible("agent"))
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
			session.config.set_tile_visible("logs", not session.config.is_tile_visible("logs"))
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


func _on_presentation_changed() -> void:
	if ambient_backdrop != null:
		ambient_backdrop.set_motion_enabled(session.config.ui_motion_enabled)
	refresh_now()

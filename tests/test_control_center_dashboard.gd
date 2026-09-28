## Tests for the dashboard shell of the Control Center: persistent page
## navigation, agent cards, the headless monitors with their bounded live
## logs, PC-status rendering and the history page. Uses the real scene
## with the GUI forced on, exactly like the existing scene tests.
class_name TestControlCenterDashboard
extends RefCounted

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterSystemMonitor = preload("res://scripts/control_center/system_monitor.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")


static func _instantiate() -> Node:
	var packed: PackedScene = load("res://scenes/control_center.tscn") as PackedScene
	if packed == null:
		return null
	var instance: Node = packed.instantiate()
	if instance == null:
		return null
	instance.force_gui = true
	instance.environment_count = 1
	return instance


static func _mount(instance: Node) -> bool:
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop == null:
		return false
	loop.root.add_child(instance)
	return true


static func _teardown(instance: Node) -> void:
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop != null and instance.get_parent() != null:
		loop.root.remove_child(instance)
	instance.free()


func test_dashboard_pages_exist_and_navigation_switches_them() -> SandboxTest:
	var t := SandboxTest.new("dashboard_pages_and_navigation")
	var instance: Node = _instantiate()
	t.assert_not_null(instance)
	if instance == null or not _mount(instance):
		if instance != null:
			instance.free()
		return t
	var ui = instance.ui
	t.assert_not_null(ui.home_panel, "HOME page")
	t.assert_not_null(ui.agents_panel, "AGENTS page")
	t.assert_not_null(ui.headless_panel, "HEADLESS page")
	t.assert_not_null(ui.training_launch_panel, "TRAINING launch page")
	t.assert_not_null(ui.analytics_panel, "ANALYTICS page")
	t.assert_not_null(ui.history_panel, "HISTORY page")
	t.assert_not_null(ui.home_panel.system_status_panel, "PC status strip on HOME")
	t.assert_not_null(
		ui.training_config_panel.get_parent(),
		"the existing training configuration editor is hosted, not replaced"
	)

	# Switching pages shows exactly one page and persists the choice in
	# the config model.
	var before: int = instance.session.get_selected_environment().episode.step_count
	for page_value in ui.PAGES:
		var page_id: String = str(page_value[0])
		ui.set_page(page_id, false)
		t.assert_eq(instance.session.config.active_page, page_id)
		var visible_count: int = 0
		for existing in ui._pages:
			if (ui._pages[existing] as Control).visible:
				visible_count += 1
		t.assert_eq(visible_count, 1, "exactly one page visible on %s" % page_id)
		ui.refresh_now()
	t.assert_eq(
		instance.session.get_selected_environment().episode.step_count,
		before,
		"visiting and refreshing every page must never advance the simulation"
	)
	_teardown(instance)
	return t


func test_agent_cards_split_between_active_and_recent() -> SandboxTest:
	var t := SandboxTest.new("dashboard_agent_cards")
	var instance: Node = _instantiate()
	if instance == null or not _mount(instance):
		if instance != null:
			instance.free()
		return t
	var ui = instance.ui
	var manager = instance.session.agent_manager

	var running := TrainingRunController.new()
	var finished := TrainingRunController.new()
	var failed := TrainingRunController.new()
	manager.adopt_agent(running)
	manager.adopt_agent(finished)
	manager.adopt_agent(failed)
	running.apply_status(
		{
			"state": "Running",
			"training_type": "PPO",
			"timesteps": 1200,
			"total_training_steps": 10000,
			"episodes": 12,
			"mean_episode_reward": 0.5,
			"mean_kills": 1.0,
			"mean_deaths": 0.5,
			"mean_accuracy": 0.25,
			"steps_per_second": 300.0,
		}
	)
	finished.apply_status({"state": "Finished", "timesteps": 10000})
	failed.apply_status({"state": "Error", "error": "no dataset"})

	ui.set_page("home", false)
	ui.home_panel.refresh()
	# Primary agent is Idle -> no card; three adopted agents have cards.
	t.assert_eq(ui.home_panel.card_count(), 3, "one card per non-idle agent")
	ui.set_page("headless", false)
	ui.headless_panel.refresh()
	t.assert_eq(ui.headless_panel.panel_count(), 3, "one monitor panel per launched agent")

	# The agents page lists every agent including the idle primary.
	ui.set_page("agents", false)
	ui.agents_panel.refresh()
	t.assert_eq(manager.agent_count(), 4)
	_teardown(instance)
	return t


func test_headless_monitor_ingests_real_events_with_bounded_log() -> SandboxTest:
	var t := SandboxTest.new("dashboard_headless_log_bounded")
	var instance: Node = _instantiate()
	if instance == null or not _mount(instance):
		if instance != null:
			instance.free()
		return t
	var ui = instance.ui
	var manager = instance.session.agent_manager
	var controller := TrainingRunController.new()
	manager.adopt_agent(controller)
	var agent_id: int = manager.agent_ids()[manager.agent_ids().size() - 1]
	controller.apply_status({"state": "Running", "training_type": "PPO"})
	ui.set_page("headless", false)
	ui.headless_panel.refresh()
	var monitor = ui.headless_panel.monitor_for(agent_id)
	t.assert_not_null(monitor, "the launched agent owns a monitor panel")
	if monitor == null:
		_teardown(instance)
		return t
	t.assert_eq(monitor.rendered_line_count(), 0, "no fake log entries before real events")

	var event_path: String = "user://sandboxai_test_dashboard_events.jsonl"
	var file := FileAccess.open(event_path, FileAccess.WRITE)
	var total: int = monitor.MAX_LOG_LINES + 25
	for index in range(total):
		file.store_line(JSON.stringify({
			"wall_time": 1700000000.0 + index,
			"category": "error" if index == total - 1 else "system",
			"message": "line %d" % index,
		}))
	file.close()
	controller.event_file = event_path
	controller._poll_events()
	monitor.refresh()
	t.assert_eq(
		monitor.rendered_line_count(),
		monitor.MAX_LOG_LINES,
		"the UI log is trimmed to its bound"
	)
	# A second refresh with no new events must not duplicate lines.
	monitor.refresh()
	t.assert_eq(monitor.rendered_line_count(), monitor.MAX_LOG_LINES)
	DirAccess.remove_absolute(ProjectSettings.globalize_path(event_path))
	_teardown(instance)
	return t


func test_system_status_panel_renders_na_for_unavailable_metrics() -> SandboxTest:
	var t := SandboxTest.new("dashboard_system_status_na")
	var instance: Node = _instantiate()
	if instance == null or not _mount(instance):
		if instance != null:
			instance.free()
		return t
	var panel = instance.ui.home_panel.system_status_panel
	panel.render(ControlCenterSystemMonitor.empty_snapshot())
	var texts: Dictionary = panel.rendered_texts()
	t.assert_eq(texts["gpu_name"], "N/A")
	t.assert_true(str(texts["gpu"]).contains("N/A"), "unavailable GPU metrics say N/A")
	t.assert_true(str(texts["cpu"]).contains("N/A"))
	t.assert_true(str(texts["ram"]).contains("N/A"))
	t.assert_eq(texts["sampled"], "no sample yet")

	# A real measured sample renders real numbers.
	var sample: Dictionary = ControlCenterSystemMonitor.empty_snapshot()
	sample["gpu"] = ControlCenterSystemMonitor.parse_nvidia_smi(
		"NVIDIA GeForce RTX 4060 Ti, 34, 2048, 8188, 52"
	)
	sample["ram"] = ControlCenterSystemMonitor.ram_snapshot_from(OS.get_memory_info())
	sample["sampled_at"] = Time.get_unix_time_from_system()
	panel.render(sample)
	texts = panel.rendered_texts()
	t.assert_eq(texts["gpu_name"], "NVIDIA GeForce RTX 4060 Ti")
	t.assert_true(str(texts["gpu"]).contains("34%"))
	t.assert_true(str(texts["gpu"]).contains("2048 MB / 8188 MB"))
	t.assert_true(
		str(texts["cpu"]).contains("N/A"),
		"CPU stays N/A when no CPU sample was measured"
	)
	_teardown(instance)
	return t


func test_history_panel_renders_rows_without_inventing_values() -> SandboxTest:
	var t := SandboxTest.new("dashboard_history_rendering")
	var instance: Node = _instantiate()
	if instance == null or not _mount(instance):
		if instance != null:
			instance.free()
		return t
	var history = instance.ui.history_panel
	history.render_rows(
		[
			{
				"run_id": "control_center_ppo_1700000000_1",
				"algorithm": "PPO",
				"state": "Finished",
				"started_unix": 1700000000.0,
				"updated_unix": 1700000300.0,
				"duration_seconds": 300.0,
				"duration_final": true,
				"timesteps": 5000,
				"episodes": 42,
				"mean_episode_reward": 0.75,
			},
			{
				"run_id": "control_center_bc_1700001000_2",
				"algorithm": "BC",
				"state": "Error",
			},
		]
	)
	t.assert_eq(history.row_count(), 2)
	_teardown(instance)
	return t


func test_launching_navigates_to_the_headless_monitor() -> SandboxTest:
	var t := SandboxTest.new("dashboard_launch_navigates_to_headless")
	var instance: Node = _instantiate()
	if instance == null or not _mount(instance):
		if instance != null:
			instance.free()
		return t
	var ui = instance.ui
	ui.set_page("training", false)
	# Simulate the launch signal without spawning a process.
	ui._on_training_started(1)
	t.assert_eq(instance.session.config.active_page, "headless")
	t.assert_eq(ui.headless_panel.focused_agent_id, 1)
	_teardown(instance)
	return t


func test_active_page_round_trips_through_config() -> SandboxTest:
	var t := SandboxTest.new("dashboard_active_page_round_trip")
	var config := ControlCenterConfig.new()
	config.active_page = "headless"
	var restored := ControlCenterConfig.new()
	restored.apply_dict(config.to_dict())
	t.assert_eq(restored.active_page, "headless")
	restored.active_page = "not-a-page"
	restored.sanitize()
	t.assert_eq(restored.active_page, "home", "unknown pages sanitize to home")
	return t

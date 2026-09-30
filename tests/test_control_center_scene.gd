## Tests for the Control Center scene itself (scenes/control_center.tscn).
##
## Two properties matter here:
##   1. In a headless process the scene boots the SIMULATION ONLY — no
##      CanvasLayer, no Controls, no EnvironmentViews. That is what keeps
##      the Control Center off the RL training path.
##   2. With the GUI explicitly forced on, the whole presentation stack
##      (status bar, docks, panels, HUD, camera rig, 3D overlay) builds
##      without errors and every panel survives a refresh.
class_name TestControlCenterScene
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


static func _instantiate(force_gui: bool, environments: int) -> Node:
	var packed: PackedScene = load("res://scenes/control_center.tscn") as PackedScene
	if packed == null:
		return null
	var instance: Node = packed.instantiate()
	if instance == null:
		return null
	instance.force_gui = force_gui
	instance.environment_count = environments
	return instance


static func _teardown(instance: Node) -> void:
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop != null and instance.get_parent() != null:
		loop.root.remove_child(instance)
	instance.free()


func test_scene_loads_and_boots_simulation_only_when_headless() -> SandboxTest:
	var t := SandboxTest.new("control_center_scene_headless_is_simulation_only")
	var instance: Node = _instantiate(false, SandboxConfig.DEFAULT_ENVIRONMENT_COUNT)
	t.assert_not_null(instance, "scenes/control_center.tscn should load and instantiate")
	if instance == null:
		return t
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	t.assert_not_null(loop)
	if loop == null:
		instance.free()
		return t

	loop.root.add_child(instance)  # triggers _ready() synchronously

	var headless: bool = DisplayServer.get_name() == "headless"
	if headless:
		t.assert_false(instance.presentation_enabled, "no display server means no presentation")
		t.assert_null(instance.ui, "the GUI must not be constructed in a headless process")
		t.assert_null(instance.get_node_or_null("ControlCenterUI"))
		t.assert_null(instance.spectator, "no camera rig in a headless process")
		t.assert_false(
			instance.session.simulation_manager.create_visuals,
			"no EnvironmentViews are built in a headless process"
		)
		for view in instance.session.simulation_manager.views:
			t.assert_null(view)

	t.assert_not_null(instance.session, "the session is always created")
	t.assert_eq(
		instance.session.simulation_manager.environments.size(),
		SandboxConfig.DEFAULT_ENVIRONMENT_COUNT,
		"the scene builds the configured environment count"
	)
	t.assert_false(
		instance.session.simulation_manager.auto_tick,
		"the session owns stepping so pause/step/speed stay exact"
	)
	_teardown(instance)
	return t


func test_full_gui_builds_and_refreshes_without_errors() -> SandboxTest:
	var t := SandboxTest.new("control_center_scene_gui_builds")
	var instance: Node = _instantiate(true, 2)
	t.assert_not_null(instance)
	if instance == null:
		return t
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop == null:
		instance.free()
		return t
	loop.root.add_child(instance)

	t.assert_true(instance.presentation_enabled, "force_gui builds the presentation layer")
	var ui = instance.ui
	t.assert_not_null(ui, "the Control Center UI root exists")
	if ui == null:
		_teardown(instance)
		return t

	t.assert_not_null(ui.status_bar, "top status bar")
	t.assert_not_null(ui.agent_panel, "live agent view panel")
	t.assert_not_null(ui.perception_panel, "what-does-the-AI-see panel")
	t.assert_not_null(ui.observation_panel, "observation inspector panel")
	t.assert_not_null(ui.results_panel, "results/metrics panel")
	t.assert_not_null(ui.metrics_panel, "research metrics panel")
	t.assert_not_null(ui.replay_panel, "replay panel")
	t.assert_not_null(ui.settings_panel, "settings panel")
	t.assert_not_null(ui.training_config_panel, "real backend training configuration")
	t.assert_not_null(ui.training_controls_panel, "training lifecycle controls")
	t.assert_not_null(ui.training_dashboard_panel, "headless training dashboard")
	t.assert_not_null(ui.training_monitor_panel, "visual-mode training metrics")
	t.assert_not_null(ui.controls_panel, "simulation controls")
	t.assert_not_null(ui.log_panel, "event log panel")
	t.assert_not_null(ui.hud, "in-world HUD")
	t.assert_not_null(instance.spectator, "camera rig")
	t.assert_not_null(instance.overlay, "3D perception overlay")

	# Stepping plus a refresh must exercise every panel without touching
	# the simulation result.
	instance.session.advance(5)
	var before: int = instance.session.get_selected_environment().episode.step_count
	for tab in range(ui._tabs.get_tab_count()):
		ui._tabs.current_tab = tab
		ui.refresh_now()
	t.assert_eq(
		instance.session.get_selected_environment().episode.step_count,
		before,
		"refreshing the UI must never advance the simulation"
	)

	# Only the selected environment is rendered.
	var manager = instance.session.simulation_manager
	t.assert_not_null(manager.views[instance.session.config.selected_environment])
	for index in range(manager.views.size()):
		var view = manager.views[index]
		if view != null and index != instance.session.config.selected_environment:
			t.assert_false(view.visible, "unselected environments are never rendered")

	_teardown(instance)
	return t


func test_gui_mode_switch_keeps_panels_alive() -> SandboxTest:
	var t := SandboxTest.new("control_center_scene_mode_switch")
	var instance: Node = _instantiate(true, 2)
	if instance == null:
		return t
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop == null:
		instance.free()
		return t
	loop.root.add_child(instance)

	for mode in [
		ControlCenterConfig.Mode.HUMAN,
		ControlCenterConfig.Mode.TRAINING,
		ControlCenterConfig.Mode.WATCH
	]:
		instance.session.set_mode(mode)
		instance.ui.refresh_now()
		t.assert_eq(instance.session.config.mode, mode)
	t.assert_true(instance.ui.hud.visible, "the HUD returns when leaving TRAINING mode")
	t.assert_eq(
		instance.session.simulation_manager.environments.size(), 2, "no rebuild on mode switch"
	)
	_teardown(instance)
	return t


## Regression: the transport panel used C's "%g" conversion, which GDScript's
## String formatter does not implement. Every Control Center build printed
## "String formatting error: unsupported format character." and the speed
## buttons/tooltips ended up with an error string instead of "0.25x".
func test_speed_labels_format_without_unsupported_conversions() -> SandboxTest:
	var t := SandboxTest.new("control_center_speed_labels_format")
	t.assert_eq(ControlCenterTheme.format_number(0.25), "0.25")
	t.assert_eq(ControlCenterTheme.format_number(1.0), "1")
	t.assert_eq(ControlCenterTheme.format_number(16.0), "16")
	t.assert_eq(ControlCenterTheme.format_number(0.05), "0.05")
	for preset_value in ControlCenterConfig.SPEED_PRESETS:
		var label: String = "%sx" % ControlCenterTheme.format_number(float(preset_value))
		t.assert_false(label.contains("error"), "speed label must render: %s" % label)
		t.assert_true(label.ends_with("x"))

	# No Control Center UI script may reintroduce an unsupported conversion.
	# Comment lines are skipped so this file's own explanation of the bug
	# does not trip the check.
	var unsupported: Array = ["%g", "%e", "%i", "%u"]
	for path in _ui_script_paths():
		var script: Script = load(path) as Script
		if script == null:
			continue
		for raw_line in script.source_code.split("\n"):
			var line: String = str(raw_line).strip_edges()
			if line.begins_with("#"):
				continue
			for token in unsupported:
				t.assert_false(
					line.contains(token),
					"%s uses the unsupported format conversion %s" % [path, token]
				)
	return t


static func _ui_script_paths() -> Array:
	var paths: Array = []
	var dir: DirAccess = DirAccess.open("res://scripts/control_center/ui")
	if dir == null:
		return paths
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if not dir.current_is_dir() and file_name.ends_with(".gd"):
			paths.append("res://scripts/control_center/ui/%s" % file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	paths.sort()
	return paths

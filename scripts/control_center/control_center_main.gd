## ControlCenterMain
##
## Entry point of the Control Center scene (`scenes/control_center.tscn`).
## It wires together the three layers and nothing else:
##
##   simulation   ControlCenterSession -> SimulationManager -> EnvironmentCore
##   presentation ControlCenterUI, SpectatorCamera, PerceptionOverlay3D
##   input        the existing HumanController, owned by the session
##
## Everything visual is created only when a real display server is present,
## so `--headless` runs (and therefore RL training) construct the
## simulation exactly as before and pay nothing for the GUI.
##
## Command line (after `--`), e.g.
##   godot --path . scenes/control_center.tscn -- --mode=human --env-count=2
class_name ControlCenterMain
extends Node3D

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterSession = preload("res://scripts/control_center/control_center_session.gd")
const ControlCenterUI = preload("res://scripts/control_center/ui/control_center_ui.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentView = preload("res://scripts/env/environment_view.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const PerceptionOverlay3D = preload("res://scripts/control_center/perception_overlay_3d.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SpectatorCamera = preload("res://scripts/control_center/spectator_camera.gd")

## The 3D perception overlay refreshes at a lower rate than the render
## loop; nothing it draws changes meaningfully faster than this.
const OVERLAY_HZ: float = 20.0

@export var start_mode: int = ControlCenterConfig.Mode.WATCH
@export var environment_count: int = SandboxConfig.DEFAULT_ENVIRONMENT_COUNT
@export var enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT
## Full curriculum range (1..11), matching ControlCenterConfig.sanitize(),
## ControlCenterSession.set_curriculum_level() and the settings panel's own
## level picker. It used to clamp to 5, which silently downgraded
## `--curriculum-level=8` even though the scenario presets go up to 10.
@export_range(1, 11) var curriculum_level: int = CurriculumConfig.Level.ENEMY_ATTACKS
@export var random_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED
## Escape hatch for tests: build the GUI even when the display server is
## headless. Off by default so training never pays for it.
@export var force_gui: bool = false

var session: ControlCenterSession
var ui: ControlCenterUI
var spectator: SpectatorCamera
var overlay: PerceptionOverlay3D
var presentation_enabled: bool = false

var _overlay_accumulator: float = 0.0
var _bound_view: EnvironmentView
var _last_camera_mode: int = -1


func _ready() -> void:
	var config := ControlCenterConfig.new(start_mode)
	# Automated/headless scene tests use force_gui and intentionally start
	# from clean exported defaults. Normal desktop launches restore operator
	# layout and training preferences before command-line overrides apply.
	if not force_gui:
		config.load_preferences()
	config.mode = start_mode
	config.environment_count = environment_count
	config.enemy_count = enemy_count
	config.curriculum_level = curriculum_level
	config.seed = random_seed
	# Parsed first: `--force-gui=1` has to be able to influence the
	# presentation decision made right below it.
	_apply_command_line(config)

	presentation_enabled = force_gui or not _is_headless_display()

	if presentation_enabled:
		_setup_world_lighting()

	session = ControlCenterSession.new()
	session.name = "ControlCenterSession"
	session.presentation_enabled = presentation_enabled
	add_child(session)
	session.setup(config)

	if not presentation_enabled:
		print("[ControlCenter] headless display detected: GUI disabled, simulation only.")
		return

	ui = ControlCenterUI.new()
	ui.name = "ControlCenterUI"
	add_child(ui)
	ui.setup(session)

	spectator = SpectatorCamera.new()
	spectator.name = "SpectatorCamera"
	add_child(spectator)

	overlay = PerceptionOverlay3D.new()
	overlay.name = "PerceptionOverlay3D"
	add_child(overlay)

	session.selection_changed.connect(_on_selection_changed)
	session.mode_changed.connect(_on_mode_changed)
	session.environments_rebuilt.connect(_on_environments_rebuilt)

	_bind_presentation_to_selection()


## Reads `--key=value` pairs passed after `--` on the command line.
## Unknown arguments are ignored so Godot's own flags stay untouched.
func _apply_command_line(config: ControlCenterConfig) -> void:
	for argument in OS.get_cmdline_user_args():
		var text: String = String(argument)
		if not text.begins_with("--") or not text.contains("="):
			continue
		var pair: PackedStringArray = text.substr(2).split("=", true, 1)
		if pair.size() != 2:
			continue
		var key: String = pair[0].strip_edges().to_lower()
		var value: String = pair[1].strip_edges()
		match key:
			"mode":
				config.mode = ControlCenterConfig.mode_from_name(value)
			"env-count", "environments":
				config.environment_count = maxi(1, int(value))
			"enemy-count", "enemies":
				config.enemy_count = maxi(1, int(value))
			"curriculum-level", "level":
				config.curriculum_level = clampi(
					int(value),
					CurriculumConfig.Level.STATIONARY_TARGET,
					CurriculumConfig.Level.AGENT_VS_AGENT
				)
			"seed":
				config.seed = int(value)
			"scenario":
				config.apply_scenario(value)
			"force-gui":
				force_gui = value != "0" and value.to_lower() != "false"
			_:
				push_warning("[ControlCenter] ignoring unknown argument: %s" % text)
	config.sanitize()


func _setup_world_lighting() -> void:
	var light := DirectionalLight3D.new()
	light.name = "SunLight"
	light.rotation_degrees = Vector3(-55.0, -30.0, 0.0)
	light.light_energy = 1.0
	light.shadow_enabled = false
	add_child(light)

	var world_env := WorldEnvironment.new()
	world_env.name = "WorldEnvironment"
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.05, 0.05, 0.08)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.55, 0.55, 0.6)
	env.ambient_light_energy = 0.7
	world_env.environment = env
	add_child(world_env)


func _is_headless_display() -> bool:
	return DisplayServer.get_name() == "headless"


## Points the camera rig and the debug overlay at the environment the
## session currently has selected. Cheap enough to call on every
## selection/mode change.
func _bind_presentation_to_selection() -> void:
	if not presentation_enabled or session.simulation_manager == null:
		return
	var index: int = session.config.selected_environment
	var view: EnvironmentView = session.simulation_manager.ensure_view(index)
	if view == null:
		return
	_bound_view = view
	var core = session.simulation_manager.environments[index]
	spectator.bind_view(view.position, view.agent_view, core.arena_half_extent)
	overlay.follow_view(view)
	_last_camera_mode = -1
	_apply_camera_mode()


func _apply_camera_mode() -> void:
	if not presentation_enabled or _bound_view == null or not is_instance_valid(_bound_view):
		return
	var mode: int = session.config.camera_mode
	if mode == _last_camera_mode:
		return
	_last_camera_mode = mode
	spectator.set_camera_mode(mode, _bound_view.get_camera())


func _process(delta: float) -> void:
	if not presentation_enabled:
		return
	_apply_camera_mode()
	# Free-flight must never fight the human player over WASD.
	spectator.input_enabled = not (session.is_human_mode() and session.human_input_enabled)

	var overlay_active: bool = (
		session.config.show_perception_overlay and not session.is_training_mode()
	)
	overlay.visible = overlay_active
	if not overlay_active:
		return
	_overlay_accumulator += delta
	if _overlay_accumulator < 1.0 / OVERLAY_HZ:
		return
	_overlay_accumulator = 0.0
	var env = session.get_selected_environment()
	if env == null:
		return
	overlay.update_from_perception(PerceptionModel.build(env))


func _unhandled_input(event: InputEvent) -> void:
	if not presentation_enabled:
		return
	if session.is_human_mode() and session.human_input_enabled:
		return
	spectator.handle_input(event)


func _on_selection_changed(_environment_index: int, _agent_slot: int) -> void:
	_bind_presentation_to_selection()


func _on_mode_changed(_mode: int) -> void:
	_bind_presentation_to_selection()


func _on_environments_rebuilt() -> void:
	_bound_view = null
	_bind_presentation_to_selection()

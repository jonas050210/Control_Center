## Main
##
## Bootstraps the playable/demo scene: minimal world lighting, a
## SimulationManager running one or more environments, a human controller
## bound to environment 0, a trivial AI stub controller bound to every other
## environment (so multi-environment operation is visible even without an
## external trainer yet), the active camera, and the debug overlay.
extends Node3D

@export var environment_count: int = SandboxConfig.DEFAULT_ENVIRONMENT_COUNT
@export var enemy_count_per_environment: int = SandboxConfig.ENEMY_COUNT_DEFAULT
@export_range(1, 5) var curriculum_level: int = CurriculumConfig.Level.ENEMY_ATTACKS
@export var human_controls_environment_zero: bool = true

var simulation_manager: SimulationManager
var debug_overlay: DebugOverlay


func _ready() -> void:
	_setup_world_lighting()

	simulation_manager = SimulationManager.new()
	simulation_manager.name = "SimulationManager"
	simulation_manager.environment_count = environment_count
	simulation_manager.enemy_count_per_environment = enemy_count_per_environment
	simulation_manager.curriculum_level = curriculum_level
	add_child(simulation_manager)  # _ready() on the child calls build() immediately

	_setup_controllers()
	_setup_camera()
	_setup_debug_overlay()


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


func _setup_controllers() -> void:
	var start_index_for_ai: int = 0
	if human_controls_environment_zero:
		var human := HumanController.new()
		human.name = "HumanController"
		add_child(human)
		simulation_manager.set_controller(0, human)
		start_index_for_ai = 1

	for i in range(start_index_for_ai, simulation_manager.environments.size()):
		var ai := AIStubController.new()
		ai.name = "AIStubController_%d" % i
		add_child(ai)
		simulation_manager.set_controller(i, ai)


func _setup_camera() -> void:
	if simulation_manager.views.is_empty():
		return
	var view: EnvironmentView = simulation_manager.views[0]
	if view == null:
		return
	var camera: Camera3D = view.get_camera()
	if camera != null:
		camera.current = true


func _setup_debug_overlay() -> void:
	debug_overlay = DebugOverlay.new()
	debug_overlay.name = "DebugOverlay"
	add_child(debug_overlay)
	debug_overlay.setup(simulation_manager, 0)

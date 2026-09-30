## Tests that the actual project scene (scenes/main.tscn) loads, boots a
## SimulationManager, and wires up the expected number of environments —
## i.e. "the Godot project opens and runs" success criterion, exercised
## headlessly.
class_name TestProjectStartup
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const HumanController = preload("res://scripts/input/human_controller.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_main_scene_loads_and_boots_simulation_manager() -> SandboxTest:
	var t := SandboxTest.new("main_scene_loads_and_boots_simulation_manager")

	var packed: PackedScene = load("res://scenes/main.tscn") as PackedScene
	t.assert_not_null(packed, "scenes/main.tscn should load")
	if packed == null:
		return t

	var instance: Node = packed.instantiate()
	t.assert_not_null(instance, "main.tscn should instantiate")
	if instance == null:
		return t

	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	t.assert_not_null(loop, "a SceneTree main loop must be running")
	if loop == null:
		instance.free()
		return t

	loop.root.add_child(instance)  # triggers _ready() synchronously

	var sim_manager: Node = instance.get_node_or_null("SimulationManager")
	t.assert_not_null(sim_manager, "Main._ready() should create a SimulationManager child")
	if sim_manager != null:
		t.assert_eq(
			sim_manager.environments.size(),
			SandboxConfig.DEFAULT_ENVIRONMENT_COUNT,
			"SimulationManager should build the configured default environment count"
		)
		t.assert_not_null(
			instance.get_node_or_null("HumanController"),
			"a HumanController should be attached for env 0"
		)

	loop.root.remove_child(instance)
	instance.free()
	return t

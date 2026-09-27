## Tests that the actual project scene (scenes/main.tscn) loads, boots a
## SimulationManager, and wires up the expected number of environments —
## i.e. "the Godot project opens and runs" success criterion, exercised
## headlessly.
class_name TestProjectStartup
extends RefCounted

# Explicit dependency: headless --script runs do not populate the editor class cache.
const TestResult = preload("res://tests/sandbox_test.gd")


func test_main_scene_loads_and_boots_simulation_manager() -> TestResult:
	var t := TestResult.new("main_scene_loads_and_boots_simulation_manager")

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

	instance.queue_free()
	return t

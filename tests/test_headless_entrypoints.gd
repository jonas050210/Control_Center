## Regression tests for the standalone SceneTree entry points that the
## normal suite never instantiates (the Python JSON bridge and the
## demonstration recorder). These scripts previously referenced engine APIs
## that do not exist in Godot 4.7 (`OS.get_stdin()`,
## `NOTIFICATION_WM_CLOSE_REQUEST` in a SceneTree script), which no test
## caught because nothing ever loaded them. Loading them here ensures they
## at least compile against the running engine version.
class_name TestHeadlessEntrypoints
extends RefCounted

const SandboxTest = preload("res://tests/sandbox_test.gd")

const ENTRY_POINT_PATHS: Array = [
	"res://scripts/rl/rl_server.gd",
	"res://scripts/recording/record_demo.gd",
	"res://tests/run_tests.gd",
]


func test_scene_tree_entry_points_compile() -> SandboxTest:
	var t := SandboxTest.new("scene_tree_entry_points_compile")
	for path in ENTRY_POINT_PATHS:
		var script: Script = load(path) as Script
		t.assert_not_null(script, "%s should load" % path)
		if script == null:
			continue
		t.assert_true(script.can_instantiate(), "%s should compile" % path)
		var base: StringName = script.get_instance_base_type()
		t.assert_eq(String(base), "SceneTree", "%s must extend SceneTree" % path)
	return t

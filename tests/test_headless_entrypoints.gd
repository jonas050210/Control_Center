## Regression tests for the standalone SceneTree entry points that the
## normal suite never instantiates (the Python JSON bridge and the
## demonstration recorder). These scripts previously referenced engine APIs
## that do not exist in Godot 4.7 (`OS.get_stdin()`,
## `NOTIFICATION_WM_CLOSE_REQUEST` in a SceneTree script), which no test
## caught because nothing ever loaded them. Loading them here ensures they
## at least compile against the running engine version.
##
## The preload-closure test below extends the same idea to every script the
## entry points pull in: an invalid DEPENDENCY does not stop its preloader
## from compiling (a failed script still loads as a resource), so checking
## only the entry point itself cannot see it. That is exactly how a broken
## call (`LightingProfile.mode(...)`, an instance variable, called through
## the script class) once made SelfPlayEnvironmentCore.new() return null and
## the whole self-play bridge silently answer `observations: []` while every
## single-agent check still passed.
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


## Every script in the transitive preload closure of every headless entry
## point must load AND compile (can_instantiate). A script that fails to
## compile still loads as an (invalid) resource, so `load()` alone proves
## nothing — can_instantiate() is what actually fails for an invalid script.
##
## The closure follows SCENES too: `record_demo.gd` boots the project
## through `res://scenes/main.tscn`, and the scripts a scene attaches to its
## nodes (here `scripts/core/main.gd`) are reachable from no preload at all.
## Scenes are checked as scenes — a PackedScene is not a Script, and
## demanding that it load as one is what used to fail this test.
func test_entry_point_preload_closures_compile() -> SandboxTest:
	var t := SandboxTest.new("entry_point_preload_closures_compile")
	var closure: Array = _preload_closure(ENTRY_POINT_PATHS)
	t.assert_gt(float(closure.size()), 20.0, "closure should cover dozens of scripts")
	var scenes_checked: int = 0
	for path in closure:
		if path.ends_with(".tscn") or path.ends_with(".scn"):
			var packed: PackedScene = load(path) as PackedScene
			t.assert_not_null(packed, "%s should load as a PackedScene" % path)
			if packed != null:
				t.assert_true(packed.can_instantiate(), "%s should instantiate" % path)
				scenes_checked += 1
			continue
		var script: Script = load(path) as Script
		t.assert_not_null(script, "%s should load" % path)
		if script == null:
			continue
		t.assert_true(
			script.can_instantiate(),
			"%s should compile (an invalid script makes .new() return null)" % path
		)
	t.assert_gte(float(scenes_checked), 1.0, "the closure must reach scenes/main.tscn")
	t.assert_true(
		closure.has("res://scripts/core/main.gd"),
		"the entry-point scene's own script must be part of the closure"
	)
	return t


## Walks every preload/load resource-literal (res:// path) starting
## from `roots` and returns the sorted, de-duplicated res:// paths.
static func _preload_closure(roots: Array) -> Array:
	var seen: Dictionary = {}
	var stack: Array = roots.duplicate()
	while not stack.is_empty():
		var path: String = stack.pop_back()
		if seen.has(path):
			continue
		seen[path] = true
		var source: String = _source_of(path)
		if source.is_empty():
			continue
		for dependency in _literal_resource_paths(source):
			if not seen.has(dependency):
				stack.append(dependency)
	var paths: Array = seen.keys()
	paths.sort()
	return paths


## Text a resource contributes to the dependency walk: a script's source, or
## a scene file's raw text (whose ext_resource lines name the scripts its
## nodes run).
static func _source_of(path: String) -> String:
	if path.ends_with(".tscn") or path.ends_with(".scn"):
		var file: FileAccess = FileAccess.open(path, FileAccess.READ)
		if file == null:
			return ""
		var text: String = file.get_as_text()
		file.close()
		return text
	var script: Script = load(path) as Script
	return "" if script == null else script.source_code


## Extracts the `res://...` string literals passed to `preload(...)` /
## `load(...)` in one script's source. Plain string scanning (rather than a
## RegEx pattern) keeps the pattern text itself from looking like a resource
## literal to the repository's static analyzer.
static func _literal_resource_paths(source: String) -> Array:
	var paths: Array = []
	for needle in ["load(\"", "path=\""]:
		var cursor := 0
		while true:
			var hit: int = source.find(needle, cursor)
			if hit < 0:
				break
			var start: int = hit + needle.length()
			var end: int = source.find("\"", start)
			if end < 0:
				break
			var candidate: String = source.substr(start, end - start)
			if candidate.begins_with("res://") and not candidate.contains("%"):
				paths.append(candidate)
			cursor = end
	return paths

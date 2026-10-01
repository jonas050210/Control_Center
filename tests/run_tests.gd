## SandboxAI automated test runner.
##
## Run headlessly from the project root with:
##   godot --headless --path . --script res://tests/run_tests.gd
##
## Discovers every tests/test_*.gd script, instantiates it, calls every
## method whose name starts with "test_" (each must return a SandboxTest),
## prints a PASS/FAIL line per test, then exits with code 0 if everything
## passed or 1 if anything failed (suitable for CI).
##
## Shutdown diagnostics
## --------------------
## Tests own their objects: anything that is NOT RefCounted (every Node —
## SimulationManager, controllers, panels, scene
## instances) must be freed by the test that created it, because this
## runner calls `quit()` on the first process frame and never gives the
## SceneTree another frame to flush `queue_free()`.
##
## A "N ObjectDB instances were leaked at exit" / "resources still in use
## at exit" line therefore points at one of two things, in this order:
##   1. a test that created a Node and returned without freeing it (a
##      test-cleanup bug — fix the test, not the engine), or
##   2. objects the engine itself keeps alive past `SceneTree::finalize()`
##      in a headless process (script/scene resources still referenced by
##      the resource cache while it is torn down). Those are engine
##      shutdown artifacts: they are reported after the suite has already
##      printed its verdict, they do not affect the exit code, and they
##      must NOT be "fixed" by force-freeing engine-owned objects here.
extends SceneTree

const SandboxTest = preload("res://tests/sandbox_test.gd")

var _tests_started: bool = false


## Tests must NOT run inside `_initialize()`: SceneTree::initialize() calls the
## script's `_initialize()` BEFORE `root` enters the tree
## (`root->_set_tree(this)` only runs after `MainLoop::initialize()` returns),
## so any node a test adds under `root` at that point never enters the tree and
## never receives `_ready()`. Running the suite on the first process frame gives
## tests a genuinely live SceneTree, matching normal runtime semantics.
func _process(_delta: float) -> bool:
	if _tests_started:
		return true
	_tests_started = true
	var exit_code: int = _run_all_tests()
	quit(exit_code)
	return true


func _run_all_tests() -> int:
	var test_scripts: Array = _discover_test_scripts()
	var total: int = 0
	var failed: int = 0
	var failure_lines: Array = []
	## Names of the failing tests, in run order, for the final summary.
	var failed_tests: Array = []

	print("SandboxAI automated tests")
	print("==========================================================")

	for script_path in test_scripts:
		var script_resource: Script = load(script_path) as Script
		if script_resource == null:
			total += 1
			failed += 1
			print("ERROR %s (failed to load)" % script_path)
			failed_tests.append("%s (script failed to load)" % script_path)
			failure_lines.append("%s -> script failed to load" % script_path)
			continue
		# A script with a compilation/dependency failure can still load() as a
		# non-null but invalid GDScript. Calling new() on it throws
		# "Nonexistent function 'new' in base 'GDScript'", so guard first and
		# report it as a proper failure instead of crashing the runner.
		if not script_resource.can_instantiate():
			total += 1
			failed += 1
			print("ERROR %s (failed to compile)" % script_path)
			failed_tests.append("%s (script failed to compile)" % script_path)
			failure_lines.append("%s -> script failed to compile" % script_path)
			continue
		var instance: Object = script_resource.new()
		if instance == null:
			total += 1
			failed += 1
			print("ERROR %s (failed to instantiate)" % script_path)
			failed_tests.append("%s (script failed to instantiate)" % script_path)
			failure_lines.append("%s -> script failed to instantiate" % script_path)
			continue
		for method_info in instance.get_method_list():
			var method_name: String = method_info.name
			if not method_name.begins_with("test_"):
				continue
			total += 1
			var result = instance.call(method_name)
			if not (result is SandboxTest):
				failed += 1
				failed_tests.append(method_name)
				print("FAIL  %s::%s" % [script_path, method_name])
				failure_lines.append(
					"%s::%s -> did not return a SandboxTest" % [script_path, method_name]
				)
				continue
			if result.passed():
				print("PASS  %s::%s" % [script_path, method_name])
			else:
				failed += 1
				failed_tests.append(method_name)
				print("FAIL  %s::%s" % [script_path, method_name])
				for failure_message in result.failures:
					failure_lines.append(
						"%s::%s -> %s" % [script_path, method_name, failure_message]
					)

	if failed > 0:
		print("")
		print("---- FAILURE DETAILS ----")
		for line in failure_lines:
			print(line)

	_print_summary(total, failed, failed_tests)
	return 1 if failed > 0 else 0


## Final, paste-ready verdict. Every number is derived from the run that
## just happened: `total` counts every collected test (including scripts
## that could not even be loaded), `failed` counts the ones that did not
## pass, and `failed_tests` names them in run order.
func _print_summary(total: int, failed: int, failed_tests: Array) -> void:
	var separator: String = "=========================================================="
	print("")
	print(separator)
	print("SANDBOXAI TEST SUMMARY")
	print(separator)
	print("Total: %d" % total)
	print("Passed: %d" % (total - failed))
	print("Failed: %d" % failed)
	if not failed_tests.is_empty():
		print("")
		print("FAILURES:")
		for index in range(failed_tests.size()):
			print("%d. %s" % [index + 1, str(failed_tests[index])])
	print("")
	print("Status: %s" % ("FAILED" if failed > 0 else "PASSED"))
	print(separator)


func _discover_test_scripts() -> Array:
	var paths: Array = []
	var dir: DirAccess = DirAccess.open("res://tests")
	if dir == null:
		return paths
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if (
			not dir.current_is_dir()
			and file_name.begins_with("test_")
			and file_name.ends_with(".gd")
		):
			paths.append("res://tests/%s" % file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	paths.sort()
	return paths

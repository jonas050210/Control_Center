## SandboxAI automated test runner.
##
## Run headlessly from the project root with:
##   godot --headless --path . --script res://tests/run_tests.gd
##
## Discovers every tests/test_*.gd script, instantiates it, calls every
## method whose name starts with "test_" (each must return a SandboxTest),
## prints a PASS/FAIL line per test, then exits with code 0 if everything
## passed or 1 if anything failed (suitable for CI).
extends SceneTree


func _initialize() -> void:
	var exit_code: int = _run_all_tests()
	quit(exit_code)


func _run_all_tests() -> int:
	var test_scripts: Array = _discover_test_scripts()
	var total: int = 0
	var failed: int = 0
	var failure_lines: Array = []

	print("SandboxAI automated tests")
	print("==========================================================")

	for script_path in test_scripts:
		var script: GDScript = load(script_path) as GDScript
		if script == null:
			print("SKIP  %s (failed to load)" % script_path)
			continue
		var instance: Object = script.new()
		for method_info in instance.get_method_list():
			var method_name: String = method_info.name
			if not method_name.begins_with("test_"):
				continue
			total += 1
			var result = instance.call(method_name)
			if not (result is SandboxTest):
				failed += 1
				failure_lines.append(
					"%s::%s -> did not return a SandboxTest" % [script_path, method_name]
				)
				continue
			if result.passed():
				print("PASS  %s::%s" % [script_path, method_name])
			else:
				failed += 1
				print("FAIL  %s::%s" % [script_path, method_name])
				for failure_message in result.failures:
					failure_lines.append(
						"%s::%s -> %s" % [script_path, method_name, failure_message]
					)

	print("==========================================================")
	print("Total: %d   Passed: %d   Failed: %d" % [total, total - failed, failed])

	if failed > 0:
		print("")
		print("---- FAILURE DETAILS ----")
		for line in failure_lines:
			print(line)

	return 1 if failed > 0 else 0


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

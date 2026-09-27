## Headless JSON-lines bridge for the Python trainer.
##
## Protocol: one JSON object per input line and one JSON object per output
## line. Commands are spaces, reset, step, metrics and close. This keeps the
## simulation process independent from Python package versions and works on
## Windows and Linux without native extensions or a renderer.
extends SceneTree

var simulation_manager: SimulationManager
var adapter: RLAdapter
var stdin: FileAccess
var stdout: FileAccess


func _initialize() -> void:
	stdin = OS.get_stdin()
	stdout = OS.get_stdout()
	var options: Dictionary = _parse_user_args(OS.get_cmdline_user_args())
	simulation_manager = SimulationManager.new()
	simulation_manager.create_visuals = false
	simulation_manager.auto_tick = false
	simulation_manager.auto_reset_on_done = true
	simulation_manager.base_seed = int(options.get("seed", SandboxConfig.DEFAULT_RANDOM_SEED))
	simulation_manager.curriculum_level = int(
		options.get("curriculum-level", CurriculumConfig.Level.ENEMY_ATTACKS)
	)
	simulation_manager.build(
		int(options.get("env-count", SandboxConfig.DEFAULT_ENVIRONMENT_COUNT)),
		int(options.get("enemy-count", SandboxConfig.ENEMY_COUNT_DEFAULT))
	)
	adapter = RLAdapter.new(simulation_manager)
	_serve_stdio()


func _serve_stdio() -> void:
	while not stdin.eof_reached():
		var line: String = stdin.get_line()
		if line.strip_edges().is_empty():
			continue
		var request = JSON.parse_string(line)
		var response: Dictionary = _handle_request(request)
		stdout.store_line(JSON.stringify(response))
		stdout.flush()
		if bool(response.get("close", false)):
			break
	quit(0)


func _handle_request(request) -> Dictionary:
	var response: Dictionary = {"ok": false, "error": "request must be a JSON object"}
	if not request is Dictionary:
		return response
	var command: String = str(request.get("cmd", ""))
	match command:
		"spaces":
			response = {
				"ok": true,
				"action_space": RLAdapter.action_space_info(),
				"observation_space": RLAdapter.observation_space_info(),
			}
		"reset":
			var seed: int = int(request.get("seed", -1))
			var observations: Array = adapter.reset(seed)
			var flat: Array = []
			for observation in observations:
				flat.append(RLAdapter._observation_to_array(observation))
			response = {
				"ok": true,
				"observations": flat,
				"infos": simulation_manager.get_metrics(),
			}
		"step":
			var actions: Array = request.get("actions", [])
			response = adapter.step(actions)
			response["ok"] = true
		"metrics":
			response = {"ok": true, "metrics": simulation_manager.get_metrics()}
		"close":
			response = {"ok": true, "close": true}
		_:
			response = {"ok": false, "error": "unknown command: %s" % command}
	return response


func _parse_user_args(args: PackedStringArray) -> Dictionary:
	var parsed: Dictionary = {}
	var index: int = 0
	while index < args.size():
		var arg: String = args[index]
		if (
			arg.begins_with("--")
			and index + 1 < args.size()
			and not args[index + 1].begins_with("--")
		):
			parsed[arg.trim_prefix("--")] = args[index + 1]
			index += 2
		else:
			index += 1
	return parsed

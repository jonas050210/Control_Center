## Headless JSON-lines bridge for the Python trainer.
##
## Protocol: one JSON object per input line and one JSON object per output
## line. Commands are spaces, reset, reset_indices, step, metrics,
## reward_breakdown, set_curriculum, set_episode_plans, episode_conditions,
## health_check, ping, and close. This keeps the simulation process
## independent from Python package versions and works on Windows and Linux
## without native extensions or a renderer.
##
## With `--self-play 1` the server instead hosts a batch of deterministic
## two-agent matches (SelfPlayEnvironmentCore) for league/checkpoint
## evaluation; see self_play_adapter.gd for the two-slot wire shapes.
extends SceneTree

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SelfPlayAdapter = preload("res://scripts/rl/self_play_adapter.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


## Maximum accepted length (in bytes) of one incoming JSON request line.
## A "step" request grows linearly with the environment count; 1 MiB gives
## enormous headroom while staying cheap to allocate per read.
const STDIN_BUFFER_SIZE: int = 1 << 20

var simulation_manager: SimulationManager
var adapter: RLAdapter
var self_play_adapter: SelfPlayAdapter

## Opt-in aggregate profiling. The normal bridge never calls a clock or adds
## fields to step responses. Python enables this with `--profile 1` and reads
## one summary at the end of training.
var profiling_enabled: bool = false
var _profile_timings: Dictionary = {}
var _profile_counters: Dictionary = {}


func _initialize() -> void:
	var options: Dictionary = _parse_user_args(OS.get_cmdline_user_args())
	profiling_enabled = int(options.get("profile", 0)) > 0
	if int(options.get("self-play", 0)) > 0:
		self_play_adapter = SelfPlayAdapter.new(
			int(options.get("env-count", 1)),
			int(options.get("seed", SandboxConfig.DEFAULT_RANDOM_SEED))
		)
		_serve_stdio()
		return
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


## Godot 4.7 exposes standard input only through
## `OS.read_string_from_stdin()` (there is no `OS.get_stdin()` FileAccess).
## On Unix it reads one line (fgets); on Windows one ReadFile chunk, which for
## this strict request/response protocol is one line but is defensively split
## anyway. Responses go through `print()`; the project enables
## `application/run/flush_stdout_on_print` so every line is flushed
## immediately in both debug and release builds.
func _serve_stdio() -> void:
	var closing: bool = false
	while not closing:
		var chunk: String = OS.read_string_from_stdin(STDIN_BUFFER_SIZE)
		if chunk.strip_edges().is_empty():
			# EOF (the Python side closed the pipe or died) or a blank line,
			# which the protocol never sends. Shut down instead of spinning.
			break
		for line in chunk.split("\n", false):
			if line.strip_edges().is_empty():
				continue
			var parse_started: int = Time.get_ticks_usec() if profiling_enabled else 0
			var request = JSON.parse_string(line)
			_profile_add_time("request_parse", parse_started)
			var command: String = str(request.get("cmd", "invalid")) if request is Dictionary else "invalid"
			var handle_started: int = Time.get_ticks_usec() if profiling_enabled else 0
			var response: Dictionary = (
				_handle_self_play_request(request)
				if self_play_adapter != null
				else _handle_request(request)
			)
			_profile_add_time("command_%s" % command, handle_started)
			var encode_started: int = Time.get_ticks_usec() if profiling_enabled else 0
			var encoded: String = JSON.stringify(response)
			_profile_add_time("response_encode", encode_started)
			var write_started: int = Time.get_ticks_usec() if profiling_enabled else 0
			print(encoded)
			_profile_add_time("response_write", write_started)
			if profiling_enabled:
				_profile_add_counter("request_bytes", line.to_utf8_buffer().size())
				_profile_add_counter("response_bytes", encoded.to_utf8_buffer().size() + 1)
			if bool(response.get("close", false)):
				closing = true
				break
	if simulation_manager != null and is_instance_valid(simulation_manager):
		simulation_manager.free()
	quit(0)


## Self-play mode (two policy slots per environment). The wire shapes add a
## slot axis of size 2 to the normal protocol; there is no auto-reset, so
## match boundaries are driven by the caller.
func _handle_self_play_request(request) -> Dictionary:
	var response: Dictionary = {"ok": false, "error": "request must be a JSON object"}
	if not (request is Dictionary):
		return response
	var command: String = str(request.get("cmd", ""))
	match command:
		"ping":
			response = {"ok": true, "pong": true}
		"spaces":
			response = {
				"ok": true,
				"action_space": RLAdapter.action_space_info(),
				"observation_space": RLAdapter.observation_space_info(),
				"policy_slots": 2,
			}
		"reset":
			if self_play_adapter.environment_count() == 0:
				# Empty environments means SelfPlayEnvironmentCore failed to
				# construct at startup (a compile error — see stderr). Answer
				# with an explicit error instead of a silent `observations: []`,
				# which the Python side can only report as a shape mismatch.
				response = {"ok": false, "error": _self_play_empty_error()}
			else:
				var seed: int = int(request.get("seed", -1))
				response = {"ok": true, "observations": self_play_adapter.reset(seed)}
		"step":
			if self_play_adapter.environment_count() == 0:
				response = {"ok": false, "error": _self_play_empty_error()}
			else:
				response = self_play_adapter.step(request.get("actions", []))
				response["ok"] = true
		"health_check":
			response = {"ok": true, "health": self_play_adapter.health_check()}
		"profile_snapshot":
			response = {"ok": true, "profile": _profile_snapshot()}
		"set_map":
			var map_id: String = str(request.get("map_id", ""))
			var ok: bool = self_play_adapter.set_map(map_id)
			response = {"ok": ok, "map_id": map_id}
		"set_layout":
			var layout_id: String = str(request.get("layout_id", ""))
			var ok: bool = self_play_adapter.set_layout(layout_id)
			response = {"ok": ok, "layout_id": layout_id}
		"set_lighting":
			var lighting: String = str(request.get("lighting", ""))
			var ok: bool = self_play_adapter.set_lighting_mode(lighting)
			response = {"ok": ok, "lighting": lighting}
		"set_curriculum":
			var level: int = int(request.get("level", CurriculumConfig.Level.AGENT_VS_AGENT))
			self_play_adapter.set_curriculum_level(level)
			response = {"ok": true, "curriculum_level": level}
		"episode_conditions":
			response = {"ok": true, "conditions": self_play_adapter.get_episode_conditions()}
		"close":
			response = {"ok": true, "close": true}
		_:
			response = {
				"ok": false,
				"error": "command %s is not supported in self-play mode" % command,
			}
	return response


## Error body used when the self-play adapter holds zero environments. That
## state is unreachable for a compiling script tree: it means
## SelfPlayEnvironmentCore.new() returned null during startup because the
## script (or one of its dependencies) failed to compile. The compile error is
## on stderr; this response makes the failure visible over the wire too.
func _self_play_empty_error() -> String:
	return (
		"self-play adapter has 0 environments: SelfPlayEnvironmentCore failed "
		+ "to construct at startup (a Godot compile error; see the engine's "
		+ "stderr output)"
	)


func _handle_request(request) -> Dictionary:
	var response: Dictionary = {"ok": false, "error": "request must be a JSON object"}
	if not (request is Dictionary):
		return response
	var command: String = str(request.get("cmd", ""))
	match command:
		"ping":
			response = {"ok": true, "pong": true}
		"spaces":
			response = {
				"ok": true,
				"action_space": RLAdapter.action_space_info(),
				"observation_space": RLAdapter.observation_space_info(),
			}
		"reset":
			var seed: int = int(request.get("seed", -1))
			var observations: Array = adapter.reset(seed)
			response = {
				"ok": true,
				"observations": observations,
				"infos": simulation_manager.get_metrics(),
			}
		"reset_indices":
			var indices: Array = request.get("indices", [])
			var seed_idx: int = int(request.get("seed", -1))
			var results: Array = adapter.reset_indices(indices, seed_idx)
			response = {
				"ok": true,
				"results": results,
			}
		"step":
			var actions: Array = request.get("actions", [])
			response = adapter.step(actions, bool(request.get("compact_infos", false)))
			response["ok"] = true
		"metrics":
			response = {"ok": true, "metrics": simulation_manager.get_metrics()}
		"reward_breakdown":
			response = {"ok": true, "breakdowns": simulation_manager.get_reward_breakdowns()}
		"set_curriculum":
			var level: int = int(request.get("level", CurriculumConfig.Level.ENEMY_ATTACKS))
			simulation_manager.set_curriculum_level(level)
			response = {"ok": true, "curriculum_level": simulation_manager.curriculum_level}
		"set_episode_plans":
			response = adapter.set_episode_plans(request.get("plans", []))
		"episode_conditions":
			response = {"ok": true, "conditions": adapter.get_episode_conditions()}
		"health_check":
			response = {"ok": true, "health": simulation_manager.health_check_all()}
		"profile_snapshot":
			response = {"ok": true, "profile": _profile_snapshot()}
		"close":
			response = {"ok": true, "close": true}
		_:
			response = {"ok": false, "error": "unknown command: %s" % command}
	return response


func _profile_add_time(name: String, started_usec: int) -> void:
	if not profiling_enabled or started_usec <= 0:
		return
	var elapsed: int = maxi(0, Time.get_ticks_usec() - started_usec)
	var entry: Dictionary = _profile_timings.get(
		name,
		{"count": 0, "total_usec": 0, "min_usec": elapsed, "max_usec": 0}
	)
	entry["count"] = int(entry["count"]) + 1
	entry["total_usec"] = int(entry["total_usec"]) + elapsed
	entry["min_usec"] = mini(int(entry["min_usec"]), elapsed)
	entry["max_usec"] = maxi(int(entry["max_usec"]), elapsed)
	_profile_timings[name] = entry


func _profile_add_counter(name: String, value: int) -> void:
	if profiling_enabled:
		_profile_counters[name] = int(_profile_counters.get(name, 0)) + value


func _profile_snapshot() -> Dictionary:
	if not profiling_enabled:
		return {"available": false, "reason": "Godot bridge profiling disabled"}
	var timings: Dictionary = {}
	for name_value in _profile_timings:
		var name: String = str(name_value)
		var source: Dictionary = _profile_timings[name]
		var count: int = int(source.get("count", 0))
		var total_usec: int = int(source.get("total_usec", 0))
		timings[name] = {
			"count": count,
			"total_seconds": float(total_usec) / 1000000.0,
			"mean_ms": float(total_usec) / float(maxi(1, count)) / 1000.0,
			"min_ms": float(source.get("min_usec", 0)) / 1000.0,
			"max_ms": float(source.get("max_usec", 0)) / 1000.0,
		}
	return {
		"available": true,
		"timings": timings,
		"counters": _profile_counters.duplicate(true),
	}


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

## ViewerMain
##
## The 3D checkpoint viewer: watch a trained policy play the real simulation
## on any map, with spectator cameras, pause/step/speed controls and a HUD.
##
## Launch it through Python so a checkpoint drives the agent:
##   python start.py view --checkpoint training/runs/<run>
## or open it standalone (built-in heuristic instead of a checkpoint):
##   godot --path . --script res://scripts/viewer/viewer_entry.gd -- --map compound
##
## The simulation is the canonical EnvironmentCore stepped at the fixed
## training dt - one tick per policy decision, exactly like the headless
## bridge. Rendering speed only changes how many ticks run per frame, never
## the physics of a tick, so what you see is what the policy experiences.
##
## User arguments (all optional): --policy-host, --policy-port, --map,
## --level, --enemies, --seed, --scenario, --lighting, --weapon, --speed,
## --camera, --max-steps, --max-episodes.
class_name ViewerMain
extends Node3D

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AIStubController = preload("res://scripts/input/ai_stub_controller.gd")
const ControllerBase = preload("res://scripts/input/controller_base.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyView = preload("res://scripts/enemy/enemy_view.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const EnvironmentView = preload("res://scripts/env/environment_view.gd")
const MapLibrary = preload("res://scripts/world/map_library.gd")
const Observation = preload("res://scripts/core/observation.gd")
const RemotePolicyClient = preload("res://scripts/viewer/remote_policy_client.gd")
const RemotePolicyController = preload("res://scripts/viewer/remote_policy_controller.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")
const ViewerCameraRig = preload("res://scripts/viewer/viewer_camera_rig.gd")
const ViewerHud = preload("res://scripts/viewer/viewer_hud.gd")

const SPEEDS: Array = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
const DEFAULT_SPEED_INDEX: int = 2
const DEFAULT_LEVEL: int = CurriculumConfig.Level.MIXED_RANDOMIZED
const EPISODE_REPORT_TIMEOUT_MS: int = 2000

var options: Dictionary = {}
var simulation_manager: SimulationManager
var controller: ControllerBase
var remote_controller: RemotePolicyController
var client: RemotePolicyClient
var camera_rig: ViewerCameraRig
var hud: ViewerHud
var policy_label: String = "built-in heuristic (no checkpoint)"
var paused: bool = false
var speed_index: int = DEFAULT_SPEED_INDEX
var map_ids: PackedStringArray = PackedStringArray()
## Index into map_ids; -1 means "no authored map" (scenario-generated arena).
var map_index: int = -1
var episode_number: int = 1
var episode_step: int = 0
var episode_reward: float = 0.0
var results: Array = []
var wins: int = 0
var total_steps: int = 0
var max_steps: int = 0
var max_episodes: int = 0
var exit_code: int = 0
var _step_accumulator: float = 0.0
var _step_once: bool = false
var _snap_camera: bool = true
var _quitting: bool = false


func _ready() -> void:
	options = parse_user_args(OS.get_cmdline_user_args())
	max_steps = maxi(0, int(options.get("max-steps", 0)))
	max_episodes = maxi(0, int(options.get("max-episodes", 0)))
	speed_index = speed_index_for(float(options.get("speed", 1.0)))
	_setup_world_lighting()
	_build_simulation()
	if not _setup_controller():
		return
	_setup_camera()
	hud = ViewerHud.new()
	hud.name = "ViewerHud"
	add_child(hud)
	get_window().title = "SandboxAI Viewer - %s" % policy_label
	print("[viewer] playing '%s' (level %d)" % [policy_label, _level()])
	_refresh_hud()


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_viewer.gd)
# ---------------------------------------------------------------------------


## `--key value` pairs after the `--` separator, like rl_server.gd.
static func parse_user_args(args: PackedStringArray) -> Dictionary:
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


## Nearest available playback speed for a requested multiplier.
static func speed_index_for(requested: float) -> int:
	var best: int = DEFAULT_SPEED_INDEX
	var best_distance: float = INF
	for index in range(SPEEDS.size()):
		var distance: float = absf(float(SPEEDS[index]) - requested)
		if distance < best_distance:
			best_distance = distance
			best = index
	return best


## Cycles through [-1 (generated arena), 0 .. count-1] in either direction.
static func next_map_index(current: int, direction: int, count: int) -> int:
	if count <= 0:
		return -1
	var slots: int = count + 1
	var slot: int = posmod(current + 1 + direction, slots)
	return slot - 1


## "win", "loss" or "timeout" from an episode's terminal metrics.
static func outcome_of(metrics: Dictionary) -> String:
	if bool(metrics.get("win", false)):
		return "win"
	if bool(metrics.get("loss", false)):
		return "loss"
	return "timeout"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


func _setup_world_lighting() -> void:
	var light := DirectionalLight3D.new()
	light.name = "SunLight"
	light.rotation_degrees = Vector3(-55.0, -30.0, 0.0)
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


func _build_simulation() -> void:
	simulation_manager = SimulationManager.new()
	simulation_manager.name = "SimulationManager"
	simulation_manager.environment_count = 1
	simulation_manager.enemy_count_per_environment = maxi(1, int(options.get("enemies", 1)))
	simulation_manager.curriculum_level = _level()
	simulation_manager.base_seed = int(options.get("seed", SandboxConfig.DEFAULT_RANDOM_SEED))
	simulation_manager.create_visuals = true
	simulation_manager.auto_tick = false
	simulation_manager.auto_reset_on_done = true
	add_child(simulation_manager)  # _ready() builds the environment and its view
	map_ids = MapLibrary.ids()
	map_index = map_ids.find(str(options.get("map", "")))
	_apply_configuration(int(options.get("seed", -1)))


func _setup_controller() -> bool:
	if not options.has("policy-port"):
		var stub := AIStubController.new()
		stub.name = "HeuristicController"
		add_child(stub)
		controller = stub
		return true
	client = RemotePolicyClient.new()
	var host: String = str(options.get("policy-host", "127.0.0.1"))
	if not client.connect_to(host, int(options["policy-port"])):
		printerr("[viewer] %s" % client.last_error)
		exit_code = 3
		_quit()
		return false
	var hello: Dictionary = (
		client
		. request(
			{
				"type": "hello",
				"observation_size": Observation.FIELD_COUNT,
				"action_nvec": Action.MULTI_DISCRETE_NVECS,
				"godot_version": str(Engine.get_version_info().get("string", "")),
				"map_ids": Array(map_ids),
			}
		)
	)
	if not bool(hello.get("ok", false)):
		printerr("[viewer] policy server refused: %s" % str(hello.get("error", client.last_error)))
		exit_code = 4
		_quit()
		return false
	policy_label = str(hello.get("policy", "checkpoint"))
	remote_controller = RemotePolicyController.new(client)
	remote_controller.name = "RemotePolicyController"
	add_child(remote_controller)
	controller = remote_controller
	return true


func _setup_camera() -> void:
	camera_rig = ViewerCameraRig.new()
	camera_rig.name = "ViewerCameraRig"
	add_child(camera_rig)
	var view: EnvironmentView = _view()
	if view != null:
		camera_rig.set_first_person_camera(view.get_camera())
	var requested: String = str(options.get("camera", "chase")).to_lower()
	for index in range(ViewerCameraRig.MODE_NAMES.size()):
		if ViewerCameraRig.mode_name(index).to_lower().begins_with(requested.left(3)):
			camera_rig.set_mode(index)
			break


## Stages the current options (+ selected map) and starts a fresh episode.
func _apply_configuration(seed_value: int) -> void:
	var plan: Dictionary = {
		"index": 0,
		"seed": seed_value,
		"map_id": map_ids[map_index] if map_index >= 0 and map_index < map_ids.size() else "",
		"scenario": str(options.get("scenario", "")),
		"lighting": str(options.get("lighting", "")),
		"weapon_profile": str(options.get("weapon", "")),
		"enemy_count": maxi(1, int(options.get("enemies", 1))),
		"curriculum_level": _level(),
	}
	var problem: String = _env().validate_episode_plan(plan)
	if not problem.is_empty():
		printerr("[viewer] ignoring invalid option: %s" % problem)
		plan["scenario"] = ""
		plan["lighting"] = ""
		plan["weapon_profile"] = ""
		plan["map_id"] = plan["map_id"] if MapLibrary.has_map(str(plan["map_id"])) else ""
	simulation_manager.pending_plans[0] = plan
	simulation_manager.reset_all(-1)
	# Later explicit resets (R key) should start NEW episodes on this setup,
	# not replay the seeded one, so the remembered plan continues the RNG.
	if simulation_manager.current_plans.has(0):
		simulation_manager.current_plans[0]["seed"] = -1
	_reset_episode_counters()
	_sync_view()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


func _physics_process(_delta: float) -> void:
	if _quitting or controller == null:
		return
	var steps: int = 0
	if paused:
		if _step_once:
			steps = 1
			_step_once = false
	else:
		_step_accumulator += float(SPEEDS[speed_index])
		steps = int(_step_accumulator)
		_step_accumulator -= float(steps)
	for _index in range(steps):
		_advance_one_step()
		if _quitting:
			return
	if steps > 0:
		_sync_view()


func _process(delta: float) -> void:
	if _quitting or simulation_manager == null:
		return
	var env: EnvironmentCore = _env()
	if camera_rig != null:
		camera_rig.update_view(
			env.agent.position,
			env.agent.get_forward_horizontal(),
			env.arena_half_extent,
			delta,
			_snap_camera
		)
		_snap_camera = false
	_refresh_hud()


func _advance_one_step() -> void:
	var env: EnvironmentCore = _env()
	var action: Action = controller.get_action(env)
	var step_results: Array = simulation_manager.step_all([action])
	var result: Dictionary = step_results[0]
	episode_step += 1
	total_steps += 1
	episode_reward += float(result.get("reward", 0.0))
	if bool(result.get("done", false)):
		_finish_episode(result)
	if max_steps > 0 and total_steps >= max_steps:
		_quit()


func _finish_episode(result: Dictionary) -> void:
	var info: Dictionary = result.get("info", {})
	var metrics: Dictionary = info.get("metrics", {})
	var outcome: String = outcome_of(metrics)
	results.append(outcome)
	if outcome == "win":
		wins += 1
	var summary: Dictionary = {
		"episode": episode_number,
		"outcome": outcome,
		"done_reason": str(info.get("done_reason", "")),
		"reward": episode_reward,
		"steps": episode_step,
		"kills": int(metrics.get("kills", 0)),
		"enemy_count": int(metrics.get("enemy_count", 0)),
		"accuracy": float(metrics.get("accuracy", 0.0)),
		"damage_dealt": float(metrics.get("damage_dealt", 0.0)),
		"damage_received": float(metrics.get("damage_received", 0.0)),
		"survival_time": float(metrics.get("survival_time", 0.0)),
		"map_id": _current_map_label(),
	}
	print(
		(
			"[viewer] episode %d: %s, reward %.2f, %d steps, kills %d/%d"
			% [
				episode_number,
				outcome,
				episode_reward,
				episode_step,
				summary["kills"],
				summary["enemy_count"],
			]
		)
	)
	if client != null and client.connected:
		client.request({"type": "episode", "summary": summary}, EPISODE_REPORT_TIMEOUT_MS)
	if hud != null:
		var colors: Dictionary = {
			"win": Color(0.45, 1.0, 0.45),
			"loss": Color(1.0, 0.4, 0.35),
			"timeout": Color(1.0, 0.85, 0.3),
		}
		hud.show_banner("Episode %d: %s" % [episode_number, outcome.to_upper()], colors[outcome])
	episode_number += 1
	episode_step = 0
	episode_reward = 0.0
	_snap_camera = true
	if max_episodes > 0 and results.size() >= max_episodes:
		_quit()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


func _unhandled_input(event: InputEvent) -> void:
	if camera_rig != null and camera_rig.handle_input(event):
		return
	if not (event is InputEventKey):
		return
	var key := event as InputEventKey
	if not key.pressed or key.echo:
		return
	match key.keycode:
		KEY_SPACE:
			paused = not paused
		KEY_N, KEY_PERIOD:
			if paused:
				_step_once = true
			else:
				paused = true
		KEY_R:
			_start_new_episode()
		KEY_M:
			_change_map(-1 if key.shift_pressed else 1)
		KEY_EQUAL, KEY_PLUS, KEY_KP_ADD:
			speed_index = mini(speed_index + 1, SPEEDS.size() - 1)
		KEY_MINUS, KEY_KP_SUBTRACT:
			speed_index = maxi(speed_index - 1, 0)
		KEY_C:
			camera_rig.cycle_mode()
		KEY_1, KEY_2, KEY_3, KEY_4:
			camera_rig.set_mode(int(key.keycode - KEY_1))
		KEY_H:
			hud.toggle()
		KEY_ESCAPE, KEY_Q:
			_quit()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_quit()


func _start_new_episode() -> void:
	simulation_manager.reset_indices([0], -1)
	_reset_episode_counters()
	_sync_view()
	hud.show_banner("New episode")


func _change_map(direction: int) -> void:
	map_index = next_map_index(map_index, direction, map_ids.size())
	_apply_configuration(-1)
	if hud != null:
		hud.show_banner("Map: %s" % _current_map_label())


func _quit() -> void:
	if _quitting:
		return
	_quitting = true
	if client != null:
		client.close()
	print("[viewer] finished: %d episodes, %d won, %d steps" % [results.size(), wins, total_steps])
	get_tree().quit(exit_code)


# ---------------------------------------------------------------------------
# Small accessors
# ---------------------------------------------------------------------------


func _env() -> EnvironmentCore:
	return simulation_manager.environments[0]


func _view() -> EnvironmentView:
	if simulation_manager.views.is_empty():
		return null
	return simulation_manager.views[0]


func _level() -> int:
	return clampi(
		int(options.get("level", DEFAULT_LEVEL)),
		CurriculumConfig.Level.STATIONARY_TARGET,
		CurriculumConfig.Level.MIXED_RANDOMIZED
	)


func _current_map_label() -> String:
	if map_index >= 0 and map_index < map_ids.size():
		return map_ids[map_index]
	return ""


func _reset_episode_counters() -> void:
	episode_step = 0
	episode_reward = 0.0
	_snap_camera = true


## Mirrors the simulation onto the scene, re-binding enemy views whenever a
## reset replaced the enemy list (enemy count or level changes).
func _sync_view() -> void:
	var view: EnvironmentView = _view()
	if view == null:
		return
	var env: EnvironmentCore = _env()
	var stale: bool = view.enemy_views.size() != env.enemies.size()
	if not stale and not env.enemies.is_empty():
		stale = (view.enemy_views[0] as EnemyView).state != env.enemies[0]
	if stale:
		view.rebind_enemy_views()
	view.sync_from_state()


func _refresh_hud() -> void:
	if hud == null or simulation_manager == null:
		return
	var env: EnvironmentCore = _env()
	var connected: bool = remote_controller == null or remote_controller.is_connected_to_policy()
	var action_values: Array = (
		remote_controller.last_action_values
		if remote_controller != null
		else simulation_manager.get_last_actions()[0].to_multidiscrete()
	)
	(
		hud
		. set_status(
			{
				"policy": policy_label,
				"connected": connected,
				"error": client.last_error if client != null else "",
				"map": _current_map_label(),
				"level": env.curriculum.level,
				"enemies": env.enemies.size(),
				"seed": env.episode_seed,
				"episode": episode_number,
				"step": episode_step,
				"reward": episode_reward,
				"health": env.agent.health,
				"kills": env.episode.kills,
				"action": action_values,
				"results": results,
				"wins": wins,
				"episodes_done": results.size(),
				"paused": paused,
				"speed": SPEEDS[speed_index],
				"camera": ViewerCameraRig.mode_name(camera_rig.mode) if camera_rig != null else "",
			}
		)
	)

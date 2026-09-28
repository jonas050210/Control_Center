# gdlint:ignore=max-public-methods
# The session is the Control Center's public facade over the simulation:
# every operator action (mode, run state, selection, settings, reset) is one
# method that forwards to an existing SimulationManager/EnvironmentCore API.
# Splitting it further would only spread that mapping across more files.
## ControlCenterSession
##
## The Control Center's simulation driver. It owns a SimulationManager,
## decides who produces actions (heuristic AI, idle, or the human input
## pipeline), and advances the simulation under explicit play/pause/step/
## speed control.
##
## Deliberately a plain Node with NO Control/UI dependency: the UI reads
## `build_snapshot()` and calls the public methods below, and every one of
## those methods forwards to an existing public SimulationManager /
## EnvironmentCore API. Nothing in this class implements gameplay.
##
## Relationship to training (Phase 15):
##   * `SimulationManager.auto_tick` is OFF; this session is the only
##     stepper, so pausing is free and does NOT abuse `Engine.time_scale`
##     (the GUI stays responsive while the simulation is frozen).
##   * In TRAINING mode every presentation cost is switched off: views are
##     hidden and never synced, the event log is disabled, and no telemetry
##     snapshot or perception model is built. Steps run in a per-frame time
##     budget for maximum throughput.
##   * The headless RL bridge (scripts/rl/rl_server.gd) never instantiates
##     this class. Real PPO training runs in a separate headless Godot
##     process with no Control Center in it at all.
##
## All three modes share one EnvironmentCore, one Action type, one
## Observation contract and one reward system; only the action source and
## the amount of presentation work differ.
class_name ControlCenterSession
extends Node

signal mode_changed(mode: int)
signal running_changed(running: bool)
signal selection_changed(environment_index: int, agent_slot: int)
signal settings_applied(changed_keys: PackedStringArray)
signal episode_recorded(record: Dictionary)
signal environments_rebuilt

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AIStubController = preload("res://scripts/input/ai_stub_controller.gd")
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterEventLog = preload("res://scripts/control_center/control_center_event_log.gd")
const ControlCenterResults = preload("res://scripts/control_center/control_center_results.gd")
const ControlCenterTelemetry = preload("res://scripts/control_center/control_center_telemetry.gd")
const ControllerBase = preload("res://scripts/input/controller_base.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const HumanController = preload("res://scripts/input/human_controller.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")

var config: ControlCenterConfig
var simulation_manager: SimulationManager
var event_log: ControlCenterEventLog
var results: ControlCenterResults

## Whether the human input pipeline is currently allowed to drive the
## agent. The UI sets this from the mouse-capture state so clicking a
## button does not also fire the weapon; when false, HUMAN mode feeds
## Action.idle() instead of silently keeping the last input.
var human_input_enabled: bool = false
## Presentation work (view sync, telemetry, logging). Forced off in
## TRAINING mode and in headless runs.
var presentation_enabled: bool = true

var running: bool = true
var total_steps: int = 0
var steps_per_second: float = 0.0

var human_controller: HumanController
var _ai_controllers: Array = []  # Array[AIStubController], one per environment
var _step_accumulator: float = 0.0
var _steps_since_report: int = 0
var _report_timer: float = 0.0
var _pending_setting_keys: PackedStringArray = PackedStringArray()
## Per-environment derived measurements for the CURRENT episode. Only the
## selected environment is instrumented (Phase 15: no per-environment
## bookkeeping for data nobody looks at).
var _episode_probe: Dictionary = {}
var _last_target_index: int = -1
var _single_step_requests: int = 0


func _init() -> void:
	config = ControlCenterConfig.new()
	event_log = ControlCenterEventLog.new()
	results = ControlCenterResults.new()


## Creates (or adopts) the SimulationManager and wires controllers.
## `p_config` is copied by reference — the UI edits the same instance.
func setup(p_config: ControlCenterConfig = null, p_manager: SimulationManager = null) -> void:
	if p_config != null:
		config = p_config
	config.sanitize()

	simulation_manager = p_manager if p_manager != null else SimulationManager.new()
	simulation_manager.name = "SimulationManager"
	simulation_manager.auto_tick = false
	simulation_manager.auto_reset_on_done = true
	simulation_manager.create_visuals = presentation_enabled
	simulation_manager.base_seed = config.seed
	simulation_manager.curriculum_level = config.curriculum_level
	simulation_manager.environment_count = config.environment_count
	simulation_manager.enemy_count_per_environment = config.enemy_count
	if presentation_enabled:
		simulation_manager.visual_environment_indices = PackedInt32Array(
			[config.selected_environment]
		)
	if simulation_manager.get_parent() == null:
		add_child(simulation_manager)
	if simulation_manager.environments.is_empty():
		simulation_manager.build(config.environment_count, config.enemy_count)

	_create_controllers()
	_apply_mode_to_runtime()
	_reset_episode_probe()
	log_system(
		(
			"Control Center session ready: %d environments, level %d, seed %d"
			% [simulation_manager.environments.size(), config.curriculum_level, config.seed]
		)
	)


# ---------------------------------------------------------------------------
# Mode / run state
# ---------------------------------------------------------------------------


func set_mode(mode: int) -> void:
	var resolved: int = clampi(
		mode, ControlCenterConfig.Mode.TRAINING, ControlCenterConfig.Mode.HUMAN
	)
	if resolved == config.mode:
		return
	config.mode = resolved
	_apply_mode_to_runtime()
	log_system("mode -> %s" % ControlCenterConfig.mode_name(resolved))
	mode_changed.emit(resolved)


func is_training_mode() -> bool:
	return config.mode == ControlCenterConfig.Mode.TRAINING


func is_human_mode() -> bool:
	return config.mode == ControlCenterConfig.Mode.HUMAN


## Telemetry/log/view work is allowed only outside TRAINING mode and only
## when this session renders at all.
func telemetry_enabled() -> bool:
	return presentation_enabled and not is_training_mode()


func set_running(value: bool) -> void:
	if running == value:
		return
	running = value
	_step_accumulator = 0.0
	log_system("simulation %s" % ("resumed" if running else "paused"))
	running_changed.emit(running)


func toggle_running() -> void:
	set_running(not running)


## Pauses and queues exactly `count` simulation steps, executed on the next
## frame so a click always advances the same amount of simulation.
func request_single_step(count: int = 1) -> void:
	_single_step_requests += maxi(1, count)
	if running:
		set_running(false)


func set_speed(value: float) -> void:
	config.simulation_speed = clampf(
		value, ControlCenterConfig.MIN_SPEED, ControlCenterConfig.MAX_SPEED
	)


func set_human_input_enabled(value: bool) -> void:
	if human_input_enabled == value:
		return
	human_input_enabled = value
	if human_controller != null:
		# Only listen for mouse/keyboard while the human actually drives.
		human_controller.set_process_unhandled_input(value)
	if presentation_enabled and not value and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		# Hand the mouse back. With its input processing disabled the human
		# controller can no longer release the capture itself, which would
		# otherwise leave the operator unable to click the GUI.
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	log_system("human input %s" % ("captured" if value else "released"))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


func select_environment(index: int) -> void:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return
	var resolved: int = clampi(index, 0, simulation_manager.environments.size() - 1)
	if resolved == config.selected_environment:
		return
	config.selected_environment = resolved
	if presentation_enabled:
		simulation_manager.ensure_view(resolved)
		_apply_view_visibility()
	_bind_controllers()
	_reset_episode_probe()
	log_system("selected environment %d" % resolved)
	selection_changed.emit(resolved, config.selected_agent_slot)


## Agent/policy slot inside the selected environment. The canonical
## EnvironmentCore exposes exactly one controllable agent; slot 1 belongs
## to the self-play foundation, which SimulationManager does not build, so
## selecting it is rejected instead of silently doing nothing visible.
func select_agent_slot(slot: int) -> bool:
	if slot == config.selected_agent_slot:
		return true
	if not is_agent_slot_available(slot):
		log_warning(
			(
				"agent slot %d is unavailable: SimulationManager builds single-agent "
				+ "EnvironmentCore instances (slot 1 requires the self-play environment)"
			)
			% slot
		)
		return false
	config.selected_agent_slot = slot
	selection_changed.emit(config.selected_environment, slot)
	return true


func is_agent_slot_available(slot: int) -> bool:
	return slot == 0


## Switches which action source drives the AI-controlled environments.
## Returns false (and changes nothing) for a source that cannot actually
## run in-engine, so the UI can revert its selector instead of implying a
## trained policy is now playing.
func set_policy_source(source: int) -> bool:
	if not ControlCenterConfig.policy_source_available(source):
		log_warning(
			(
				"policy source '%s' is unavailable in-engine: Godot has no inference "
				+ "runtime, trained checkpoints run in the Python trainer over the "
				+ "JSON-lines bridge"
			)
			% ControlCenterConfig.policy_source_name(source)
		)
		return false
	if source == config.policy_source:
		return true
	config.policy_source = source
	_bind_controllers()
	log_system("policy source -> %s" % ControlCenterConfig.policy_source_name(source))
	return true


## Camera selection is presentation-only; the simulation never reads it.
func set_camera_mode(mode: int) -> void:
	var resolved: int = clampi(
		mode, ControlCenterConfig.CameraMode.FIRST_PERSON, ControlCenterConfig.CameraMode.TOP_DOWN
	)
	if resolved == config.camera_mode:
		return
	config.camera_mode = resolved
	log_system("camera -> %s" % ControlCenterConfig.camera_mode_name(resolved))


func available_agent_slots() -> Array:
	return [
		{
			"slot": 0,
			"label": "Agent 0 (primary)",
			"available": true,
			"note": "the controllable AgentState of the selected EnvironmentCore",
		},
		{
			"slot": 1,
			"label": "Agent 1 (self-play opponent)",
			"available": false,
			"note": "only exists in SelfPlayEnvironmentCore, which is not built here",
		},
	]


func get_selected_environment() -> EnvironmentCore:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return null
	var index: int = clampi(config.selected_environment, 0, simulation_manager.environments.size() - 1)
	return simulation_manager.environments[index]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


## Curriculum level is the one difficulty setting the simulation can change
## on live environments (SimulationManager.set_curriculum_level rebuilds the
## enemy lists in place), so it is applied immediately.
func set_curriculum_level(level: int) -> void:
	var resolved: int = clampi(
		level,
		CurriculumConfig.Level.STATIONARY_TARGET,
		CurriculumConfig.Level.AGENT_VS_AGENT
	)
	if resolved == config.curriculum_level and simulation_manager.curriculum_level == resolved:
		return
	config.curriculum_level = resolved
	config.scenario_id = "custom"
	simulation_manager.set_curriculum_level(resolved)
	config.enemy_count = simulation_manager.enemy_count_per_environment
	_apply_view_visibility()
	log_system(
		"curriculum level -> %d (%s)" % [resolved, CurriculumConfig.level_name(resolved)]
	)


## Records a setting that cannot be applied to running environments.
## Returns true when a rebuild is now pending.
func request_setting(key: String, value) -> bool:
	match key:
		"environment_count":
			config.environment_count = clampi(
				int(value), 1, ControlCenterConfig.MAX_ENVIRONMENT_COUNT
			)
		"enemy_count":
			config.enemy_count = clampi(int(value), 1, ControlCenterConfig.MAX_ENEMY_COUNT)
			config.scenario_id = "custom"
		"seed":
			config.seed = maxi(0, int(value))
		_:
			log_warning("unknown pending setting: %s" % key)
			return false
	if not _pending_setting_keys.has(key):
		_pending_setting_keys.append(key)
	return true


func pending_setting_keys() -> PackedStringArray:
	return _pending_setting_keys.duplicate()


func has_pending_settings() -> bool:
	return not _pending_setting_keys.is_empty()


## Queues a new random seed as a PENDING setting (it is applied when the
## user hits Apply, like every other edited field).
func randomize_seed() -> int:
	var rng := RandomNumberGenerator.new()
	rng.randomize()
	var value: int = rng.randi_range(0, 1_000_000)
	request_setting("seed", value)
	return value


## Draws a fresh seed and immediately restarts the selected environment
## with it.
##
## The "Reset (random)" button used to call `randomize_seed()` followed by
## `reset_selected_environment(false)`. The first call only QUEUED the new
## seed and the second passed -1 ("continue the current RNG stream"), so
## the freshly drawn seed was never applied to anything and the button was
## indistinguishable from a plain non-deterministic reset. This applies the
## seed for real and returns it so the UI can report it.
func reset_selected_environment_with_random_seed() -> int:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return -1
	var rng := RandomNumberGenerator.new()
	rng.randomize()
	var value: int = rng.randi_range(0, 1_000_000)
	config.seed = value
	var pending_index: int = _pending_setting_keys.find("seed")
	if pending_index >= 0:
		_pending_setting_keys.remove_at(pending_index)
	var index: int = config.selected_environment
	simulation_manager.base_seed = value
	simulation_manager.reset_indices([index], value)
	_reset_episode_probe()
	log_system("reset environment %d (random seed %d)" % [index, value + index])
	return value


## Applies a scenario preset. Curriculum level takes effect immediately;
## an enemy-count change becomes a pending rebuild, exactly like editing
## the field by hand.
func apply_scenario(scenario_id: String) -> PackedStringArray:
	var changed: PackedStringArray = config.apply_scenario(scenario_id)
	if changed.has("curriculum_level"):
		var level: int = config.curriculum_level
		config.scenario_id = scenario_id
		simulation_manager.set_curriculum_level(level)
		config.enemy_count = simulation_manager.enemy_count_per_environment
	if changed.has("enemy_count"):
		if not _pending_setting_keys.has("enemy_count"):
			_pending_setting_keys.append("enemy_count")
	config.scenario_id = scenario_id
	log_system("scenario -> %s" % scenario_id)
	return changed


## Rebuilds the environments so pending settings take effect. This restarts
## every episode by design; the UI labels those settings "requires reset".
func apply_pending_settings() -> PackedStringArray:
	var changed: PackedStringArray = _pending_setting_keys.duplicate()
	_pending_setting_keys = PackedStringArray()
	if simulation_manager == null:
		return changed
	config.sanitize()
	simulation_manager.base_seed = config.seed
	simulation_manager.curriculum_level = config.curriculum_level
	if presentation_enabled:
		simulation_manager.visual_environment_indices = PackedInt32Array(
			[config.selected_environment]
		)
	simulation_manager.build(config.environment_count, config.enemy_count)
	config.enemy_count = simulation_manager.enemy_count_per_environment
	config.selected_environment = clampi(
		config.selected_environment, 0, simulation_manager.environments.size() - 1
	)
	_create_controllers()
	_apply_mode_to_runtime()
	_reset_episode_probe()
	log_system(
		(
			"applied settings [%s]: %d environments, %d enemies, seed %d"
			% [
				", ".join(changed),
				simulation_manager.environments.size(),
				simulation_manager.enemy_count_per_environment,
				config.seed,
			]
		)
	)
	settings_applied.emit(changed)
	environments_rebuilt.emit()
	return changed


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


## Restarts the selected environment. `deterministic` replays the exact
## same episode (seed + index), which is what makes HUMAN vs AI runs
## comparable; otherwise the environment continues its RNG stream.
func reset_selected_environment(deterministic: bool = true) -> void:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return
	var index: int = config.selected_environment
	simulation_manager.reset_indices([index], config.seed if deterministic else -1)
	_reset_episode_probe()
	log_system(
		(
			"reset environment %d (%s)"
			% [index, "seed %d" % (config.seed + index) if deterministic else "continued RNG"]
		)
	)


func reset_all_environments(deterministic: bool = true) -> void:
	if simulation_manager == null:
		return
	simulation_manager.reset_all(config.seed if deterministic else -1)
	_reset_episode_probe()
	log_system("reset all environments")


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


func _physics_process(delta: float) -> void:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return

	var executed: int = 0
	if _single_step_requests > 0:
		executed = mini(_single_step_requests, ControlCenterConfig.MAX_STEPS_PER_FRAME_INTERACTIVE)
		_single_step_requests -= executed
		advance(executed)
	elif running:
		if is_training_mode():
			executed = _advance_training_budget()
		else:
			_step_accumulator += config.simulation_speed
			var steps: int = int(floor(_step_accumulator))
			_step_accumulator -= float(steps)
			steps = mini(steps, ControlCenterConfig.MAX_STEPS_PER_FRAME_INTERACTIVE)
			if steps > 0:
				advance(steps)
			executed = steps

	if telemetry_enabled() and executed > 0:
		simulation_manager.sync_view(config.selected_environment)

	_report_timer += delta
	if _report_timer >= 0.5:
		steps_per_second = float(_steps_since_report) / _report_timer
		_steps_since_report = 0
		_report_timer = 0.0


## Advances the whole batch by `steps` simulation ticks using the same
## fixed dt as headless training, so trajectories are identical regardless
## of frame rate, speed multiplier or pauses.
func advance(steps: int = 1) -> void:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return
	for _index in range(maxi(1, steps)):
		var step_results: Array = simulation_manager.step_all(
			_collect_actions(), SandboxConfig.SIMULATION_DT
		)
		var count: int = simulation_manager.environments.size()
		total_steps += count
		_steps_since_report += count
		if telemetry_enabled():
			_observe_step(step_results)


## TRAINING mode: run as many batched steps as fit in the frame budget.
func _advance_training_budget() -> int:
	var deadline_usec: int = (
		Time.get_ticks_usec() + int(ControlCenterConfig.TRAINING_FRAME_BUDGET_MS * 1000.0)
	)
	var executed: int = 0
	while Time.get_ticks_usec() < deadline_usec:
		advance(1)
		executed += 1
	return executed


## Builds one Action per environment from the bound controllers. This is
## the same collection loop SimulationManager runs in auto-tick mode, kept
## here because the Control Center drives stepping itself.
func _collect_actions() -> Array:
	var actions: Array = []
	for index in range(simulation_manager.environments.size()):
		var controller: ControllerBase = simulation_manager.controllers[index]
		if controller == null:
			actions.append(Action.idle())
			continue
		if controller == human_controller and not human_input_enabled:
			# HUMAN mode with input released (mouse visible / GUI focused):
			# feed a genuine idle action instead of stale input.
			actions.append(Action.idle())
			continue
		actions.append(controller.get_action(simulation_manager.environments[index]))
	return actions


# ---------------------------------------------------------------------------
# Observation of results (presentation only)
# ---------------------------------------------------------------------------


## Reads the step results the Control Center already has in hand: logging
## discrete events for the SELECTED environment and recording finished
## episodes for every environment. Never runs in TRAINING mode.
func _observe_step(step_results: Array) -> void:
	var selected: int = config.selected_environment
	for index in range(step_results.size()):
		var result: Dictionary = step_results[index]
		var info: Dictionary = result.get("info", {})
		var events: Dictionary = info.get("events", {})
		if index == selected:
			_track_selected_events(events)
		if bool(result.get("done", false)):
			_record_episode(index, info)
	if selected < simulation_manager.environments.size():
		_track_target_change(simulation_manager.environments[selected])


func _track_selected_events(events: Dictionary) -> void:
	var probe: Dictionary = _episode_probe
	if bool(events.get("shot_fired", false)):
		probe["shots"] = int(probe.get("shots", 0)) + 1
		if int(probe.get("first_shot_step", -1)) < 0:
			probe["first_shot_step"] = int(probe.get("steps", 0))
	if bool(events.get("hit", false)):
		event_log.log_event(
			ControlCenterEventLog.Category.COMBAT,
			"hit enemy for %.0f damage" % float(events.get("damage_dealt", 0.0)),
			{},
			"hit"
		)
	if bool(events.get("kill", false)):
		event_log.log_event(ControlCenterEventLog.Category.COMBAT, "enemy eliminated")
	if float(events.get("damage_taken", 0.0)) > 0.0:
		event_log.log_event(
			ControlCenterEventLog.Category.COMBAT,
			"took %.0f damage" % float(events.get("damage_taken", 0.0)),
			{},
			"damage_taken"
		)
	if bool(events.get("useless_shot", false)):
		probe["useless_shots"] = int(probe.get("useless_shots", 0)) + 1
		event_log.log_event(
			ControlCenterEventLog.Category.REWARD,
			"useless trigger pull (weapon on cooldown or no live target)",
			{},
			"useless_shot"
		)
	if bool(events.get("missed_shot", false)):
		probe["missed_shots"] = int(probe.get("missed_shots", 0)) + 1
		event_log.log_event(
			ControlCenterEventLog.Category.REWARD, "shot missed a live target", {}, "missed_shot"
		)
	if bool(events.get("died", false)):
		event_log.log_event(ControlCenterEventLog.Category.COMBAT, "agent died")
	probe["steps"] = int(probe.get("steps", 0)) + 1


## Target changes are a perception event: the observation's primary slot
## now refers to a different enemy.
func _track_target_change(env) -> void:
	var target_index: int = PerceptionModel.current_target_index(env)
	if target_index == _last_target_index:
		return
	if _last_target_index >= 0 and target_index >= 0:
		_episode_probe["target_switches"] = int(_episode_probe.get("target_switches", 0)) + 1
	event_log.log_event(
		ControlCenterEventLog.Category.PERCEPTION,
		(
			"target -> enemy #%d" % target_index
			if target_index >= 0
			else "target lost (no alive enemy in observation)"
		),
		{"previous": _last_target_index, "current": target_index}
	)
	_last_target_index = target_index

	# First moment an enemy is inside weapon range starts the reaction clock.
	if (
		int(_episode_probe.get("engagement_step", -1)) < 0
		and target_index >= 0
		and env.agent.position.distance_to(env.enemies[target_index].position)
		<= SandboxConfig.WEAPON_RANGE
	):
		_episode_probe["engagement_step"] = int(_episode_probe.get("steps", 0))


func _record_episode(env_index: int, info: Dictionary) -> void:
	var metrics: Dictionary = info.get("metrics", {})
	if metrics.is_empty():
		return
	var is_selected: bool = env_index == config.selected_environment
	var record: Dictionary = {
		"source": _action_source_for(env_index),
		"mode": ControlCenterConfig.mode_name(config.mode),
		"env_index": env_index,
		"episode": int(
			simulation_manager.environments[env_index].episode.episode_count
		),
		"seed": config.seed + env_index,
		"curriculum_level": config.curriculum_level,
		"enemy_count": int(metrics.get("enemy_count", 0)),
		"reward": float(metrics.get("episode_reward", 0.0)),
		"episode_length": int(metrics.get("episode_length", 0)),
		"kills": int(metrics.get("kills", 0)),
		"deaths": int(metrics.get("deaths", 0)),
		"damage_dealt": float(metrics.get("damage_dealt", 0.0)),
		"damage_received": float(metrics.get("damage_received", 0.0)),
		"shots_fired": int(metrics.get("shots_fired", 0)),
		"shots_hit": int(metrics.get("shots_hit", 0)),
		"accuracy": float(metrics.get("accuracy", 0.0)),
		"survival_time": float(metrics.get("survival_time", 0.0)),
		"win": bool(metrics.get("win", false)),
		"loss": bool(metrics.get("loss", false)),
		"done_reason": str(metrics.get("done_reason", "")),
		"reward_breakdown": metrics.get("reward_breakdown", {}),
		"wall_time": float(Time.get_ticks_msec()) / 1000.0,
	}
	if is_selected:
		record["reaction_time"] = _measured_reaction_time()
		record["useless_shots"] = int(_episode_probe.get("useless_shots", 0))
		record["missed_shots"] = int(_episode_probe.get("missed_shots", 0))
		record["target_switches"] = int(_episode_probe.get("target_switches", 0))
	else:
		record["reaction_time"] = -1.0

	var stored: Dictionary = results.record(record)
	event_log.log_event(
		ControlCenterEventLog.Category.SYSTEM,
		(
			"env %d episode end (%s): reward %.2f, kills %d, accuracy %.0f%%"
			% [
				env_index,
				str(record["done_reason"]),
				float(record["reward"]),
				int(record["kills"]),
				float(record["accuracy"]) * 100.0,
			]
		)
	)
	if is_selected:
		_reset_episode_probe()
	episode_recorded.emit(stored)


## Seconds between the first tick with an enemy inside weapon range and the
## first shot fired after that. Returns -1.0 when either never happened,
## so "not measured" is never reported as a real number.
func _measured_reaction_time() -> float:
	var engagement: int = int(_episode_probe.get("engagement_step", -1))
	var first_shot: int = int(_episode_probe.get("first_shot_step", -1))
	if engagement < 0 or first_shot < engagement:
		return -1.0
	return float(first_shot - engagement) * SandboxConfig.SIMULATION_DT


func _reset_episode_probe() -> void:
	_episode_probe = {
		"steps": 0,
		"shots": 0,
		"useless_shots": 0,
		"missed_shots": 0,
		"target_switches": 0,
		"engagement_step": -1,
		"first_shot_step": -1,
	}
	_last_target_index = -1


func _action_source_for(env_index: int) -> String:
	var controller: ControllerBase = simulation_manager.controllers[env_index]
	if controller == human_controller and controller != null:
		return "human"
	if controller == null:
		return "idle"
	if config.policy_source == ControlCenterConfig.PolicySource.IDLE:
		return "idle"
	return "ai"


# ---------------------------------------------------------------------------
# Controllers / views
# ---------------------------------------------------------------------------


func _create_controllers() -> void:
	for controller_value in _ai_controllers:
		var existing: Node = controller_value
		if existing != null and is_instance_valid(existing):
			existing.queue_free()
	_ai_controllers.clear()

	for index in range(simulation_manager.environments.size()):
		var ai := AIStubController.new()
		ai.name = "AIController_%d" % index
		add_child(ai)
		_ai_controllers.append(ai)

	if human_controller == null:
		human_controller = HumanController.new()
		human_controller.name = "HumanController"
		# Never grab the mouse on construction: the Control Center captures
		# input only when the operator switches to HUMAN mode.
		human_controller.start_mouse_captured = false
		add_child(human_controller)
		human_controller.set_process_unhandled_input(false)
	_bind_controllers()


## Binds one controller per environment according to the active mode.
## HUMAN mode swaps ONLY the selected environment to the human pipeline;
## every other environment keeps its AI controller, so the comparison runs
## on identical machinery.
func _bind_controllers() -> void:
	if simulation_manager == null:
		return
	for index in range(simulation_manager.environments.size()):
		var controller: ControllerBase = null
		if config.policy_source != ControlCenterConfig.PolicySource.IDLE:
			controller = _ai_controllers[index] if index < _ai_controllers.size() else null
		if is_human_mode() and index == config.selected_environment:
			controller = human_controller
		simulation_manager.set_controller(index, controller)


func _apply_mode_to_runtime() -> void:
	var training: bool = is_training_mode()
	event_log.enabled = telemetry_enabled()
	if not is_human_mode():
		# Leaving HUMAN mode always disarms input and releases the mouse.
		set_human_input_enabled(false)
	_bind_controllers()
	_apply_view_visibility()
	if presentation_enabled and not training:
		simulation_manager.ensure_view(config.selected_environment)
		simulation_manager.sync_view(config.selected_environment)


## Only the selected environment is ever visible, and only outside TRAINING
## mode. Hidden Node3D subtrees are not rendered at all, and this session
## never syncs them either.
func _apply_view_visibility() -> void:
	if simulation_manager == null:
		return
	var show_views: bool = presentation_enabled and not is_training_mode()
	for index in range(simulation_manager.views.size()):
		var view: Node3D = simulation_manager.views[index]
		if view != null and is_instance_valid(view):
			view.visible = show_views and index == config.selected_environment


# ---------------------------------------------------------------------------
# Snapshot / status
# ---------------------------------------------------------------------------


func get_status() -> Dictionary:
	var environment_count: int = (
		simulation_manager.environments.size() if simulation_manager != null else 0
	)
	return {
		"mode": config.mode,
		"mode_name": ControlCenterConfig.mode_name(config.mode),
		"running": running,
		"speed": config.simulation_speed,
		"steps_per_second": steps_per_second,
		"total_steps": total_steps,
		"render_fps": Engine.get_frames_per_second(),
		"environment_count": environment_count,
		"selected_environment": config.selected_environment,
		"selected_agent_slot": config.selected_agent_slot,
		"policy_source": config.policy_source,
		"policy_source_name": ControlCenterConfig.policy_source_name(config.policy_source),
		"curriculum_level": config.curriculum_level,
		"curriculum_name": CurriculumConfig.level_name(config.curriculum_level),
		"enemy_count": config.enemy_count,
		"seed": config.seed,
		"scenario_id": config.scenario_id,
		"human_input_enabled": human_input_enabled,
		"presentation_enabled": presentation_enabled,
		"telemetry_enabled": telemetry_enabled(),
		"pending_settings": _pending_setting_keys.duplicate(),
		"log_dropped": event_log.dropped_count,
		"episodes_recorded": results.size(),
	}


## Full UI snapshot for the selected environment. Returns {} in TRAINING
## mode (where telemetry is intentionally disabled) so a caller cannot
## accidentally pay for it on the throughput path.
func build_snapshot(options: Dictionary = {}) -> Dictionary:
	if not telemetry_enabled():
		return {}
	return ControlCenterTelemetry.build(simulation_manager, config, get_status(), options)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


func log_system(message: String) -> void:
	event_log.log_event(ControlCenterEventLog.Category.SYSTEM, message)


func log_warning(message: String) -> void:
	event_log.log_event(ControlCenterEventLog.Category.ERROR, message)
	push_warning("[ControlCenter] %s" % message)


## Exact headless training command for the current settings. Shown in the
## TRAINING panel because real PPO updates happen in the Python trainer,
## not in this window.
func training_command_line() -> String:
	return (
		"sandboxai train --env-count %d --enemy-count %d --curriculum-level %d --seed %d"
		% [config.environment_count, config.enemy_count, config.curriculum_level, config.seed]
	)

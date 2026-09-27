## SimulationManager
## Owns and ticks independent, render-free EnvironmentCore instances. Views
## and controllers are optional presentation/input adapters.
class_name SimulationManager
extends Node

signal environment_reset(env_index: int)
signal environment_done(env_index: int, reason: String)

@export var environment_count: int = SandboxConfig.DEFAULT_ENVIRONMENT_COUNT
@export var enemy_count_per_environment: int = SandboxConfig.ENEMY_COUNT_DEFAULT
@export var base_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED
@export var curriculum_level: int = CurriculumConfig.Level.ENEMY_ATTACKS
@export var create_visuals: bool = true
@export var auto_tick: bool = true
@export var auto_reset_on_done: bool = true
@export var view_cell_spacing: float = SandboxConfig.ARENA_HALF_EXTENT * 2.5 + 6.0

var environments: Array = []  # Array[EnvironmentCore]
var views: Array = []  # Array[EnvironmentView] or null
var controllers: Array = []  # Array[ControllerBase] or null
var recorders: Dictionary = {}  # env index -> DemoRecorder-like object
## Two named policy slots are an explicit self-play hook. A slot can contain
## a live controller or a frozen checkpoint path; the current single-agent
## combat environment uses slot 0, while slot 1 is available to the
## SelfPlayEnvironmentCore and future multi-agent modes.
var policy_slots: Array = []

var steps_per_second: float = 0.0
var _steps_since_report: int = 0
var _report_timer: float = 0.0
var _last_step_rewards: Array = []
var _last_step_dones: Array = []


func _ready() -> void:
	build(environment_count, enemy_count_per_environment)


func build(count: int, enemies_per_env: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	_clear()
	environment_count = maxi(1, count)
	enemy_count_per_environment = maxi(1, enemies_per_env)
	if (
		curriculum_level >= CurriculumConfig.Level.MULTIPLE_ENEMIES
		and curriculum_level < CurriculumConfig.Level.AGENT_VS_AGENT
	):
		enemy_count_per_environment = maxi(2, enemy_count_per_environment)
	policy_slots = [
		{"slot": 0, "controller": null, "frozen_checkpoint": "", "seed": base_seed},
		{"slot": 1, "controller": null, "frozen_checkpoint": "", "seed": base_seed + 1000003},
	]

	for i in range(environment_count):
		var env: EnvironmentCore = EnvironmentCore.new(i, enemy_count_per_environment)
		env.set_curriculum_level(curriculum_level)
		env.reset(base_seed + i)
		environments.append(env)
		controllers.append(null)
		_last_step_rewards.append(0.0)
		_last_step_dones.append(false)

		var view: EnvironmentView = null
		if create_visuals:
			view = EnvironmentView.new()
			view.name = "Environment_%d" % i
			add_child(view)
			view.setup(env)
			view.position = _cell_offset(i)
			views.append(view)
		else:
			views.append(null)


func _clear() -> void:
	for view in views:
		if view != null and is_instance_valid(view):
			view.queue_free()
	environments.clear()
	views.clear()
	controllers.clear()
	recorders.clear()
	_last_step_rewards.clear()
	_last_step_dones.clear()


func _cell_offset(index: int) -> Vector3:
	var columns: int = maxi(1, ceili(sqrt(float(environment_count))))
	var col: int = index % columns
	var row: int = index / columns
	return Vector3(col * view_cell_spacing, 0.0, row * view_cell_spacing)


func reset_all(seed_base_value: int = -1) -> Array:
	var observations: Array = []
	for i in range(environments.size()):
		var seed_value: int = (seed_base_value + i) if seed_base_value >= 0 else -1
		var env: EnvironmentCore = environments[i]
		env.reset(seed_value)
		_last_step_rewards[i] = 0.0
		_last_step_dones[i] = false
		observations.append(env.get_observations())
		environment_reset.emit(i)
	return observations


## Resets only selected environments, preserving all other trajectories.
func reset_indices(indices: Array, seed_base_value: int = -1) -> Array:
	var observations: Array = []
	for offset in range(indices.size()):
		var index: int = int(indices[offset])
		if index < 0 or index >= environments.size():
			continue
		var seed_value: int = (seed_base_value + offset) if seed_base_value >= 0 else -1
		var env: EnvironmentCore = environments[index]
		env.reset(seed_value)
		_last_step_rewards[index] = 0.0
		_last_step_dones[index] = false
		observations.append({"index": index, "observation": env.get_observations()})
		environment_reset.emit(index)
	return observations


## Steps every environment. When auto-reset is enabled, the returned result
## retains the terminal observation under terminal_observation and returns the
## fresh reset observation in observation, matching Gym vector semantics.
func step_all(actions: Array, dt: float = SandboxConfig.SIMULATION_DT) -> Array:
	var results: Array = []
	for i in range(environments.size()):
		var env: EnvironmentCore = environments[i]
		var action: Action = (
			actions[i] if i < actions.size() and actions[i] != null else Action.idle()
		)
		var pre_observation: PackedFloat32Array = env.get_observations().to_array()
		var result: Dictionary = env.step(action, dt)
		_last_step_rewards[i] = float(result.get("reward", 0.0))
		_last_step_dones[i] = bool(result.get("done", false))
		if recorders.has(i) and recorders[i] != null:
			var recorder = recorders[i]
			if recorder.has_method("record_transition"):
				recorder.record_transition(i, pre_observation, action, result)
		if result.done:
			environment_done.emit(i, env.episode.done_reason)
			if auto_reset_on_done:
				var terminal_observation = result.observation
				env.reset(-1)
				result["terminal_observation"] = terminal_observation
				result["observation"] = env.get_observations()
				result["auto_reset"] = true
		results.append(result)
	_steps_since_report += environments.size()
	return results


func run_headless_steps(n: int, action_provider: Callable = Callable()) -> void:
	for _step_index in range(n):
		var actions: Array = []
		for i in range(environments.size()):
			if action_provider.is_valid():
				actions.append(action_provider.call(i))
			elif controllers[i] != null:
				actions.append(controllers[i].get_action(environments[i]))
			else:
				actions.append(Action.idle())
		step_all(actions)


func get_observations() -> Array:
	var observations: Array = []
	for env in environments:
		observations.append((env as EnvironmentCore).get_observations())
	return observations


func get_rewards() -> Array:
	# Keep the reward from the most recent batch even when auto-reset has
	# already cleared the new episode's EpisodeState counters.
	return _last_step_rewards.duplicate()


func is_done_all() -> Array:
	# This reports the done flags from the latest transition, including a
	# terminal transition that was immediately auto-reset for vector training.
	return _last_step_dones.duplicate()


func get_metrics() -> Array:
	var metrics: Array = []
	for env in environments:
		metrics.append((env as EnvironmentCore).get_metrics())
	return metrics


func set_curriculum_level(level: int) -> void:
	curriculum_level = clampi(
		level, CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT
	)
	for env in environments:
		(env as EnvironmentCore).set_curriculum_level(curriculum_level)


func set_controller(env_index: int, controller: ControllerBase) -> void:
	if env_index >= 0 and env_index < controllers.size():
		controllers[env_index] = controller


func attach_recorder(env_index: int, recorder) -> void:
	if env_index >= 0 and env_index < environments.size():
		recorders[env_index] = recorder


func configure_policy_slot(
	slot_index: int, controller = null, frozen_checkpoint: String = "", seed: int = 0
) -> void:
	if slot_index < 0 or slot_index >= policy_slots.size():
		return
	policy_slots[slot_index] = {
		"slot": slot_index,
		"controller": controller,
		"frozen_checkpoint": frozen_checkpoint,
		"seed": seed,
	}


# ---------------------------------------------------------------------------
# Engine-driven tick for visual/human/demo mode.
# ---------------------------------------------------------------------------
func _physics_process(delta: float) -> void:
	if not auto_tick:
		return

	var actions: Array = []
	for i in range(environments.size()):
		var controller: ControllerBase = controllers[i]
		actions.append(
			controller.get_action(environments[i]) if controller != null else Action.idle()
		)
	step_all(actions, SandboxConfig.SIMULATION_DT)

	for i in range(views.size()):
		var view: EnvironmentView = views[i]
		if view != null:
			view.sync_from_state()

	_report_timer += delta
	if _report_timer >= 1.0:
		steps_per_second = _steps_since_report / _report_timer
		_steps_since_report = 0
		_report_timer = 0.0

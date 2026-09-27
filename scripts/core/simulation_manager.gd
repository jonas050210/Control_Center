## SimulationManager
##
## Owns and ticks N independent EnvironmentCore instances. This is the class
## a future RL trainer (or the human-play bootstrap in scripts/core/main.gd)
## talks to. Each EnvironmentCore is a fully independent object graph (its
## own AgentState, EnemyState array, EpisodeState, RNG) — nothing is shared
## between environments, so resetting or stepping one never affects another.
##
## Rendering is optional and decoupled from simulation: when `create_visuals`
## is true, a lightweight EnvironmentView (Node3D) is spawned per
## environment purely to make it visible/playable; the simulation tick
## itself never depends on those views existing. This is what allows
## `run_headless_steps()` to advance simulation far faster than real time
## regardless of whether a window/renderer is present.
class_name SimulationManager
extends Node

signal environment_reset(env_index: int)
signal environment_done(env_index: int, reason: String)

@export var environment_count: int = SandboxConfig.DEFAULT_ENVIRONMENT_COUNT
@export var enemy_count_per_environment: int = SandboxConfig.ENEMY_COUNT_DEFAULT
@export var base_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED
@export var create_visuals: bool = true
@export var auto_tick: bool = true  # tick every _physics_process (visual/human/demo mode)
@export var auto_reset_on_done: bool = true
@export var view_cell_spacing: float = SandboxConfig.ARENA_HALF_EXTENT * 2.5 + 6.0

var environments: Array = []  # Array[EnvironmentCore]
var views: Array = []  # Array[EnvironmentView] (or null entries if headless)
var controllers: Array = []  # Array[ControllerBase] (or null -> idle action)

var steps_per_second: float = 0.0
var _steps_since_report: int = 0
var _report_timer: float = 0.0


func _ready() -> void:
	build(environment_count, enemy_count_per_environment)


## (Re)builds the whole set of environments from scratch. Safe to call again
## later to change the environment count at runtime.
func build(count: int, enemies_per_env: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	_clear()
	environment_count = maxi(1, count)
	enemy_count_per_environment = maxi(1, enemies_per_env)

	for i in range(environment_count):
		var env: EnvironmentCore = EnvironmentCore.new(i, enemy_count_per_environment)
		env.reset(base_seed + i)
		environments.append(env)
		controllers.append(null)

		var view: EnvironmentView = null
		if create_visuals:
			view = EnvironmentView.new()
			view.name = "Environment_%d" % i
			add_child(view)
			view.setup(env)
			view.position = _cell_offset(i)
		views.append(view)


func _clear() -> void:
	for v in views:
		if v != null and is_instance_valid(v):
			v.queue_free()
	environments.clear()
	views.clear()
	controllers.clear()


func _cell_offset(index: int) -> Vector3:
	# Simple grid layout so multiple environments are visually separated;
	# purely cosmetic, has no effect on simulation coordinates or logic.
	var columns: int = maxi(1, ceili(sqrt(float(environment_count))))
	var col: int = index % columns
	var row: int = index / columns
	return Vector3(col * view_cell_spacing, 0.0, row * view_cell_spacing)


# ---------------------------------------------------------------------------
# Batch RL-style interface
# ---------------------------------------------------------------------------


func reset_all(seed_base: int = -1) -> Array:
	var observations: Array = []
	for i in range(environments.size()):
		var seed_value: int = (seed_base + i) if seed_base >= 0 else -1
		var env: EnvironmentCore = environments[i]
		observations.append(env.reset(seed_value))
		environment_reset.emit(i)
	return observations


## Steps every environment with the given per-environment action (or an
## idle action if `actions` is shorter than the environment list / entries
## are null). Automatically resets any environment that finished, unless
## `auto_reset_on_done` is false.
func step_all(actions: Array, dt: float = SandboxConfig.SIMULATION_DT) -> Array:
	var results: Array = []
	for i in range(environments.size()):
		var env: EnvironmentCore = environments[i]
		var action: Action = (
			actions[i] if i < actions.size() and actions[i] != null else Action.idle()
		)
		var result: Dictionary = env.step(action, dt)
		results.append(result)
		if result.done:
			environment_done.emit(i, env.episode.done_reason)
			if auto_reset_on_done:
				env.reset(-1)
	_steps_since_report += environments.size()
	return results


## Advances every environment `n` times back-to-back with no regard for
## real time or rendering — the "fast/headless" path used for training
## throughput or for automated tests.
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
	var obs: Array = []
	for env in environments:
		obs.append((env as EnvironmentCore).get_observations())
	return obs


func get_rewards() -> Array:
	var rewards: Array = []
	for env in environments:
		rewards.append((env as EnvironmentCore).get_rewards())
	return rewards


func is_done_all() -> Array:
	var flags: Array = []
	for env in environments:
		flags.append((env as EnvironmentCore).is_done())
	return flags


func set_controller(env_index: int, controller: ControllerBase) -> void:
	if env_index >= 0 and env_index < controllers.size():
		controllers[env_index] = controller


# ---------------------------------------------------------------------------
# Engine-driven tick (visual / human-play / demo mode)
# ---------------------------------------------------------------------------


func _physics_process(delta: float) -> void:
	if not auto_tick:
		return

	var actions: Array = []
	for i in range(environments.size()):
		var controller: ControllerBase = controllers[i]
		if controller != null:
			actions.append(controller.get_action(environments[i]))
		else:
			actions.append(Action.idle())
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

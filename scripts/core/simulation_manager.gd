## SimulationManager
## Owns and ticks independent, render-free EnvironmentCore instances. Views
## and controllers are optional presentation/input adapters.
class_name SimulationManager
extends Node

signal environment_reset(env_index: int)
signal environment_done(env_index: int, reason: String)
## Emitted after build() finishes recreating the environment array, so
## observers (views, the Control Center, the debug overlay) can re-bind
## instead of silently holding references to freed environments.
signal environments_rebuilt(environment_count: int, enemies_per_environment: int)

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const ControllerBase = preload("res://scripts/input/controller_base.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const EnvironmentView = preload("res://scripts/env/environment_view.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

@export var environment_count: int = SandboxConfig.DEFAULT_ENVIRONMENT_COUNT
@export var enemy_count_per_environment: int = SandboxConfig.ENEMY_COUNT_DEFAULT
@export var base_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED
@export var curriculum_level: int = CurriculumConfig.Level.ENEMY_ATTACKS
@export var create_visuals: bool = true
## When `create_visuals` is true, restricts view creation to these
## environment indices. Empty (the default) keeps the historical behavior:
## one EnvironmentView per environment. The Control Center sets this to the
## single environment it currently renders so a 16-environment session does
## not allocate 16 arenas' worth of meshes; `ensure_view()` creates the
## others lazily if the user selects them. Purely presentational — it never
## changes simulation results.
@export var visual_environment_indices: PackedInt32Array = PackedInt32Array()
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

## Per-environment episode plans staged by the training pipeline (the
## `set_episode_plans` bridge command). A plan is a Dictionary with the
## EpisodePlan.replay_header_fields() field names: seed, map_id, scenario,
## lighting, enemy_count, curriculum_level.
##
## Lifecycle, deliberately deterministic:
##   * `pending_plans[i]` holds the NEXT episode to start in environment i.
##     Any reset of that environment (an explicit reset_all/reset_indices,
##     or the automatic reset after `done`) CONSUMES the pending plan and
##     records it as `current_plans[i]`.
##   * An explicit reset with no pending plan REPLAYS `current_plans[i]`
##     (the plan carries its own seed, so the replay is exact). This keeps
##     repeated VecEnv resets idempotent for the training loop.
##   * The auto-reset with no pending plan falls back to the legacy
##     behaviour (env.reset(-1)), i.e. continuing that environment's seeded
##     RNG stream. With no plans staged at all, behaviour is bit-for-bit
##     the pre-plan behaviour.
var pending_plans: Dictionary = {}
var current_plans: Dictionary = {}

var steps_per_second: float = 0.0
var _steps_since_report: int = 0
var _report_timer: float = 0.0
var _last_step_rewards: Array = []
var _last_step_dones: Array = []
## Mirrors the last resolved Action passed to each environment. Debug/GUI
## and telemetry consumers only; never read by the simulation itself.
var _last_actions: Array = []


func _ready() -> void:
	build(environment_count, enemy_count_per_environment)


## (Re)creates the environment array.
##
## Controllers are PRESERVED across a rebuild. Previously `build()` called
## `_clear()`, which emptied `controllers`, and nothing re-attached them —
## so changing the enemy count from the debug overlay silently detached the
## HumanController installed by main.gd and the player lost all input until
## the scene was reloaded. Controllers are keyed by environment index, so
## they are captured here and re-attached to the environments that still
## exist afterwards.
func build(count: int, enemies_per_env: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	var preserved_controllers: Array = controllers.duplicate()
	_clear()
	environment_count = maxi(1, count)
	enemy_count_per_environment = maxi(1, enemies_per_env)
	if (
		curriculum_level >= CurriculumConfig.Level.MULTIPLE_ENEMIES
		and curriculum_level < CurriculumConfig.Level.AGENT_VS_AGENT
	):
		# Keep the manager's recorded per-environment count consistent with
		# what the environments will actually contain (CurriculumConfig
		# enforces the same minimum via effective_enemy_count()).
		enemy_count_per_environment = maxi(
			CurriculumConfig.MULTIPLE_ENEMIES_MIN_COUNT, enemy_count_per_environment
		)
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
		_last_actions.append(Action.idle())

		views.append(null)
		if create_visuals and _should_create_view(i):
			_create_view(i)

	for i in range(mini(preserved_controllers.size(), controllers.size())):
		var controller = preserved_controllers[i]
		if controller != null:
			controllers[i] = controller
	environments_rebuilt.emit(environment_count, enemy_count_per_environment)


## Whether environment `index` gets a view up front. An empty
## `visual_environment_indices` means "every environment", preserving the
## original behavior for the existing scenes and tests.
func _should_create_view(index: int) -> bool:
	if visual_environment_indices.is_empty():
		return true
	return visual_environment_indices.has(index)


func _create_view(index: int) -> EnvironmentView:
	var view := EnvironmentView.new()
	view.name = "Environment_%d" % index
	add_child(view)
	view.setup(environments[index])
	view.position = _cell_offset(index)
	views[index] = view
	return view


## Returns the EnvironmentView for `index`, creating it on demand when
## visuals are enabled but this environment was skipped by
## `visual_environment_indices`. Returns null in a visual-free (headless)
## manager. Presentation-only; it does not touch environment state.
func ensure_view(index: int) -> EnvironmentView:
	if not create_visuals or index < 0 or index >= environments.size():
		return null
	var existing: EnvironmentView = views[index]
	if existing != null and is_instance_valid(existing):
		return existing
	return _create_view(index)


## Mirrors one environment's state onto its view, if that view exists.
## Used by callers that drive stepping themselves (auto_tick = false) and
## only want to pay the sync cost for the environment actually on screen.
func sync_view(index: int) -> void:
	if index < 0 or index >= views.size():
		return
	var view: EnvironmentView = views[index]
	if view != null and is_instance_valid(view):
		view.sync_from_state()


func _clear() -> void:
	for view in views:
		if view != null and is_instance_valid(view):
			view.queue_free()
	environments.clear()
	views.clear()
	controllers.clear()
	recorders.clear()
	pending_plans.clear()
	current_plans.clear()
	_last_step_rewards.clear()
	_last_step_dones.clear()
	_last_actions.clear()


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
		_reset_environment(i, seed_value)
		observations.append(env.get_observations())
	return observations


## Resets only selected environments, preserving all other trajectories.
## Seeding is by ENVIRONMENT index (seed_base + index), matching reset_all(),
## so the same environment always receives the same seed for a given seed
## base regardless of which other environments are reset in the same call.
func reset_indices(indices: Array, seed_base_value: int = -1) -> Array:
	var observations: Array = []
	for offset in range(indices.size()):
		var index: int = int(indices[offset])
		if index < 0 or index >= environments.size():
			continue
		var seed_value: int = (seed_base_value + index) if seed_base_value >= 0 else -1
		var env: EnvironmentCore = environments[index]
		_reset_environment(index, seed_value)
		observations.append({"index": index, "observation": env.get_observations()})
	return observations


## Core reset path for one environment. A pending episode plan (staged via
## set_episode_plan) is consumed and applied; otherwise the current plan is
## replayed exactly; otherwise the legacy seed path runs.
func _reset_environment(index: int, seed_value: int) -> void:
	var env: EnvironmentCore = environments[index]
	if pending_plans.has(index):
		var plan: Dictionary = pending_plans[index]
		pending_plans.erase(index)
		current_plans[index] = plan
		_apply_episode_plan(env, plan)
	elif current_plans.has(index):
		_apply_episode_plan(env, current_plans[index])
	else:
		env.reset(seed_value)
	_last_step_rewards[index] = 0.0
	_last_step_dones[index] = false
	environment_reset.emit(index)


## Applies one staged episode plan to an environment, in the same order
## EpisodePlan.environment_commands() documents on the Python side:
## curriculum level, enemy count, map, lighting, scenario, reset(seed).
## Plans are validated when staged, so this path cannot half-configure an
## environment on a typo'd id.
func _apply_episode_plan(env: EnvironmentCore, plan: Dictionary) -> void:
	env.set_curriculum_level(int(plan.get("curriculum_level", curriculum_level)))
	env.set_enemy_count(int(plan.get("enemy_count", env.enemy_count)))
	env.set_map(str(plan.get("map_id", "")))
	env.set_lighting_mode(str(plan.get("lighting", "")))
	env.set_scenario(str(plan.get("scenario", "")))
	env.set_weapon_profile(str(plan.get("weapon_profile", "")))
	env.reset(int(plan.get("seed", -1)))


## The episode configuration each environment actually resolved (the
## ground truth of what ran), per environment.
func get_episode_conditions() -> Array:
	var conditions: Array = []
	for env in environments:
		conditions.append((env as EnvironmentCore).get_episode_condition())
	return conditions


## True while every environment has a staged next episode. Diagnostics only.
func all_environments_planned() -> bool:
	for i in range(environments.size()):
		if not pending_plans.has(i):
			return false
	return true


## Steps every environment. When auto-reset is enabled, the returned result
## retains the terminal observation under terminal_observation and returns the
## fresh reset observation in observation, matching Gym vector semantics.
func step_all(
	actions: Array, dt: float = SandboxConfig.SIMULATION_DT, compact_info: bool = false
) -> Array:
	var count: int = environments.size()
	var results: Array = []
	results.resize(count)
	var has_recorders: bool = not recorders.is_empty()
	for i in range(count):
		var env: EnvironmentCore = environments[i]
		var action: Action = (
			actions[i] if i < actions.size() and actions[i] != null else Action.idle()
		)
		# The pre-step observation is only needed by an attached recorder.
		# Building the 84-float array for every environment on every step
		# would be pure wasted work on the headless training path, where no
		# recorder ever exists, so it is computed lazily here.
		var recorder = recorders.get(i) if has_recorders else null
		var pre_observation: PackedFloat32Array = (
			env.get_observations().to_array()
			if recorder != null and recorder.has_method("record_transition")
			else PackedFloat32Array()
		)
		var result: Dictionary = env.step(action, dt, compact_info)
		_last_step_rewards[i] = float(result.get("reward", 0.0))
		_last_step_dones[i] = bool(result.get("done", false))
		_last_actions[i] = action
		if recorder != null and recorder.has_method("record_transition"):
			recorder.record_transition(i, pre_observation, action, result)
		if result.done:
			environment_done.emit(i, env.episode.done_reason)
			if auto_reset_on_done:
				var terminal_observation = result.observation
				if pending_plans.has(i):
					# The staged plan is consumed exactly once, here, so the
					# fresh observation returned in `result.observation` is
					# already from the NEW episode. Without this the
					# trainer would see one stray observation from the old
					# configuration at every episode boundary.
					var plan: Dictionary = pending_plans[i]
					pending_plans.erase(i)
					current_plans[i] = plan
					_apply_episode_plan(env, plan)
				else:
					env.reset(-1)
				result["terminal_observation"] = terminal_observation
				result["observation"] = env.get_observations()
				result["auto_reset"] = true
		results[i] = result
	_steps_since_report += count
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
	return _last_step_rewards.duplicate()


## Last resolved Action applied to each environment (debug/GUI/telemetry
## only). Returns Action.idle() for an environment that has not stepped yet.
func get_last_actions() -> Array:
	return _last_actions.duplicate()


func is_done_all() -> Array:
	return _last_step_dones.duplicate()


func get_metrics() -> Array:
	var metrics: Array = []
	for env in environments:
		metrics.append((env as EnvironmentCore).get_metrics())
	return metrics


func get_reward_breakdowns() -> Array:
	var breakdowns: Array = []
	for env in environments:
		breakdowns.append((env as EnvironmentCore).episode.get_reward_breakdown())
	return breakdowns


func health_check_all() -> Array:
	var reports: Array = []
	for env in environments:
		reports.append((env as EnvironmentCore).health_check())
	return reports


func set_curriculum_level(level: int) -> void:
	curriculum_level = clampi(
		level, CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT
	)
	for env in environments:
		(env as EnvironmentCore).set_curriculum_level(curriculum_level)
	# A level change can rebuild each environment's enemy list; refresh any
	# attached views so they mirror the live EnemyState objects.
	for i in range(views.size()):
		var view: EnvironmentView = views[i]
		if view != null and is_instance_valid(view):
			view.rebind_enemy_views()


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

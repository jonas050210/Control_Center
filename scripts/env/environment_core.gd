# gdlint:ignore=max-public-methods
# The public surface is intentionally wide: it is the RL interface
# (reset/step/get_*), plus the seven read-only introspection hooks
# PerceptionModel probes by name for the Control Center. Splitting the
# hooks into a helper object would mean the Control Center could no longer
# discover them with has_method() on the environment, which is the whole
# mechanism that keeps the debug GUI optional.
## EnvironmentCore
##
## The render-independent RL environment. All gameplay state is local to this
## object graph, so one instance can be reset or stepped without affecting
## another instance. The public contract is reset(seed) -> Observation and
## step(Action) -> {observation, reward, done, info}.
##
## Layering (added with the world/perception milestone):
##
##   ArenaWorld        static seeded geometry, collision + ray queries
##   CharacterMotor    shared gravity/jump/collision integration
##   SoundBus          transient audible events
##   AgentPerception   what the POLICY is allowed to know (FOV/LOS/memory)
##   EnemyBrain        what the OPPONENTS know and do (same rules)
##
## Every one of those is optional and driven by `CurriculumConfig` flags.
## With them all off (levels 1-4) this class executes exactly the original
## analytic code path: no world, no perception, no sound, no allocations
## beyond what it always made. That is deliberate — it keeps the cheap
## levels cheap for massively parallel training and keeps previously
## trained policies reproducible.
class_name EnvironmentCore
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentPerception = preload("res://scripts/perception/agent_perception.gd")
const AgentState = preload("res://scripts/agent/agent_state.gd")
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyBrain = preload("res://scripts/enemy/enemy_brain.gd")
const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EnvironmentIntrospection = preload("res://scripts/env/environment_introspection.gd")
const EnvironmentReset = preload("res://scripts/env/environment_reset.gd")
const EpisodeState = preload("res://scripts/core/episode_state.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const MapAnalyzer = preload("res://scripts/exploration/map_analyzer.gd")
const MapLibrary = preload("res://scripts/world/map_library.gd")
const NavigationGraph = preload("res://scripts/world/navigation_graph.gd")
const Observation = preload("res://scripts/core/observation.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const ReactionProfile = preload("res://scripts/perception/reaction_profile.gd")
const RewardSystem = preload("res://scripts/reward/reward_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const ScenarioLibrary = preload("res://scripts/scenario/scenario_library.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")
const SpatialMemory = preload("res://scripts/exploration/spatial_memory.gd")
const TargetSelector = preload("res://scripts/perception/target_selector.gd")

## Sound source id reserved for the agent. Enemies use their `enemy_id`.
const AGENT_SOUND_SOURCE: int = -1

## Default scenario played by each world-backed curriculum level. Level 10
## ignores this and draws a random scenario per episode.
const LEVEL_SCENARIOS: Dictionary = {
	CurriculumConfig.Level.OBSTACLES_COVER: "cover_fight",
	CurriculumConfig.Level.FOV_LOS: "corner_fight",
	CurriculumConfig.Level.SOUND: "sound_only",
	CurriculumConfig.Level.MEMORY_LOST_TARGETS: "target_disappears",
	CurriculumConfig.Level.VERTICAL_COMBAT: "vertical_encounter",
}

var env_id: int = 0
var arena_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
## The arena size this environment was configured with. A map may override
## `arena_half_extent` for its episode; clearing the map restores this.
var configured_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var max_steps: int = SandboxConfig.MAX_EPISODE_STEPS
var enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT
var curriculum: CurriculumConfig = CurriculumConfig.new()

var agent: AgentState = AgentState.new()
var enemies: Array = []  # Array[EnemyState]
var episode: EpisodeState = EpisodeState.new()
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

## Static geometry for this episode. `null` on the obstacle-free levels,
## which is the signal every downstream system uses to take the cheap path.
var world: ArenaWorld = null
## Walkable navigation graph for the current geometry. Built LAZILY on the
## first tick an enemy is actually blocked, so obstacle-free levels and
## episodes where nobody ever gets stuck never pay the bake cost.
var navigation: NavigationGraph = null
## Transient audible events. Only ticked when the curriculum enables sound.
var sound_bus: SoundBus = SoundBus.create()
## The agent's perception state (contact timers, memory, heard events).
var perception: AgentPerception = AgentPerception.create()

## Explicit map override. Empty means "let the scenario generate its own
## geometry" (the pre-map behavior). When set, `MapLibrary` owns the
## geometry, the arena size and the lighting, and the scenario only places
## the characters into it.
var map_id: String = ""
## Resolved map descriptor for the current episode ({} when none).
var map_instance: Dictionary = {}
## Explicit lighting override (a LightingProfile mode id). Empty means "use
## the map's lighting", which for a map-less episode is NORMAL.
var lighting_mode_id: String = ""
## Environmental visibility for this episode. Never null.
var lighting: LightingProfile = LightingProfile.create()

## Persistent, perception-built knowledge of the map (Map Analyzer). Null
## unless `exploration_tracking` is on, so combat training pays nothing for
## it. It is fed ONLY from what the agent perceives.
var exploration: MapAnalyzer = null
## Maintain the spatial memory alongside normal play.
var exploration_tracking: bool = false
## Dedicated Map Analyzer episode: the objective is coverage, not combat.
## Implies `exploration_tracking`, spawns no enemies unless the caller asks
## for them, and ends when the map is explored.
var exploration_mode: bool = false
## In exploration mode, whether the map is empty of enemies.
var exploration_solo: bool = true

## Explicit scenario override. Empty means "use the curriculum's layout".
var scenario_id: String = ""
## The resolved scenario spec for the current episode ({} when none).
var scenario: Dictionary = {}

## When true, perception is evaluated every step even if the curriculum
## does not gate the observation with it. This exists purely so the Control
## Center can draw FOV cones and LOS rays at any level; it defaults to
## false and headless training never turns it on, so the hot path never
## pays for the extra ray casts.
var debug_perception: bool = false

## Enemy id that most recently damaged the agent; biases target selection.
var last_damage_source: int = -1
## Seed the current episode was generated from (-1 = unseeded stream).
var episode_seed: int = -1

var _last_observation: Observation = null
var _has_reset: bool = false
## Latest belief list, kept for the Control Center and target reporting.
var _beliefs: Array = []
## Reused per-step scratch dictionary handed to EnemyBrain.update(), so the
## tactical path allocates nothing per environment per tick.
var _brain_context: Dictionary = {}
var _target_reason: String = "no target"
## Latest target-selection result (TargetSelector.select()).
var _selection: Dictionary = {}
## Enemy id of the contact currently in the primary slot (-1 = none).
var _target_id: int = -1
## Seconds since the primary slot last changed to a different contact.
var _time_since_target_switch: float = SandboxConfig.TARGET_SWITCH_RECENT_WINDOW
## Reused scratch dictionary for the Map Analyzer update (no per-tick alloc).
var _exploration_context: Dictionary = {}
## Whether the exploration-complete bonus has already been paid this episode.
var _exploration_paid: bool = false


func _init(p_env_id: int = 0, p_enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	env_id = p_env_id
	configured_half_extent = arena_half_extent
	enemy_count = maxi(1, p_enemy_count)
	curriculum = CurriculumConfig.new(CurriculumConfig.Level.ENEMY_ATTACKS, enemy_count)
	_rebuild_enemies(enemy_count)


func _rebuild_enemies(count: int) -> void:
	enemies.clear()
	for i in range(count):
		var enemy: EnemyState = EnemyState.new()
		enemy.enemy_id = i
		enemies.append(enemy)


# ---------------------------------------------------------------------------
# RL interface
# ---------------------------------------------------------------------------


func set_curriculum_level(level: int) -> void:
	curriculum.level = clampi(
		level, CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT
	)
	var target_count: int = curriculum.effective_enemy_count()
	if enemies.size() != target_count:
		enemy_count = target_count
		_rebuild_enemies(enemy_count)

	for enemy in enemies:
		_apply_enemy_difficulty(enemy)


## Forces a specific scenario for subsequent resets. Pass "" to go back to
## the curriculum-driven layout. Unknown ids are rejected so a typo fails
## loudly here instead of silently training on the wrong encounter.
func set_scenario(p_scenario_id: String) -> bool:
	if p_scenario_id.is_empty():
		scenario_id = ""
		return true
	if not ScenarioLibrary.has_scenario(p_scenario_id):
		return false
	scenario_id = p_scenario_id
	return true


## Forces a specific authored map for subsequent resets. Pass "" to go back
## to scenario-generated geometry. Unknown ids are rejected so a typo fails
## here instead of silently training on the wrong environment.
##
## The map id is deliberately NOT part of the observation: it selects an
## environment, it is not a label the policy may condition on.
func set_map(p_map_id: String) -> bool:
	if p_map_id.is_empty():
		map_id = ""
		return true
	if not MapLibrary.has_map(p_map_id):
		return false
	map_id = p_map_id
	return true


## Forces a lighting mode for subsequent resets ("" = use the map default).
func set_lighting_mode(mode_id: String) -> bool:
	if mode_id.is_empty():
		lighting_mode_id = ""
		return true
	if not LightingProfile.MODE_IDS.has(mode_id):
		return false
	lighting_mode_id = mode_id
	return true


## Validates an episode plan (set_episode_plan wire format: seed, map_id,
## scenario, lighting, enemy_count, curriculum_level). Returns "" when the
## plan can be applied, otherwise a human-readable reason. Validating at
## staging time means a misconfigured training distribution fails loudly
## before a single step runs instead of silently training on defaults.
func validate_episode_plan(plan: Dictionary) -> String:
	return EnvironmentReset.validate_episode_plan(plan, curriculum.level)


## Sets the configured enemy count for subsequent resets. The curriculum
## minimum (MULTIPLE_ENEMIES at level 4+) still applies through
## effective_enemy_count(), so the resolved count can be higher than the
## requested one; callers that need the truth read get_episode_condition().
func set_enemy_count(count: int) -> void:
	curriculum.configured_enemy_count = maxi(1, count)
	var target_count: int = curriculum.effective_enemy_count()
	if enemies.size() != target_count:
		enemy_count = target_count
		_rebuild_enemies(enemy_count)
	for enemy in enemies:
		_apply_enemy_difficulty(enemy)


## The episode configuration the engine actually resolved, in the
## EpisodePlan.replay_header_fields() field names. This is the ground truth
## of what ran: the RESOLVED enemy count (after curriculum minimums) and
## the RESOLVED lighting (explicit override, else map, else normal).
func get_episode_condition() -> Dictionary:
	return EnvironmentReset.episode_condition(self)


## Turns the spatial memory on/off without changing the objective. Useful to
## give a combat policy map knowledge, or to draw the Map Analyzer view
## while a normal fight is running.
func set_exploration_tracking(enabled: bool) -> void:
	exploration_tracking = enabled
	if not enabled and not exploration_mode:
		exploration = null


## Switches the whole episode into Map Analyzer mode. Takes effect on the
## next reset(). `solo` removes the enemies so exploration is measured on
## its own; pass false to explore a populated map.
func set_exploration_mode(enabled: bool, solo: bool = true) -> void:
	exploration_mode = enabled
	exploration_solo = solo
	if enabled:
		exploration_tracking = true


## Applies every curriculum-derived per-enemy parameter to one enemy. Used
## both at reset() and by set_curriculum_level() so a mid-episode level
## change takes effect consistently on the existing enemy list (previously
## only the radius was updated immediately, leaving speed/cooldown/damage at
## the previous level's values until the next reset).
func _apply_enemy_difficulty(enemy: EnemyState) -> void:
	enemy.radius = SandboxConfig.ENEMY_RADIUS * curriculum.target_radius_scale()
	enemy.move_speed = SandboxConfig.ENEMY_MOVE_SPEED * curriculum.enemy_speed_scale()
	enemy.attack_damage = SandboxConfig.ENEMY_ATTACK_DAMAGE
	enemy.attack_cooldown_time = (
		SandboxConfig.ENEMY_ATTACK_COOLDOWN * curriculum.enemy_cooldown_scale()
	)
	enemy.reaction.apply_archetype(curriculum.enemy_archetype())
	enemy.weapon.damage = SandboxConfig.ENEMY_FIRE_DAMAGE
	enemy.weapon.cooldown_time = SandboxConfig.ENEMY_FIRE_COOLDOWN
	enemy.weapon.range_m = SandboxConfig.ENEMY_FIRE_RANGE


## Deterministically (re)starts an episode. Passing the same seed produces
## the same world, the same spawn positions and the same first observation.
## Passing a negative seed keeps the current RNG stream: a fresh
## RandomNumberGenerator is already randomized at construction, so unseeded
## use stays random, while a previously seeded environment continues its
## deterministic sequence (essential for reproducible training across
## auto-resets).
func reset(seed_value: int = -1) -> Observation:
	if seed_value >= 0:
		rng.seed = seed_value
	episode_seed = seed_value

	var target_count: int = curriculum.effective_enemy_count()
	if exploration_mode and exploration_solo:
		target_count = 0
	if enemies.size() != target_count:
		enemy_count = target_count
		_rebuild_enemies(enemy_count)

	sound_bus.clear()
	navigation = null
	map_instance = {}
	lighting = LightingProfile.create()
	perception.reset()
	perception.configure(
		curriculum.perception_enabled() or exploration_mode,
		curriculum.sound_enabled(),
		curriculum.memory_enabled()
	)
	last_damage_source = -1
	_beliefs = []
	_target_reason = "no target"
	_selection = {}
	_target_id = -1
	_time_since_target_switch = SandboxConfig.TARGET_SWITCH_RECENT_WINDOW

	if _world_enabled():
		_reset_with_world()
	else:
		world = null
		scenario = {}
		# NOTE: no rng.randi() here. The obstacle-free levels must consume
		# exactly the same number of random draws they always did, or
		# previously trained policies stop reproducing.
		_apply_lighting(maxi(0, episode_seed))
		_reset_legacy()
	perception.set_lighting(lighting)
	_reset_exploration()

	episode.start_new_episode()
	_has_reset = true
	_last_observation = _build_observation()
	return _last_observation


## (Re)builds the Map Analyzer for the episode. Always starts empty: map
## knowledge is earned per episode and never carried in from outside.
func _reset_exploration() -> void:
	if not exploration_tracking and not exploration_mode:
		exploration = null
		return
	if exploration == null:
		exploration = MapAnalyzer.create(arena_half_extent)
	else:
		exploration.configure(arena_half_extent, SandboxConfig.EXPLORATION_CELL_SIZE)
	_exploration_paid = false


## Episode setup lives in EnvironmentReset (see that file for why). These
## wrappers keep the call sites readable and the hot loop untouched.
func _reset_legacy() -> void:
	EnvironmentReset.reset_legacy(self)


func _reset_with_world() -> void:
	EnvironmentReset.reset_with_world(self)


func _apply_lighting(layout_seed: int) -> void:
	EnvironmentReset.apply_lighting(self, layout_seed)


func _world_enabled() -> bool:
	return (
		curriculum.obstacles_enabled()
		or not scenario_id.is_empty()
		or not map_id.is_empty()
	)


func step(action: Action, dt: float = SandboxConfig.SIMULATION_DT) -> Dictionary:
	if not _has_reset:
		reset(SandboxConfig.DEFAULT_RANDOM_SEED)

	if episode.done:
		return _make_step_result(
			0.0,
			{
				"already_done": true,
				"done_reason": episode.done_reason,
				"TimeLimit.truncated": episode.done_reason == "timeout",
				"metrics": get_metrics()
			}
		)
	if action == null:
		action = Action.idle()

	var alive_before: bool = agent.alive
	var prev_enemy: EnemyState = _nearest_alive_enemy(agent.position)
	var prev_distance: float = (
		agent.position.distance_to(prev_enemy.position) if prev_enemy != null else 0.0
	)
	var prev_alignment: float = _target_alignment(prev_enemy)

	var sound_on: bool = curriculum.sound_enabled()
	var motion: Dictionary = agent.apply_action(action, dt, arena_half_extent, world)
	if sound_on:
		_emit_motion_sounds(motion, agent.position, AGENT_SOUND_SOURCE)

	# Positioning reward: closure caused by the agent's OWN motion only,
	# measured immediately after the agent moved and before enemies advance.
	# (Measuring after enemy movement let a standing-still agent farm reward
	# for letting an enemy walk up to it.) Only measured against the same
	# enemy while it is still alive, preventing false penalties when an
	# enemy is killed and the target switches.
	var positioning_delta: float = 0.0
	if prev_enemy != null and prev_enemy.alive and prev_distance > SandboxConfig.ENEMY_ATTACK_RANGE:
		positioning_delta = prev_distance - agent.position.distance_to(prev_enemy.position)

	var shot: Dictionary = _resolve_agent_shot(action, sound_on)
	var aiming_delta: float = _target_alignment(prev_enemy) - prev_alignment
	var meaningful_action: bool = bool(shot["shot_fired"]) or aiming_delta > 0.0 or positioning_delta > 0.0

	var damage_taken: float = _update_enemies(dt, sound_on)
	if damage_taken > 0.0:
		episode.record_damage_taken(damage_taken)

	if sound_on:
		sound_bus.tick(dt)
	if curriculum.perception_enabled() or debug_perception or exploration != null:
		_select_target(
			perception.update(agent, enemies, world, sound_bus if sound_on else null, dt), dt
		)

	var exploration_summary: Dictionary = {}
	if exploration != null:
		exploration_summary = _update_exploration(dt, damage_taken)

	var died: bool = alive_before and not agent.alive
	if died:
		episode.record_death()

	var events: Dictionary = {
		"hit": bool(shot["hit"]),
		"kill": bool(shot["kill"]),
		"damage_taken": damage_taken,
		"damage_dealt": float(shot["damage_dealt"]),
		"died": died,
		"useless_shot": bool(shot["useless_shot"]),
		"missed_shot": bool(shot["missed_shot"]),
		"shot_fired": bool(shot["shot_fired"]),
		"positioning_delta": positioning_delta,
		"aiming_delta": aiming_delta,
		"valid_target": prev_enemy != null and prev_enemy.is_targetable(),
		"meaningful_action": meaningful_action,
		"alive": agent.alive,
	}
	# Exploration only pays in the dedicated Map Analyzer mode. With
	# tracking enabled during a normal fight the map knowledge is still
	# built, but it must not distort the combat reward.
	if exploration_mode and exploration != null:
		events["exploration_gain"] = exploration.exploration_reward()
		events["exploration_complete"] = (
			bool(exploration_summary.get("completed", false)) and not _exploration_paid
		)
		if bool(events["exploration_complete"]):
			_exploration_paid = true
	var reward_components: Dictionary = RewardSystem.compute_components(events)
	var reward: float = RewardSystem.components_total(reward_components)
	episode.record_step(reward)
	episode.record_reward_breakdown(events)

	var done: bool = false
	var reason: String = ""
	if SandboxConfig.END_EPISODE_ON_AGENT_DEATH and not agent.alive:
		done = true
		reason = "agent_died"
	elif exploration_mode and exploration != null and exploration.completed:
		done = true
		reason = "map_explored"
	elif (
		SandboxConfig.END_EPISODE_ON_ALL_ENEMIES_DEAD
		and enemies.size() > 0
		and _all_enemies_dead()
	):
		done = true
		reason = "all_enemies_eliminated"
	elif episode.is_timeout(max_steps):
		done = true
		reason = "timeout"
	if done:
		episode.mark_done(reason)

	_last_observation = _build_observation()
	return _make_step_result(
		reward,
		{
			"events": events,
			"reward_components": reward_components,
			"done_reason": episode.done_reason,
			"TimeLimit.truncated": episode.done_reason == "timeout",
			"metrics": get_metrics()
		}
	)


# ---------------------------------------------------------------------------
# Step sub-steps
# ---------------------------------------------------------------------------


func _target_alignment(target: EnemyState) -> float:
	if target == null or not target.is_targetable():
		return 0.0
	var direction: Vector3 = (target.get_chest_position() - agent.get_eye_position()).normalized()
	return agent.get_forward_vector().dot(direction)


## Resolves the agent's trigger pull.
##
## Two corrections relative to the original implementation:
##   * a DEAD agent can no longer fire (previously the shoot branch ran
##     regardless of `agent.alive`, letting a corpse score kills on the tick
##     it died);
##   * when a world exists, a shot is blocked by geometry, so you cannot
##     shoot an enemy through a wall.
func _resolve_agent_shot(action: Action, sound_on: bool) -> Dictionary:
	var result: Dictionary = {
		"hit": false,
		"kill": false,
		"useless_shot": false,
		"missed_shot": false,
		"shot_fired": false,
		"damage_dealt": 0.0,
	}
	if not action.shoot or not agent.alive:
		return result

	var shot_fired: bool = agent.weapon.try_fire()
	if not shot_fired:
		result["useless_shot"] = true
		return result
	result["shot_fired"] = true
	if sound_on:
		sound_bus.emit_sound(SoundBus.Category.SHOT, agent.position, AGENT_SOUND_SOURCE)

	var any_alive: bool = false
	var eye: Vector3 = agent.get_eye_position()
	var forward: Vector3 = agent.get_forward_vector()

	# Closest targetable enemy along the ray trajectory.
	var best_hit_enemy: EnemyState = null
	var best_hit_distance: float = INF
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if not enemy.is_targetable():
			continue
		any_alive = true
		var hit_dist: float = agent.weapon.ray_hit_distance(
			eye, forward, enemy.get_chest_position()
		)
		if hit_dist < 0.0 or hit_dist >= best_hit_distance:
			continue
		if world != null and world.segment_blocked(eye, enemy.get_chest_position()):
			continue
		best_hit_distance = hit_dist
		best_hit_enemy = enemy

	if best_hit_enemy != null:
		var applied: float = best_hit_enemy.take_damage(agent.weapon.damage)
		if applied > 0.0:
			result["hit"] = true
			result["damage_dealt"] = applied
			episode.record_damage_dealt(applied)
			if sound_on:
				sound_bus.emit_sound(
					SoundBus.Category.IMPACT, best_hit_enemy.position, best_hit_enemy.enemy_id
				)
			if not best_hit_enemy.alive:
				result["kill"] = true
				episode.record_kill()
				_on_enemy_died(best_hit_enemy, sound_on)

	# A real miss against a live target is a genuine aiming attempt
	# (cheap PENALTY_MISSED_SHOT); only pulls that cannot connect at
	# all are "useless" (main PENALTY_USELESS_SHOT). This keeps the
	# expected value of shooting positive while aim is being learned.
	result["missed_shot"] = any_alive and not bool(result["hit"])
	result["useless_shot"] = not any_alive
	episode.record_shot(bool(result["hit"]))
	return result


## Death bookkeeping. A corpse must stop being a target, stop being
## perceived and stop being remembered by anyone, immediately.
func _on_enemy_died(enemy: EnemyState, sound_on: bool) -> void:
	enemy.mark_dead(episode.step_count * SandboxConfig.SIMULATION_DT)
	if sound_on:
		sound_bus.emit_sound(SoundBus.Category.DEATH, enemy.death_position, enemy.enemy_id)
	perception.forget(enemy.enemy_id)
	if last_damage_source == enemy.enemy_id:
		last_damage_source = -1


## Advances every enemy and returns the total damage applied to the agent.
func _update_enemies(dt: float, sound_on: bool) -> float:
	var damage_taken: float = 0.0
	var tactical: bool = curriculum.tactical_enemies_enabled()
	# The brain context is a REUSED member dictionary rather than a fresh
	# literal per step: with 64 parallel environments at 60 Hz this would
	# otherwise allocate ~230k short-lived dictionaries per simulated
	# second, all of which the GC then has to sweep.
	var context: Dictionary = _brain_context
	if tactical:
		context["world"] = world
		context["navigation"] = _ensure_navigation()
		context["lighting"] = lighting
		context["sound_bus"] = sound_bus if sound_on else null
		context["rng"] = rng
		context["dt"] = dt
		context["arena_half_extent"] = arena_half_extent
		context["agent_position"] = agent.position
		context["agent_eye"] = agent.get_eye_position()
		context["agent_height"] = agent.height
		context["agent_alive"] = agent.alive
		context["allow_movement"] = curriculum.enemy_movement_enabled()
		context["allow_attack"] = curriculum.enemy_attacks_enabled()
		context["allow_ranged"] = curriculum.ranged_enemies_enabled()
		context["allow_strafe"] = curriculum.strafing_enabled()
		context["allow_jump"] = curriculum.vertical_enabled()

	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if not enemy.is_targetable():
			continue
		var damage: float = 0.0
		if tactical:
			var brain_events: Dictionary = EnemyBrain.update(enemy, context)
			damage = float(brain_events["damage"])
			if sound_on:
				if bool(brain_events["shot"]):
					sound_bus.emit_sound(SoundBus.Category.SHOT, enemy.position, enemy.enemy_id)
				_emit_motion_sounds(brain_events, enemy.position, enemy.enemy_id)
		else:
			damage = enemy.update_ai(
				dt,
				agent.position,
				arena_half_extent,
				curriculum.enemy_movement_enabled(),
				curriculum.enemy_attacks_enabled(),
				curriculum.strafing_enabled()
			)
		if damage > 0.0:
			var applied: float = agent.take_damage(damage)
			if applied > 0.0:
				damage_taken += applied
				last_damage_source = enemy.enemy_id
				if sound_on:
					sound_bus.emit_sound(
						SoundBus.Category.IMPACT, agent.position, AGENT_SOUND_SOURCE
					)
	return damage_taken


## Turns a CharacterMotor/EnemyBrain motion event dictionary into sounds.
## Runs target selection over this tick's beliefs.
##
## `unreachable` is only populated when the navigation graph has ALREADY
## been baked by a stuck enemy; selection never forces the bake, so a
## perception tick keeps costing what it always cost.
func _select_target(beliefs: Array, dt: float) -> void:
	var unreachable: Array = []
	if navigation != null and navigation.is_ready():
		for belief_value in beliefs:
			var belief: Dictionary = belief_value
			if not navigation.is_reachable(agent.position, belief["position"]):
				unreachable.append(int(belief["id"]))
	_selection = TargetSelector.select(
		beliefs,
		{
			"damage_source": last_damage_source,
			"previous_target_id": _target_id,
			"unreachable": unreachable,
		}
	)
	_beliefs = _selection["ranked"]
	_target_reason = str(_selection["reason"])
	var winner: Dictionary = _selection["target"]
	var winner_id: int = int(winner.get("id", -1)) if not winner.is_empty() else -1
	if winner_id != _target_id:
		_time_since_target_switch = 0.0
		_target_id = winner_id
	else:
		_time_since_target_switch += dt


## Feeds one simulation step into the Map Analyzer.
##
## Everything handed over is something the agent itself perceived: its own
## pose, the lighting it stands in, the sounds it heard, and the fact that
## it took damage HERE. No enemy ground truth crosses this boundary; the
## shooter's real position is deliberately not passed, because the agent
## does not know it.
func _update_exploration(dt: float, damage_taken: float) -> Dictionary:
	_exploration_context["position"] = agent.position
	_exploration_context["forward"] = agent.get_forward_horizontal()
	_exploration_context["eye_height"] = SandboxConfig.AGENT_EYE_HEIGHT
	_exploration_context["world"] = world
	_exploration_context["lighting"] = lighting
	_exploration_context["fov_deg"] = perception.fov_deg
	_exploration_context["vision_range"] = perception.vision_range
	_exploration_context["local_illumination"] = perception.local_illumination
	_exploration_context["sounds"] = perception.heard
	_exploration_context["damage_taken_from"] = agent.position if damage_taken > 0.0 else null
	return exploration.update(dt, _exploration_context)


func _emit_motion_sounds(motion: Dictionary, position: Vector3, source_id: int) -> void:
	if bool(motion.get("footstep", false)):
		sound_bus.emit_sound(SoundBus.Category.FOOTSTEP, position, source_id)
	if bool(motion.get("jumped", false)):
		sound_bus.emit_sound(SoundBus.Category.JUMP, position, source_id)
	if bool(motion.get("landed", false)):
		sound_bus.emit_sound(SoundBus.Category.LAND, position, source_id)


## Assembles the observation, feeding the perception context only when the
## curriculum actually gates on it. `debug_perception` deliberately does NOT
## feed the context: the Control Center may look, but it may never change
## what the policy sees.
func _build_observation() -> Observation:
	if not curriculum.perception_enabled():
		return Observation.build(agent, enemies, arena_half_extent)
	return Observation.build(
		agent,
		enemies,
		arena_half_extent,
		{
			"beliefs": _beliefs,
			"sounds": perception.heard,
			"sound_summary": perception.sound_summary,
			"world": world,
			"forward_clearance": perception.forward_clearance,
			"in_cover": perception.in_cover,
			"corpse_count": get_corpse_count(),
			"enemy_slots": enemies.size(),
			"local_illumination": perception.local_illumination,
			"contact_summary": AgentPerception.summarize_contacts(
				_beliefs, Observation.MAX_TRACKED_ENEMIES
			),
			"target_priority_norm": float(_selection.get("priority_norm", 0.0)),
			"target_switch_recent": (
				_time_since_target_switch < SandboxConfig.TARGET_SWITCH_RECENT_WINDOW
			),
			"exploration": _exploration_observation(),
		}
	)


## The part of the agent's map knowledge that goes into the observation.
##
## Only locally meaningful quantities: how much of the map it has seen, the
## state of the ground under its feet, and the bearing/distance of the
## nearest cover and the nearest place it got hurt. Never the grid itself,
## never anything it has not observed. Returns {} when map tracking is off,
## which the observation reads as "no map knowledge".
func _exploration_observation() -> Dictionary:
	if exploration == null or exploration.memory == null:
		return {}
	var memory: SpatialMemory = exploration.memory
	var cover: Dictionary = memory.nearest_remembered_cover(agent.position)
	var danger: Dictionary = memory.nearest_remembered_danger(agent.position)
	var forward: Vector3 = agent.get_forward_horizontal()
	return {
		"explored_fraction": memory.coverage_fraction(),
		"area_known": memory.is_known(agent.position),
		"time_since_visit": memory.time_since_visit(agent.position),
		"cover_distance": float(cover.get("distance", -1.0)) if not cover.is_empty() else -1.0,
		"cover_bearing_deg": (
			PerceptionSystem.bearing_deg(forward, agent.position, cover["position"])
			if not cover.is_empty()
			else 0.0
		),
		"danger_distance": float(danger.get("distance", -1.0)) if not danger.is_empty() else -1.0,
		"danger_bearing_deg": (
			PerceptionSystem.bearing_deg(forward, agent.position, danger["position"])
			if not danger.is_empty()
			else 0.0
		),
	}


func get_observations() -> Observation:
	if _last_observation == null:
		return _build_observation()
	return _last_observation


func get_rewards() -> float:
	return episode.last_reward


func is_done() -> bool:
	return episode.done


func get_metrics() -> Dictionary:
	var won: bool = episode.done_reason == "all_enemies_eliminated"
	return episode.to_metrics(SandboxConfig.SIMULATION_DT, enemies.size(), won)


func health_check() -> Dictionary:
	return EnvironmentIntrospection.health_check(self)


# ---------------------------------------------------------------------------
# Debug / Control Center introspection hooks
#
# `PerceptionModel` probes for these by name with has_method(), so simply
# declaring them lights up the REAL WORLD / AI PERCEPTION / AI MEMORY /
# SOUND panels. They are read-only and are never called from step().
# ---------------------------------------------------------------------------


## Agent FOV cone parameters for the overlay.
func get_agent_field_of_view() -> Dictionary:
	return EnvironmentIntrospection.field_of_view(self)


## Map Analyzer state for the Control Center EXPLORATION view. Read-only,
## and privileged only in the sense that the UI sees the agent's beliefs all
## at once; every cell in it was earned by the agent looking at it.
func get_exploration_state() -> Dictionary:
	return EnvironmentIntrospection.exploration_state(self)


## Aggregate hearing state for the SOUND panel.
func get_sound_summary() -> Dictionary:
	return EnvironmentIntrospection.sound_summary(self)


## Raw line-of-sight query against the current geometry.
func has_line_of_sight(from_position: Vector3, to_position: Vector3) -> bool:
	if world == null:
		return true
	return not world.segment_blocked(from_position, to_position)


## Currently audible events from the AGENT's point of view.
func get_sound_events() -> Array:
	return EnvironmentIntrospection.sound_events(self)


## The agent's memory tracks plus the live belief list and the reason the
## current target was chosen.
func get_target_memory() -> Dictionary:
	return EnvironmentIntrospection.target_memory(self)


## The ranked belief list the policy acted on this step.
func get_beliefs() -> Array:
	return _beliefs


## Human-readable reason the current target was chosen (debug only).
func get_target_reason() -> String:
	return _target_reason


## Builds (once per episode) and returns the navigation graph for the
## current geometry, or null when there is no geometry at all.
##
## Baked for the ENEMY footprint: the enemies are the only navigation
## consumers, they are wider than the agent, and a graph baked for the
## wider body is conservatively valid for a narrower one. Sharing one graph
## per environment instead of one per enemy is what keeps the cost O(1) in
## the enemy count.
func _ensure_navigation() -> NavigationGraph:
	if world == null:
		return null
	if navigation == null:
		navigation = NavigationGraph.build(
			world, SandboxConfig.ENEMY_RADIUS, SandboxConfig.AGENT_HEIGHT
		)
	return navigation


## Navigation graph summary for the Control Center. Read-only: calling it
## does force the lazy bake, which costs time but cannot change simulation
## state, and the graph itself is a pure function of the geometry.
func get_navigation_graph_info() -> Dictionary:
	var graph: NavigationGraph = _ensure_navigation()
	if graph == null:
		return {"available": false, "node_count": 0}
	var info: Dictionary = graph.to_dict()
	info["available"] = true
	return info


## Environmental conditions for the Control Center. Includes the map's
## human-facing metadata, which the POLICY never receives.
func get_environment_conditions() -> Dictionary:
	return EnvironmentIntrospection.environment_conditions(self)


## Static geometry description for the overlay: one Dictionary per box.
func get_obstacles() -> Array:
	if world == null:
		return []
	return world.to_dict()["obstacles"]


## Full arena description (layout id, seed, bounds and boxes).
func get_world_description() -> Dictionary:
	if world == null:
		return {"layout_id": "none", "obstacle_count": 0, "obstacles": []}
	return world.to_dict()


## Layout/scenario metadata plus per-enemy tactical state.
func get_navigation_state() -> Dictionary:
	return EnvironmentIntrospection.navigation_state(self)


## Corpses, as pure environmental information. Never targetable.
func get_dead_bodies() -> Array:
	return EnvironmentIntrospection.dead_bodies(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


func _make_step_result(reward: float, info: Dictionary) -> Dictionary:
	return {
		"observation": get_observations(),
		"reward": reward,
		"done": episode.done,
		"info": info,
	}


func _nearest_alive_enemy(from_position: Vector3) -> EnemyState:
	var best: EnemyState = null
	var best_dist: float = INF
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if enemy.is_targetable():
			var distance: float = enemy.position.distance_squared_to(from_position)
			if distance < best_dist:
				best_dist = distance
				best = enemy
	return best


func _all_enemies_dead() -> bool:
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if enemy.alive:
			return false
	return true


func get_primary_enemy() -> EnemyState:
	return _nearest_alive_enemy(agent.position)


func get_alive_enemy_count() -> int:
	var count: int = 0
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if enemy.alive:
			count += 1
	return count


func get_corpse_count() -> int:
	var count: int = 0
	for enemy_value in enemies:
		if (enemy_value as EnemyState).corpse:
			count += 1
	return count


## The belief the policy is currently acting on ({} when it has no contact).
func get_current_target_belief() -> Dictionary:
	return _beliefs[0] if _beliefs.size() > 0 else {}

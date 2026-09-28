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
const EpisodeState = preload("res://scripts/core/episode_state.gd")
const Observation = preload("res://scripts/core/observation.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const ReactionProfile = preload("res://scripts/perception/reaction_profile.gd")
const RewardSystem = preload("res://scripts/reward/reward_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const ScenarioLibrary = preload("res://scripts/scenario/scenario_library.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")

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
## Transient audible events. Only ticked when the curriculum enables sound.
var sound_bus: SoundBus = SoundBus.create()
## The agent's perception state (contact timers, memory, heard events).
var perception: AgentPerception = AgentPerception.create()

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
var _target_reason: String = "no target"


func _init(p_env_id: int = 0, p_enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	env_id = p_env_id
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
	if enemies.size() != target_count:
		enemy_count = target_count
		_rebuild_enemies(enemy_count)

	sound_bus.clear()
	perception.reset()
	perception.configure(
		curriculum.perception_enabled(), curriculum.sound_enabled(), curriculum.memory_enabled()
	)
	last_damage_source = -1
	_beliefs = []
	_target_reason = "no target"

	if _world_enabled():
		_reset_with_world()
	else:
		world = null
		scenario = {}
		_reset_legacy()

	episode.start_new_episode()
	_has_reset = true
	_last_observation = _build_observation()
	return _last_observation


## Original obstacle-free reset. Untouched so curriculum levels 1-4 keep
## their exact spawn distributions for a given seed.
func _reset_legacy() -> void:
	agent.reset(SandboxConfig.AGENT_SPAWN_POSITION, SandboxConfig.AGENT_SPAWN_YAW_DEG)
	agent.weapon.hit_radius = SandboxConfig.WEAPON_HIT_RADIUS * curriculum.target_radius_scale()

	var spawn_variety: bool = curriculum.spawn_variety_enabled()
	var strafing: bool = curriculum.strafing_enabled()
	var angle_spread_deg: float = curriculum.spawn_angle_spread_deg()
	var distance_range: Vector2 = curriculum.spawn_distance_range()
	var spread: float = 2.5
	for i in range(enemies.size()):
		var enemy: EnemyState = enemies[i]
		var spawn: Vector3
		if spawn_variety:
			spawn = _random_spawn_position(angle_spread_deg, distance_range)
		else:
			# Legacy fixed layout: enemies lined up directly ahead of the
			# agent with only small positional jitter (curriculum level 1).
			var lateral: float = (i - (enemies.size() - 1) / 2.0) * spread
			var jitter_x: float = rng.randf_range(-0.5, 0.5)
			var jitter_z: float = rng.randf_range(-0.5, 0.5)
			spawn = SandboxConfig.ENEMY_SPAWN_POSITION + Vector3(lateral + jitter_x, 0.0, jitter_z)
		_reset_enemy(enemy, i, spawn, strafing)


## World-backed reset: geometry, scenario spawns and tactical enemies.
func _reset_with_world() -> void:
	var resolved_id: String = scenario_id
	if resolved_id.is_empty():
		resolved_id = _scenario_for_level()
	# The layout seed is drawn from the environment RNG rather than reusing
	# `seed_value` directly, so consecutive auto-resets of a seeded env
	# produce a varied but fully reproducible sequence of arenas.
	var layout_seed: int = rng.randi()
	scenario = ScenarioLibrary.resolve(
		resolved_id, layout_seed, enemies.size(), arena_half_extent
	)
	world = scenario["world"]

	agent.reset(scenario["agent_spawn"], float(scenario["agent_yaw_deg"]))
	agent.weapon.hit_radius = SandboxConfig.WEAPON_HIT_RADIUS * curriculum.target_radius_scale()

	var strafing: bool = curriculum.strafing_enabled()
	var spawns: Array = scenario["enemy_spawns"]
	for i in range(enemies.size()):
		var spawn: Vector3
		if i < spawns.size():
			spawn = spawns[i]
		else:
			spawn = world.sample_free_position(
				rng, SandboxConfig.ENEMY_RADIUS, SandboxConfig.AGENT_HEIGHT
			)
		_reset_enemy(enemies[i], i, spawn, strafing)


func _reset_enemy(enemy: EnemyState, index: int, spawn: Vector3, strafing: bool) -> void:
	enemy.enemy_id = index
	enemy.reset(spawn)
	_apply_enemy_difficulty(enemy)
	if strafing:
		enemy.strafe_direction = 1.0 if rng.randf() > 0.5 else -1.0
		enemy.strafe_phase = rng.randf_range(0.0, TAU)
	else:
		enemy.strafe_direction = 1.0
		enemy.strafe_phase = 0.0
	# Face the agent's spawn so the FOV cone starts somewhere sensible.
	enemy.yaw_deg = rad_to_deg(
		atan2(agent.position.x - spawn.x, -(agent.position.z - spawn.z))
	)


## Which scenario the current curriculum level plays. Level 10 draws a new
## one from the seeded RNG every episode (Phase 9/10 "mixed randomized").
func _scenario_for_level() -> String:
	if curriculum.randomized_scenarios_enabled():
		var ids: PackedStringArray = ScenarioLibrary.ids()
		return ids[rng.randi_range(0, ids.size() - 1)]
	return str(LEVEL_SCENARIOS.get(curriculum.level, "open_arena"))


func _world_enabled() -> bool:
	return curriculum.obstacles_enabled() or not scenario_id.is_empty()


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

	var damage_taken: float = _update_enemies(dt, sound_on)
	if damage_taken > 0.0:
		episode.record_damage_taken(damage_taken)

	if sound_on:
		sound_bus.tick(dt)
	if curriculum.perception_enabled() or debug_perception:
		_beliefs = AgentPerception.rank_beliefs(
			perception.update(agent, enemies, world, sound_bus if sound_on else null, dt),
			last_damage_source
		)
		_target_reason = AgentPerception.selection_reason(
			_beliefs[0] if _beliefs.size() > 0 else {}, last_damage_source
		)

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
		"alive": agent.alive,
	}
	var reward: float = RewardSystem.compute(events)
	episode.record_step(reward)
	episode.record_reward_breakdown(events)

	var done: bool = false
	var reason: String = ""
	if SandboxConfig.END_EPISODE_ON_AGENT_DEATH and not agent.alive:
		done = true
		reason = "agent_died"
	elif SandboxConfig.END_EPISODE_ON_ALL_ENEMIES_DEAD and _all_enemies_dead():
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
			"done_reason": episode.done_reason,
			"TimeLimit.truncated": episode.done_reason == "timeout",
			"metrics": get_metrics()
		}
	)


# ---------------------------------------------------------------------------
# Step sub-steps
# ---------------------------------------------------------------------------


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
	var context: Dictionary = {}
	if tactical:
		context = {
			"world": world,
			"sound_bus": sound_bus if sound_on else null,
			"rng": rng,
			"dt": dt,
			"arena_half_extent": arena_half_extent,
			"agent_position": agent.position,
			"agent_eye": agent.get_eye_position(),
			"agent_height": agent.height,
			"agent_alive": agent.alive,
			"allow_movement": curriculum.enemy_movement_enabled(),
			"allow_attack": curriculum.enemy_attacks_enabled(),
			"allow_ranged": curriculum.ranged_enemies_enabled(),
			"allow_strafe": curriculum.strafing_enabled(),
			"allow_jump": curriculum.vertical_enabled(),
		}

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
			"world": world,
			"forward_clearance": perception.forward_clearance,
			"in_cover": perception.in_cover,
			"corpse_count": get_corpse_count(),
			"enemy_slots": enemies.size(),
		}
	)


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
	var healthy: bool = true
	var issues: Array = []

	if is_nan(agent.position.x) or is_nan(agent.position.y) or is_nan(agent.position.z):
		healthy = false
		issues.append("agent position has NaN")
	if is_nan(agent.health) or agent.health < 0.0 or agent.health > agent.max_health:
		healthy = false
		issues.append("agent health invalid: %f" % agent.health)

	for i in range(enemies.size()):
		var enemy: EnemyState = enemies[i]
		if is_nan(enemy.position.x) or is_nan(enemy.position.z):
			healthy = false
			issues.append("enemy %d position has NaN" % i)
		if is_nan(enemy.health) or enemy.health < 0.0:
			healthy = false
			issues.append("enemy %d health invalid: %f" % [i, enemy.health])

	return {
		"healthy": healthy,
		"env_id": env_id,
		"agent_alive": agent.alive,
		"alive_enemies": get_alive_enemy_count(),
		"total_enemies": enemies.size(),
		"issues": issues,
	}


# ---------------------------------------------------------------------------
# Debug / Control Center introspection hooks
#
# `PerceptionModel` probes for these by name with has_method(), so simply
# declaring them lights up the REAL WORLD / AI PERCEPTION / AI MEMORY /
# SOUND panels. They are read-only and are never called from step().
# ---------------------------------------------------------------------------


## Agent FOV cone parameters for the overlay.
func get_agent_field_of_view() -> Dictionary:
	return {
		"origin": agent.get_eye_position(),
		"forward": agent.get_forward_horizontal(),
		"fov_deg": perception.fov_deg,
		"range": perception.vision_range,
		"enabled": curriculum.perception_enabled(),
		"forward_clearance": perception.forward_clearance,
	}


## Raw line-of-sight query against the current geometry.
func has_line_of_sight(from_position: Vector3, to_position: Vector3) -> bool:
	if world == null:
		return true
	return not world.segment_blocked(from_position, to_position)


## Currently audible events from the AGENT's point of view.
func get_sound_events() -> Array:
	if not curriculum.sound_enabled() and not debug_perception:
		return []
	return sound_bus.sample(
		agent.position,
		agent.get_forward_horizontal(),
		world,
		AGENT_SOUND_SOURCE,
		SandboxConfig.SOUND_DETECTION_DELAY
	)


## The agent's memory tracks plus the live belief list and the reason the
## current target was chosen.
func get_target_memory() -> Dictionary:
	return {
		"tracks": perception.memory.to_dict(),
		"beliefs": _beliefs,
		"target_reason": _target_reason,
		"memory_enabled": curriculum.memory_enabled(),
		"half_life": SandboxConfig.MEMORY_HALF_LIFE,
	}


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
	var enemy_states: Array = []
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		enemy_states.append(
			{
				"id": enemy.enemy_id,
				"state": EnemyState.ai_state_name(enemy.ai_state),
				"reason": enemy.tactical_reason,
				"destination": enemy.tactical_destination,
				"has_destination": enemy.has_tactical_destination,
				"target_confirmed": enemy.target_confirmed,
				"time_since_visual": enemy.time_since_visual,
				"reaction": enemy.reaction.to_dict(),
			}
		)
	return {
		"layout_id": world.layout_id if world != null else "none",
		"layout_seed": world.layout_seed if world != null else -1,
		"scenario": scenario.get("id", ""),
		"scenario_label": scenario.get("label", ""),
		"spawn_rule": scenario.get("spawn_rule", ""),
		"episode_seed": episode_seed,
		"enemies": enemy_states,
	}


## Corpses, as pure environmental information. Never targetable.
func get_dead_bodies() -> Array:
	var bodies: Array = []
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if enemy.corpse:
			bodies.append(enemy.to_corpse_dict())
	return bodies


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


## Deterministically (given the environment's seeded RNG) picks a spawn
## position at a random distance/angle around the agent's spawn point, so
## enemies are not always directly ahead. `angle_spread_deg` is the half-width
## of the horizontal arc around the agent's forward-facing direction
## (0deg = enemy spawn's original -Z direction). Results are clamped inside
## the arena walls.
func _random_spawn_position(angle_spread_deg: float, distance_range: Vector2) -> Vector3:
	var angle_deg: float = rng.randf_range(-angle_spread_deg, angle_spread_deg)
	var distance: float = rng.randf_range(distance_range.x, distance_range.y)
	var base_direction := Vector3(0.0, 0.0, -1.0)  # matches AGENT_SPAWN_YAW_DEG == 0 forward
	var rotated: Vector3 = base_direction.rotated(Vector3.UP, deg_to_rad(angle_deg))
	var spawn: Vector3 = SandboxConfig.AGENT_SPAWN_POSITION + rotated * distance
	var limit: float = arena_half_extent - SandboxConfig.ENEMY_RADIUS
	spawn.x = clampf(spawn.x, -limit, limit)
	spawn.z = clampf(spawn.z, -limit, limit)
	spawn.y = 0.0
	return spawn


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

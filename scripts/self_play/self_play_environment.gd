# gdlint:ignore=max-public-methods
## SelfPlayEnvironmentCore
##
## Render-free two-agent match with full ArenaWorld, AgentPerception, SoundBus
## and LightingProfile parity. Slot 0 and Slot 1 represent two independent
## agents acting simultaneously with zero privileged information leakage.
class_name SelfPlayEnvironmentCore
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentPerception = preload("res://scripts/perception/agent_perception.gd")
const AgentState = preload("res://scripts/agent/agent_state.gd")
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EpisodeState = preload("res://scripts/core/episode_state.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const MapLibrary = preload("res://scripts/world/map_library.gd")
const Observation = preload("res://scripts/core/observation.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const RewardSystem = preload("res://scripts/reward/reward_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")

var arena_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var max_steps: int = SandboxConfig.MAX_EPISODE_STEPS

var agent_a: AgentState = AgentState.new()
var agent_b: AgentState = AgentState.new()
var proxy_a: EnemyState = EnemyState.new()
var proxy_b: EnemyState = EnemyState.new()
var episode_a: EpisodeState = EpisodeState.new()
var episode_b: EpisodeState = EpisodeState.new()
var rng_a: RandomNumberGenerator = RandomNumberGenerator.new()
var rng_b: RandomNumberGenerator = RandomNumberGenerator.new()

var world: ArenaWorld = null
var sound_bus: SoundBus = SoundBus.create()
var perception_a: AgentPerception = AgentPerception.create()
var perception_b: AgentPerception = AgentPerception.create()

var map_id: String = ""
var layout_id: String = ""
var lighting_mode_id: String = ""
var lighting: LightingProfile = LightingProfile.create()
var curriculum_level: int = CurriculumConfig.Level.AGENT_VS_AGENT

var done: bool = false
var done_reason: String = ""
var _beliefs_a: Array = []
var _beliefs_b: Array = []
var _time_seconds: float = 0.0


func set_map(p_map_id: String) -> bool:
	if p_map_id.is_empty():
		map_id = ""
		return true
	if not MapLibrary.has_map(p_map_id):
		return false
	map_id = p_map_id
	return true


func set_layout(p_layout_id: String) -> bool:
	if p_layout_id.is_empty():
		layout_id = ""
		return true
	if not WorldGenerator.layout_ids().has(p_layout_id):
		return false
	layout_id = p_layout_id
	return true


func set_lighting_mode(mode_id: String) -> bool:
	if mode_id.is_empty():
		lighting_mode_id = ""
		return true
	if not LightingProfile.MODE_IDS.has(mode_id):
		return false
	lighting_mode_id = mode_id
	return true


func set_curriculum_level(level: int) -> void:
	curriculum_level = clampi(
		level, CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT
	)


func get_episode_condition() -> Dictionary:
	return {
		"map_id": map_id,
		"layout_id": layout_id,
		"lighting": lighting_mode_id if not lighting_mode_id.is_empty() else "normal",
		"curriculum_level": curriculum_level,
		"enemy_count": 1,
	}


func reset(seed_a: int = SandboxConfig.DEFAULT_RANDOM_SEED, seed_b: int = -1) -> Array:
	if seed_b < 0:
		seed_b = seed_a + 1000003
	rng_a.seed = seed_a
	rng_b.seed = seed_b
	_time_seconds = 0.0
	sound_bus.clear()

	var world_on: bool = _world_enabled()
	if world_on:
		if not map_id.is_empty():
			var map_def: Dictionary = MapLibrary.definition(map_id)
			var variants: Array = map_def.get("variants", ["open_arena"])
			var chosen_layout: String = (
				variants[rng_a.randi() % variants.size()]
				if not variants.is_empty()
				else "open_arena"
			)
			world = WorldGenerator.build(
				chosen_layout,
				seed_a,
				float(map_def.get("half_extent", SandboxConfig.ARENA_HALF_EXTENT))
			)
			arena_half_extent = world.half_extent
			if lighting_mode_id.is_empty():
				lighting = LightingProfile.from_id(
					str(map_def.get("lighting", "normal")), seed_a
				)
			else:
				lighting = LightingProfile.from_id(lighting_mode_id, seed_a)
		else:
			var l_id: String = layout_id if not layout_id.is_empty() else "scattered_cover"
			world = WorldGenerator.build(l_id, seed_a, arena_half_extent)
			lighting = (
				LightingProfile.from_id(lighting_mode_id, seed_a)
				if not lighting_mode_id.is_empty()
				else LightingProfile.create()
			)
	else:
		world = null
		arena_half_extent = SandboxConfig.ARENA_HALF_EXTENT
		lighting = (
			LightingProfile.from_id(lighting_mode_id, seed_a)
			if not lighting_mode_id.is_empty()
			else LightingProfile.create()
		)

	var perception_on: bool = _perception_enabled()
	var sound_on: bool = _sound_enabled()
	var memory_on: bool = _memory_enabled()

	perception_a.reset()
	perception_a.configure(perception_on, sound_on, memory_on)
	perception_a.set_lighting(lighting)

	perception_b.reset()
	perception_b.configure(perception_on, sound_on, memory_on)
	perception_b.set_lighting(lighting)

	if world != null:
		var pos_a: Vector3 = world.sample_free_position(rng_a, agent_a.radius, agent_a.height)
		var pos_b: Vector3 = world.sample_free_position(rng_b, agent_b.radius, agent_b.height)
		agent_a.reset(pos_a, 0.0)
		agent_b.reset(pos_b, 180.0)
	else:
		var jitter_a := Vector3(rng_a.randf_range(-0.5, 0.5), 0.0, rng_a.randf_range(-0.5, 0.5))
		var jitter_b := Vector3(rng_b.randf_range(-0.5, 0.5), 0.0, rng_b.randf_range(-0.5, 0.5))
		agent_a.reset(SandboxConfig.AGENT_SPAWN_POSITION + jitter_a, 0.0)
		agent_b.reset(SandboxConfig.ENEMY_SPAWN_POSITION + jitter_b, 180.0)

	episode_a.start_new_episode()
	episode_b.start_new_episode()
	done = false
	done_reason = ""
	_sync_proxies()

	if perception_on:
		_beliefs_a = perception_a.update(agent_a, [proxy_b], world, sound_bus, 0.0, 0)
		_beliefs_b = perception_b.update(agent_b, [proxy_a], world, sound_bus, 0.0, 1)
	else:
		_beliefs_a = []
		_beliefs_b = []

	return get_observations()


func step(actions: Array, dt: float = SandboxConfig.SIMULATION_DT) -> Dictionary:
	if done:
		return {
			"observations": get_observations(),
			"rewards": [0.0, 0.0],
			"done": true,
			"infos": [
				{
					"already_done": true,
					"done_reason": done_reason,
					"TimeLimit.truncated": done_reason == "timeout",
					"metrics": episode_a.to_metrics(
						SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_a_win"
					),
				},
				{
					"already_done": true,
					"done_reason": done_reason,
					"TimeLimit.truncated": done_reason == "timeout",
					"metrics": episode_b.to_metrics(
						SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_b_win"
					),
				},
			],
		}

	_time_seconds += dt
	var sound_on: bool = _sound_enabled()
	if sound_on:
		sound_bus.tick(dt)

	var action_a: Action = (
		actions[0] if actions.size() > 0 and actions[0] is Action else Action.idle()
	)
	var action_b: Action = (
		actions[1] if actions.size() > 1 and actions[1] is Action else Action.idle()
	)

	var prev_a: Vector3 = agent_a.position
	var prev_b: Vector3 = agent_b.position

	agent_a.apply_action(action_a, dt, arena_half_extent)
	agent_b.apply_action(action_b, dt, arena_half_extent)

	if world != null:
		agent_a.position = world.resolve_move(
			prev_a, agent_a.position, agent_a.radius, agent_a.height
		)
		agent_b.position = world.resolve_move(
			prev_b, agent_b.position, agent_b.radius, agent_b.height
		)

	if sound_on:
		if action_a.move_axis != 0 or action_a.strafe_axis != 0:
			sound_bus.emit_sound(SoundBus.Category.FOOTSTEP, agent_a.position, 0)
		if action_a.jump:
			sound_bus.emit_sound(SoundBus.Category.JUMP, agent_a.position, 0)
		if action_b.move_axis != 0 or action_b.strafe_axis != 0:
			sound_bus.emit_sound(SoundBus.Category.FOOTSTEP, agent_b.position, 1)
		if action_b.jump:
			sound_bus.emit_sound(SoundBus.Category.JUMP, agent_b.position, 1)

	var hit_a: bool = false
	var hit_b: bool = false
	var kill_a: bool = false
	var kill_b: bool = false
	var shot_a: bool = false
	var shot_b: bool = false
	var damage_a: float = 0.0
	var damage_b: float = 0.0

	var eye_a: Vector3 = agent_a.get_eye_position()
	var chest_b: Vector3 = _chest_position(agent_b)
	var eye_b: Vector3 = agent_b.get_eye_position()
	var chest_a: Vector3 = _chest_position(agent_a)

	if action_a.shoot and agent_a.alive and agent_a.weapon.try_fire():
		shot_a = true
		if sound_on:
			sound_bus.emit_sound(SoundBus.Category.SHOT, agent_a.position, 0)
		var hits_sphere_a: bool = agent_a.weapon.ray_hits_sphere(
			eye_a, agent_a.get_forward_vector(), chest_b
		)
		var occluded_a: bool = world != null and world.segment_blocked(eye_a, chest_b)
		if hits_sphere_a and not occluded_a:
			damage_a = agent_b.take_damage(agent_a.weapon.damage)
			hit_a = damage_a > 0.0
			kill_a = hit_a and not agent_b.alive
			if hit_a and sound_on:
				sound_bus.emit_sound(SoundBus.Category.IMPACT, agent_b.position, 1)
			if kill_a and sound_on:
				sound_bus.emit_sound(SoundBus.Category.DEATH, agent_b.position, 1)

	if action_b.shoot and agent_b.alive and agent_b.weapon.try_fire():
		shot_b = true
		if sound_on:
			sound_bus.emit_sound(SoundBus.Category.SHOT, agent_b.position, 1)
		var hits_sphere_b: bool = agent_b.weapon.ray_hits_sphere(
			eye_b, agent_b.get_forward_vector(), chest_a
		)
		var occluded_b: bool = world != null and world.segment_blocked(eye_b, chest_a)
		if hits_sphere_b and not occluded_b:
			damage_b = agent_a.take_damage(agent_b.weapon.damage)
			hit_b = damage_b > 0.0
			kill_b = hit_b and not agent_a.alive
			if hit_b and sound_on:
				sound_bus.emit_sound(SoundBus.Category.IMPACT, agent_a.position, 0)
			if kill_b and sound_on:
				sound_bus.emit_sound(SoundBus.Category.DEATH, agent_a.position, 0)

	if shot_a:
		episode_a.record_shot(hit_a)
	if shot_b:
		episode_b.record_shot(hit_b)
	if damage_a > 0.0:
		episode_a.record_damage_dealt(damage_a)
		episode_b.record_damage_taken(damage_a)
	if damage_b > 0.0:
		episode_b.record_damage_dealt(damage_b)
		episode_a.record_damage_taken(damage_b)

	var events_a := {
		"hit": hit_a,
		"kill": kill_a,
		"damage_taken": damage_b,
		"damage_dealt": damage_a,
		"died": not agent_a.alive,
		"shot_fired": shot_a,
		"missed_shot": shot_a and not hit_a,
		"alive": agent_a.alive,
	}
	var events_b := {
		"hit": hit_b,
		"kill": kill_b,
		"damage_taken": damage_a,
		"damage_dealt": damage_b,
		"died": not agent_b.alive,
		"shot_fired": shot_b,
		"missed_shot": shot_b and not hit_b,
		"alive": agent_b.alive,
	}

	var reward_a: float = RewardSystem.compute(events_a)
	var reward_b: float = RewardSystem.compute(events_b)
	episode_a.record_step(reward_a)
	episode_b.record_step(reward_b)
	episode_a.record_reward_breakdown(events_a)
	episode_b.record_reward_breakdown(events_b)

	if kill_a:
		episode_a.record_kill()
	if kill_b:
		episode_b.record_kill()
	if not agent_a.alive:
		episode_a.record_death()
	if not agent_b.alive:
		episode_b.record_death()

	if not agent_a.alive or not agent_b.alive:
		done = true
		if not agent_b.alive and agent_a.alive:
			done_reason = "agent_a_win"
		elif not agent_a.alive and agent_b.alive:
			done_reason = "agent_b_win"
		else:
			done_reason = "draw"
	elif episode_a.is_timeout(max_steps):
		done = true
		done_reason = "timeout"

	if done:
		episode_a.mark_done(done_reason)
		episode_b.mark_done(done_reason)

	_sync_proxies()

	if _perception_enabled():
		_beliefs_a = perception_a.update(agent_a, [proxy_b], world, sound_bus, dt, 0)
		_beliefs_b = perception_b.update(agent_b, [proxy_a], world, sound_bus, dt, 1)

	return {
		"observations": get_observations(),
		"rewards": [reward_a, reward_b],
		"done": done,
		"infos": [
			{
				"events": events_a,
				"done_reason": done_reason,
				"TimeLimit.truncated": done_reason == "timeout",
				"metrics": episode_a.to_metrics(
					SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_a_win"
				),
			},
			{
				"events": events_b,
				"done_reason": done_reason,
				"TimeLimit.truncated": done_reason == "timeout",
				"metrics": episode_b.to_metrics(
					SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_b_win"
				),
			},
		],
	}


func get_observations() -> Array:
	if not _perception_enabled():
		return [
			Observation.build(agent_a, [proxy_b], arena_half_extent),
			Observation.build(agent_b, [proxy_a], arena_half_extent),
		]
	return [
		Observation.build(
			agent_a,
			[proxy_b],
			arena_half_extent,
			_build_perception_context(perception_a, _beliefs_a)
		),
		Observation.build(
			agent_b,
			[proxy_a],
			arena_half_extent,
			_build_perception_context(perception_b, _beliefs_b)
		),
	]


func _build_perception_context(perception: AgentPerception, beliefs: Array) -> Dictionary:
	return {
		"beliefs": beliefs,
		"sounds": perception.heard,
		"sound_summary": perception.sound_summary,
		"world": world,
		"forward_clearance": perception.forward_clearance,
		"in_cover": perception.in_cover,
		"corpse_count": 0,
		"enemy_slots": 1,
		"local_illumination": perception.local_illumination,
		"contact_summary": AgentPerception.summarize_contacts(
			beliefs, Observation.MAX_TRACKED_ENEMIES
		),
		"target_priority_norm": 1.0 if not beliefs.is_empty() else 0.0,
		"target_switch_recent": false,
		"exploration": {},
	}


func _world_enabled() -> bool:
	return (
		not map_id.is_empty()
		or not layout_id.is_empty()
		or curriculum_level >= CurriculumConfig.Level.OBSTACLES_COVER
	)


func _perception_enabled() -> bool:
	return _world_enabled() or curriculum_level >= CurriculumConfig.Level.FOV_LOS


func _sound_enabled() -> bool:
	return _world_enabled() or curriculum_level >= CurriculumConfig.Level.SOUND


func _memory_enabled() -> bool:
	return _world_enabled() or curriculum_level >= CurriculumConfig.Level.MEMORY_LOST_TARGETS


func _sync_proxies() -> void:
	proxy_a.enemy_id = 0
	proxy_a.position = agent_a.position
	proxy_a.health = agent_a.health
	proxy_a.max_health = agent_a.max_health
	proxy_a.alive = agent_a.alive
	proxy_a.radius = agent_a.radius

	proxy_b.enemy_id = 1
	proxy_b.position = agent_b.position
	proxy_b.health = agent_b.health
	proxy_b.max_health = agent_b.max_health
	proxy_b.alive = agent_b.alive
	proxy_b.radius = agent_b.radius


func health_check() -> Dictionary:
	var a_ok: bool = not is_nan(agent_a.position.x) and not is_nan(agent_a.health)
	var b_ok: bool = not is_nan(agent_b.position.x) and not is_nan(agent_b.health)
	return {
		"healthy": a_ok and b_ok,
		"agent_a_alive": agent_a.alive,
		"agent_b_alive": agent_b.alive,
		"done": done,
		"done_reason": done_reason,
	}


## Chest aim point, matching EnvironmentCore's enemy hit height so the
## self-play and single-agent accuracy/win metrics stay comparable.
static func _chest_position(agent) -> Vector3:
	return agent.position + Vector3(0.0, SandboxConfig.ENEMY_CHEST_HEIGHT, 0.0)

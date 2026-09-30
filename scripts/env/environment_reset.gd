## EnvironmentReset
##
## Episode setup for `EnvironmentCore`: spawning, geometry selection,
## lighting resolution and the per-enemy reset.
##
## Split out of `EnvironmentCore` so the file that runs the simulation loop
## stays readable. The split is along a real seam: everything here runs
## exactly once per episode and is allowed to allocate, while the code left
## behind in `EnvironmentCore` runs every tick and is not.
##
## The two reset paths are deliberately separate. `reset_legacy()` is the
## original obstacle-free spawn logic and must keep consuming the exact
## same random draws in the exact same order, or previously trained
## curriculum level 1-4 policies stop reproducing. `reset_with_world()` is
## the world/map path.
##
## `env` is intentionally untyped: typing it would require preloading
## `environment_core.gd`, which preloads this file.
class_name EnvironmentReset
extends RefCounted

const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const MapLibrary = preload("res://scripts/world/map_library.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const ScenarioLibrary = preload("res://scripts/scenario/scenario_library.gd")
const WeaponState = preload("res://scripts/weapon/weapon_state.gd")


## Validates an episode plan in the `set_episode_plans` wire format (seed,
## map_id, scenario, lighting, enemy_count, curriculum_level), returning ""
## when it can be applied and a human-readable reason otherwise. Kept next
## to the reset path because a plan's semantics ARE the reset's inputs; the
## level is passed explicitly (rather than read from an environment) so the
## check is honest about which level the plan requests.
static func validate_episode_plan(plan: Dictionary, current_level: int) -> String:
	var level: int = int(plan.get("curriculum_level", current_level))
	if (
		level < CurriculumConfig.Level.STATIONARY_TARGET
		or level > CurriculumConfig.Level.AGENT_VS_AGENT
	):
		return "curriculum_level %d out of range" % level
	var p_map: String = str(plan.get("map_id", ""))
	if not p_map.is_empty() and not MapLibrary.has_map(p_map):
		return "unknown map_id: %s" % p_map
	var p_lighting: String = str(plan.get("lighting", ""))
	if not p_lighting.is_empty() and not LightingProfile.MODE_IDS.has(p_lighting):
		return "unknown lighting: %s" % p_lighting
	var p_scenario: String = str(plan.get("scenario", ""))
	if not p_scenario.is_empty() and not ScenarioLibrary.has_scenario(p_scenario):
		return "unknown scenario: %s" % p_scenario
	var p_weapon: String = str(plan.get("weapon_profile", ""))
	if not p_weapon.is_empty() and not WeaponState.has_profile(p_weapon):
		return "unknown weapon_profile: %s" % p_weapon
	if int(plan.get("enemy_count", 1)) < 1:
		return "enemy_count must be >= 1"
	return ""


## The episode configuration `env` actually resolved, in the same field
## names EpisodePlan.replay_header_fields() uses on the Python side. This
## is the ground truth of what ran: the requested map/scenario could be
## empty (curriculum-driven defaults), the enemy count is the RESOLVED
## count after the curriculum minimum, and lighting is the RESOLVED mode
## (explicit override, else map default, else normal).
static func episode_condition(env) -> Dictionary:
	return {
		"seed": env.episode_seed,
		"map_id": str((env.map_instance as Dictionary).get("map_id", env.map_id)),
		"scenario": str((env.scenario as Dictionary).get("id", env.scenario_id)),
		"lighting": LightingProfile.mode_id((env.lighting as LightingProfile).mode),
		"weapon_profile": env.agent.weapon.profile_id,
		"enemy_count": env.enemies.size(),
		"curriculum_level": env.curriculum.level,
	}


## Original obstacle-free reset. Untouched so curriculum levels 1-4 keep
## their exact spawn distributions for a given seed.
static func reset_legacy(env) -> void:
	env.agent.reset(SandboxConfig.AGENT_SPAWN_POSITION, SandboxConfig.AGENT_SPAWN_YAW_DEG)
	env._apply_agent_weapon_profile()

	var spawn_variety: bool = env.curriculum.spawn_variety_enabled()
	var strafing: bool = env.curriculum.strafing_enabled()
	var angle_spread_deg: float = env.curriculum.spawn_angle_spread_deg()
	var distance_range: Vector2 = env.curriculum.spawn_distance_range()
	var spread: float = 2.5
	for i in range(env.enemies.size()):
		var enemy: EnemyState = env.enemies[i]
		var spawn: Vector3
		if spawn_variety:
			spawn = random_spawn_position(env, angle_spread_deg, distance_range)
		else:
			# Legacy fixed layout: enemies lined up directly ahead of the
			# agent with only small positional jitter (curriculum level 1).
			var lateral: float = (i - (env.enemies.size() - 1) / 2.0) * spread
			var jitter_x: float = env.rng.randf_range(-0.5, 0.5)
			var jitter_z: float = env.rng.randf_range(-0.5, 0.5)
			spawn = SandboxConfig.ENEMY_SPAWN_POSITION + Vector3(lateral + jitter_x, 0.0, jitter_z)
		reset_enemy(env, enemy, i, spawn, strafing)


## World-backed reset: geometry, scenario spawns and tactical enemies.
static func reset_with_world(env) -> void:
	var resolved_id: String = env.scenario_id
	if resolved_id.is_empty():
		resolved_id = scenario_for_level(env)
	# The layout seed is drawn from the environment RNG rather than reusing
	# `seed_value` directly, so consecutive auto-resets of a seeded env
	# produce a varied but fully reproducible sequence of arenas.
	var layout_seed: int = env.rng.randi()
	if env.map_id.is_empty():
		env.arena_half_extent = env.configured_half_extent
		env.scenario = ScenarioLibrary.resolve(
			resolved_id, layout_seed, env.enemies.size(), env.arena_half_extent
		)
		env.map_instance = {}
	else:
		# A map owns the geometry, the arena size and the lighting; the
		# scenario only decides where the characters start inside it.
		env.map_instance = MapLibrary.resolve(env.map_id, layout_seed)
		env.arena_half_extent = float(env.map_instance["half_extent"])
		env.scenario = ScenarioLibrary.resolve_on_world(
			resolved_id, layout_seed, env.map_instance["world"], env.enemies.size()
		)
	apply_lighting(env, layout_seed)
	env.sound_bus.configure_ambience(env.map_instance.get("ambient_sources", []))
	env.world = env.scenario["world"]

	env.agent.reset(env.scenario["agent_spawn"], float(env.scenario["agent_yaw_deg"]))
	env._apply_agent_weapon_profile()

	var strafing: bool = env.curriculum.strafing_enabled()
	var spawns: Array = env.scenario["enemy_spawns"]
	for i in range(env.enemies.size()):
		var spawn: Vector3
		if i < spawns.size():
			spawn = spawns[i]
		else:
			spawn = env.world.sample_free_position(
				env.rng, SandboxConfig.ENEMY_RADIUS, SandboxConfig.AGENT_HEIGHT
			)
		reset_enemy(env, env.enemies[i], i, spawn, strafing)


static func reset_enemy(env, enemy: EnemyState, index: int, spawn: Vector3, strafing: bool) -> void:
	enemy.enemy_id = index
	enemy.reset(spawn)
	env._apply_enemy_difficulty(enemy)
	if strafing:
		enemy.strafe_direction = 1.0 if env.rng.randf() > 0.5 else -1.0
		enemy.strafe_phase = env.rng.randf_range(0.0, TAU)
	else:
		enemy.strafe_direction = 1.0
		enemy.strafe_phase = 0.0
	# Face the agent's spawn so the FOV cone starts somewhere sensible.
	enemy.yaw_deg = rad_to_deg(
		atan2(env.agent.position.x - spawn.x, -(env.agent.position.z - spawn.z))
	)


## Resolves the episode's lighting profile: an explicit override wins, then
## the map's declared lighting, then NORMAL. The profile seed is derived
## from the layout seed so light/dark patches are reproducible.
static func apply_lighting(env, layout_seed: int) -> void:
	var mode_id: String = env.lighting_mode_id
	if mode_id.is_empty() and not env.map_instance.is_empty():
		mode_id = str(env.map_instance.get("lighting_id", SandboxConfig.LIGHTING_DEFAULT_MODE_ID))
	if mode_id.is_empty():
		mode_id = SandboxConfig.LIGHTING_DEFAULT_MODE_ID
	env.lighting = LightingProfile.from_id(mode_id, layout_seed)


## Which scenario the current curriculum level plays. Level 10 draws a new
## one from the seeded RNG every episode (Phase 9/10 "mixed randomized").
static func scenario_for_level(env) -> String:
	if env.curriculum.randomized_scenarios_enabled():
		var ids: PackedStringArray = ScenarioLibrary.training_ids()
		return ids[env.rng.randi_range(0, ids.size() - 1)]
	return str(env.LEVEL_SCENARIOS.get(env.curriculum.level, "open_arena"))


## Deterministically (given the environment's seeded RNG) picks a spawn
## position at a random distance/angle around the agent's spawn point, so
## enemies are not always directly ahead. `angle_spread_deg` is the half-width
## of the horizontal arc around the agent's forward-facing direction
## (0deg = enemy spawn's original -Z direction). Results are clamped inside
## the arena walls.
static func random_spawn_position(env, angle_spread_deg: float, distance_range: Vector2) -> Vector3:
	var angle_deg: float = env.rng.randf_range(-angle_spread_deg, angle_spread_deg)
	var distance: float = env.rng.randf_range(distance_range.x, distance_range.y)
	var base_direction := Vector3(0.0, 0.0, -1.0)  # matches AGENT_SPAWN_YAW_DEG == 0 forward
	var rotated: Vector3 = base_direction.rotated(Vector3.UP, deg_to_rad(angle_deg))
	var spawn: Vector3 = SandboxConfig.AGENT_SPAWN_POSITION + rotated * distance
	var limit: float = maxf(
		0.0, env.arena_half_extent - SandboxConfig.ENEMY_RADIUS - SandboxConfig.ARENA_BOUNDS_EPSILON
	)
	spawn.x = clampf(spawn.x, -limit, limit)
	spawn.z = clampf(spawn.z, -limit, limit)
	spawn.y = 0.0
	return spawn

## ScenarioLibrary
##
## Modular, seedable encounter definitions. A scenario is pure DATA: an
## arena layout id, an enemy count, a spawn rule and the perception
## capabilities that must be active. `EnvironmentCore.apply_scenario()`
## consumes the resulting spec; there is no scenario-specific gameplay code
## anywhere, which is what keeps twelve scenarios from becoming twelve
## divergent simulations.
##
## Seeding contract: `resolve(id, seed)` is a pure function. The same
## (id, seed) pair always produces the same layout AND the same spawn
## points, because both are derived from the same seeded
## RandomNumberGenerator stream in a fixed order.
class_name ScenarioLibrary
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const ReactionProfile = preload("res://scripts/perception/reaction_profile.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")

## Enemy placement strategies. Each one is implemented once in
## `_place_enemies()` and reused by any scenario that asks for it.
const SPAWN_AHEAD: String = "ahead"
const SPAWN_RING: String = "ring"
const SPAWN_OUT_OF_SIGHT: String = "out_of_sight"
const SPAWN_BEHIND_COVER: String = "behind_cover"
const SPAWN_SURROUND: String = "surround"
const SPAWN_ELEVATED: String = "elevated"
const SPAWN_RANDOM: String = "random"

## The twelve shipped scenarios. `flags` lists the capabilities the
## scenario needs; anything absent falls back to the curriculum level's
## own setting, so a scenario can be run at any level without lying about
## what the agent can perceive.
const SCENARIOS: Array = [
	{
		"id": "open_arena",
		"label": "Open arena",
		"description": "Empty box, one moving enemy. The classic aim/movement baseline.",
		"layout": "open_arena",
		"enemy_count": 1,
		"spawn": SPAWN_AHEAD,
		"level": CurriculumConfig.Level.MOVING_TARGET,
		"flags": {"obstacles": false, "perception": false, "sound": false, "memory": false},
	},
	{
		"id": "single_target",
		"label": "Single target",
		"description": "One stationary target directly ahead: pure aiming practice.",
		"layout": "open_arena",
		"enemy_count": 1,
		"spawn": SPAWN_AHEAD,
		"level": CurriculumConfig.Level.STATIONARY_TARGET,
		"flags": {"obstacles": false, "perception": false, "sound": false, "memory": false},
	},
	{
		"id": "multiple_targets",
		"label": "Multiple targets",
		"description": "Three strafing enemies in the open: forces target selection.",
		"layout": "open_arena",
		"enemy_count": 3,
		"spawn": SPAWN_RING,
		"level": CurriculumConfig.Level.MULTIPLE_ENEMIES,
		"flags": {"obstacles": false, "perception": false, "sound": false, "memory": false},
	},
	{
		"id": "corner_fight",
		"label": "Corner fight",
		"description": "L-shaped wall junction: the enemy starts out of sight around it.",
		"layout": "corner",
		"enemy_count": 1,
		"spawn": SPAWN_OUT_OF_SIGHT,
		"level": CurriculumConfig.Level.FOV_LOS,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "cover_fight",
		"label": "Cover fight",
		"description": "Staggered low/high cover. Both sides can break line of sight.",
		"layout": "cover_field",
		"enemy_count": 2,
		"spawn": SPAWN_BEHIND_COVER,
		"level": CurriculumConfig.Level.OBSTACLES_COVER,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "corridor_fight",
		"label": "Corridor fight",
		"description": "Long lane with one side opening: a single contested sight line.",
		"layout": "corridor",
		"enemy_count": 2,
		"spawn": SPAWN_OUT_OF_SIGHT,
		"level": CurriculumConfig.Level.FOV_LOS,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "ambush",
		"label": "Ambush",
		"description": "Two rooms; enemies wait in the alcove, unseen and silent.",
		"layout": "rooms",
		"enemy_count": 2,
		"spawn": SPAWN_OUT_OF_SIGHT,
		"level": CurriculumConfig.Level.MEMORY_LOST_TARGETS,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "target_disappears",
		"label": "Target breaks contact",
		"description": "The enemy spawns visible behind reachable cover and breaks LOS.",
		"layout": "cover_field",
		"enemy_count": 1,
		"spawn": SPAWN_AHEAD,
		"level": CurriculumConfig.Level.MEMORY_LOST_TARGETS,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "sound_only",
		"label": "Sound-only contact",
		"description": "Enemy starts fully occluded: footsteps and shots are the only cue.",
		"layout": "rooms",
		"enemy_count": 1,
		"spawn": SPAWN_OUT_OF_SIGHT,
		"level": CurriculumConfig.Level.SOUND,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "multi_direction",
		"label": "Enemies from all sides",
		"description": "Four enemies surrounding the agent among pillars.",
		"layout": "pillars",
		"enemy_count": 4,
		"spawn": SPAWN_SURROUND,
		"level": CurriculumConfig.Level.MEMORY_LOST_TARGETS,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "vertical_encounter",
		"label": "Vertical encounter",
		"description": "Platforms at two heights; at least one enemy starts elevated.",
		"layout": "vertical",
		"enemy_count": 2,
		"spawn": SPAWN_ELEVATED,
		"level": CurriculumConfig.Level.VERTICAL_COMBAT,
		"flags":
		{"obstacles": true, "perception": true, "sound": true, "memory": true, "vertical": true},
	},
	{
		"id": "rifle_lane_drill",
		"label": "Rifle lane drill",
		"description": "TTK-style rifle test: medium lanes, cover breaks and clean optics.",
		"layout": "combat_complex",
		"enemy_count": 2,
		"spawn": SPAWN_BEHIND_COVER,
		"level": CurriculumConfig.Level.FOV_LOS,
		"weapon_profile": "rifle",
		"training_pool": false,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "shotgun_breach_drill",
		"label": "Shotgun breach drill",
		"description": "TTK-style close-room drill with a pellet shotgun and occluded targets.",
		"layout": "rooms",
		"enemy_count": 2,
		"spawn": SPAWN_OUT_OF_SIGHT,
		"level": CurriculumConfig.Level.FOV_LOS,
		"weapon_profile": "shotgun",
		"training_pool": false,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "sidearm_finish_drill",
		"label": "Sidearm finish drill",
		"description": "TTK-style pistol cleanup drill: short lanes and wounded-target pacing.",
		"layout": "corner",
		"enemy_count": 1,
		"spawn": SPAWN_OUT_OF_SIGHT,
		"level": CurriculumConfig.Level.FOV_LOS,
		"weapon_profile": "pistol",
		"training_pool": false,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "smg_tracking_drill",
		"label": "SMG tracking drill",
		"description":
		"Close moving-target drill: high cadence, sharp falloff and sustained tracking.",
		"layout": "pillars",
		"enemy_count": 3,
		"spawn": SPAWN_RING,
		"level": CurriculumConfig.Level.MULTIPLE_ENEMIES,
		"weapon_profile": "smg",
		"training_pool": false,
		"flags": {"obstacles": true, "perception": true, "sound": true, "memory": true},
	},
	{
		"id": "randomized_arena",
		"label": "Randomized arena",
		"description": "Layout, enemy count and spawn rule are all drawn from the seed.",
		"layout": "randomized",
		"enemy_count": 3,
		"spawn": SPAWN_RANDOM,
		"level": CurriculumConfig.Level.MIXED_RANDOMIZED,
		"flags":
		{"obstacles": true, "perception": true, "sound": true, "memory": true, "vertical": true},
	},
]


static func ids() -> PackedStringArray:
	var out := PackedStringArray()
	for entry_value in SCENARIOS:
		out.append(str((entry_value as Dictionary)["id"]))
	return out


static func training_ids() -> PackedStringArray:
	var out := PackedStringArray()
	for entry_value in SCENARIOS:
		var entry: Dictionary = entry_value
		if bool(entry.get("training_pool", true)):
			out.append(str(entry["id"]))
	return out


static func definition(scenario_id: String) -> Dictionary:
	for entry_value in SCENARIOS:
		var entry: Dictionary = entry_value
		if str(entry["id"]) == scenario_id:
			return entry.duplicate(true)
	return {}


static func has_scenario(scenario_id: String) -> bool:
	return not definition(scenario_id).is_empty()


## Builds a concrete, reproducible encounter.
##
## Returns:
##   {
##     "id", "label", "layout_id", "level", "flags",
##     "world": ArenaWorld,
##     "agent_spawn": Vector3, "agent_yaw_deg": float,
##     "enemy_spawns": Array[Vector3],
##     "enemy_archetype": int,
##   }
## An unknown id resolves to "open_arena" rather than failing, so a stale
## saved session degrades instead of crashing a training run.
static func resolve(
	scenario_id: String,
	seed_value: int,
	enemy_count_override: int = -1,
	half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
) -> Dictionary:
	var entry: Dictionary = definition(scenario_id)
	if entry.is_empty():
		entry = definition("open_arena")

	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value

	var layout: String = str(entry["layout"])
	# The world uses its own derived seed so that changing the enemy count
	# does not change the geometry for the same scenario seed.
	var world: ArenaWorld = WorldGenerator.build(layout, seed_value, half_extent)
	return _populate(entry, world, rng, seed_value, enemy_count_override)


## Places the agent and the enemies of `scenario_id` into an ALREADY BUILT
## world.
##
## This is what lets a `MapLibrary` map own the geometry while a scenario
## still owns the encounter: the same "enemy waits out of sight" situation
## can be played on a corridor, a compound or a foggy field. Splitting it
## out (instead of duplicating the placement code) is what keeps the spawn
## rules identical in both paths.
static func resolve_on_world(
	scenario_id: String, seed_value: int, world: ArenaWorld, enemy_count_override: int = -1
) -> Dictionary:
	var entry: Dictionary = definition(scenario_id)
	if entry.is_empty():
		entry = definition("open_arena")
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	return _populate(entry, world, rng, seed_value, enemy_count_override)


static func _populate(
	entry: Dictionary,
	world: ArenaWorld,
	rng: RandomNumberGenerator,
	seed_value: int,
	enemy_count_override: int
) -> Dictionary:
	var layout: String = str(entry["layout"])
	var enemy_count: int = (
		enemy_count_override if enemy_count_override > 0 else int(entry["enemy_count"])
	)
	var spawn_rule: String = str(entry["spawn"])
	if spawn_rule == SPAWN_RANDOM:
		var rules: Array = [
			SPAWN_AHEAD, SPAWN_RING, SPAWN_OUT_OF_SIGHT, SPAWN_BEHIND_COVER, SPAWN_SURROUND
		]
		spawn_rule = str(rules[rng.randi_range(0, rules.size() - 1)])

	var agent_spawn: Vector3 = _place_agent(world, rng, layout)
	var enemy_spawns: Array = _place_enemies(world, rng, agent_spawn, enemy_count, spawn_rule)

	return {
		"id": str(entry["id"]),
		"label": str(entry["label"]),
		"description": str(entry["description"]),
		"layout_id": layout,
		"spawn_rule": spawn_rule,
		"level": int(entry["level"]),
		"flags": (entry["flags"] as Dictionary).duplicate(),
		"world": world,
		"agent_spawn": agent_spawn,
		"agent_yaw_deg": _facing_yaw(agent_spawn, Vector3.ZERO),
		"enemy_spawns": enemy_spawns,
		"enemy_archetype": ReactionProfile.Archetype.REGULAR,
		"weapon_profile": str(entry.get("weapon_profile", "rifle")),
		"seed": seed_value,
	}


static func _place_agent(world: ArenaWorld, rng: RandomNumberGenerator, layout: String) -> Vector3:
	if layout == "open_arena":
		# Keep the historical fixed spawn for the obstacle-free layouts so
		# the legacy curriculum levels reproduce exactly.
		return SandboxConfig.AGENT_SPAWN_POSITION
	if world.spawn_points_for("agent").size() > 0:
		return world.sample_spawn_point(
			"agent", rng, SandboxConfig.AGENT_RADIUS, SandboxConfig.AGENT_HEIGHT
		)
	var spawn: Vector3 = world.sample_free_position(
		rng, SandboxConfig.AGENT_RADIUS, SandboxConfig.AGENT_HEIGHT
	)
	return spawn


static func _place_enemies(
	world: ArenaWorld, rng: RandomNumberGenerator, agent_spawn: Vector3, count: int, rule: String
) -> Array:
	var spawns: Array = []
	for index in range(maxi(1, count)):
		spawns.append(_place_one_enemy(world, rng, agent_spawn, index, count, rule))
	return spawns


static func _place_one_enemy(
	world: ArenaWorld,
	rng: RandomNumberGenerator,
	agent_spawn: Vector3,
	index: int,
	count: int,
	rule: String
) -> Vector3:
	var radius: float = SandboxConfig.ENEMY_RADIUS
	var height: float = SandboxConfig.AGENT_HEIGHT
	var agent_eye: Vector3 = agent_spawn + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0)

	# Complex authored maps declare multiple enemy spawn anchors. Use them for
	# scenario rules that are about positioning rather than the legacy fixed
	# "ahead" setup; the helper still enforces distance/occlusion and falls
	# back to the procedural samplers when no anchor fits.
	if world.spawn_points_for("enemy").size() > 0 and rule != SPAWN_AHEAD:
		var require_occluded: bool = rule in [SPAWN_OUT_OF_SIGHT, SPAWN_BEHIND_COVER]
		var authored: Vector3 = _authored_enemy_spawn(
			world, rng, agent_spawn, agent_eye, radius, height, require_occluded
		)
		if authored != Vector3.INF:
			return authored

	match rule:
		SPAWN_AHEAD:
			var lateral: float = (float(index) - float(count - 1) * 0.5) * 2.5
			var candidate: Vector3 = (
				SandboxConfig.ENEMY_SPAWN_POSITION
				+ Vector3(lateral + rng.randf_range(-0.5, 0.5), 0.0, rng.randf_range(-0.5, 0.5))
			)
			if world.is_position_free(candidate, radius, height):
				return candidate
			return world.sample_free_position_near(rng, candidate, 0.5, 3.0, radius, height)
		SPAWN_RING:
			return _ring_spawn(world, rng, agent_spawn, index, count, 5.0, 11.0, radius, height)
		SPAWN_SURROUND:
			return _ring_spawn(world, rng, agent_spawn, index, count, 6.0, 10.0, radius, height)
		SPAWN_OUT_OF_SIGHT:
			return _occluded_spawn(world, rng, agent_spawn, agent_eye, radius, height)
		SPAWN_BEHIND_COVER:
			return _occluded_spawn(world, rng, agent_spawn, agent_eye, radius, height)
		SPAWN_ELEVATED:
			return _elevated_spawn(world, rng, agent_spawn, index, radius, height)
		_:
			return world.sample_free_position(rng, radius, height)


## Samples one of a layout's authored enemy anchors. Returning Vector3.INF
## means "no authored point satisfied this scenario".
static func _authored_enemy_spawn(
	world: ArenaWorld,
	rng: RandomNumberGenerator,
	agent_spawn: Vector3,
	agent_eye: Vector3,
	radius: float,
	height: float,
	require_occluded: bool
) -> Vector3:
	var points: Array = world.spawn_points_for("enemy")
	if points.is_empty():
		return Vector3.INF
	var start: int = rng.randi_range(0, points.size() - 1)
	for offset in range(points.size()):
		var point: Dictionary = points[(start + offset) % points.size()]
		var candidate: Vector3 = point.get("position", Vector3.ZERO)
		candidate.y = world.ground_height(candidate, radius, candidate.y)
		if candidate.distance_to(agent_spawn) < 4.0:
			continue
		if not world.is_position_free(candidate, radius, height):
			continue
		if (
			require_occluded
			and PerceptionSystem.has_line_of_sight(world, agent_eye, candidate, height)
		):
			continue
		return candidate
	return Vector3.INF


static func _ring_spawn(
	world: ArenaWorld,
	rng: RandomNumberGenerator,
	center: Vector3,
	index: int,
	count: int,
	min_distance: float,
	max_distance: float,
	radius: float,
	height: float
) -> Vector3:
	var base_angle: float = TAU * float(index) / float(maxi(1, count))
	var angle: float = base_angle + rng.randf_range(-0.35, 0.35)
	var distance: float = rng.randf_range(min_distance, max_distance)
	var candidate := Vector3(
		center.x + cos(angle) * distance, 0.0, center.z + sin(angle) * distance
	)
	var limit: float = world.half_extent - radius - 0.2
	candidate.x = clampf(candidate.x, -limit, limit)
	candidate.z = clampf(candidate.z, -limit, limit)
	candidate.y = world.ground_height(candidate, radius, 0.0)
	if world.is_position_free(candidate, radius, height):
		return candidate
	return world.sample_free_position_near(rng, candidate, 0.5, 3.0, radius, height)


## Samples until the spawn point is NOT visible from the agent's eye. This
## is what makes the corner/ambush/sound-only scenarios actually start with
## no visual contact instead of merely hoping for it.
static func _occluded_spawn(
	world: ArenaWorld,
	rng: RandomNumberGenerator,
	agent_spawn: Vector3,
	agent_eye: Vector3,
	radius: float,
	height: float
) -> Vector3:
	var fallback: Vector3 = agent_spawn
	for _attempt in range(32):
		var candidate: Vector3 = world.sample_free_position(rng, radius, height)
		if candidate.distance_to(agent_spawn) < 4.0:
			continue
		fallback = candidate
		if not PerceptionSystem.has_line_of_sight(world, agent_eye, candidate, height):
			return candidate
	return fallback


## Prefers the top of a standable platform for the first enemy, so a
## vertical scenario reliably contains an elevation difference.
static func _elevated_spawn(
	world: ArenaWorld,
	rng: RandomNumberGenerator,
	agent_spawn: Vector3,
	index: int,
	radius: float,
	height: float
) -> Vector3:
	if index == 0:
		for obstacle_value in world.obstacles:
			var obstacle = obstacle_value
			if not obstacle.standable or obstacle.top_y() < 0.6:
				continue
			var candidate := Vector3(obstacle.center.x, obstacle.top_y(), obstacle.center.z)
			if world.is_position_free(candidate, radius, height):
				return candidate
	return _ring_spawn(world, rng, agent_spawn, index, 3, 5.0, 10.0, radius, height)


## Yaw (degrees) that makes a character at `from_position` face `target`.
## yaw = 0 corresponds to forward = (0, 0, -1), matching AgentState.
static func _facing_yaw(from_position: Vector3, target: Vector3) -> float:
	var delta := Vector3(target.x - from_position.x, 0.0, target.z - from_position.z)
	if delta.is_zero_approx():
		return SandboxConfig.AGENT_SPAWN_YAW_DEG
	return rad_to_deg(atan2(delta.x, -delta.z))

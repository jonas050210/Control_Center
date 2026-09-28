## PerceptionModel
##
## Answers "what does the AI actually see?" by building two clearly
## separated views of the same instant:
##
##   REAL WORLD    - ground truth read from EnvironmentCore (every enemy,
##                   alive or dead, wherever it is). Debug information for
##                   the human operator only.
##   AI PERCEPTION - decoded from the observation vector the policy
##                   receives, and from nothing else. If a value is not in
##                   the observation, it is not in this section.
##
## The difference between the two sections is the interesting part: enemies
## that exist in the world but are NOT represented in the observation are
## listed under `hidden_from_ai` with the concrete reason. Nothing here
## leaks extra information INTO the observation — this module only reads.
##
## Perception features the simulation does not implement yet (field-of-view
## gating, line-of-sight/occlusion, sound events, target memory/last-known
## positions, cover/obstacles, navigation, corpses) are reported as
## explicitly unavailable instead of being faked. `capabilities()` probes
## EnvironmentCore for the optional hooks a future world/perception
## milestone would add, so this view starts showing real data the moment
## those methods exist, without duplicating any perception logic in the GUI.
class_name PerceptionModel
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

## Optional perception features and the EnvironmentCore method that would
## provide each one. Probed with `has_method()` so the Control Center
## consumes a perception system when it lands instead of re-implementing it.
##
## EnvironmentCore now implements all seven. The notes below are still used
## verbatim for any OTHER environment type that does not (for example the
## self-play environment), which is exactly why the probe is dynamic
## instead of a compile-time assumption.
const OPTIONAL_FEATURES: Array = [
	{
		"id": "field_of_view",
		"label": "Field of view gating",
		"method": "get_agent_field_of_view",
		"unavailable_note":
		"This environment exposes no FOV hook, so the policy sees tracked "
		+ "enemies regardless of facing.",
	},
	{
		"id": "line_of_sight",
		"label": "Line of sight / occlusion",
		"method": "has_line_of_sight",
		"unavailable_note":
		"This environment exposes no line-of-sight hook, so nothing can be "
		+ "hidden by geometry.",
	},
	{
		"id": "sound_events",
		"label": "Sound events / hearing",
		"method": "get_sound_events",
		"unavailable_note": "This environment exposes no sound-event hook.",
	},
	{
		"id": "target_memory",
		"label": "Target memory / last-known position",
		"method": "get_target_memory",
		"unavailable_note":
		"This environment exposes no memory hook, so the observation is "
		+ "memoryless: current positions only, no decay or confidence.",
	},
	{
		"id": "obstacles_cover",
		"label": "Obstacles / cover",
		"method": "get_obstacles",
		"unavailable_note": "This environment exposes no obstacle hook.",
	},
	{
		"id": "navigation",
		"label": "Navigation / pathfinding",
		"method": "get_navigation_state",
		"unavailable_note":
		"This environment exposes no navigation hook. Movement is analytic; "
		+ "there is no navmesh or path planner in any case.",
	},
	{
		"id": "dead_bodies",
		"label": "Corpses / dead bodies",
		"method": "get_dead_bodies",
		"unavailable_note":
		"This environment exposes no corpse hook, so dead enemies are only "
		+ "flagged not-alive and keep no body.",
	},
]

const SLOT_LABELS: PackedStringArray = PackedStringArray(["primary", "secondary", "tertiary"])


## Which optional perception hooks the current EnvironmentCore provides.
## Returns feature id -> {"available", "label", "method", "note"}.
static func capabilities(env) -> Dictionary:
	var result: Dictionary = {}
	for feature_value in OPTIONAL_FEATURES:
		var feature: Dictionary = feature_value
		var method_name: String = str(feature["method"])
		var available: bool = env != null and env.has_method(method_name)
		result[str(feature["id"])] = {
			"label": str(feature["label"]),
			"method": method_name,
			"available": available,
			"note": "" if available else str(feature["unavailable_note"]),
		}
	return result


## Full REAL WORLD vs AI PERCEPTION snapshot for one environment.
## `observation` defaults to the environment's current observation.
static func build(env, observation = null) -> Dictionary:
	if env == null:
		return {}
	var obs = observation if observation != null else env.get_observations()
	var caps: Dictionary = capabilities(env)
	var real_world: Dictionary = _build_real_world(env)
	var ai_perception: Dictionary = _build_ai_perception(obs)
	var tracked_indices: Array = _tracked_enemy_indices(env)

	var hidden: Array = []
	for entry_value in (real_world["enemies"] as Array):
		var entry: Dictionary = entry_value
		if not bool(entry["alive"]):
			continue
		if tracked_indices.has(int(entry["index"])):
			continue
		hidden.append(
			{
				"index": int(entry["index"]),
				"distance_m": float(entry["distance_m"]),
				"bearing_deg": float(entry["bearing_deg"]),
				"reason": "beyond_tracked_enemy_budget",
				"detail":
				(
					"only the %d nearest alive enemies are in the observation contract"
					% Observation.MAX_TRACKED_ENEMIES
				),
			}
		)

	var unavailable: Array = []
	for feature_id in caps.keys():
		var capability: Dictionary = caps[feature_id]
		if not bool(capability["available"]):
			unavailable.append(
				{
					"id": str(feature_id),
					"label": str(capability["label"]),
					"note": str(capability["note"]),
				}
			)

	return {
		"capabilities": caps,
		"real_world": real_world,
		"ai_perception": ai_perception,
		"hidden_from_ai": hidden,
		"tracked_enemy_indices": tracked_indices,
		"target_index": int(tracked_indices[0]) if tracked_indices.size() > 0 else -1,
		"unavailable_features": unavailable,
		"max_tracked_enemies": Observation.MAX_TRACKED_ENEMIES,
	}


## Ground-truth enemy/agent state. Debug-only: the policy never sees this.
static func _build_real_world(env) -> Dictionary:
	var agent = env.agent
	var enemies: Array = []
	for index in range(env.enemies.size()):
		var enemy: EnemyState = env.enemies[index]
		var to_enemy: Vector3 = enemy.position - agent.position
		to_enemy.y = 0.0
		enemies.append(
			{
				"index": index,
				"position": enemy.position,
				"distance_m": to_enemy.length(),
				"bearing_deg": _bearing_deg(agent, enemy.position),
				"alive": enemy.alive,
				"health": enemy.health,
				"max_health": enemy.max_health,
				"ai_state": EnemyState.ai_state_name(enemy.ai_state),
				"in_weapon_range": to_enemy.length() <= SandboxConfig.WEAPON_RANGE,
			}
		)
	return {
		"agent_position": agent.position,
		"agent_yaw_deg": agent.yaw_deg,
		"agent_pitch_deg": agent.pitch_deg,
		"agent_forward": agent.get_forward_vector(),
		"agent_health": agent.health,
		"agent_max_health": agent.max_health,
		"agent_alive": agent.alive,
		"weapon_range": SandboxConfig.WEAPON_RANGE,
		"arena_half_extent": env.arena_half_extent,
		"enemies": enemies,
		"alive_enemy_count": env.get_alive_enemy_count(),
		"total_enemy_count": env.enemies.size(),
	}


## Decoded exclusively from the observation vector: this is the AI's entire
## world model. Distances/bearings are de-normalized back to metres/degrees
## purely so a human can read them.
##
## The decode goes through `Observation.to_array()` + `field_names()` on
## purpose. Reading the Observation object's member variables would be
## easier but WRONG for this panel: some members (e.g. `enemy_alive`) are
## bookkeeping that never reaches the policy, and showing them here would
## quietly turn "what the AI sees" into "what the simulation knows".
static func _build_ai_perception(obs) -> Dictionary:
	var slots: Array = []
	if obs == null:
		return {"source": "observation_vector", "slots": slots}
	var vector: Dictionary = _vector_by_name(obs)

	# The primary slot always holds the nearest ALIVE enemy when one exists,
	# so "is the slot occupied?" is decodable from the vector itself through
	# the alive-enemy count; there is no separate primary-alive field.
	var primary_alive: bool = float(vector.get("alive_enemy_count_norm", 0.0)) > 0.0
	slots.append(_slot(vector, SLOT_LABELS[0], "primary_enemy", primary_alive))
	slots.append(
		_slot(
			vector,
			SLOT_LABELS[1],
			"secondary_enemy",
			float(vector.get("secondary_enemy_alive", 0.0)) > 0.5
		)
	)
	slots.append(
		_slot(
			vector,
			SLOT_LABELS[2],
			"tertiary_enemy",
			float(vector.get("tertiary_enemy_alive", 0.0)) > 0.5
		)
	)
	return {
		"source": "observation_vector",
		"slots": slots,
		"agent_health_norm": float(vector.get("agent_health_norm", 0.0)),
		"weapon_ready": float(vector.get("weapon_ready", 0.0)) > 0.5,
		"in_combat": float(vector.get("in_combat", 0.0)) > 0.5,
		"alive_enemy_count_norm": float(vector.get("alive_enemy_count_norm", 0.0)),
		"memory": "none (memoryless observation)",
		"confidence": "exact (no noise model)",
	}


## field name -> value for every index of the observation vector. Names come
## from Observation.FIELD_SPEC, so a contract change renames these keys
## instead of silently shifting an index.
static func _vector_by_name(obs) -> Dictionary:
	var values: PackedFloat32Array = obs.to_array()
	var names: PackedStringArray = Observation.field_names()
	var by_name: Dictionary = {}
	for index in range(names.size()):
		by_name[names[index]] = float(values[index]) if index < values.size() else 0.0
	return by_name


## One decoded enemy slot. `prefix` is the FIELD_SPEC name prefix of that
## slot ("primary_enemy", "secondary_enemy", "tertiary_enemy").
static func _slot(vector: Dictionary, label: String, prefix: String, alive: bool) -> Dictionary:
	var relative := Vector3(
		float(vector.get("%s_relative_position_norm.x" % prefix, 0.0)),
		float(vector.get("%s_relative_position_norm.y" % prefix, 0.0)),
		float(vector.get("%s_relative_position_norm.z" % prefix, 0.0))
	)
	return {
		"slot": label,
		"alive": alive,
		"distance_m": (
			float(vector.get("%s_distance_norm" % prefix, 0.0)) * SandboxConfig.ARENA_MAX_DISTANCE
		),
		"bearing_deg": float(vector.get("%s_bearing_norm" % prefix, 0.0)) * 180.0,
		"health_norm": float(vector.get("%s_health_norm" % prefix, 0.0)),
		"relative_position_norm": relative,
	}


## Indices (into env.enemies) of the enemies that the observation contract
## actually reports, in rank order. Uses Observation.rank_alive_enemies so
## the GUI cannot drift from the contract's own ranking rule.
static func _tracked_enemy_indices(env) -> Array:
	var ranked: Array = Observation.rank_alive_enemies(env.enemies, env.agent.position)
	var indices: Array = []
	for rank in range(mini(ranked.size(), Observation.MAX_TRACKED_ENEMIES)):
		var enemy: EnemyState = ranked[rank]
		for index in range(env.enemies.size()):
			if env.enemies[index] == enemy:
				indices.append(index)
				break
	return indices


## Index of the enemy the observation's primary slot refers to, or -1.
static func current_target_index(env) -> int:
	var tracked: Array = _tracked_enemy_indices(env)
	return int(tracked[0]) if tracked.size() > 0 else -1


static func _bearing_deg(agent, target_position: Vector3) -> float:
	var to_target: Vector3 = target_position - agent.position
	to_target.y = 0.0
	if to_target.length_squared() < 0.000001:
		return 0.0
	var desired_yaw_deg: float = rad_to_deg(atan2(to_target.x, -to_target.z))
	return wrapf(desired_yaw_deg - agent.yaw_deg, -180.0, 180.0)


## Compact, human-readable summary lines for a text panel.
static func format_lines(perception: Dictionary) -> PackedStringArray:
	var lines := PackedStringArray()
	if perception.is_empty():
		lines.append("no environment selected")
		return lines

	var real_world: Dictionary = perception["real_world"]
	var ai_perception: Dictionary = perception["ai_perception"]

	lines.append("REAL WORLD (ground truth, debug only)")
	lines.append(
		(
			"  agent hp %.0f/%.0f   alive enemies %d/%d"
			% [
				float(real_world["agent_health"]),
				float(real_world["agent_max_health"]),
				int(real_world["alive_enemy_count"]),
				int(real_world["total_enemy_count"]),
			]
		)
	)
	for entry_value in (real_world["enemies"] as Array):
		var entry: Dictionary = entry_value
		lines.append(
			(
				"  #%d %-6s %5.1fm %+6.1f deg hp %3.0f%s"
				% [
					int(entry["index"]),
					str(entry["ai_state"]),
					float(entry["distance_m"]),
					float(entry["bearing_deg"]),
					float(entry["health"]),
					"" if bool(entry["alive"]) else "  (dead)",
				]
			)
		)

	lines.append("")
	lines.append("AI PERCEPTION (decoded from the observation vector)")
	for slot_value in (ai_perception["slots"] as Array):
		var slot: Dictionary = slot_value
		if bool(slot["alive"]):
			lines.append(
				(
					"  %-9s %5.1fm %+6.1f deg  hp %.0f%%"
					% [
						str(slot["slot"]),
						float(slot["distance_m"]),
						float(slot["bearing_deg"]),
						float(slot["health_norm"]) * 100.0,
					]
				)
			)
		else:
			lines.append("  %-9s (empty slot: no enemy reported)" % str(slot["slot"]))
	lines.append(
		(
			"  weapon_ready %s   in_combat %s   memory: %s"
			% [
				str(ai_perception["weapon_ready"]),
				str(ai_perception["in_combat"]),
				str(ai_perception["memory"]),
			]
		)
	)

	var hidden: Array = perception["hidden_from_ai"]
	lines.append("")
	if hidden.is_empty():
		lines.append("HIDDEN FROM AI: none — every alive enemy is in the observation")
	else:
		lines.append("HIDDEN FROM AI (exists in world, absent from observation)")
		for entry_value in hidden:
			var entry: Dictionary = entry_value
			lines.append(
				(
					"  #%d %5.1fm %+6.1f deg  reason: %s"
					% [
						int(entry["index"]),
						float(entry["distance_m"]),
						float(entry["bearing_deg"]),
						str(entry["reason"]),
					]
				)
			)
	return lines

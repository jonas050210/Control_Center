## EnvironmentIntrospection
##
## Read-only projections of an `EnvironmentCore` for the Control Center.
##
## These live outside `EnvironmentCore` for two reasons: the simulation file
## was growing past a reviewable size, and physically separating "what the
## simulation is" from "what the debug UI is shown" makes it obvious that
## none of this can write to the simulation. Every function here takes the
## environment, reads it, and returns a fresh Dictionary/Array.
##
## `EnvironmentCore` keeps one-line wrappers for each of these, because
## `PerceptionModel` discovers the panels with `has_method()` on the
## environment object.
##
## `env` is intentionally untyped: typing it would require preloading
## `environment_core.gd`, which preloads this file.
class_name EnvironmentIntrospection
extends RefCounted

const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


## Agent FOV cone parameters for the overlay.
static func field_of_view(env) -> Dictionary:
	return {
		"origin": env.agent.get_eye_position(),
		"forward": env.agent.get_forward_horizontal(),
		"fov_deg": env.perception.fov_deg,
		"range": env.perception.vision_range,
		"enabled": env.curriculum.perception_enabled(),
		"forward_clearance": env.perception.forward_clearance,
	}


## Map Analyzer state for the EXPLORATION view. Privileged only in that the
## UI sees every belief at once; each cell was earned by the agent looking.
static func exploration_state(env) -> Dictionary:
	if env.exploration == null:
		return {"enabled": false, "mode": env.exploration_mode}
	var payload: Dictionary = env.exploration.to_dict()
	payload["enabled"] = true
	payload["mode"] = env.exploration_mode
	return payload


## Currently audible events from the AGENT's point of view.
static func sound_events(env) -> Array:
	if not env.curriculum.sound_enabled() and not env.debug_perception:
		return []
	return env.sound_bus.sample(
		env.agent.position,
		env.agent.get_forward_horizontal(),
		env.world,
		env.AGENT_SOUND_SOURCE,
		SandboxConfig.SOUND_DETECTION_DELAY
	)


## Memory tracks plus the live belief list and the target-selection reason.
static func target_memory(env) -> Dictionary:
	return {
		"tracks": env.perception.memory.to_dict(),
		"beliefs": env.get_beliefs(),
		"target_reason": env.get_target_reason(),
		"memory_enabled": env.curriculum.memory_enabled(),
		"half_life": SandboxConfig.MEMORY_HALF_LIFE,
	}


## Environmental conditions. Includes the map's human-facing metadata,
## which the POLICY never receives.
static func environment_conditions(env) -> Dictionary:
	var conditions: Dictionary = {
		"lighting": env.lighting.to_dict(),
		"local_illumination": env.perception.local_illumination,
		"map_id": env.map_id,
		"arena_half_extent": env.arena_half_extent,
	}
	if (env.map_instance as Dictionary).is_empty():
		conditions["map"] = {"id": "", "label": "(scenario-generated)", "known": false}
	else:
		conditions["map"] = env.map_instance["metadata"]
	return conditions


## Layout/scenario metadata plus per-enemy tactical state.
static func navigation_state(env) -> Dictionary:
	var enemy_states: Array = []
	for enemy_value in env.enemies:
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
				"navigation": enemy.navigation.to_dict(),
			}
		)
	return {
		"layout_id": env.world.layout_id if env.world != null else "none",
		"layout_seed": env.world.layout_seed if env.world != null else -1,
		"graph": env.get_navigation_graph_info(),
		"scenario": (env.scenario as Dictionary).get("id", ""),
		"scenario_label": (env.scenario as Dictionary).get("label", ""),
		"spawn_rule": (env.scenario as Dictionary).get("spawn_rule", ""),
		"episode_seed": env.episode_seed,
		"enemies": enemy_states,
	}


## Corpses, as pure environmental information. Never targetable.
static func dead_bodies(env) -> Array:
	var bodies: Array = []
	for enemy_value in env.enemies:
		var enemy: EnemyState = enemy_value
		if enemy.corpse:
			bodies.append(enemy.to_corpse_dict())
	return bodies


## Invariant check over the simulation state. Diagnostic only: it reports
## problems, it never repairs them, so a corrupted state stays visible.
static func health_check(env) -> Dictionary:
	var healthy: bool = true
	var issues: Array = []
	var agent = env.agent

	if is_nan(agent.position.x) or is_nan(agent.position.y) or is_nan(agent.position.z):
		healthy = false
		issues.append("agent position has NaN")
	if is_nan(agent.health) or agent.health < 0.0 or agent.health > agent.max_health:
		healthy = false
		issues.append("agent health invalid: %f" % agent.health)

	for i in range((env.enemies as Array).size()):
		var enemy: EnemyState = env.enemies[i]
		if is_nan(enemy.position.x) or is_nan(enemy.position.z):
			healthy = false
			issues.append("enemy %d position has NaN" % i)
		if is_nan(enemy.health) or enemy.health < 0.0:
			healthy = false
			issues.append("enemy %d health invalid: %f" % [i, enemy.health])

	return {
		"healthy": healthy,
		"env_id": env.env_id,
		"agent_alive": agent.alive,
		"alive_enemies": env.get_alive_enemy_count(),
		"total_enemies": (env.enemies as Array).size(),
		"issues": issues,
	}

## ControlCenterTelemetry
##
## Builds the read-only snapshot the Control Center UI renders, for ONE
## selected environment at a time.
##
## Design rules (Phase 15):
##   * Pure function over existing public state. It reads; it never writes
##     to the simulation, and nothing in the simulation depends on it.
##   * Only the selected environment is inspected. Nothing is serialised
##     for the other environments beyond the small status counters they
##     already maintain.
##   * Expensive sections (observation rows, perception model) are opt-in
##     through `options`, so a hidden panel costs nothing.
##   * It reuses `DebugOverlay.build_telemetry_dict()` instead of
##     re-implementing the same telemetry collection a second time.
class_name ControlCenterTelemetry
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const DebugOverlay = preload("res://scripts/debug/debug_overlay.gd")
const ObservationInspector = preload("res://scripts/control_center/observation_inspector.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const DEFAULT_OPTIONS: Dictionary = {
	"observation": true,
	"perception": true,
	"reward": true,
	"action": true,
}


## `manager` is a SimulationManager, `config` a ControlCenterConfig and
## `status` the session's own run-state dictionary. Returns {} when there
## is nothing to inspect.
static func build(manager, config, status: Dictionary, options: Dictionary = {}) -> Dictionary:
	if manager == null or manager.environments.is_empty() or config == null:
		return {}

	var resolved: Dictionary = DEFAULT_OPTIONS.duplicate()
	for key in options.keys():
		resolved[key] = options[key]

	var index: int = clampi(config.selected_environment, 0, manager.environments.size() - 1)
	var env = manager.environments[index]
	var snapshot: Dictionary = {
		"status": status.duplicate(),
		"selected_environment": index,
		"selected_agent_slot": config.selected_agent_slot,
		"environment_count": manager.environments.size(),
		"base": DebugOverlay.build_telemetry_dict(manager, index),
	}

	var observation = env.get_observations()
	if bool(resolved["observation"]):
		snapshot["observation"] = {
			"vector": Array(observation.to_array()),
			"rows": ObservationInspector.build_rows(observation),
			"grouped_rows": ObservationInspector.build_grouped_rows(observation),
			"field_count": observation.to_array().size(),
		}
	if bool(resolved["perception"]):
		snapshot["perception"] = PerceptionModel.build(env, observation)
	if bool(resolved["reward"]):
		snapshot["reward"] = {
			"rows": ObservationInspector.build_reward_rows(env.episode.get_reward_breakdown()),
			"breakdown": env.episode.get_reward_breakdown(),
		}
	if bool(resolved["action"]):
		var actions: Array = manager.get_last_actions()
		var action = actions[index] if index < actions.size() else null
		snapshot["action"] = {
			"rows": ObservationInspector.build_action_rows(action),
			"continuous_rows": ObservationInspector.build_continuous_action_rows(action),
			"shooting": action != null and action.shoot,
			"moving": (
				action != null and (action.move_axis != 0 or action.strafe_axis != 0)
			),
			"turning": action != null and (action.look_yaw_axis != 0 or action.look_pitch_axis != 0),
		}

	snapshot["agent"] = {
		"position": env.agent.position,
		"velocity": env.agent.velocity,
		"forward": env.agent.get_forward_vector(),
		"yaw_deg": env.agent.yaw_deg,
		"pitch_deg": env.agent.pitch_deg,
		"on_ground": env.agent.on_ground,
		"in_combat": observation.in_combat,
		"health": env.agent.health,
		"max_health": env.agent.max_health,
		"alive": env.agent.alive,
		"weapon_ready": env.agent.weapon.is_ready(),
		"weapon_cooldown": env.agent.weapon.cooldown_remaining,
		"weapon_damage": env.agent.weapon.damage,
		"weapon_range": env.agent.weapon.range_m,
	}
	snapshot["episode"] = {
		"episode": env.episode.episode_count,
		"step": env.episode.step_count,
		"max_steps": env.max_steps,
		"time_seconds": float(env.episode.step_count) * SandboxConfig.SIMULATION_DT,
		"survival_time": float(env.episode.survival_steps) * SandboxConfig.SIMULATION_DT,
		"reward": env.episode.cumulative_reward,
		"last_reward": env.episode.last_reward,
		"kills": env.episode.kills,
		"deaths": env.episode.deaths,
		"damage_dealt": env.episode.damage_dealt,
		"damage_received": env.episode.damage_taken,
		"shots_fired": env.episode.shots_fired,
		"shots_hit": env.episode.shots_hit,
		"accuracy": (
			float(env.episode.shots_hit) / float(env.episode.shots_fired)
			if env.episode.shots_fired > 0
			else 0.0
		),
		"done": env.episode.done,
		"done_reason": env.episode.done_reason,
		"alive_enemies": env.get_alive_enemy_count(),
		"total_enemies": env.enemies.size(),
	}
	snapshot["target"] = _target_info(env)
	return snapshot


## Which enemy the observation's primary slot currently refers to. Uses the
## contract's own ranking (PerceptionModel -> Observation.rank_alive_enemies).
static func _target_info(env) -> Dictionary:
	var index: int = PerceptionModel.current_target_index(env)
	if index < 0:
		return {"index": -1, "has_target": false}
	var enemy = env.enemies[index]
	var to_enemy: Vector3 = enemy.position - env.agent.position
	to_enemy.y = 0.0
	var distance: float = to_enemy.length()
	return {
		"index": index,
		"has_target": true,
		"distance_m": distance,
		"in_range": distance <= SandboxConfig.WEAPON_RANGE,
		"health": enemy.health,
		"max_health": enemy.max_health,
		"position": enemy.position,
		"direction": to_enemy.normalized() if not to_enemy.is_zero_approx() else Vector3.ZERO,
	}

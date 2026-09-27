## SelfPlayEnvironmentCore
##
## Small render-free two-agent match using the same AgentState, Action,
## WeaponState, Observation and reward primitives as the canonical FPS
## environment. It is deliberately a foundation rather than a population
## algorithm: callers provide one Action per policy slot and may freeze either
## policy externally.
class_name SelfPlayEnvironmentCore
extends RefCounted

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
var done: bool = false
var done_reason: String = ""


func reset(seed_a: int = SandboxConfig.DEFAULT_RANDOM_SEED, seed_b: int = -1) -> Array:
	if seed_b < 0:
		seed_b = seed_a + 1000003
	rng_a.seed = seed_a
	rng_b.seed = seed_b
	var jitter_a := Vector3(rng_a.randf_range(-0.5, 0.5), 0.0, rng_a.randf_range(-0.5, 0.5))
	var jitter_b := Vector3(rng_b.randf_range(-0.5, 0.5), 0.0, rng_b.randf_range(-0.5, 0.5))
	agent_a.reset(SandboxConfig.AGENT_SPAWN_POSITION + jitter_a, 0.0)
	agent_b.reset(SandboxConfig.ENEMY_SPAWN_POSITION + jitter_b, 180.0)
	episode_a.start_new_episode()
	episode_b.start_new_episode()
	done = false
	done_reason = ""
	_sync_proxies()
	return get_observations()


func step(actions: Array, dt: float = SandboxConfig.SIMULATION_DT) -> Dictionary:
	if done:
		return {
			"observations": get_observations(),
			"rewards": [0.0, 0.0],
			"done": true,
			"infos":
			[
				{
					"already_done": true,
					"done_reason": done_reason,
					"TimeLimit.truncated": done_reason == "timeout",
					"metrics":
					episode_a.to_metrics(
						SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_a_win"
					)
				},
				{
					"already_done": true,
					"done_reason": done_reason,
					"TimeLimit.truncated": done_reason == "timeout",
					"metrics":
					episode_b.to_metrics(
						SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_b_win"
					)
				}
			],
		}
	var action_a: Action = (
		actions[0] if actions.size() > 0 and actions[0] is Action else Action.idle()
	)
	var action_b: Action = (
		actions[1] if actions.size() > 1 and actions[1] is Action else Action.idle()
	)
	agent_a.apply_action(action_a, dt, arena_half_extent)
	agent_b.apply_action(action_b, dt, arena_half_extent)
	var hit_a: bool = false
	var hit_b: bool = false
	var kill_a: bool = false
	var kill_b: bool = false
	var shot_a: bool = false
	var shot_b: bool = false
	var damage_a: float = 0.0
	var damage_b: float = 0.0

	if action_a.shoot and agent_a.weapon.try_fire():
		shot_a = true
		if agent_a.weapon.ray_hits_sphere(
			agent_a.get_eye_position(), agent_a.get_forward_vector(), agent_b.get_eye_position()
		):
			damage_a = agent_b.take_damage(agent_a.weapon.damage)
			hit_a = damage_a > 0.0
			kill_a = hit_a and not agent_b.alive
	if action_b.shoot and agent_b.weapon.try_fire():
		shot_b = true
		if agent_b.weapon.ray_hits_sphere(
			agent_b.get_eye_position(), agent_b.get_forward_vector(), agent_a.get_eye_position()
		):
			damage_b = agent_a.take_damage(agent_b.weapon.damage)
			hit_b = damage_b > 0.0
			kill_b = hit_b and not agent_a.alive

	if shot_a:
		episode_a.record_shot(hit_a)
	if shot_b:
		episode_b.record_shot(hit_b)
	if damage_a > 0.0:
		episode_a.record_damage_dealt(damage_a)
	if damage_b > 0.0:
		episode_b.record_damage_dealt(damage_b)
	if damage_b > 0.0:
		episode_a.record_damage_taken(damage_b)
	if damage_a > 0.0:
		episode_b.record_damage_taken(damage_a)

	var events_a := {
		"hit": hit_a,
		"kill": kill_a,
		"damage_taken": damage_b,
		"damage_dealt": damage_a,
		"died": not agent_a.alive,
		"shot_fired": shot_a,
		"useless_shot": shot_a and not hit_a,
		"alive": agent_a.alive,
	}
	var events_b := {
		"hit": hit_b,
		"kill": kill_b,
		"damage_taken": damage_a,
		"damage_dealt": damage_b,
		"died": not agent_b.alive,
		"shot_fired": shot_b,
		"useless_shot": shot_b and not hit_b,
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

	return {
		"observations": get_observations(),
		"rewards": [reward_a, reward_b],
		"done": done,
		"infos":
		[
			{
				"events": events_a,
				"done_reason": done_reason,
				"TimeLimit.truncated": done_reason == "timeout",
				"metrics":
				episode_a.to_metrics(
					SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_a_win"
				)
			},
			{
				"events": events_b,
				"done_reason": done_reason,
				"TimeLimit.truncated": done_reason == "timeout",
				"metrics":
				episode_b.to_metrics(
					SandboxConfig.SIMULATION_DT, 1, done_reason == "agent_b_win"
				)
			},
		],
	}


func get_observations() -> Array:
	return [
		Observation.build(agent_a, [proxy_b], arena_half_extent),
		Observation.build(agent_b, [proxy_a], arena_half_extent),
	]


func _sync_proxies() -> void:
	proxy_a.position = agent_a.position
	proxy_a.health = agent_a.health
	proxy_a.alive = agent_a.alive
	proxy_b.position = agent_b.position
	proxy_b.health = agent_b.health
	proxy_b.alive = agent_b.alive


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

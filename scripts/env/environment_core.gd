## EnvironmentCore
##
## The heart of one independent RL environment instance: an agent, one or
## more enemies, episode bookkeeping and reward calculation. This class has
## NO Node/scene-tree dependency at all, which is what makes it possible to
## run dozens of these purely as data objects (for fast/headless training)
## while an optional `EnvironmentView` (Node3D) renders a subset of them for
## human play / debugging.
##
## Exposes the standard RL-style interface requested by the milestone:
##   reset(seed) -> Observation
##   step(action) -> Dictionary{observation, reward, done, info}
##   get_observations() / get_rewards() / is_done()
class_name EnvironmentCore
extends RefCounted

var env_id: int = 0
var arena_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var max_steps: int = SandboxConfig.MAX_EPISODE_STEPS
var enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT

var agent: AgentState = AgentState.new()
var enemies: Array = []  # Array[EnemyState]
var episode: EpisodeState = EpisodeState.new()
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

var _last_observation: Observation = null
var _has_reset: bool = false


func _init(p_env_id: int = 0, p_enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	env_id = p_env_id
	enemy_count = maxi(1, p_enemy_count)
	for i in range(enemy_count):
		enemies.append(EnemyState.new())


# ---------------------------------------------------------------------------
# RL interface
# ---------------------------------------------------------------------------


## Deterministically (re)starts an episode. Passing the same `seed_value`
## always produces the same initial agent/enemy configuration and therefore
## the same first observation.
func reset(seed_value: int = -1) -> Observation:
	if seed_value >= 0:
		rng.seed = seed_value
	else:
		rng.randomize()

	agent.reset(SandboxConfig.AGENT_SPAWN_POSITION, SandboxConfig.AGENT_SPAWN_YAW_DEG)

	var spread: float = 2.5
	for i in range(enemies.size()):
		var enemy: EnemyState = enemies[i]
		var lateral: float = (i - (enemies.size() - 1) / 2.0) * spread
		var jitter_x: float = rng.randf_range(-0.5, 0.5)
		var jitter_z: float = rng.randf_range(-0.5, 0.5)
		var spawn: Vector3 = (
			SandboxConfig.ENEMY_SPAWN_POSITION + Vector3(lateral + jitter_x, 0.0, jitter_z)
		)
		enemy.reset(spawn)

	episode.start_new_episode()
	_has_reset = true
	_last_observation = Observation.build(agent, enemies, arena_half_extent)
	return _last_observation


## Advances the simulation by exactly one deterministic tick given a
## structured Action. Returns a dictionary with the standard RL step
## outputs: observation, reward, done, info.
func step(action: Action, dt: float = SandboxConfig.SIMULATION_DT) -> Dictionary:
	if not _has_reset:
		reset(SandboxConfig.DEFAULT_RANDOM_SEED)

	if episode.done:
		return _make_step_result(0.0, {"already_done": true})

	var alive_before: bool = agent.alive
	var prev_enemy: EnemyState = _nearest_alive_enemy(agent.position)
	var prev_distance: float = (
		agent.position.distance_to(prev_enemy.position) if prev_enemy != null else 0.0
	)

	agent.apply_action(action, dt, arena_half_extent)

	var hit: bool = false
	var kill: bool = false
	var useless_shot: bool = false

	if action.shoot:
		var fired: bool = agent.weapon.try_fire()
		if not fired:
			useless_shot = true
		else:
			var any_alive: bool = false
			var eye: Vector3 = agent.get_eye_position()
			var forward: Vector3 = agent.get_forward_vector()
			for e in enemies:
				var enemy: EnemyState = e
				if not enemy.alive:
					continue
				any_alive = true
				if agent.weapon.ray_hits_sphere(eye, forward, enemy.get_chest_position()):
					var applied: float = enemy.take_damage(agent.weapon.damage)
					if applied > 0.0:
						hit = true
					if not enemy.alive:
						kill = true
						episode.total_kills += 1
					break
			if not any_alive:
				useless_shot = true

	var damage_taken: float = 0.0
	for e in enemies:
		var enemy: EnemyState = e
		var dmg: float = enemy.update_ai(dt, agent.position, arena_half_extent)
		if dmg > 0.0:
			damage_taken += agent.take_damage(dmg)

	var died: bool = alive_before and not agent.alive
	if died:
		episode.total_deaths += 1

	var next_enemy: EnemyState = _nearest_alive_enemy(agent.position)
	var positioning_delta: float = 0.0
	if next_enemy != null and prev_enemy != null:
		var next_distance: float = agent.position.distance_to(next_enemy.position)
		if prev_distance > SandboxConfig.ENEMY_ATTACK_RANGE:
			positioning_delta = prev_distance - next_distance

	var events: Dictionary = {
		"hit": hit,
		"kill": kill,
		"damage_taken": damage_taken,
		"died": died,
		"useless_shot": useless_shot,
		"positioning_delta": positioning_delta,
		"alive": agent.alive,
	}
	var reward: float = RewardSystem.compute(events)
	episode.record_step(reward)

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

	_last_observation = Observation.build(agent, enemies, arena_half_extent)
	return _make_step_result(reward, {"events": events, "done_reason": episode.done_reason})


func get_observations() -> Observation:
	if _last_observation == null:
		return Observation.build(agent, enemies, arena_half_extent)
	return _last_observation


func get_rewards() -> float:
	return episode.last_reward


func is_done() -> bool:
	return episode.done


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
	for e in enemies:
		var enemy: EnemyState = e
		if enemy.alive:
			var d: float = enemy.position.distance_squared_to(from_position)
			if d < best_dist:
				best_dist = d
				best = enemy
	return best


func _all_enemies_dead() -> bool:
	for e in enemies:
		var enemy: EnemyState = e
		if enemy.alive:
			return false
	return true


## Public accessor for the nearest alive enemy relative to the agent (used
## by AI controllers and by the debug overlay). Returns null if no enemy is
## alive.
func get_primary_enemy() -> EnemyState:
	return _nearest_alive_enemy(agent.position)


func get_alive_enemy_count() -> int:
	var count: int = 0
	for e in enemies:
		var enemy: EnemyState = e
		if enemy.alive:
			count += 1
	return count

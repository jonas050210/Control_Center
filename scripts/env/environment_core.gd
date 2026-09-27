## EnvironmentCore
##
## The render-independent RL environment. All gameplay state is local to this
## object graph, so one instance can be reset or stepped without affecting
## another instance. The public contract is reset(seed) -> Observation and
## step(Action) -> {observation, reward, done, info}.
class_name EnvironmentCore
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentState = preload("res://scripts/agent/agent_state.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EpisodeState = preload("res://scripts/core/episode_state.gd")
const Observation = preload("res://scripts/core/observation.gd")
const RewardSystem = preload("res://scripts/reward/reward_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


var env_id: int = 0
var arena_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var max_steps: int = SandboxConfig.MAX_EPISODE_STEPS
var enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT
var curriculum: CurriculumConfig = CurriculumConfig.new()

var agent: AgentState = AgentState.new()
var enemies: Array = []  # Array[EnemyState]
var episode: EpisodeState = EpisodeState.new()
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

var _last_observation: Observation = null
var _has_reset: bool = false


func _init(p_env_id: int = 0, p_enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT) -> void:
	env_id = p_env_id
	enemy_count = maxi(1, p_enemy_count)
	curriculum = CurriculumConfig.new(CurriculumConfig.Level.ENEMY_ATTACKS, enemy_count)
	_rebuild_enemies(enemy_count)


func _rebuild_enemies(count: int) -> void:
	enemies.clear()
	for _i in range(count):
		enemies.append(EnemyState.new())


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
		var state: EnemyState = enemy
		state.radius = SandboxConfig.ENEMY_RADIUS * curriculum.target_radius_scale()


## Deterministically (re)starts an episode. Passing the same seed produces
## the same spawn positions and first observation. Passing a negative seed
## keeps the current RNG stream: a fresh RandomNumberGenerator is already
## randomized at construction, so unseeded use stays random, while a
## previously seeded environment continues its deterministic sequence
## (essential for reproducible training across auto-resets).
func reset(seed_value: int = -1) -> Observation:
	if seed_value >= 0:
		rng.seed = seed_value

	var target_count: int = curriculum.effective_enemy_count()
	if enemies.size() != target_count:
		enemy_count = target_count
		_rebuild_enemies(enemy_count)

	agent.reset(SandboxConfig.AGENT_SPAWN_POSITION, SandboxConfig.AGENT_SPAWN_YAW_DEG)
	agent.weapon.hit_radius = SandboxConfig.WEAPON_HIT_RADIUS * curriculum.target_radius_scale()
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
		enemy.radius = SandboxConfig.ENEMY_RADIUS * curriculum.target_radius_scale()
		enemy.move_speed = SandboxConfig.ENEMY_MOVE_SPEED
		enemy.attack_damage = SandboxConfig.ENEMY_ATTACK_DAMAGE

	episode.start_new_episode()
	_has_reset = true
	_last_observation = Observation.build(agent, enemies, arena_half_extent)
	return _last_observation


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

	agent.apply_action(action, dt, arena_half_extent)

	# Positioning reward: closure caused by the agent's OWN motion only,
	# measured immediately after the agent moved and before enemies advance.
	# (Measuring after enemy movement let a standing-still agent farm reward
	# for letting an enemy walk up to it.) Only measured against the same
	# enemy while it is still alive, preventing false penalties when an
	# enemy is killed and the target switches.
	var positioning_delta: float = 0.0
	if prev_enemy != null and prev_enemy.alive and prev_distance > SandboxConfig.ENEMY_ATTACK_RANGE:
		positioning_delta = prev_distance - agent.position.distance_to(prev_enemy.position)

	var hit: bool = false
	var kill: bool = false
	var useless_shot: bool = false
	var missed_shot: bool = false
	var shot_fired: bool = false
	var damage_dealt: float = 0.0

	if action.shoot:
		shot_fired = agent.weapon.try_fire()
		if not shot_fired:
			useless_shot = true
		else:
			var any_alive: bool = false
			var eye: Vector3 = agent.get_eye_position()
			var forward: Vector3 = agent.get_forward_vector()

			# Find the closest alive enemy along the ray trajectory
			var best_hit_enemy: EnemyState = null
			var best_hit_distance: float = INF

			for enemy_value in enemies:
				var enemy: EnemyState = enemy_value
				if not enemy.alive:
					continue
				any_alive = true
				var hit_dist: float = agent.weapon.ray_hit_distance(
					eye, forward, enemy.get_chest_position()
				)
				if hit_dist >= 0.0 and hit_dist < best_hit_distance:
					best_hit_distance = hit_dist
					best_hit_enemy = enemy

			if best_hit_enemy != null:
				var applied: float = best_hit_enemy.take_damage(agent.weapon.damage)
				if applied > 0.0:
					hit = true
					damage_dealt += applied
					episode.record_damage_dealt(applied)
					if not best_hit_enemy.alive:
						kill = true
						episode.record_kill()

			# A real miss against a live target is a genuine aiming attempt
			# (cheap PENALTY_MISSED_SHOT); only pulls that cannot connect at
			# all are "useless" (main PENALTY_USELESS_SHOT). This keeps the
			# expected value of shooting positive while aim is being learned.
			missed_shot = any_alive and not hit
			useless_shot = not any_alive
		if shot_fired:
			episode.record_shot(hit)

	var damage_taken: float = 0.0
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		var damage: float = enemy.update_ai(
			dt,
			agent.position,
			arena_half_extent,
			curriculum.enemy_movement_enabled(),
			curriculum.enemy_attacks_enabled()
		)
		if damage > 0.0:
			damage_taken += agent.take_damage(damage)
	if damage_taken > 0.0:
		episode.record_damage_taken(damage_taken)

	var died: bool = alive_before and not agent.alive
	if died:
		episode.record_death()

	var events: Dictionary = {
		"hit": hit,
		"kill": kill,
		"damage_taken": damage_taken,
		"damage_dealt": damage_dealt,
		"died": died,
		"useless_shot": useless_shot,
		"missed_shot": missed_shot,
		"shot_fired": shot_fired,
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

	_last_observation = Observation.build(agent, enemies, arena_half_extent)
	return _make_step_result(
		reward,
		{
			"events": events,
			"done_reason": episode.done_reason,
			"TimeLimit.truncated": episode.done_reason == "timeout",
			"metrics": get_metrics()
		}
	)


func get_observations() -> Observation:
	if _last_observation == null:
		return Observation.build(agent, enemies, arena_half_extent)
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
		if enemy.alive:
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

## EnvironmentCore
##
## The render-independent RL environment. All gameplay state is local to this
## object graph, so one instance can be reset or stepped without affecting
## another instance. The public contract is reset(seed) -> Observation and
## step(Action) -> {observation, reward, done, info}.
class_name EnvironmentCore
extends RefCounted

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
	for _i in range(enemy_count):
		enemies.append(EnemyState.new())


# ---------------------------------------------------------------------------
# RL interface
# ---------------------------------------------------------------------------


func set_curriculum_level(level: int) -> void:
	curriculum.level = clampi(
		level, CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT
	)
	for enemy in enemies:
		var state: EnemyState = enemy
		state.radius = SandboxConfig.ENEMY_RADIUS * curriculum.target_radius_scale()


## Deterministically (re)starts an episode. Passing the same seed produces
## the same spawn positions and first observation.
func reset(seed_value: int = -1) -> Observation:
	if seed_value >= 0:
		rng.seed = seed_value
	else:
		rng.randomize()

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
		return _make_step_result(0.0, {"already_done": true, "metrics": get_metrics()})
	if action == null:
		action = Action.idle()

	var alive_before: bool = agent.alive
	var prev_enemy: EnemyState = _nearest_alive_enemy(agent.position)
	var prev_distance: float = (
		agent.position.distance_to(prev_enemy.position) if prev_enemy != null else 0.0
	)

	agent.apply_action(action, dt, arena_half_extent)

	var hit: bool = false
	var kill: bool = false
	var useless_shot: bool = false
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
			for enemy_value in enemies:
				var enemy: EnemyState = enemy_value
				if not enemy.alive:
					continue
				any_alive = true
				if agent.weapon.ray_hits_sphere(eye, forward, enemy.get_chest_position()):
					var applied: float = enemy.take_damage(agent.weapon.damage)
					if applied > 0.0:
						hit = true
						damage_dealt += applied
						episode.record_damage_dealt(applied)
						if not enemy.alive:
							kill = true
							episode.record_kill()
						break
			# A fired miss is also useless for shaping purposes. A blocked shot
			# cannot be distinguished from a miss in the analytic hit-test.
			useless_shot = not hit or not any_alive
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
		"damage_dealt": damage_dealt,
		"died": died,
		"useless_shot": useless_shot,
		"shot_fired": shot_fired,
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
	return _make_step_result(
		reward, {"events": events, "done_reason": episode.done_reason, "metrics": get_metrics()}
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

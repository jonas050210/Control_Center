## Multi-enemy combat validation: enemy counts 1..5+, deterministic spawning,
## nearest-alive tracking stability when enemies die, observation bounds,
## dead-enemy exclusion from rewards, termination/reset semantics and
## curriculum parameter consistency.
class_name TestMultiEnemy
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


## Small tolerance for float32 comparisons (observation values may sit a few
## ulps outside a strict [-1, 1] check after normalization arithmetic).
const BOUNDS_EPSILON: float = 0.000001


## Every observation value produced across randomized multi-enemy episodes
## (levels 2-4, 1-5 enemies, including deaths, timeouts and auto-resets)
## must stay inside the documented [-1, 1] contract.
func test_observation_values_stay_within_documented_bounds() -> SandboxTest:
	var t := SandboxTest.new("multi_enemy_observation_values_within_bounds")
	for enemy_count in range(1, 6):
		for level in range(2, 5):
			var env := EnvironmentCore.new(0, enemy_count)
			env.set_curriculum_level(level)
			env.max_steps = 40  # force frequent terminations/resets
			env.reset(1000 + enemy_count * 10 + level)
			var rng := RandomNumberGenerator.new()
			rng.seed = 77 + enemy_count
			for _step in range(120):
				var action := Action.new(
					rng.randi_range(-1, 1),
					rng.randi_range(-1, 1),
					rng.randi_range(-1, 1),
					rng.randi_range(-1, 1),
					rng.randf() > 0.5
				)
				var result: Dictionary = env.step(action)
				var values: PackedFloat32Array = (result.observation as Observation).to_array()
				t.assert_eq(
					values.size(), Observation.FIELD_COUNT, "observation width is fixed at Observation.FIELD_COUNT"
				)
				for i in range(values.size()):
					if values[i] < -1.0 - BOUNDS_EPSILON or values[i] > 1.0 + BOUNDS_EPSILON:
						t.fail(
							(
								"observation value out of [-1,1] contract: enemies=%d level=%d index=%d value=%f"
								% [enemy_count, level, i, values[i]]
							)
						)
						return t
				if env.is_done():
					env.reset(-1)
	return t


## Killing one enemy out of several must not end the episode, must keep the
## nearest-alive tracking pointing at the next-nearest LIVE enemy, and must
## be reflected in the alive-count fraction.
func test_killing_one_of_several_enemies_keeps_tracking_stable() -> SandboxTest:
	var t := SandboxTest.new("killing_one_of_several_keeps_tracking_stable")
	var env := EnvironmentCore.new(0, 3)
	env.set_curriculum_level(CurriculumConfig.Level.ENEMY_ATTACKS)
	env.reset(21)
	# Controlled geometry: nearest at z=-2.5, mid at z=-5, far at z=-9, agent
	# at origin (all outside the 2 m melee range, so no attacks interfere).
	env.agent.position = Vector3.ZERO
	env.agent.yaw_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, -2.5)
	env.enemies[1].position = Vector3(0.0, 0.0, -5.0)
	env.enemies[2].position = Vector3(0.0, 0.0, -9.0)
	for enemy_value in env.enemies:
		var enemy: EnemyState = enemy_value
		enemy.move_speed = 0.0  # freeze positions: isolate tracking behavior

	var before: Observation = env.get_observations()
	t.assert_almost_eq(before.alive_enemy_count_norm, 1.0, 0.0001)

	env.enemies[0].take_damage(1000.0)
	var result: Dictionary = env.step(Action.idle())
	t.assert_false(result.done, "killing 1 of 3 enemies must not end the episode")
	var after: Observation = result.observation
	t.assert_true(after.enemy_alive, "primary slot must still report a live enemy")
	t.assert_almost_eq(
		after.enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE,
		5.0,
		0.01,
		"primary target must switch to the next-nearest ALIVE enemy"
	)
	t.assert_true(after.secondary_enemy_alive, "2nd slot must now report the far enemy")
	t.assert_almost_eq(
		after.secondary_enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE,
		9.0,
		0.01,
		"2nd slot must track the remaining alive enemy"
	)
	t.assert_false(after.tertiary_enemy_alive, "3rd slot must be empty with 2 alive")
	t.assert_almost_eq(after.alive_enemy_count_norm, 2.0 / 3.0, 0.0001)
	return t


## Dead enemies must be excluded from positioning reward: closing distance to
## a DEAD enemy must not pay approach reward while a live enemy elsewhere is
## the actual reference target.
func test_dead_enemies_do_not_grant_positioning_reward() -> SandboxTest:
	var t := SandboxTest.new("dead_enemies_do_not_grant_positioning_reward")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.ENEMY_ATTACKS)
	env.reset(33)
	env.agent.position = Vector3(0.0, 0.0, 5.0)
	env.agent.yaw_deg = 0.0
	# A dead enemy directly ahead, a live enemy far to the right (outside attack range).
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)
	env.enemies[0].take_damage(1000.0)
	env.enemies[1].position = Vector3(9.0, 0.0, 5.0)
	env.enemies[1].move_speed = 0.0

	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))
	t.assert_lte(
		result.info.events.positioning_delta,
		0.0,
		"walking toward a dead enemy must not earn approach reward; "
		+ "positioning is measured against the nearest ALIVE enemy only"
	)
	return t


## Level 4 with a configured count above 3: all enemies exist and affect the
## alive-count fraction, but only 3 are individually tracked.
func test_five_enemies_all_counted_but_only_three_tracked() -> SandboxTest:
	var t := SandboxTest.new("five_enemies_all_counted_three_tracked")
	var env := EnvironmentCore.new(0, 5)
	env.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env.reset(7)
	t.assert_eq(env.enemies.size(), 5, "level 4 must respect an explicit count above the minimum")
	var obs: Observation = env.get_observations()
	t.assert_almost_eq(
		obs.alive_enemy_count_norm, 1.0, 0.0001, "all five enemies must be counted"
	)
	t.assert_true(obs.enemy_alive and obs.secondary_enemy_alive and obs.tertiary_enemy_alive)
	# Kill the two untracked-by-rank enemies is impractical to identify from
	# the observation; instead verify the structural invariant that killing
	# any one enemy drops the fraction by exactly 1/N.
	env.enemies[0].take_damage(1000.0)
	var after: Observation = env.step(Action.idle()).observation
	t.assert_almost_eq(after.alive_enemy_count_norm, 4.0 / 5.0, 0.0001)
	return t


## Eliminating every enemy in a multi-enemy episode terminates it with the
## win reason, and the terminal step pays no further rewards afterwards.
func test_all_enemies_eliminated_ends_multi_enemy_episode() -> SandboxTest:
	var t := SandboxTest.new("all_enemies_eliminated_ends_multi_enemy_episode")
	var env := EnvironmentCore.new(0, 4)
	env.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env.reset(11)
	for enemy_value in env.enemies:
		(enemy_value as EnemyState).take_damage(1000.0)
	var result: Dictionary = env.step(Action.idle())
	t.assert_true(result.done, "eliminating all enemies must end the episode")
	t.assert_eq(env.episode.done_reason, "all_enemies_eliminated")
	t.assert_true(result.info.metrics.win, "metrics must report a win")
	# Dead enemies cannot keep paying survive ticks after termination.
	var extra: Dictionary = env.step(Action.idle())
	t.assert_true(extra.info.get("already_done", false))
	t.assert_almost_eq(float(extra.reward), 0.0, 0.000001)
	return t


## Deterministic multi-enemy reset at a count above the tracked-enemy budget.
func test_five_enemy_reset_is_deterministic_under_seed() -> SandboxTest:
	var t := SandboxTest.new("five_enemy_reset_deterministic_under_seed")
	var env_a := EnvironmentCore.new(0, 5)
	var env_b := EnvironmentCore.new(1, 5)
	env_a.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env_b.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env_a.reset(4242)
	env_b.reset(4242)
	for i in range(env_a.enemies.size()):
		t.assert_vec_almost_eq(
			env_a.enemies[i].position,
			env_b.enemies[i].position,
			0.00001,
			"same seed must reproduce identical 5-enemy spawn positions"
		)
		t.assert_almost_eq(
			env_a.enemies[i].strafe_phase,
			env_b.enemies[i].strafe_phase,
			0.00001,
			"same seed must reproduce identical strafe phases"
		)
	return t


## Regression: a curriculum level change must apply ALL level-derived enemy
## parameters to the existing enemies immediately (previously only the hit
## radius was updated; speed/cooldown stayed at the old level's values until
## the next reset).
func test_curriculum_change_applies_enemy_parameters_immediately() -> SandboxTest:
	var t := SandboxTest.new("curriculum_change_applies_enemy_parameters_immediately")
	var env := EnvironmentCore.new(0, 3)
	env.set_curriculum_level(CurriculumConfig.Level.ENEMY_ATTACKS)
	env.reset(5)
	env.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	var expected_speed: float = SandboxConfig.ENEMY_MOVE_SPEED * 1.15
	var expected_cooldown: float = SandboxConfig.ENEMY_ATTACK_COOLDOWN * 0.85
	for enemy_value in env.enemies:
		var enemy: EnemyState = enemy_value
		t.assert_almost_eq(
			enemy.move_speed, expected_speed, 0.0001, "level-4 speed must apply immediately"
		)
		t.assert_almost_eq(
			enemy.attack_cooldown_time,
			expected_cooldown,
			0.0001,
			"level-4 cooldown must apply immediately"
		)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	for enemy_value in env.enemies:
		var enemy2: EnemyState = enemy_value
		t.assert_almost_eq(
			enemy2.radius,
			SandboxConfig.ENEMY_RADIUS * 1.5,
			0.0001,
			"level-1 bigger target radius must apply immediately"
		)
	return t


## A single environment stepping alongside a 4-environment batch of the same
## seed base must observe the same multi-enemy determinism guarantees
## (reset_all seeding stays index-based for every enemy count).
func test_reset_all_seeding_is_index_based_across_enemy_counts() -> SandboxTest:
	var t := SandboxTest.new("reset_all_seeding_index_based_across_enemy_counts")
	var env_a := EnvironmentCore.new(0, 4)
	var env_b := EnvironmentCore.new(1, 4)
	env_a.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env_b.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env_a.reset(909)
	env_b.reset(910)  # base + index offset, as SimulationManager.reset_all does
	t.assert_true(
		env_a.enemies[0].position != env_b.enemies[0].position
			or env_a.enemies[1].position != env_b.enemies[1].position,
		"different env-index seeds should generally produce different spawns"
	)
	env_b.reset(909)
	for i in range(env_a.enemies.size()):
		t.assert_vec_almost_eq(
			env_a.enemies[i].position, env_b.enemies[i].position, 0.00001, "same seed must reproduce spawns"
		)
	return t

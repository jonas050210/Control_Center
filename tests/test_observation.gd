## Tests for the Observation builder.
class_name TestObservation
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const AgentState = preload("res://scripts/agent/agent_state.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_observation_array_has_documented_field_count() -> SandboxTest:
	var t := SandboxTest.new("observation_array_has_documented_field_count")
	var agent := AgentState.new()
	agent.reset()
	var enemy := EnemyState.new()
	enemy.reset()
	var obs := Observation.build(agent, [enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_eq(
		obs.to_array().size(), Observation.FIELD_COUNT, "array length must match FIELD_COUNT"
	)
	return t


func test_observation_picks_nearest_alive_enemy() -> SandboxTest:
	var t := SandboxTest.new("observation_picks_nearest_alive_enemy")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)

	var far_enemy := EnemyState.new()
	far_enemy.reset(Vector3(0.0, 0.0, -9.0))

	var near_enemy := EnemyState.new()
	near_enemy.reset(Vector3(0.0, 0.0, -1.0))

	var obs := Observation.build(agent, [far_enemy, near_enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(
		obs.enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE,
		1.0,
		0.01,
		"should report the nearer enemy's distance"
	)
	return t


func test_observation_falls_back_to_dead_enemy_when_none_alive() -> SandboxTest:
	var t := SandboxTest.new("observation_falls_back_when_all_dead")
	var agent := AgentState.new()
	agent.reset()
	var enemy := EnemyState.new()
	enemy.reset()
	enemy.take_damage(1000.0)
	var obs := Observation.build(agent, [enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_false(obs.enemy_alive, "reported enemy should be flagged dead")
	t.assert_almost_eq(obs.enemy_health_norm, 0.0, 0.001)
	return t


func test_agent_health_normalizes_to_unit_range() -> SandboxTest:
	var t := SandboxTest.new("agent_health_normalizes_to_unit_range")
	var agent := AgentState.new()
	agent.reset()
	agent.take_damage(25.0)
	var enemy := EnemyState.new()
	enemy.reset()
	var obs := Observation.build(agent, [enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(obs.agent_health_norm, 0.75, 0.001)
	return t


func test_observation_tracks_up_to_three_alive_enemies_nearest_first() -> SandboxTest:
	var t := SandboxTest.new("observation_tracks_up_to_three_alive_enemies_nearest_first")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)

	var near := EnemyState.new()
	near.reset(Vector3(0.0, 0.0, -2.0))
	var mid := EnemyState.new()
	mid.reset(Vector3(0.0, 0.0, -5.0))
	var far := EnemyState.new()
	far.reset(Vector3(0.0, 0.0, -9.0))

	# Deliberately shuffled array order: ranking must be by distance, not index.
	var obs := Observation.build(agent, [far, near, mid], SandboxConfig.ARENA_HALF_EXTENT)

	t.assert_almost_eq(obs.enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE, 2.0, 0.01)
	t.assert_true(obs.secondary_enemy_alive)
	t.assert_almost_eq(
		obs.secondary_enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE, 5.0, 0.01
	)
	t.assert_true(obs.tertiary_enemy_alive)
	t.assert_almost_eq(
		obs.tertiary_enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE, 9.0, 0.01
	)
	return t


func test_observation_missing_enemies_report_not_alive() -> SandboxTest:
	var t := SandboxTest.new("observation_missing_enemies_report_not_alive")
	var agent := AgentState.new()
	agent.reset()
	var enemy := EnemyState.new()
	enemy.reset()
	var obs := Observation.build(agent, [enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_false(obs.secondary_enemy_alive, "only one enemy exists; slot 2 must be empty")
	t.assert_false(obs.tertiary_enemy_alive, "only one enemy exists; slot 3 must be empty")
	return t


func test_observation_bearing_is_zero_when_enemy_dead_center() -> SandboxTest:
	var t := SandboxTest.new("observation_bearing_is_zero_when_enemy_dead_center")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)  # yaw 0 => forward is (0,0,-1)
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -5.0))  # directly ahead
	var obs := Observation.build(agent, [enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(obs.enemy_bearing_norm, 0.0, 0.01, "an enemy dead-ahead has zero bearing")
	return t


func test_observation_bearing_is_signed_for_left_and_right() -> SandboxTest:
	var t := SandboxTest.new("observation_bearing_is_signed_for_left_and_right")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var enemy_right := EnemyState.new()
	enemy_right.reset(Vector3(5.0, 0.0, 0.0))
	var enemy_left := EnemyState.new()
	enemy_left.reset(Vector3(-5.0, 0.0, 0.0))
	var obs_right := Observation.build(agent, [enemy_right], SandboxConfig.ARENA_HALF_EXTENT)
	var obs_left := Observation.build(agent, [enemy_left], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_gt(obs_right.enemy_bearing_norm, 0.0, "an enemy to the right must have positive bearing")
	t.assert_lt(obs_left.enemy_bearing_norm, 0.0, "an enemy to the left must have negative bearing")
	return t


func test_observation_alive_enemy_count_norm_reflects_alive_fraction() -> SandboxTest:
	var t := SandboxTest.new("observation_alive_enemy_count_norm_reflects_alive_fraction")
	var agent := AgentState.new()
	agent.reset()
	var alive_enemy := EnemyState.new()
	alive_enemy.reset()
	var dead_enemy := EnemyState.new()
	dead_enemy.reset()
	dead_enemy.take_damage(1000.0)
	var obs := Observation.build(agent, [alive_enemy, dead_enemy], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(obs.alive_enemy_count_norm, 0.5, 0.001)
	return t

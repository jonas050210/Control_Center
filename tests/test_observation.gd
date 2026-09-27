## Tests for the Observation builder.
class_name TestObservation
extends RefCounted


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

## Tests for EnemyState: idle/chase/attack behavior and damage/death.
class_name TestEnemyState
extends RefCounted

# Explicit dependency: headless --script runs do not populate the editor class cache.
const TestResult = preload("res://tests/sandbox_test.gd")


func test_enemy_chases_agent_when_far_away() -> TestResult:
	var t := TestResult.new("enemy_chases_agent_when_far")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -8.0))
	var agent_pos := Vector3(0.0, 0.0, 8.0)
	var start_distance: float = enemy.position.distance_to(agent_pos)
	enemy.update_ai(0.5, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	var new_distance: float = enemy.position.distance_to(agent_pos)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.CHASE, "enemy should be chasing when far away")
	t.assert_lt(new_distance, start_distance, "chasing enemy should close the distance")
	return t


func test_enemy_attacks_when_within_attack_range() -> TestResult:
	var t := TestResult.new("enemy_attacks_when_close")
	var enemy := EnemyState.new()
	var close_pos := Vector3(0.0, 0.0, -1.0)
	enemy.reset(close_pos)
	var agent_pos := Vector3.ZERO  # within ENEMY_ATTACK_RANGE
	var damage := enemy.update_ai(0.1, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.ATTACK, "enemy should switch to attack state")
	t.assert_almost_eq(
		damage, SandboxConfig.ENEMY_ATTACK_DAMAGE, 0.001, "attack should deal configured damage"
	)
	return t


func test_enemy_attack_respects_cooldown() -> TestResult:
	var t := TestResult.new("enemy_attack_respects_cooldown")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -1.0))
	var agent_pos := Vector3.ZERO
	var first := enemy.update_ai(0.01, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	var second := enemy.update_ai(0.01, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_gt(first, 0.0, "first attack in range should deal damage")
	t.assert_almost_eq(second, 0.0, 0.001, "second attack immediately after should be on cooldown")
	return t


func test_enemy_take_damage_and_death() -> TestResult:
	var t := TestResult.new("enemy_take_damage_and_death")
	var enemy := EnemyState.new()
	enemy.reset()
	enemy.take_damage(60.0)
	t.assert_almost_eq(enemy.health, enemy.max_health - 60.0)
	t.assert_true(enemy.alive)

	enemy.take_damage(1000.0)
	t.assert_almost_eq(enemy.health, 0.0, 0.001)
	t.assert_false(enemy.alive, "enemy should die once health reaches zero")
	t.assert_eq(enemy.ai_state, EnemyState.AIState.DEAD, "dead enemy AI state should be DEAD")
	return t


func test_dead_enemy_does_not_move_or_attack() -> TestResult:
	var t := TestResult.new("dead_enemy_inert")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -1.0))
	enemy.take_damage(1000.0)
	var pos_before: Vector3 = enemy.position
	var damage := enemy.update_ai(1.0, Vector3.ZERO, SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(damage, 0.0, 0.001, "dead enemy should not deal damage")
	t.assert_vec_almost_eq(enemy.position, pos_before, 0.001, "dead enemy should not move")
	return t

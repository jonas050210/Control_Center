## Tests for EnemyState: idle/chase/attack behavior and damage/death.
class_name TestEnemyState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_enemy_chases_agent_when_far_away() -> SandboxTest:
	var t := SandboxTest.new("enemy_chases_agent_when_far")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -8.0))
	var agent_pos := Vector3(0.0, 0.0, 8.0)
	var start_distance: float = enemy.position.distance_to(agent_pos)
	enemy.update_ai(0.5, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	var new_distance: float = enemy.position.distance_to(agent_pos)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.CHASE, "enemy should be chasing when far away")
	t.assert_lt(new_distance, start_distance, "chasing enemy should close the distance")
	return t


func test_enemy_attacks_when_within_attack_range() -> SandboxTest:
	var t := SandboxTest.new("enemy_attacks_when_close")
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


func test_enemy_attack_respects_cooldown() -> SandboxTest:
	var t := SandboxTest.new("enemy_attack_respects_cooldown")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -1.0))
	var agent_pos := Vector3.ZERO
	var first := enemy.update_ai(0.01, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	var second := enemy.update_ai(0.01, agent_pos, SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_gt(first, 0.0, "first attack in range should deal damage")
	t.assert_almost_eq(second, 0.0, 0.001, "second attack immediately after should be on cooldown")
	return t


func test_enemy_take_damage_and_death() -> SandboxTest:
	var t := SandboxTest.new("enemy_take_damage_and_death")
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


func test_dead_enemy_does_not_move_or_attack() -> SandboxTest:
	var t := SandboxTest.new("dead_enemy_inert")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -1.0))
	enemy.take_damage(1000.0)
	var pos_before: Vector3 = enemy.position
	var damage := enemy.update_ai(1.0, Vector3.ZERO, SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(damage, 0.0, 0.001, "dead enemy should not deal damage")
	t.assert_vec_almost_eq(enemy.position, pos_before, 0.001, "dead enemy should not move")
	return t


## Regression: without strafing enabled (the default), chase motion must stay
## a pure straight line toward the agent, exactly as before this milestone.
func test_chase_without_strafe_moves_in_a_straight_line() -> SandboxTest:
	var t := SandboxTest.new("chase_without_strafe_moves_in_a_straight_line")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -8.0))
	var agent_pos := Vector3(0.0, 0.0, 8.0)
	for _i in range(10):
		enemy.update_ai(0.1, agent_pos, SandboxConfig.ARENA_HALF_EXTENT, true, true, false)
	t.assert_almost_eq(enemy.position.x, 0.0, 0.0001, "no strafe means no lateral drift")
	return t


## Strafing enemies must weave laterally (nonzero perpendicular displacement)
## instead of walking in a straight line at the agent.
func test_strafing_enemy_moves_laterally() -> SandboxTest:
	var t := SandboxTest.new("strafing_enemy_moves_laterally")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -8.0))
	enemy.strafe_direction = 1.0
	enemy.strafe_phase = 0.0
	var agent_pos := Vector3(0.0, 0.0, 8.0)
	var max_lateral_drift: float = 0.0
	for _i in range(60):
		enemy.update_ai(0.05, agent_pos, SandboxConfig.ARENA_HALF_EXTENT, true, true, true)
		max_lateral_drift = maxf(max_lateral_drift, absf(enemy.position.x))
	t.assert_gt(max_lateral_drift, 0.05, "strafing enemy should drift laterally, not walk straight")
	return t


## Deterministic seeded strafing: identical strafe_direction/strafe_phase and
## identical update sequence must produce byte-for-byte identical positions,
## since strafing is a pure function of elapsed time, not further RNG calls.
func test_strafing_is_deterministic_given_same_phase_and_direction() -> SandboxTest:
	var t := SandboxTest.new("strafing_is_deterministic_given_same_phase_and_direction")
	var enemy_a := EnemyState.new()
	var enemy_b := EnemyState.new()
	enemy_a.reset(Vector3(0.0, 0.0, -8.0))
	enemy_b.reset(Vector3(0.0, 0.0, -8.0))
	enemy_a.strafe_direction = -1.0
	enemy_b.strafe_direction = -1.0
	enemy_a.strafe_phase = 1.23
	enemy_b.strafe_phase = 1.23
	var agent_pos := Vector3(0.0, 0.0, 8.0)
	for _i in range(40):
		enemy_a.update_ai(0.05, agent_pos, SandboxConfig.ARENA_HALF_EXTENT, true, true, true)
		enemy_b.update_ai(0.05, agent_pos, SandboxConfig.ARENA_HALF_EXTENT, true, true, true)
	t.assert_vec_almost_eq(
		enemy_a.position, enemy_b.position, 0.00001, "identical strafe seeds must stay in lockstep"
	)
	return t


## Even while strafing, the enemy must still eventually close the distance
## and reach attack range instead of orbiting forever — strafing should
## make the approach harder to predict, not prevent it entirely.
func test_strafing_enemy_still_eventually_reaches_attack_range() -> SandboxTest:
	var t := SandboxTest.new("strafing_enemy_still_eventually_reaches_attack_range")
	var enemy := EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -8.0))
	enemy.strafe_direction = 1.0
	enemy.strafe_phase = 0.0
	var agent_pos := Vector3(0.0, 0.0, 8.0)
	var reached: bool = false
	for _i in range(2000):
		enemy.update_ai(0.05, agent_pos, SandboxConfig.ARENA_HALF_EXTENT, true, false, true)
		if enemy.position.distance_to(agent_pos) <= SandboxConfig.ENEMY_ATTACK_RANGE:
			reached = true
			break
	t.assert_true(reached, "strafing enemy should still eventually approach the agent")
	return t

## Tests for AgentState: movement, look/aim, bounds clamping, damage/death.
class_name TestAgentState
extends RefCounted

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_move_forward_advances_position_along_facing_direction() -> SandboxTest:
	var t := SandboxTest.new("agent_move_forward_advances_position")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)  # yaw 0 -> forward is -Z
	var dt := 0.5
	agent.apply_action(
		Action.from_discrete(Action.Discrete.MOVE_FORWARD), dt, SandboxConfig.ARENA_HALF_EXTENT
	)
	var expected := Vector3(0.0, 0.0, -agent.move_speed * dt)
	t.assert_vec_almost_eq(agent.position, expected, 0.001, "forward movement distance/direction")
	return t


func test_move_backward_moves_opposite_of_forward() -> SandboxTest:
	var t := SandboxTest.new("agent_move_backward")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	agent.apply_action(
		Action.from_discrete(Action.Discrete.MOVE_BACKWARD), 0.5, SandboxConfig.ARENA_HALF_EXTENT
	)
	t.assert_gt(agent.position.z, 0.0, "moving backward from yaw=0 should increase z")
	return t


func test_strafe_left_and_right_are_perpendicular_to_forward() -> SandboxTest:
	var t := SandboxTest.new("agent_strafe_left_right")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	agent.apply_action(
		Action.from_discrete(Action.Discrete.STRAFE_RIGHT), 0.5, SandboxConfig.ARENA_HALF_EXTENT
	)
	t.assert_gt(agent.position.x, 0.0, "strafing right at yaw=0 should move +X")

	var agent2 := AgentState.new()
	agent2.reset(Vector3.ZERO, 0.0)
	agent2.apply_action(
		Action.from_discrete(Action.Discrete.STRAFE_LEFT), 0.5, SandboxConfig.ARENA_HALF_EXTENT
	)
	t.assert_lt(agent2.position.x, 0.0, "strafing left at yaw=0 should move -X")
	return t


func test_look_left_and_right_change_yaw() -> SandboxTest:
	var t := SandboxTest.new("agent_look_left_right_changes_yaw")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var dt := 0.2
	agent.apply_action(
		Action.from_discrete(Action.Discrete.LOOK_RIGHT), dt, SandboxConfig.ARENA_HALF_EXTENT
	)
	t.assert_almost_eq(agent.yaw_deg, agent.turn_speed_deg * dt, 0.01, "look right increases yaw")

	var agent2 := AgentState.new()
	agent2.reset(Vector3.ZERO, 0.0)
	agent2.apply_action(
		Action.from_discrete(Action.Discrete.LOOK_LEFT), dt, SandboxConfig.ARENA_HALF_EXTENT
	)
	var expected_yaw: float = wrapf(-agent2.turn_speed_deg * dt, 0.0, 360.0)
	t.assert_almost_eq(agent2.yaw_deg, expected_yaw, 0.01, "look left decreases yaw")
	return t


func test_look_up_and_down_change_and_clamp_pitch() -> SandboxTest:
	var t := SandboxTest.new("agent_look_up_down_clamps_pitch")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	agent.apply_action(
		Action.from_discrete(Action.Discrete.LOOK_UP), 0.2, SandboxConfig.ARENA_HALF_EXTENT
	)
	t.assert_gt(agent.pitch_deg, 0.0, "look up increases pitch")

	# Hammer look-up for a long time; pitch must clamp, never exceed limit.
	var agent2 := AgentState.new()
	agent2.reset(Vector3.ZERO, 0.0)
	for _i in range(1000):
		agent2.apply_action(
			Action.from_discrete(Action.Discrete.LOOK_UP), 0.1, SandboxConfig.ARENA_HALF_EXTENT
		)
	t.assert_almost_eq(
		agent2.pitch_deg, SandboxConfig.AGENT_PITCH_LIMIT_DEG, 0.01, "pitch clamps at limit"
	)
	return t


func test_arena_bounds_clamp_agent_inside_walls() -> SandboxTest:
	var t := SandboxTest.new("agent_arena_bounds_clamp")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)  # facing -Z
	for _i in range(2000):
		agent.apply_action(
			Action.from_discrete(Action.Discrete.MOVE_FORWARD), 0.1, SandboxConfig.ARENA_HALF_EXTENT
		)
	var limit: float = SandboxConfig.ARENA_HALF_EXTENT - agent.radius
	t.assert_true(agent.position.z >= -limit - 0.001, "agent should not pass through the wall")
	t.assert_almost_eq(agent.position.z, -limit, 0.01, "agent should be clamped at the wall")
	return t


func test_take_damage_reduces_health_and_kills_at_zero() -> SandboxTest:
	var t := SandboxTest.new("agent_take_damage_and_death")
	var agent := AgentState.new()
	agent.reset()
	var applied := agent.take_damage(30.0)
	t.assert_almost_eq(applied, 30.0)
	t.assert_almost_eq(agent.health, agent.max_health - 30.0)
	t.assert_true(agent.alive, "agent should still be alive above zero health")

	agent.take_damage(1000.0)
	t.assert_almost_eq(agent.health, 0.0, 0.001, "health should clamp at zero")
	t.assert_false(agent.alive, "agent should be dead at zero health")

	var post_death_damage := agent.take_damage(10.0)
	t.assert_almost_eq(post_death_damage, 0.0, 0.001, "no further damage can be applied once dead")
	return t

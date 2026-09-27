## Tests for the Action struct / discrete action mapping.
class_name TestAction
extends RefCounted

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_idle_default() -> SandboxTest:
	var t := SandboxTest.new("action_idle_default")
	var a := Action.idle()
	t.assert_eq(a.move_axis, 0)
	t.assert_eq(a.strafe_axis, 0)
	t.assert_eq(a.look_yaw_axis, 0)
	t.assert_eq(a.look_pitch_axis, 0)
	t.assert_false(a.shoot)
	return t


func test_from_discrete_maps_all_ten_actions() -> SandboxTest:
	var t := SandboxTest.new("action_from_discrete_maps_all_ten_actions")

	var forward := Action.from_discrete(Action.Discrete.MOVE_FORWARD)
	t.assert_eq(forward.move_axis, 1, "move forward")

	var backward := Action.from_discrete(Action.Discrete.MOVE_BACKWARD)
	t.assert_eq(backward.move_axis, -1, "move backward")

	var strafe_l := Action.from_discrete(Action.Discrete.STRAFE_LEFT)
	t.assert_eq(strafe_l.strafe_axis, -1, "strafe left")

	var strafe_r := Action.from_discrete(Action.Discrete.STRAFE_RIGHT)
	t.assert_eq(strafe_r.strafe_axis, 1, "strafe right")

	var look_l := Action.from_discrete(Action.Discrete.LOOK_LEFT)
	t.assert_eq(look_l.look_yaw_axis, -1, "look left")

	var look_r := Action.from_discrete(Action.Discrete.LOOK_RIGHT)
	t.assert_eq(look_r.look_yaw_axis, 1, "look right")

	var look_u := Action.from_discrete(Action.Discrete.LOOK_UP)
	t.assert_eq(look_u.look_pitch_axis, 1, "look up")

	var look_d := Action.from_discrete(Action.Discrete.LOOK_DOWN)
	t.assert_eq(look_d.look_pitch_axis, -1, "look down")

	var shoot_action := Action.from_discrete(Action.Discrete.SHOOT)
	t.assert_true(shoot_action.shoot, "shoot")

	var idle_action := Action.from_discrete(Action.Discrete.IDLE)
	t.assert_eq(idle_action.move_axis, 0, "idle has no move")
	t.assert_false(idle_action.shoot, "idle does not shoot")

	t.assert_eq(Action.DISCRETE_COUNT, 10, "exactly 10 discrete actions")
	return t

## Action
##
## Canonical action representation shared by human controllers, Godot's RL
## adapter, demonstration recording and external policies.  The first five
## values are deliberately fixed: four ternary axes and a binary trigger.
## Continuous mouse-look remains an optional sixth/seventh logging field, but
## is not part of the PPO action space.
class_name Action
extends RefCounted

enum Discrete {
	IDLE = 0,
	MOVE_FORWARD = 1,
	MOVE_BACKWARD = 2,
	STRAFE_LEFT = 3,
	STRAFE_RIGHT = 4,
	LOOK_LEFT = 5,
	LOOK_RIGHT = 6,
	LOOK_UP = 7,
	LOOK_DOWN = 8,
	SHOOT = 9,
}

const DISCRETE_COUNT: int = 10
## MultiDiscrete cardinalities in the order returned by to_multidiscrete().
const MULTI_DISCRETE_NVECS: Array = [3, 3, 3, 3, 2]
const MULTI_DISCRETE_SIZE: int = 5

## -1 backward, 0 none, +1 forward.
var move_axis: int = 0
## -1 left, 0 none, +1 right.
var strafe_axis: int = 0
## -1 look left, 0 none, +1 look right.
var look_yaw_axis: int = 0
## -1 look down, 0 none, +1 look up.
var look_pitch_axis: int = 0
## Whether the weapon trigger is held this tick.
var shoot: bool = false
## Optional continuous mouse-look delta in degrees. Not used by the default
## PPO action space, but preserved so human demonstrations remain lossless.
var look_delta: Vector2 = Vector2.ZERO


func _init(
	p_move_axis: int = 0,
	p_strafe_axis: int = 0,
	p_look_yaw_axis: int = 0,
	p_look_pitch_axis: int = 0,
	p_shoot: bool = false,
	p_look_delta: Vector2 = Vector2.ZERO
) -> void:
	move_axis = clampi(p_move_axis, -1, 1)
	strafe_axis = clampi(p_strafe_axis, -1, 1)
	look_yaw_axis = clampi(p_look_yaw_axis, -1, 1)
	look_pitch_axis = clampi(p_look_pitch_axis, -1, 1)
	shoot = p_shoot
	look_delta = p_look_delta


static func from_discrete(discrete_action: int) -> Action:
	match discrete_action:
		Discrete.MOVE_FORWARD:
			return Action.new(1, 0, 0, 0, false)
		Discrete.MOVE_BACKWARD:
			return Action.new(-1, 0, 0, 0, false)
		Discrete.STRAFE_LEFT:
			return Action.new(0, -1, 0, 0, false)
		Discrete.STRAFE_RIGHT:
			return Action.new(0, 1, 0, 0, false)
		Discrete.LOOK_LEFT:
			return Action.new(0, 0, -1, 0, false)
		Discrete.LOOK_RIGHT:
			return Action.new(0, 0, 1, 0, false)
		Discrete.LOOK_UP:
			return Action.new(0, 0, 0, 1, false)
		Discrete.LOOK_DOWN:
			return Action.new(0, 0, 0, -1, false)
		Discrete.SHOOT:
			return Action.new(0, 0, 0, 0, true)
		_:
			return Action.idle()


## Converts an external MultiDiscrete action [0..2, 0..2, 0..2, 0..2, 0..1]
## to the canonical -1..1/boolean representation.
static func from_multidiscrete(values: Array) -> Action:
	if values.size() < MULTI_DISCRETE_SIZE:
		return Action.idle()
	return Action.new(
		clampi(int(values[0]), 0, 2) - 1,
		clampi(int(values[1]), 0, 2) - 1,
		clampi(int(values[2]), 0, 2) - 1,
		clampi(int(values[3]), 0, 2) - 1,
		int(values[4]) > 0
	)


## Returns the canonical MultiDiscrete representation used by Python PPO.
func to_multidiscrete() -> Array:
	return [
		move_axis + 1,
		strafe_axis + 1,
		look_yaw_axis + 1,
		look_pitch_axis + 1,
		1 if shoot else 0,
	]


static func idle() -> Action:
	return Action.new()


## Flat logging representation. The two final values are continuous look
## deltas and are intentionally excluded from MULTI_DISCRETE_NVECS.
func to_array() -> Array:
	return [
		move_axis,
		strafe_axis,
		look_yaw_axis,
		look_pitch_axis,
		1 if shoot else 0,
		look_delta.x,
		look_delta.y,
	]


func _to_string() -> String:
	return (
		"Action(move=%d strafe=%d yaw=%d pitch=%d shoot=%s look_delta=%s)"
		% [move_axis, strafe_axis, look_yaw_axis, look_pitch_axis, str(shoot), str(look_delta)]
	)

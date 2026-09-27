## Action
##
## Structured action representation shared by the human controller and any
## AI/RL controller. Kept deliberately simple (small integer/bool fields)
## so it serializes trivially to/from a flat array for an external RL
## framework, while still allowing multiple sub-actions to be combined in a
## single tick (e.g. "move forward" + "shoot" at the same time), which real
## FPS play requires.
##
## Extensibility: `look_delta` is reserved for a future continuous
## mouse-aiming mode. It defaults to Vector2.ZERO and is ignored by
## AgentState.apply_action() unless SandboxConfig / the caller opts into
## continuous look. This lets continuous aiming be introduced later without
## reshaping the discrete action interface.
class_name Action
extends RefCounted

## The 10 discrete actions requested by the milestone spec. Exactly one of
## these is selected when the environment is driven in single-choice
## discrete mode via `Action.from_discrete()`.
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

## -1 backward, 0 none, +1 forward
var move_axis: int = 0
## -1 left, 0 none, +1 right
var strafe_axis: int = 0
## -1 look left, 0 none, +1 look right
var look_yaw_axis: int = 0
## -1 look down, 0 none, +1 look up
var look_pitch_axis: int = 0
## Reserved for future continuous mouse-look input (pixels/normalized delta).
var look_delta: Vector2 = Vector2.ZERO
## Whether the weapon trigger is held this tick.
var shoot: bool = false


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


## Builds a structured Action from one of the 10 discrete choices.
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
			return Action.new()


## Returns an idle action (no movement, no look, no shoot).
static func idle() -> Action:
	return Action.new()


## Flat numeric encoding, useful for logging / a future RL adapter that
## expects arrays instead of objects.
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

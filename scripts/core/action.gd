## Action
##
## Canonical action representation shared by human controllers, Godot's RL
## adapter, demonstration recording and external policies.  The first five
## values are deliberately fixed: four ternary axes and a binary trigger.
## Continuous mouse-look remains an optional trailing logging field pair, but
## is not part of the PPO action space.
##
## Contract v2 adds a sixth binary field, `jump`, so the policy can control
## vertical movement (Phase 8). The four ternary axes and the shoot trigger
## keep their index and meaning; `jump` is appended, which is why the
## MultiDiscrete nvec grew from [3,3,3,3,2] to [3,3,3,3,2,2] instead of
## being reordered.
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
	JUMP = 10,
}

const DISCRETE_COUNT: int = 11
## MultiDiscrete cardinalities in the order returned by to_multidiscrete().
const MULTI_DISCRETE_NVECS: Array = [3, 3, 3, 3, 2, 2]
const MULTI_DISCRETE_SIZE: int = 6
## Number of values in to_array(): 4 axes + shoot + jump + 2 look deltas.
const LOG_ARRAY_SIZE: int = 8

## Absolute path to this very script. The static factory methods below build
## new instances through `load(SELF_PATH).new(...)` instead of `Action.new(...)`.
##
## Referencing the script's own `class_name` in a *value* context requires the
## global class to be registered in the editor class cache
## (.godot/global_script_class_cache.cfg). That cache does not exist during a
## standalone `godot --headless --script ...` run on a freshly checked-out
## project, so `Action.new()` fails with "Identifier not found: Action".
## `preload(self)` cannot be used either — it forms a compile-time cyclic
## reference. `load()` is resolved at runtime (when this script is already
## compiled and cached), so it is both safe and cheap.
const SELF_PATH: String = "res://scripts/core/action.gd"

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
## Whether a jump is requested this tick. Only takes effect when the
## character is standing on the floor or a standable box.
var jump: bool = false
## Optional continuous mouse-look delta in degrees. Not used by the default
## PPO action space, but preserved so human demonstrations remain lossless.
var look_delta: Vector2 = Vector2.ZERO


func _init(
	p_move_axis: int = 0,
	p_strafe_axis: int = 0,
	p_look_yaw_axis: int = 0,
	p_look_pitch_axis: int = 0,
	p_shoot: bool = false,
	p_look_delta: Vector2 = Vector2.ZERO,
	p_jump: bool = false
) -> void:
	move_axis = clampi(p_move_axis, -1, 1)
	strafe_axis = clampi(p_strafe_axis, -1, 1)
	look_yaw_axis = clampi(p_look_yaw_axis, -1, 1)
	look_pitch_axis = clampi(p_look_pitch_axis, -1, 1)
	shoot = p_shoot
	look_delta = p_look_delta
	jump = p_jump


static func _make(
	p_move_axis: int = 0,
	p_strafe_axis: int = 0,
	p_look_yaw_axis: int = 0,
	p_look_pitch_axis: int = 0,
	p_shoot: bool = false,
	p_look_delta: Vector2 = Vector2.ZERO,
	p_jump: bool = false
) -> Action:
	var action_script := load(SELF_PATH) as GDScript
	return (
		action_script.new(
			p_move_axis,
			p_strafe_axis,
			p_look_yaw_axis,
			p_look_pitch_axis,
			p_shoot,
			p_look_delta,
			p_jump
		)
		as Action
	)


static func from_discrete(discrete_action: int) -> Action:
	match discrete_action:
		Discrete.MOVE_FORWARD:
			return _make(1, 0, 0, 0, false)
		Discrete.MOVE_BACKWARD:
			return _make(-1, 0, 0, 0, false)
		Discrete.STRAFE_LEFT:
			return _make(0, -1, 0, 0, false)
		Discrete.STRAFE_RIGHT:
			return _make(0, 1, 0, 0, false)
		Discrete.LOOK_LEFT:
			return _make(0, 0, -1, 0, false)
		Discrete.LOOK_RIGHT:
			return _make(0, 0, 1, 0, false)
		Discrete.LOOK_UP:
			return _make(0, 0, 0, 1, false)
		Discrete.LOOK_DOWN:
			return _make(0, 0, 0, -1, false)
		Discrete.SHOOT:
			return _make(0, 0, 0, 0, true)
		Discrete.JUMP:
			return _make(0, 0, 0, 0, false, Vector2.ZERO, true)
		_:
			return idle()


## Converts an external MultiDiscrete action
## [0..2, 0..2, 0..2, 0..2, 0..1, 0..1] to the canonical -1..1/boolean
## representation. A 5-value action (contract v1, no jump) is still
## accepted and simply never jumps, so old checkpoints and recorded
## datasets keep working against the extended environment.
static func from_multidiscrete(values: Array) -> Action:
	if values.size() < MULTI_DISCRETE_SIZE - 1:
		return idle()
	var jump_value: bool = values.size() >= MULTI_DISCRETE_SIZE and int(values[5]) > 0
	return _make(
		clampi(int(values[0]), 0, 2) - 1,
		clampi(int(values[1]), 0, 2) - 1,
		clampi(int(values[2]), 0, 2) - 1,
		clampi(int(values[3]), 0, 2) - 1,
		int(values[4]) > 0,
		Vector2.ZERO,
		jump_value
	)


## Returns the canonical MultiDiscrete representation used by Python PPO.
func to_multidiscrete() -> Array:
	return [
		move_axis + 1,
		strafe_axis + 1,
		look_yaw_axis + 1,
		look_pitch_axis + 1,
		1 if shoot else 0,
		1 if jump else 0,
	]


static func idle() -> Action:
	return _make()


## Flat logging representation (LOG_ARRAY_SIZE values). The two final
## values are continuous look deltas and are intentionally excluded from
## MULTI_DISCRETE_NVECS.
func to_array() -> Array:
	return [
		move_axis,
		strafe_axis,
		look_yaw_axis,
		look_pitch_axis,
		1 if shoot else 0,
		1 if jump else 0,
		look_delta.x,
		look_delta.y,
	]


func _to_string() -> String:
	return (
		"Action(move=%d strafe=%d yaw=%d pitch=%d shoot=%s jump=%s look_delta=%s)"
		% [
			move_axis,
			strafe_axis,
			look_yaw_axis,
			look_pitch_axis,
			str(shoot),
			str(jump),
			str(look_delta)
		]
	)

## HumanController
##
## Human debug/play controller. Converts WASD + mouse-look + left-click into
## exactly the same `Action` struct an AI controller would produce, so the
## human plays through the identical action/state pipeline used for RL.
## This is also the intended hook point for a later "record human
## demonstrations" milestone: every Action produced here already has the
## shape a demonstration-recording buffer would want to log.
##
## Raw `Input.is_key_pressed` / `Input.is_mouse_button_pressed` calls are
## used instead of the project's Input Map so this controller has zero
## dependency on project.godot input bindings (keeps the project file small
## and avoids the risk of a mis-authored input map breaking project load).
class_name HumanController
extends ControllerBase

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const ControllerBase = preload("res://scripts/input/controller_base.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")


@export var mouse_sensitivity_deg_per_px: float = 0.15
@export var start_mouse_captured: bool = true
@export var arrow_key_look_enabled: bool = true

var _pending_look_delta: Vector2 = Vector2.ZERO


func _ready() -> void:
	if start_mouse_captured:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var key_event: InputEventKey = event
		if key_event.pressed and not key_event.echo and key_event.keycode == KEY_ESCAPE:
			_toggle_mouse_capture()
	elif event is InputEventMouseButton:
		var button_event: InputEventMouseButton = event
		if button_event.pressed and button_event.button_index == MOUSE_BUTTON_LEFT:
			if Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
				Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif event is InputEventMouseMotion:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			var motion: InputEventMouseMotion = event
			_pending_look_delta.x += motion.relative.x * mouse_sensitivity_deg_per_px
			_pending_look_delta.y += -motion.relative.y * mouse_sensitivity_deg_per_px


func _toggle_mouse_capture() -> void:
	if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	else:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func get_action(_env: EnvironmentCore) -> Action:
	var move_axis: int = 0
	if Input.is_key_pressed(KEY_W):
		move_axis += 1
	if Input.is_key_pressed(KEY_S):
		move_axis -= 1

	var strafe_axis: int = 0
	if Input.is_key_pressed(KEY_D):
		strafe_axis += 1
	if Input.is_key_pressed(KEY_A):
		strafe_axis -= 1

	var look_yaw_axis: int = 0
	var look_pitch_axis: int = 0
	if arrow_key_look_enabled:
		if Input.is_key_pressed(KEY_RIGHT):
			look_yaw_axis += 1
		if Input.is_key_pressed(KEY_LEFT):
			look_yaw_axis -= 1
		if Input.is_key_pressed(KEY_UP):
			look_pitch_axis += 1
		if Input.is_key_pressed(KEY_DOWN):
			look_pitch_axis -= 1

	var shoot: bool = (
		Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT) or Input.is_key_pressed(KEY_SPACE)
	)

	var action: Action = Action.new(
		move_axis, strafe_axis, look_yaw_axis, look_pitch_axis, shoot, _pending_look_delta
	)
	_pending_look_delta = Vector2.ZERO
	return action

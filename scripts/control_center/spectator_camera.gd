## SpectatorCamera
##
## Presentation-only camera rig (Phase 13). It offers four ways to look at
## the selected environment:
##
##   FIRST_PERSON  reuses the AgentView camera (exactly what a human plays
##                 through, and exactly what WATCH shows of the AI)
##   THIRD_PERSON  a chase camera behind and above the agent
##   FREE          right-drag to look, WASD/QE to fly
##   TOP_DOWN      an orthogonal-ish overhead view of the whole arena
##
## The rig NEVER writes to the simulation: it only reads the agent
## transform that the views already mirror. Free-flight input is ignored
## while a human is driving the agent so the two never fight over WASD.
class_name SpectatorCamera
extends Node3D

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const FREE_SPEED: float = 12.0
const FREE_SPEED_FAST: float = 30.0
const LOOK_SENSITIVITY: float = 0.25
const THIRD_PERSON_BACK: float = 5.5
const THIRD_PERSON_UP: float = 2.6

var camera: Camera3D
var mode: int = ControlCenterConfig.CameraMode.FIRST_PERSON
var input_enabled: bool = true

var _agent_view: Node3D
var _view_origin: Vector3 = Vector3.ZERO
var _arena_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var _free_position: Vector3 = Vector3(0.0, 12.0, 18.0)
var _free_yaw: float = 0.0
var _free_pitch: float = -0.45
var _looking: bool = false
var _initialized_free: bool = false


func _init() -> void:
	camera = Camera3D.new()
	camera.fov = 70.0
	camera.name = "SpectatorCamera3D"
	add_child(camera)


## Binds the rig to the currently rendered environment cell.
## `view_origin` is the cell's world offset, `agent_view` the node that
## mirrors AgentState (used for chase/first-person framing).
func bind_view(view_origin: Vector3, agent_view: Node3D, arena_half_extent: float) -> void:
	_view_origin = view_origin
	_agent_view = agent_view
	_arena_half_extent = arena_half_extent
	_initialized_free = false


## Switches camera mode and returns the camera that should now be current.
## FIRST_PERSON hands control back to the agent's own camera.
func set_camera_mode(value: int, agent_camera: Camera3D) -> Camera3D:
	mode = value
	if mode == ControlCenterConfig.CameraMode.FIRST_PERSON and agent_camera != null:
		camera.current = false
		agent_camera.current = true
		return agent_camera
	if agent_camera != null:
		agent_camera.current = false
	camera.current = true
	_apply_mode_defaults()
	return camera


func _apply_mode_defaults() -> void:
	match mode:
		ControlCenterConfig.CameraMode.TOP_DOWN:
			camera.global_rotation = Vector3(-PI * 0.5, 0.0, 0.0)
			camera.global_position = _view_origin + Vector3(
				0.0, _arena_half_extent * 2.1, 0.01
			)
		ControlCenterConfig.CameraMode.FREE:
			if not _initialized_free:
				_free_position = _view_origin + Vector3(
					0.0, _arena_half_extent * 0.9, _arena_half_extent * 1.4
				)
				_free_yaw = 0.0
				_free_pitch = -0.45
				_initialized_free = true
		_:
			pass


func _process(delta: float) -> void:
	match mode:
		ControlCenterConfig.CameraMode.THIRD_PERSON:
			_update_third_person()
		ControlCenterConfig.CameraMode.TOP_DOWN:
			_update_top_down()
		ControlCenterConfig.CameraMode.FREE:
			_update_free(delta)
		_:
			pass


func _update_third_person() -> void:
	if _agent_view == null or not is_instance_valid(_agent_view):
		return
	var agent_global: Vector3 = _agent_view.global_position
	var yaw: float = _agent_view.global_rotation.y
	var back := Vector3(sin(yaw), 0.0, cos(yaw)) * THIRD_PERSON_BACK
	camera.global_position = agent_global + back + Vector3(0.0, THIRD_PERSON_UP, 0.0)
	camera.look_at(agent_global + Vector3(0.0, 1.4, 0.0), Vector3.UP)


func _update_top_down() -> void:
	var centre: Vector3 = _view_origin
	if _agent_view != null and is_instance_valid(_agent_view):
		centre = Vector3(_agent_view.global_position.x, 0.0, _agent_view.global_position.z)
		centre = centre.lerp(_view_origin, 0.5)
	camera.global_position = centre + Vector3(0.0, _arena_half_extent * 2.1, 0.01)
	camera.global_rotation = Vector3(-PI * 0.5, 0.0, 0.0)


func _update_free(delta: float) -> void:
	if not _initialized_free:
		_apply_mode_defaults()
	var direction := Vector3.ZERO
	if input_enabled:
		if Input.is_key_pressed(KEY_W):
			direction.z -= 1.0
		if Input.is_key_pressed(KEY_S):
			direction.z += 1.0
		if Input.is_key_pressed(KEY_A):
			direction.x -= 1.0
		if Input.is_key_pressed(KEY_D):
			direction.x += 1.0
		if Input.is_key_pressed(KEY_E):
			direction.y += 1.0
		if Input.is_key_pressed(KEY_Q):
			direction.y -= 1.0
	if direction != Vector3.ZERO:
		var speed: float = (
			FREE_SPEED_FAST if Input.is_key_pressed(KEY_SHIFT) else FREE_SPEED
		)
		# `yaw_basis`, not `basis`: a local named `basis` would shadow the
		# Node3D property of the same name.
		var yaw_basis := Basis(Vector3.UP, _free_yaw)
		_free_position += yaw_basis * direction.normalized() * speed * delta
	camera.global_position = _free_position
	camera.global_rotation = Vector3(_free_pitch, _free_yaw, 0.0)


## Mouse look for FREE mode. The owner forwards unhandled input only when
## the Control Center is not giving the mouse to a human player.
func handle_input(event: InputEvent) -> void:
	if mode != ControlCenterConfig.CameraMode.FREE or not input_enabled:
		return
	var button := event as InputEventMouseButton
	if button != null and button.button_index == MOUSE_BUTTON_RIGHT:
		_looking = button.pressed
		return
	var motion := event as InputEventMouseMotion
	if motion != null and _looking:
		_free_yaw -= motion.relative.x * LOOK_SENSITIVITY * 0.01
		_free_pitch = clampf(
			_free_pitch - motion.relative.y * LOOK_SENSITIVITY * 0.01, -1.5, 1.5
		)

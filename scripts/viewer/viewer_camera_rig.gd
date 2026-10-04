## ViewerCameraRig
##
## Spectator cameras for the checkpoint viewer. Purely presentational: it
## reads the agent's position/facing and never touches simulation state.
##
## Modes (cycle with C, or pick with 1-4):
##   CHASE         behind and above the agent, smoothly following
##   ORBIT         free orbit around the agent (drag with the mouse, wheel zooms)
##   TOP_DOWN      the whole arena from above, ideal for reading tactics
##   FIRST_PERSON  the agent's own eye camera (same FOV as its perception cone)
class_name ViewerCameraRig
extends Node3D

enum Mode { CHASE, ORBIT, TOP_DOWN, FIRST_PERSON }

const MODE_NAMES: Array = ["Chase", "Orbit", "Top-down", "First person"]
const CHASE_DISTANCE: float = 6.5
const CHASE_HEIGHT: float = 3.4
const LOOK_HEIGHT: float = 1.2
const FOLLOW_SHARPNESS: float = 6.0
const ORBIT_MIN_DISTANCE: float = 3.0
const ORBIT_MAX_DISTANCE: float = 80.0
const ORBIT_MIN_PITCH: float = -85.0
const ORBIT_MAX_PITCH: float = -5.0
const MOUSE_DEGREES_PER_PIXEL: float = 0.3
const ZOOM_STEP: float = 1.12
const TOP_DOWN_MARGIN: float = 2.4

var mode: int = Mode.CHASE
var camera: Camera3D
var first_person_camera: Camera3D
var orbit_yaw_deg: float = 35.0
var orbit_pitch_deg: float = -35.0
var orbit_distance: float = 16.0
var top_down_zoom: float = 1.0
var _focus: Vector3 = Vector3.ZERO
var _dragging: bool = false


func _ready() -> void:
	camera = Camera3D.new()
	camera.name = "SpectatorCamera"
	camera.fov = 70.0
	camera.near = 0.05
	camera.far = 600.0
	add_child(camera)
	camera.current = true


static func mode_name(value: int) -> String:
	if value < 0 or value >= MODE_NAMES.size():
		return "?"
	return str(MODE_NAMES[value])


func set_first_person_camera(p_camera: Camera3D) -> void:
	first_person_camera = p_camera
	if mode == Mode.FIRST_PERSON:
		set_mode(mode)


func cycle_mode() -> void:
	set_mode((mode + 1) % MODE_NAMES.size())


func set_mode(value: int) -> void:
	mode = clampi(value, 0, MODE_NAMES.size() - 1)
	var use_first_person: bool = (
		mode == Mode.FIRST_PERSON
		and first_person_camera != null
		and is_instance_valid(first_person_camera)
	)
	if use_first_person:
		first_person_camera.current = true
	elif camera != null:
		camera.current = true


## Handles mouse orbit/zoom. Returns true when the event was consumed.
func handle_input(event: InputEvent) -> bool:
	if event is InputEventMouseButton:
		var button := event as InputEventMouseButton
		if button.button_index == MOUSE_BUTTON_WHEEL_UP and button.pressed:
			_zoom(1.0 / ZOOM_STEP)
			return true
		if button.button_index == MOUSE_BUTTON_WHEEL_DOWN and button.pressed:
			_zoom(ZOOM_STEP)
			return true
		if button.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT]:
			_dragging = button.pressed
			if _dragging and mode != Mode.ORBIT:
				set_mode(Mode.ORBIT)
			return true
	if event is InputEventMouseMotion and _dragging and mode == Mode.ORBIT:
		var motion := event as InputEventMouseMotion
		orbit_yaw_deg = wrapf(
			orbit_yaw_deg - motion.relative.x * MOUSE_DEGREES_PER_PIXEL, -180.0, 180.0
		)
		orbit_pitch_deg = clampf(
			orbit_pitch_deg - motion.relative.y * MOUSE_DEGREES_PER_PIXEL,
			ORBIT_MIN_PITCH,
			ORBIT_MAX_PITCH
		)
		return true
	return false


## Moves the spectator camera for this frame. `snap` skips the smoothing
## (used right after a reset so the camera does not fly across the map).
func update_view(
	agent_position: Vector3,
	agent_forward: Vector3,
	arena_half_extent: float,
	delta: float,
	snap: bool = false
) -> void:
	if camera == null:
		return
	var weight: float = 1.0 if snap else clampf(1.0 - exp(-FOLLOW_SHARPNESS * delta), 0.0, 1.0)
	_focus = _focus.lerp(agent_position, weight)
	var look_target: Vector3 = _focus + Vector3.UP * LOOK_HEIGHT
	match mode:
		Mode.CHASE:
			var forward: Vector3 = agent_forward
			forward.y = 0.0
			if forward.length() < 0.001:
				forward = Vector3(0.0, 0.0, -1.0)
			var desired: Vector3 = (
				_focus - forward.normalized() * CHASE_DISTANCE + Vector3.UP * CHASE_HEIGHT
			)
			camera.global_position = (
				desired if snap else camera.global_position.lerp(desired, weight)
			)
			_look_at_safely(look_target, Vector3.UP)
		Mode.ORBIT:
			var orbit_basis := Basis(Vector3.UP, deg_to_rad(orbit_yaw_deg))
			orbit_basis = orbit_basis * Basis(Vector3.RIGHT, deg_to_rad(orbit_pitch_deg))
			camera.global_position = look_target + orbit_basis * Vector3(0.0, 0.0, orbit_distance)
			_look_at_safely(look_target, Vector3.UP)
		Mode.TOP_DOWN:
			var height: float = maxf(12.0, arena_half_extent * TOP_DOWN_MARGIN) * top_down_zoom
			camera.global_position = Vector3(0.0, height, 0.0)
			_look_at_safely(Vector3.ZERO, Vector3(0.0, 0.0, -1.0))
		_:
			pass  # FIRST_PERSON renders through the agent's own camera


func _zoom(factor: float) -> void:
	if mode == Mode.TOP_DOWN:
		top_down_zoom = clampf(top_down_zoom * factor, 0.3, 3.0)
	else:
		orbit_distance = clampf(orbit_distance * factor, ORBIT_MIN_DISTANCE, ORBIT_MAX_DISTANCE)
		if mode == Mode.CHASE:
			set_mode(Mode.ORBIT)


func _look_at_safely(target: Vector3, up: Vector3) -> void:
	var direction: Vector3 = target - camera.global_position
	if direction.length() < 0.001:
		return
	if absf(direction.normalized().dot(up.normalized())) > 0.999:
		return
	camera.look_at(target, up)

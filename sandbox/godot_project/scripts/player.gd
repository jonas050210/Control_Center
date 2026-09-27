extends CharacterBody3D
class_name SandboxPlayer

signal shot_fired(hit_target: bool)

@export var move_speed: float = 6.0
@export var sprint_multiplier: float = 1.5
@export var mouse_sensitivity: float = 0.003
@export var jump_velocity: float = 4.5
@export var gravity: float = 9.8

@onready var camera_pivot: Node3D = $CameraPivot
@onready var camera: Camera3D = $CameraPivot/Camera3D
@onready var weapon_ray: RayCast3D = $CameraPivot/WeaponRay
@onready var obs_viewport: SubViewport = $ObsViewport
@onready var obs_camera: Camera3D = $ObsViewport/ObsCamera

var target_hit_count: int = 0
var last_shot_hit: bool = false

func _ready() -> void:
	obs_camera.global_transform = camera.global_transform

func _physics_process(delta: float) -> void:
	# Synchronize observation camera transform with primary camera
	obs_camera.global_transform = camera.global_transform
	if not is_on_floor():
		velocity.y -= gravity * delta

func apply_movement_input(move_dir_2d: Vector2, is_sprinting: bool, do_jump: bool) -> void:
	var speed = move_speed * (sprint_multiplier if is_sprinting else 1.0)
	var forward = -camera_pivot.global_transform.basis.z
	var right = camera_pivot.global_transform.basis.x
	forward.y = 0.0
	right.y = 0.0
	forward = forward.normalized()
	right = right.normalized()

	var direction = (right * move_dir_2d.x + forward * move_dir_2d.y).normalized()
	if direction.length_squared() > 0.01:
		velocity.x = direction.x * speed
		velocity.z = direction.z * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed)
		velocity.z = move_toward(velocity.z, 0, speed)

	if do_jump and is_on_floor():
		velocity.y = jump_velocity

	move_and_slide()

func apply_rotation_input(yaw_delta: float, pitch_delta: float) -> void:
	camera_pivot.rotate_y(-yaw_delta)
	camera.rotate_x(-pitch_delta)
	camera.rotation.x = clamp(camera.rotation.x, deg_to_rad(-80.0), deg_to_rad(80.0))

func shoot() -> bool:
	last_shot_hit = false
	if weapon_ray.is_colliding():
		var collider = weapon_ray.get_collider()
		if collider is TargetDummy:
			collider.take_damage(50.0)
			last_shot_hit = true
			target_hit_count += 1
		elif collider.has_method("take_damage"):
			collider.take_damage(50.0)
			last_shot_hit = true
			target_hit_count += 1
	emit_signal("shot_fired", last_shot_hit)
	return last_shot_hit

func reset_player(spawn_pos: Vector3 = Vector3(0, 1.0, 0)) -> void:
	global_position = spawn_pos
	velocity = Vector3.ZERO
	camera_pivot.rotation = Vector3.ZERO
	camera.rotation = Vector3.ZERO
	last_shot_hit = false

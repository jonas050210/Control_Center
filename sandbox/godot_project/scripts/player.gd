extends CharacterBody3D
class_name SandboxPlayer

const WEAPON_CONFIG: Script = preload("res://scripts/weapon_config.gd")

signal shot_fired(hit_target: bool, killed_target: bool)
signal ammo_changed(magazine: int, reserve: int)
signal health_changed(health: float)

@export var move_speed: float = 6.0
@export var sprint_multiplier: float = 1.5
@export var mouse_sensitivity: float = 0.003
@export var jump_velocity: float = 4.5
@export var gravity: float = 9.8
@export var magazine_size: int = 20
@export var starting_reserve: int = 60
@export var reload_seconds: float = 1.2
@export var fire_interval: float = 0.15

@onready var camera_pivot: Node3D = $CameraPivot
@onready var camera: Camera3D = $CameraPivot/Camera3D
@onready var weapon_ray: RayCast3D = $CameraPivot/WeaponRay
@onready var obs_viewport: SubViewport = $ObsViewport
@onready var obs_camera: Camera3D = $ObsViewport/ObsCamera
@onready var hud_status: Label = $HUD/Status
@onready var hud_help: Label = $HUD/Help

var agent_controlled: bool = false
var target_hit_count: int = 0
var kill_count: int = 0
var last_shot_hit: bool = false
var health: float = 100.0
var ammo: int = 20
var reserve_ammo: int = 60
var reload_timer: float = 0.0
var fire_timer: float = 0.0
var is_ads: bool = false
var is_crouching: bool = false
var weapon_catalog: Dictionary = {}
var current_weapon_id: String = "rifle"
var current_weapon: SandboxWeapon
var muzzle_flash_timer: float = 0.0

func _ready() -> void:
	add_to_group("players")
	ammo = magazine_size
	reserve_ammo = starting_reserve
	obs_viewport.world_3d = get_viewport().world_3d
	obs_camera.global_transform = camera.global_transform
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	weapon_catalog = SandboxWeapon.catalog()
	equip_weapon(current_weapon_id)

func _process(_delta: float) -> void:
	if hud_status:
		var reload_text := "  RELOADING" if reload_timer > 0.0 else ""
		hud_status.text = "HEALTH %03d    AMMO %02d / %02d%s    HITS %d  KILLS %d" % [int(health), ammo, reserve_ammo, reload_text, target_hit_count, kill_count]
	if hud_help:
		hud_help.visible = not agent_controlled

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	if agent_controlled:
		return
	if event is InputEventMouseButton and event.pressed and Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
		return
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		apply_rotation_input(event.relative.x * mouse_sensitivity, event.relative.y * mouse_sensitivity)
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		shoot()

func _physics_process(delta: float) -> void:
	obs_camera.global_transform = camera.global_transform
	if agent_controlled:
		return
	advance_simulation_timers(delta)
	if health <= 0.0:
		velocity = Vector3.ZERO
		if Input.is_physical_key_pressed(KEY_ENTER):
			reset_player()
		return
	var move_input := Vector2(
		float(Input.is_physical_key_pressed(KEY_D)) - float(Input.is_physical_key_pressed(KEY_A)),
		float(Input.is_physical_key_pressed(KEY_W)) - float(Input.is_physical_key_pressed(KEY_S))
	)
	is_ads = Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT)
	is_crouching = Input.is_physical_key_pressed(KEY_CTRL) or Input.is_physical_key_pressed(KEY_C)
	camera.fov = lerpf(camera.fov, 55.0 if is_ads else 75.0, minf(1.0, delta * 14.0))
	apply_movement_input(
		move_input,
		Input.is_physical_key_pressed(KEY_SHIFT) and not is_ads,
		Input.is_physical_key_pressed(KEY_SPACE),
		is_crouching
	)
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		shoot()
	if Input.is_physical_key_pressed(KEY_R):
		start_reload()

func set_agent_controlled(enabled: bool) -> void:
	agent_controlled = enabled
	if enabled:
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)

func advance_simulation_timers(delta: float) -> void:
	fire_timer = maxf(0.0, fire_timer - delta)
	if reload_timer > 0.0:
		reload_timer = maxf(0.0, reload_timer - delta)
		if reload_timer == 0.0:
			_finish_reload()
	if not is_on_floor():
		velocity.y -= gravity * delta

func apply_movement_input(move_dir_2d: Vector2, sprinting: bool, do_jump: bool, crouching: bool = false) -> void:
	is_crouching = crouching
	var speed := move_speed
	if crouching:
		speed *= 0.55
	elif sprinting:
		speed *= sprint_multiplier
	var forward := -camera_pivot.global_transform.basis.z
	var right := camera_pivot.global_transform.basis.x
	forward.y = 0.0
	right.y = 0.0
	forward = forward.normalized()
	right = right.normalized()
	var direction := (right * move_dir_2d.x + forward * move_dir_2d.y).normalized()
	if direction.length_squared() > 0.01:
		velocity.x = direction.x * speed
		velocity.z = direction.z * speed
	else:
		velocity.x = move_toward(velocity.x, 0.0, speed)
		velocity.z = move_toward(velocity.z, 0.0, speed)
	if do_jump and is_on_floor() and not crouching:
		velocity.y = jump_velocity
	move_and_slide()

func apply_rotation_input(yaw_delta: float, pitch_delta: float) -> void:
	camera_pivot.rotate_y(-yaw_delta)
	camera.rotate_x(-pitch_delta)
	camera.rotation.x = clampf(camera.rotation.x, deg_to_rad(-80.0), deg_to_rad(80.0))

func set_ads(enabled: bool) -> void:
	is_ads = enabled
	camera.fov = 55.0 if enabled else 75.0
	obs_camera.fov = camera.fov

func equip_weapon(weapon_id: String) -> bool:
	if not weapon_catalog.has(weapon_id): return false
	current_weapon_id = weapon_id
	current_weapon = weapon_catalog[weapon_id] as SandboxWeapon
	magazine_size = current_weapon.magazine_size
	starting_reserve = current_weapon.reserve_ammo
	reload_seconds = current_weapon.reload_seconds
	fire_interval = current_weapon.fire_interval
	ammo = magazine_size; reserve_ammo = starting_reserve
	return true

func switch_weapon(weapon_id: String) -> bool:
	if reload_timer > 0.0: return false
	return equip_weapon(weapon_id)

func start_reload() -> bool:
	if reload_timer > 0.0 or ammo >= magazine_size or reserve_ammo <= 0:
		return false
	reload_timer = reload_seconds
	return true

func _finish_reload() -> void:
	var load_count := mini(magazine_size - ammo, reserve_ammo)
	ammo += load_count
	reserve_ammo -= load_count
	ammo_changed.emit(ammo, reserve_ammo)

func shoot() -> bool:
	if fire_timer > 0.0 or reload_timer > 0.0:
		return false
	if ammo <= 0:
		start_reload()
		return false
	fire_timer = fire_interval
	ammo -= 1
	ammo_changed.emit(ammo, reserve_ammo)
	last_shot_hit = false
	var killed := false
	weapon_ray.force_raycast_update()
	if weapon_ray.is_colliding():
		var collider := weapon_ray.get_collider()
		if collider != null and collider.has_method("take_damage"):
			killed = bool(collider.take_damage(50.0 if is_ads else 40.0))
			last_shot_hit = true
			target_hit_count += 1
			if killed:
				kill_count += 1
	# Small deterministic recoil; AI learns to correct through pitch actions.
	camera.rotate_x(deg_to_rad(0.4 if is_ads else 0.9))
	shot_fired.emit(last_shot_hit, killed)
	return last_shot_hit

func take_damage(amount: float) -> void:
	health = maxf(0.0, health - amount)
	health_changed.emit(health)

func reset_player(spawn_pos: Vector3 = Vector3.ZERO) -> void:
	global_position = spawn_pos
	velocity = Vector3.ZERO
	camera_pivot.rotation = Vector3.ZERO
	camera.rotation = Vector3.ZERO
	last_shot_hit = false
	target_hit_count = 0
	kill_count = 0
	health = 100.0
	ammo = magazine_size
	reserve_ammo = starting_reserve
	reload_timer = 0.0
	fire_timer = 0.0
	ammo_changed.emit(ammo, reserve_ammo)
	health_changed.emit(health)

extends Area3D
class_name TargetDummy

const MODEL_FACTORY: Script = preload("res://scripts/model_factory.gd")

signal target_hit(dummy: TargetDummy, killed: bool)

@export var max_health: float = 100.0
@export var respawn_radius: float = 15.0
@export var patrol_radius: float = 1.2
@export var enemy_enabled: bool = true
@export var enemy_damage: float = 7.0
var health: float = 100.0
var rng := RandomNumberGenerator.new()
var spawn_origin := Vector3.ZERO
var patrol_phase: float = 0.0
var patrol_enabled: bool = true
var agent_controlled: bool = false
var fire_timer: float = 1.0
var is_moving: bool = false
var player: SandboxPlayer

func _ready() -> void:
	health = max_health
	spawn_origin = global_position
	add_to_group("targets")
	player = get_tree().get_first_node_in_group("players") as SandboxPlayer
	_build_visual()

func _build_visual() -> void:
	var authored_mesh: Node = get_node_or_null("MeshInstance3D")
	if authored_mesh != null:
		authored_mesh.free()
	var body: StandardMaterial3D = MODEL_FACTORY.material(Color(0.24, 0.28, 0.31), 0.35, 0.55)
	var plate: StandardMaterial3D = MODEL_FACTORY.material(Color(0.68, 0.18, 0.10), 0.25, 0.42)
	var sensor: StandardMaterial3D = MODEL_FACTORY.material(Color(0.08, 0.70, 0.80), 0.32, 0.22, Color(0.01, 0.20, 0.28))
	MODEL_FACTORY.add_capsule(self, "Body", 0.34, 1.32, Vector3(0, 0.92, 0), body)
	MODEL_FACTORY.add_sphere(self, "Head", 0.24, Vector3(0, 1.72, 0), body)
	MODEL_FACTORY.add_box(self, "TargetPlate", Vector3(0.44, 0.46, 0.10), Vector3(0, 1.02, -0.28), plate)
	MODEL_FACTORY.add_box(self, "TargetSensor", Vector3(0.16, 0.12, 0.025), Vector3(0, 1.04, -0.345), sensor)
	MODEL_FACTORY.add_box(self, "ArmLeft", Vector3(0.14, 0.50, 0.14), Vector3(-0.40, 0.86, 0), body)
	MODEL_FACTORY.add_box(self, "ArmRight", Vector3(0.14, 0.50, 0.14), Vector3(0.40, 0.86, 0), body)
	MODEL_FACTORY.add_box(self, "Base", Vector3(0.55, 0.16, 0.42), Vector3(0, 0.17, 0), body)

func _physics_process(delta: float) -> void:
	if agent_controlled:
		return
	advance_simulation(delta)

func advance_simulation(delta: float) -> void:
	if patrol_enabled:
		is_moving = true
		patrol_phase += delta * 0.65
		global_position.x = spawn_origin.x + sin(patrol_phase) * patrol_radius
		global_position.z = spawn_origin.z + cos(patrol_phase * 0.83) * patrol_radius
	else:
		is_moving = false
	if not enemy_enabled or player == null or player.health <= 0.0:
		return
	fire_timer -= delta
	if fire_timer > 0.0 or global_position.distance_to(player.global_position) > 18.0:
		return
	fire_timer = rng.randf_range(0.8, 1.5)
	var query := PhysicsRayQueryParameters3D.create(global_position, player.global_position + Vector3.UP)
	query.collide_with_areas = false
	query.collide_with_bodies = true
	var result := get_world_3d().direct_space_state.intersect_ray(query)
	if result.get("collider") == player and rng.randf() < 0.35:
		player.take_damage(enemy_damage)

func configure_seed(value: int) -> void:
	rng.seed = value
	patrol_phase = rng.randf_range(0.0, TAU)
	fire_timer = rng.randf_range(0.8, 1.5)

func take_damage(amount: float) -> bool:
	health -= amount
	var killed := health <= 0.0
	target_hit.emit(self, killed)
	if killed:
		respawn()
	return killed

func respawn() -> void:
	health = max_health
	var angle := rng.randf_range(0.0, TAU)
	var distance := rng.randf_range(6.0, respawn_radius)
	spawn_origin = Vector3(cos(angle) * distance, 0.9, sin(angle) * distance)
	global_position = spawn_origin

extends Area3D
class_name TargetDummy

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
var player: SandboxPlayer

func _ready() -> void:
	health = max_health
	spawn_origin = global_position
	add_to_group("targets")
	player = get_tree().get_first_node_in_group("players") as SandboxPlayer

func _physics_process(delta: float) -> void:
	if agent_controlled:
		return
	advance_simulation(delta)

func advance_simulation(delta: float) -> void:
	if patrol_enabled:
		patrol_phase += delta * 0.65
		global_position.x = spawn_origin.x + sin(patrol_phase) * patrol_radius
		global_position.z = spawn_origin.z + cos(patrol_phase * 0.83) * patrol_radius
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

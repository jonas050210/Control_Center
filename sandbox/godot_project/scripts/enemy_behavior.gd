extends Area3D
class_name SandboxEnemy

signal target_hit(enemy: SandboxEnemy, killed: bool)
@export_enum("stationary", "patrol", "aggressive", "cover") var behavior: String = "stationary"
@export var max_health: float = 100.0
@export var enemy_damage: float = 7.0
@export var patrol_radius: float = 3.0
@export var move_speed: float = 2.2
var health: float = 100.0
var spawn_origin: Vector3
var patrol_phase: float = 0.0
var fire_timer: float = 1.0
var agent_controlled: bool = false
var is_moving: bool = false
var player: SandboxPlayer
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

func _ready() -> void:
	health = max_health; spawn_origin = global_position; add_to_group("targets")
	player = get_tree().get_first_node_in_group("players") as SandboxPlayer

func configure_seed(value: int) -> void:
	rng.seed = value; patrol_phase = rng.randf_range(0.0, TAU); fire_timer = rng.randf_range(0.7, 1.5)

func _physics_process(delta: float) -> void:
	if agent_controlled: return
	advance_simulation(delta)

func advance_simulation(delta: float) -> void:
	if player == null: player = get_tree().get_first_node_in_group("players") as SandboxPlayer
	is_moving = false
	match behavior:
		"patrol":
			patrol_phase += delta * 0.6; global_position = spawn_origin + Vector3(sin(patrol_phase) * patrol_radius, 0.0, cos(patrol_phase * 0.8) * patrol_radius); is_moving = true
		"aggressive":
			if player and player.health > 0.0:
				var direction: Vector3 = player.global_position - global_position; direction.y = 0.0
				if direction.length() > 7.0: global_position += direction.normalized() * move_speed * delta; is_moving = true
		"cover":
			if player and player.health > 0.0 and global_position.distance_to(player.global_position) < 10.0:
				var away: Vector3 = global_position - player.global_position; away.y = 0.0
				if away.length() > 0.1: global_position += away.normalized() * move_speed * delta; is_moving = true
	if player == null or not is_instance_valid(player) or player.health <= 0.0: return
	fire_timer -= delta
	if fire_timer > 0.0 or global_position.distance_to(player.global_position) > 22.0: return
	fire_timer = rng.randf_range(0.8, 1.6)
	var query: PhysicsRayQueryParameters3D = PhysicsRayQueryParameters3D.create(global_position + Vector3.UP, player.global_position + Vector3.UP)
	query.collide_with_areas = false; query.collide_with_bodies = true
	var result: Dictionary = get_world_3d().direct_space_state.intersect_ray(query)
	if result.get("collider") == player and rng.randf() < 0.35: player.take_damage(enemy_damage)

func take_damage(amount: float) -> bool:
	health -= amount; var killed: bool = health <= 0.0; target_hit.emit(self, killed)
	if killed: respawn()
	return killed

func respawn() -> void:
	health = max_health; global_position = spawn_origin; is_moving = false

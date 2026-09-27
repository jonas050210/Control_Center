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
var reaction_timer: float = 0.0
var aim_quality: float = 0.7
var target_visible: bool = false
var navigation_agent: NavigationAgent3D
var last_target_position: Vector3 = Vector3.ZERO
var cover_anchor: Vector3 = Vector3.ZERO
var decision_timer: float = 0.0
var target_lost_seconds: float = 0.0

func _ready() -> void:
	health = max_health; spawn_origin = global_position; add_to_group("targets")
	player = get_tree().get_first_node_in_group("players") as SandboxPlayer
	navigation_agent = NavigationAgent3D.new()
	navigation_agent.path_height_offset = 0.5
	navigation_agent.path_desired_distance = 0.7
	navigation_agent.target_desired_distance = 1.8
	add_child(navigation_agent)

func configure_seed(value: int) -> void:
	rng.seed = value; patrol_phase = rng.randf_range(0.0, TAU); fire_timer = rng.randf_range(0.7, 1.5); reaction_timer = rng.randf_range(0.15, 0.65); aim_quality = rng.randf_range(0.55, 0.92)

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
				if direction.length() > 7.0: _navigate_to(player.global_position, delta)
		"cover":
			if player and player.health > 0.0 and global_position.distance_to(player.global_position) < 10.0:
				if cover_anchor == Vector3.ZERO: cover_anchor = _find_cover_anchor()
				_navigate_to(cover_anchor if cover_anchor != Vector3.ZERO else global_position + (global_position - player.global_position).normalized() * 4.0, delta)
	if player == null or not is_instance_valid(player) or player.health <= 0.0: return
	var query: PhysicsRayQueryParameters3D = PhysicsRayQueryParameters3D.create(global_position + Vector3.UP, player.global_position + Vector3.UP)
	query.collide_with_areas = false; query.collide_with_bodies = true
	var visibility: Dictionary = get_world_3d().direct_space_state.intersect_ray(query)
	target_visible = visibility.get("collider") == player
	if not target_visible: reaction_timer = minf(0.65, reaction_timer + delta); return
	reaction_timer = maxf(0.0, reaction_timer - delta)
	fire_timer -= delta
	if reaction_timer > 0.0 or fire_timer > 0.0 or global_position.distance_to(player.global_position) > 22.0: return
	fire_timer = rng.randf_range(0.8, 1.6)
	if target_visible and rng.randf() < aim_quality: player.take_damage(enemy_damage)

func _navigate_to(destination: Vector3, delta: float) -> void:
	if navigation_agent == null: return
	navigation_agent.target_position = destination
	var next_point: Vector3 = navigation_agent.get_next_path_position()
	if next_point == Vector3.ZERO: next_point = destination
	var direction: Vector3 = next_point - global_position; direction.y = 0.0
	if direction.length_squared() > 0.01:
		global_position += direction.normalized() * move_speed * delta
		look_at(global_position + direction, Vector3.UP)
		is_moving = true

func _find_cover_anchor() -> Vector3:
	var away: Vector3 = (global_position - player.global_position).normalized()
	var candidate: Vector3 = global_position + Vector3(away.x, 0.0, away.z) * 4.0
	var query: PhysicsRayQueryParameters3D = PhysicsRayQueryParameters3D.create(candidate + Vector3.UP * 2.0, candidate - Vector3.UP * 2.0)
	query.collide_with_bodies = true
	var hit: Dictionary = get_world_3d().direct_space_state.intersect_ray(query)
	return candidate if not hit.is_empty() else global_position

func take_damage(amount: float) -> bool:
	health -= amount; var killed: bool = health <= 0.0; target_hit.emit(self, killed)
	if killed: respawn()
	return killed

func respawn() -> void:
	health = max_health; global_position = spawn_origin; is_moving = false

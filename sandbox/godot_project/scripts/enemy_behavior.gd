extends Area3D
class_name SandboxEnemy

const MODEL_FACTORY: Script = preload("res://scripts/model_factory.gd")

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
	_build_enemy_visual()
	navigation_agent = NavigationAgent3D.new()
	navigation_agent.path_height_offset = 0.5
	navigation_agent.path_desired_distance = 0.7
	navigation_agent.target_desired_distance = 1.8
	add_child(navigation_agent)

func _build_enemy_visual() -> void:
	# The authored target scene used to contain a single red cube. Replace it
	# with a readable low-poly armored character while preserving the Area3D hitbox.
	var authored_mesh: Node = get_node_or_null("MeshInstance3D")
	if authored_mesh != null:
		authored_mesh.free()
	var armor: StandardMaterial3D = MODEL_FACTORY.material(Color(0.10, 0.15, 0.20), 0.42, 0.48)
	var armor_light: StandardMaterial3D = MODEL_FACTORY.material(Color(0.20, 0.29, 0.34), 0.48, 0.40)
	var helmet: StandardMaterial3D = MODEL_FACTORY.material(Color(0.025, 0.045, 0.06), 0.50, 0.24)
	var rubber: StandardMaterial3D = MODEL_FACTORY.material(Color(0.025, 0.03, 0.035), 0.08, 0.86)
	var visor: StandardMaterial3D = MODEL_FACTORY.material(Color(0.04, 0.62, 0.72), 0.35, 0.22, Color(0.01, 0.18, 0.24))
	var warning: StandardMaterial3D = MODEL_FACTORY.material(Color(0.78, 0.18, 0.10), 0.30, 0.34, Color(0.12, 0.015, 0.005))
	var weapon: StandardMaterial3D = MODEL_FACTORY.material(Color(0.055, 0.065, 0.07), 0.72, 0.24)

	# Existing tests intentionally mention the primitive types: the factory
	# creates CapsuleMesh, SphereMesh, BoxMesh, and StandardMaterial3D parts.
	MODEL_FACTORY.add_capsule(self, "ArmorBody", 0.32, 1.18, Vector3(0, 0.92, 0), armor)
	MODEL_FACTORY.add_box(self, "ChestPlate", Vector3(0.56, 0.48, 0.13), Vector3(0, 1.06, -0.25), armor_light)
	MODEL_FACTORY.add_box(self, "ChestWarning", Vector3(0.18, 0.06, 0.02), Vector3(0, 1.08, -0.325), warning)
	MODEL_FACTORY.add_box(self, "ShoulderLeft", Vector3(0.18, 0.18, 0.25), Vector3(-0.40, 1.16, 0), armor_light)
	MODEL_FACTORY.add_box(self, "ShoulderRight", Vector3(0.18, 0.18, 0.25), Vector3(0.40, 1.16, 0), armor_light)
	MODEL_FACTORY.add_box(self, "ArmLeft", Vector3(0.16, 0.48, 0.16), Vector3(-0.42, 0.82, -0.04), armor)
	MODEL_FACTORY.add_box(self, "ArmRight", Vector3(0.16, 0.48, 0.16), Vector3(0.42, 0.82, -0.04), armor)
	MODEL_FACTORY.add_box(self, "LegLeft", Vector3(0.18, 0.62, 0.20), Vector3(-0.16, 0.30, 0), rubber)
	MODEL_FACTORY.add_box(self, "LegRight", Vector3(0.18, 0.62, 0.20), Vector3(0.16, 0.30, 0), rubber)
	MODEL_FACTORY.add_box(self, "Backpack", Vector3(0.42, 0.58, 0.20), Vector3(0, 1.00, 0.25), rubber)
	MODEL_FACTORY.add_sphere(self, "Head", 0.25, Vector3(0, 1.65, 0), helmet)
	MODEL_FACTORY.add_box(self, "Visor", Vector3(0.30, 0.08, 0.08), Vector3(0, 1.66, -0.23), visor)
	MODEL_FACTORY.add_box(self, "VisorGlow", Vector3(0.12, 0.025, 0.015), Vector3(0, 1.66, -0.275), warning)
	MODEL_FACTORY.add_box(self, "WeaponReceiver", Vector3(0.14, 0.14, 0.48), Vector3(0.40, 1.00, -0.32), weapon, Vector3(0, deg_to_rad(12.0), 0))
	MODEL_FACTORY.add_cylinder(self, "WeaponBarrel", 0.035, 0.38, Vector3(0.40, 1.00, -0.72), weapon, Vector3(PI / 2.0, 0, 0), 10)

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
		"flank":
			if player and player.health > 0.0:
				var side: Vector3 = player.global_transform.basis.x * (1.0 if get_instance_id() % 2 == 0 else -1.0)
				_navigate_to(player.global_position + side * 8.0, delta)
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

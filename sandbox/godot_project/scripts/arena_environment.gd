extends Node3D
class_name SandboxArenaEnvironment

## Original modular layouts. They use the same gameplay principles as public
## tactical test maps without reproducing their geometry or assets.
@export_enum("facility", "research_complex", "compound") var map_id: String = "facility"
@export_enum("day", "evening", "night") var time_of_day: String = "day"
@export var randomize_episode: bool = true
var episode_seed: int = 42
var rng: RandomNumberGenerator = RandomNumberGenerator.new()
var sun: DirectionalLight3D
var environment: WorldEnvironment

func _ready() -> void:
	sun = get_node_or_null("DirectionalLight3D") as DirectionalLight3D
	environment = get_node_or_null("WorldEnvironment") as WorldEnvironment
	apply_time_of_day(time_of_day)
	_build_original_layout()
	_build_navigation_surface()

func _build_original_layout() -> void:
	for old: Node in get_tree().get_nodes_in_group("generated_map_geometry"):
		old.queue_free()
	# Lightweight modular cover pieces create distinct sightline rhythms per map.
	var layouts: Dictionary = {
		"facility": [Vector3(-6, 1.2, -4), Vector3(6, 1.2, -4), Vector3(-2, 1.2, 5), Vector3(3, 2.8, 0)],
		"research_complex": [Vector3(-8, 1.2, -8), Vector3(8, 1.2, -8), Vector3(-5, 3.5, 2), Vector3(5, 3.5, 6), Vector3(0, 1.2, 10)],
		"compound": [Vector3(-10, 1.2, -5), Vector3(0, 1.2, -10), Vector3(10, 1.2, -2), Vector3(-5, 1.2, 8), Vector3(7, 2.8, 8)]
	}
	if map_id == "research_complex":
		for step: int in range(5): _add_box(Vector3(-10.0 + step * 0.8, step * 0.35, 4.0), Vector3(2.0, 0.7, 4.0), Color(0.24, 0.30, 0.35))
	if map_id == "compound":
		_add_box(Vector3(0, 2.5, -12), Vector3(8, 5, 1), Color(0.30, 0.22, 0.16))
	for position: Vector3 in layouts.get(map_id, layouts["facility"]):
		var body: StaticBody3D = StaticBody3D.new()
		body.position = position
		body.add_to_group("generated_map_geometry")
		add_child(body)
		var cover: MeshInstance3D = MeshInstance3D.new()
		var mesh: BoxMesh = BoxMesh.new()
		var cover_size: Vector3 = Vector3(3.2, 2.4 if position.y < 2.0 else 4.5, 1.4)
		mesh.size = cover_size
		var collision: CollisionShape3D = CollisionShape3D.new()
		var shape: BoxShape3D = BoxShape3D.new()
		shape.size = cover_size
		collision.shape = shape
		body.add_child(collision)
		var material: StandardMaterial3D = StandardMaterial3D.new()
		material.albedo_color = Color(0.20, 0.28, 0.32) if map_id != "compound" else Color(0.34, 0.25, 0.18)
		material.roughness = 0.82; mesh.material = material; cover.mesh = mesh; body.add_child(cover)

func _build_navigation_surface() -> void:
	var region: NavigationRegion3D = NavigationRegion3D.new()
	var nav: NavigationMesh = NavigationMesh.new()
	nav.vertices = PackedVector3Array([Vector3(-18, 0.02, -18), Vector3(18, 0.02, -18), Vector3(18, 0.02, 18), Vector3(-18, 0.02, 18)])
	nav.add_polygon(PackedInt32Array([0, 1, 2, 3]))
	region.navigation_mesh = nav
	region.add_to_group("generated_map_geometry")
	add_child(region)

func _add_box(position: Vector3, size: Vector3, color: Color) -> void:
	var body: StaticBody3D = StaticBody3D.new(); body.position = position; body.add_to_group("generated_map_geometry"); add_child(body)
	var mesh_instance: MeshInstance3D = MeshInstance3D.new(); var mesh: BoxMesh = BoxMesh.new(); mesh.size = size
	var material: StandardMaterial3D = StandardMaterial3D.new(); material.albedo_color = color; material.roughness = 0.8; mesh.material = material; mesh_instance.mesh = mesh; body.add_child(mesh_instance)
	var collider: CollisionShape3D = CollisionShape3D.new(); var shape: BoxShape3D = BoxShape3D.new(); shape.size = size; collider.shape = shape; body.add_child(collider)

func is_valid_spawn(position: Vector3, radius: float = 0.45) -> bool:
	var query: PhysicsShapeQueryParameters3D = PhysicsShapeQueryParameters3D.new()
	var shape: CapsuleShape3D = CapsuleShape3D.new(); shape.radius = radius; shape.height = 1.8; query.shape = shape
	query.transform = Transform3D(Basis.IDENTITY, position + Vector3.UP * 0.9)
	return get_world_3d().direct_space_state.intersect_shape(query, 1).is_empty()

func configure(seed_value: int, requested_map: String = "") -> void:
	episode_seed = seed_value; rng.seed = seed_value
	if requested_map != "": map_id = requested_map
	elif randomize_episode: map_id = ["facility", "research_complex", "compound"][rng.randi_range(0, 2)]
	apply_time_of_day(["day", "evening", "night"][rng.randi_range(0, 2)] if randomize_episode else time_of_day)
	_build_original_layout()
	_build_navigation_surface()

func apply_time_of_day(value: String) -> void:
	time_of_day = value
	if sun == null: return
	match value:
		"day":
			sun.light_energy = 1.15; sun.light_color = Color(1.0, 0.95, 0.85); sun.rotation_degrees = Vector3(-48, -25, 0)
		"evening":
			sun.light_energy = 0.55; sun.light_color = Color(1.0, 0.55, 0.35); sun.rotation_degrees = Vector3(-18, 30, 0)
		"night":
			sun.light_energy = 0.10; sun.light_color = Color(0.30, 0.40, 0.75); sun.rotation_degrees = Vector3(-35, 80, 0)
	if environment and environment.environment:
		environment.environment.ambient_light_energy = {"day": 0.75, "evening": 0.38, "night": 0.12}.get(value, 0.75)
		environment.environment.ambient_light_color = {"day": Color(0.75, 0.8, 0.9), "evening": Color(0.55, 0.38, 0.32), "night": Color(0.16, 0.20, 0.38)}.get(value, Color.WHITE)

extends Node3D
class_name SandboxArenaEnvironment

const MODEL_FACTORY: Script = preload("res://scripts/model_factory.gd")

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
	_decorate_authored_geometry()
	_build_original_layout()
	_build_navigation_surface()

func _decorate_authored_geometry() -> void:
	# Add trim and inset panels to the authored floor, perimeter, and cover
	# bodies. Their collision shapes stay untouched; these are visual-only parts.
	var floor_body: Node = get_node_or_null("Floor")
	if floor_body != null:
		var floor_surface: StandardMaterial3D = MODEL_FACTORY.material(Color(0.12, 0.17, 0.20), 0.12, 0.86)
		MODEL_FACTORY.add_box(floor_body, "FloorInset", Vector3(38.0, 0.035, 38.0), Vector3(0, 0.52, 0), floor_surface)
		for index: int in range(-3, 4):
			var strip_material: StandardMaterial3D = MODEL_FACTORY.material(Color(0.16, 0.28, 0.30), 0.20, 0.58)
			MODEL_FACTORY.add_box(floor_body, "FloorStrip" + str(index), Vector3(0.025, 0.012, 38.0), Vector3(index * 5.0, 0.545, 0), strip_material)
	for wall_name: String in ["WallNorth", "WallSouth", "WallEast", "WallWest"]:
		var wall: Node = get_node_or_null(wall_name)
		if wall == null: continue
		var trim: StandardMaterial3D = MODEL_FACTORY.material(Color(0.20, 0.48, 0.52), 0.32, 0.38, Color(0.01, 0.06, 0.07))
		MODEL_FACTORY.add_box(wall, "WallCap", Vector3(40.2, 0.10, 1.10), Vector3(0, 3.06, 0), trim)
		MODEL_FACTORY.add_box(wall, "WallBand", Vector3(36.0, 0.16, 0.035), Vector3(0, 1.15, -0.53), trim)
	for cover_name: String in ["CoverCenter", "CoverWest", "CoverEast", "CoverSouth"]:
		var cover: Node = get_node_or_null(cover_name)
		if cover == null: continue
		var cover_trim: StandardMaterial3D = MODEL_FACTORY.material(Color(0.36, 0.55, 0.55), 0.26, 0.48)
		var cover_dark: StandardMaterial3D = MODEL_FACTORY.material(Color(0.08, 0.14, 0.16), 0.08, 0.88)
		MODEL_FACTORY.add_box(cover, "CoverCap", Vector3(4.14, 0.08, 1.64), Vector3(0, 1.34, 0), cover_trim)
		MODEL_FACTORY.add_box(cover, "CoverPanel", Vector3(2.8, 1.0, 0.035), Vector3(0, 0.15, -0.77), cover_dark)

func _build_original_layout() -> void:
	# Reset is driven by the bridge between commands, outside physics
	# callbacks. Free generated geometry immediately so old cover/navigation
	# nodes cannot overlap the newly configured map for one frame.
	for old: Node in get_tree().get_nodes_in_group("generated_map_geometry"):
		if is_instance_valid(old): old.free()
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
		var cover_size: Vector3 = Vector3(3.2, 2.4 if position.y < 2.0 else 4.5, 1.4)
		var cover_color: Color = Color(0.20, 0.28, 0.32) if map_id != "compound" else Color(0.34, 0.25, 0.18)
		_add_box(position, cover_size, cover_color)

func _build_navigation_surface() -> void:
	var region: NavigationRegion3D = NavigationRegion3D.new()
	var nav: NavigationMesh = NavigationMesh.new()
	nav.vertices = PackedVector3Array([Vector3(-18, 0.02, -18), Vector3(18, 0.02, -18), Vector3(18, 0.02, 18), Vector3(-18, 0.02, 18)])
	nav.add_polygon(PackedInt32Array([0, 1, 2, 3]))
	region.navigation_mesh = nav
	region.add_to_group("generated_map_geometry")
	add_child(region)

func _add_box(position: Vector3, size: Vector3, color: Color) -> void:
	var body: StaticBody3D = StaticBody3D.new()
	body.position = position
	body.add_to_group("generated_map_geometry")
	add_child(body)
	var surface: StandardMaterial3D = MODEL_FACTORY.material(color, 0.18, 0.78)
	var trim: StandardMaterial3D = MODEL_FACTORY.material(color.lightened(0.18), 0.24, 0.52)
	var shadow: StandardMaterial3D = MODEL_FACTORY.material(color.darkened(0.24), 0.10, 0.92)
	MODEL_FACTORY.add_box(body, "MainVolume", size, Vector3.ZERO, surface)
	var top_y: float = size.y * 0.5 + 0.035
	MODEL_FACTORY.add_box(body, "TopTrim", Vector3(size.x + 0.10, 0.07, size.z + 0.10), Vector3(0, top_y, 0), trim)
	MODEL_FACTORY.add_box(body, "FrontPanel", Vector3(size.x * 0.68, size.y * 0.42, 0.035), Vector3(0, 0, -size.z * 0.51), shadow)
	if size.x > 1.0:
		MODEL_FACTORY.add_box(body, "SupportLeft", Vector3(0.10, size.y * 0.82, 0.10), Vector3(-size.x * 0.42, 0, -size.z * 0.53), trim)
		MODEL_FACTORY.add_box(body, "SupportRight", Vector3(0.10, size.y * 0.82, 0.10), Vector3(size.x * 0.42, 0, -size.z * 0.53), trim)
	var collider: CollisionShape3D = CollisionShape3D.new()
	var shape: BoxShape3D = BoxShape3D.new()
	shape.size = size
	collider.shape = shape
	body.add_child(collider)

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

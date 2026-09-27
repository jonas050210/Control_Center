extends RefCounted
class_name SandboxModelFactory

## Small procedural low-poly model builder used by the player, enemies, and map.
## Keeping the geometry here makes the visual pass consistent without requiring
## external art assets, imports, or a network connection.

static func material(
	color: Color,
	metallic: float = 0.0,
	roughness: float = 0.72,
	emission: Color = Color.BLACK
) -> StandardMaterial3D:
	var result: StandardMaterial3D = StandardMaterial3D.new()
	result.albedo_color = color
	result.metallic = metallic
	result.roughness = troughness
	if emission != Color.BLACK:
		result.emission_enabled = true
		result.emission = emission
		result.emission_energy_multiplier = 1.8
	return result

static func box_mesh(size: Vector3, surface: Material) -> BoxMesh:
	var mesh: BoxMesh = BoxMesh.new()
	mesh.size = size
	mesh.material = surface
	return mesh

static func cylinder_mesh(
	radius: float,
	height: float,
	surface: Material,
	segments: int = 12
) -> CylinderMesh:
	var mesh: CylinderMesh = CylinderMesh.new()
	mesh.top_radius = radius
	mesh.bottom_radius = radius
	mesh.height = height
	mesh.radial_segments = segments
	mesh.material = surface
	return mesh

static func sphere_mesh(radius: float, surface: Material) -> SphereMesh:
	var mesh: SphereMesh = SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.radial_segments = 12
	mesh.rings = 6
	mesh.material = surface
	return mesh

static func capsule_mesh(radius: float, height: float, surface: Material) -> CapsuleMesh:
	var mesh: CapsuleMesh = CapsuleMesh.new()
	mesh.radius = radius
	mesh.height = height
	mesh.radial_segments = 12
	mesh.rings = 4
	mesh.material = surface
	return mesh

static func add_box(
	parent: Node,
	part_name: String,
	size: Vector3,
	position: Vector3,
	surface: Material,
	rotation: Vector3 = Vector3.ZERO
) -> MeshInstance3D:
	var part: MeshInstance3D = MeshInstance3D.new()
	part.name = part_name
	part.mesh = box_mesh(size, surface)
	part.position = position
	part.rotation = rotation
	parent.add_child(part)
	return part

static func add_cylinder(
	parent: Node,
	part_name: String,
	radius: float,
	height: float,
	position: Vector3,
	surface: Material,
	rotation: Vector3 = Vector3.ZERO,
	segments: int = 12
) -> MeshInstance3D:
	var part: MeshInstance3D = MeshInstance3D.new()
	part.name = part_name
	part.mesh = cylinder_mesh(radius, height, surface, segments)
	part.position = position
	part.rotation = rotation
	parent.add_child(part)
	return part

static func add_sphere(
	parent: Node,
	part_name: String,
	radius: float,
	position: Vector3,
	surface: Material
) -> MeshInstance3D:
	var part: MeshInstance3D = MeshInstance3D.new()
	part.name = part_name
	part.mesh = sphere_mesh(radius, surface)
	part.position = position
	parent.add_child(part)
	return part

static func add_capsule(
	parent: Node,
	part_name: String,
	radius: float,
	height: float,
	position: Vector3,
	surface: Material
) -> MeshInstance3D:
	var part: MeshInstance3D = MeshInstance3D.new()
	part.name = part_name
	part.mesh = capsule_mesh(radius, height, surface)
	part.position = position
	parent.add_child(part)
	return part

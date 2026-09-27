## EnemyView
##
## Purely visual mirror of one EnemyState. No gameplay logic lives here.
class_name EnemyView
extends Node3D

var state: EnemyState
var body_mesh: MeshInstance3D


func setup(p_state: EnemyState) -> void:
	state = p_state

	body_mesh = MeshInstance3D.new()
	var capsule := CapsuleMesh.new()
	capsule.radius = state.radius
	capsule.height = 1.8
	body_mesh.mesh = capsule
	body_mesh.position = Vector3(0.0, 0.9, 0.0)
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.9, 0.2, 0.2)
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	body_mesh.material_override = mat
	add_child(body_mesh)

	sync()


func sync() -> void:
	if state == null:
		return
	position = state.position
	visible = state.alive

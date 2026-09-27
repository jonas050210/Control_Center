## EnvironmentView
##
## Purely visual "cell" for one EnvironmentCore: a small floor + perimeter
## walls plus an AgentView and one EnemyView per enemy. Contains zero
## gameplay logic — everything it draws is a direct mirror of the
## corresponding EnvironmentCore/AgentState/EnemyState. Global lighting is
## intentionally NOT created here (see scripts/core/main.gd) since one
## DirectionalLight3D + WorldEnvironment already lights every cell.
class_name EnvironmentView
extends Node3D

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const AgentView = preload("res://scripts/agent/agent_view.gd")
const EnemyView = preload("res://scripts/enemy/enemy_view.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


var core: EnvironmentCore
var agent_view: AgentView
var enemy_views: Array = []  # Array[EnemyView]


func setup(p_core: EnvironmentCore) -> void:
	core = p_core
	_build_arena()

	agent_view = AgentView.new()
	add_child(agent_view)
	agent_view.setup(core.agent)

	for enemy_state in core.enemies:
		var enemy_view := EnemyView.new()
		add_child(enemy_view)
		enemy_view.setup(enemy_state)
		enemy_views.append(enemy_view)

	sync_from_state()


func _build_arena() -> void:
	var half: float = core.arena_half_extent
	var wall_height: float = SandboxConfig.ARENA_WALL_HEIGHT

	var floor_mat := StandardMaterial3D.new()
	floor_mat.albedo_color = Color(0.22, 0.22, 0.26)
	floor_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	var floor_mesh_instance := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(half * 2.0, half * 2.0)
	floor_mesh_instance.mesh = plane
	floor_mesh_instance.material_override = floor_mat
	add_child(floor_mesh_instance)

	var floor_body := StaticBody3D.new()
	var floor_shape := CollisionShape3D.new()
	var box_shape := BoxShape3D.new()
	box_shape.size = Vector3(half * 2.0, 0.1, half * 2.0)
	floor_shape.shape = box_shape
	floor_shape.position = Vector3(0.0, -0.05, 0.0)
	floor_body.add_child(floor_shape)
	add_child(floor_body)

	var wall_mat := StandardMaterial3D.new()
	wall_mat.albedo_color = Color(0.32, 0.32, 0.38)
	wall_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	var wall_defs: Array = [
		{
			"pos": Vector3(0.0, wall_height * 0.5, -half),
			"size": Vector3(half * 2.0, wall_height, 0.2)
		},
		{
			"pos": Vector3(0.0, wall_height * 0.5, half),
			"size": Vector3(half * 2.0, wall_height, 0.2)
		},
		{
			"pos": Vector3(-half, wall_height * 0.5, 0.0),
			"size": Vector3(0.2, wall_height, half * 2.0)
		},
		{
			"pos": Vector3(half, wall_height * 0.5, 0.0),
			"size": Vector3(0.2, wall_height, half * 2.0)
		},
	]
	for wall_def in wall_defs:
		var wall_mesh := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = wall_def.size
		wall_mesh.mesh = box
		wall_mesh.position = wall_def.pos
		wall_mesh.material_override = wall_mat
		add_child(wall_mesh)


## Recreates the per-enemy views from the core's CURRENT enemy list. Needed
## after a curriculum-level change rebuilds `core.enemies`, which would
## otherwise leave these views mirroring orphaned EnemyState objects.
func rebind_enemy_views() -> void:
	if core == null:
		return
	for enemy_view in enemy_views:
		if enemy_view != null and is_instance_valid(enemy_view):
			(enemy_view as EnemyView).queue_free()
	enemy_views.clear()
	for enemy_state in core.enemies:
		var enemy_view := EnemyView.new()
		add_child(enemy_view)
		enemy_view.setup(enemy_state)
		enemy_views.append(enemy_view)


func sync_from_state() -> void:
	if agent_view != null:
		agent_view.sync()
	for enemy_view in enemy_views:
		(enemy_view as EnemyView).sync()


func get_camera() -> Camera3D:
	if agent_view != null:
		return agent_view.camera
	return null

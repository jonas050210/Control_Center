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
## Mesh instances mirroring `core.world`'s obstacles. Rebuilt whenever the
## layout changes; empty on the obstacle-free curriculum levels.
var obstacle_meshes: Array = []  # Array[MeshInstance3D]
## Layout the current obstacle meshes were built for, so a reset that keeps
## the same layout does not pointlessly rebuild the geometry.
var _obstacle_layout_key: String = ""


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

	rebuild_obstacles()
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


## Rebuilds the interior geometry meshes from `core.world`.
##
## Called on setup and after every reset, but it early-outs unless the
## layout id or seed actually changed — regenerating dozens of BoxMeshes on
## every episode boundary would make the GUI stutter for no reason.
## Colour-codes by obstacle kind so cover reads at a glance: low cover is
## the one you can shoot over, high cover is not.
func rebuild_obstacles() -> void:
	var key: String = "none"
	if core != null and core.world != null:
		key = "%s:%d" % [core.world.layout_id, core.world.layout_seed]
	if key == _obstacle_layout_key:
		return
	_obstacle_layout_key = key

	for mesh_value in obstacle_meshes:
		if is_instance_valid(mesh_value):
			(mesh_value as Node).queue_free()
	obstacle_meshes.clear()
	if core == null or core.world == null:
		return

	for obstacle_value in core.world.obstacles:
		var obstacle = obstacle_value
		var mesh_instance := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = obstacle.half_extents * 2.0
		mesh_instance.mesh = box
		mesh_instance.position = obstacle.center
		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		material.albedo_color = _obstacle_color(obstacle)
		mesh_instance.material_override = material
		add_child(mesh_instance)
		obstacle_meshes.append(mesh_instance)


static func _obstacle_color(obstacle) -> Color:
	if not obstacle.blocks_sight:
		return Color(0.40, 0.46, 0.34)  # low cover: shoot over it
	if obstacle.standable:
		return Color(0.34, 0.40, 0.52)  # platform: stand on it
	return Color(0.30, 0.30, 0.36)  # wall / high cover


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


## Mirrors the simulation onto the scene. Also refreshes the interior
## geometry, because a reset can regenerate the arena layout and the view
## would otherwise keep displaying the previous episode's cover.
func sync_from_state() -> void:
	rebuild_obstacles()
	if agent_view != null:
		agent_view.sync()
	for enemy_view in enemy_views:
		(enemy_view as EnemyView).sync()


func get_camera() -> Camera3D:
	if agent_view != null:
		return agent_view.camera
	return null

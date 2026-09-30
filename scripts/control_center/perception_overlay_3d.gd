## PerceptionOverlay3D
##
## Optional in-world debug drawing for the selected environment (Phase 5 /
## Phase 14). It renders:
##
##   * a cyan line to every enemy the observation vector actually tracks
##   * a red dashed line to every alive enemy that is NOT in the
##     observation (hidden from the AI)
##   * a yellow marker ring on the current target
##   * the agent's forward vector and weapon-range circle
##   * the agent's FOV cone, clipped to the vision range
##   * the footprint of every sight-blocking obstacle
##   * a violet marker at every remembered last-known position, with the
##     ring radius shrinking as confidence decays
##   * an orange ring for every audible sound event, sized by loudness
##   * a grey cross on every corpse
##
## Everything is drawn from data the PerceptionModel already produced; this
## node performs no perception logic and never feeds anything back into the
## simulation. It is positioned on the selected environment's cell so it
## can draw in that cell's local space, and it is hidden in TRAINING mode
## and never created at all in headless runs.
class_name PerceptionOverlay3D
extends Node3D

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

const COLOR_TRACKED: Color = Color(0.35, 0.85, 1.0, 1.0)
const COLOR_HIDDEN: Color = Color(1.0, 0.3, 0.3, 1.0)
const COLOR_TARGET: Color = Color(1.0, 0.85, 0.25, 1.0)
const COLOR_FORWARD: Color = Color(0.5, 1.0, 0.6, 1.0)
const COLOR_RANGE: Color = Color(0.45, 0.55, 0.75, 0.75)
const COLOR_FOV: Color = Color(0.35, 0.85, 1.0, 0.45)
const COLOR_OBSTACLE: Color = Color(0.65, 0.65, 0.7, 0.9)
const COLOR_MEMORY: Color = Color(0.75, 0.45, 1.0, 0.9)
const COLOR_SOUND: Color = Color(1.0, 0.6, 0.2, 0.9)
const COLOR_CORPSE: Color = Color(0.5, 0.5, 0.5, 0.9)
const FOV_EDGE_STEPS: int = 12
const EYE_HEIGHT: float = 1.6
const RANGE_SEGMENTS: int = 48
const DASH_SEGMENTS: int = 10

var _mesh_instance: MeshInstance3D
var _mesh: ImmediateMesh
var _material: StandardMaterial3D


func _init() -> void:
	_mesh = ImmediateMesh.new()
	_material = StandardMaterial3D.new()
	_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_material.vertex_color_use_as_albedo = true
	_material.disable_receive_shadows = true
	_material.no_depth_test = true
	_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_mesh_instance = MeshInstance3D.new()
	_mesh_instance.mesh = _mesh
	_mesh_instance.material_override = _material
	_mesh_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_mesh_instance)


## Moves the overlay onto `view`'s environment cell WITHOUT re-parenting it.
## The overlay stays owned by the Control Center, so rebuilding the
## environments (which frees every EnvironmentView) can never free the
## overlay along with them. Matching the cell offset makes the view's local
## coordinates valid inside this node.
func follow_view(view: Node3D) -> void:
	if view == null or not is_instance_valid(view):
		return
	position = view.position


## Redraws from a PerceptionModel dictionary. Passing {} clears the overlay.
func update_from_perception(perception: Dictionary) -> void:
	_mesh.clear_surfaces()
	if perception.is_empty() or not visible:
		return
	if not perception.has("real_world"):
		return

	var real_world: Dictionary = perception["real_world"]
	var agent_position: Vector3 = real_world["agent_position"]
	var eye: Vector3 = agent_position + Vector3(0.0, EYE_HEIGHT, 0.0)
	var tracked: Array = perception["tracked_enemy_indices"]
	var target_index: int = int(perception.get("target_index", -1))

	_mesh.surface_begin(Mesh.PRIMITIVE_LINES)
	for enemy_value in real_world["enemies"]:
		var enemy: Dictionary = enemy_value
		if not bool(enemy["alive"]):
			continue
		var enemy_position: Vector3 = enemy["position"]
		var enemy_eye: Vector3 = enemy_position + Vector3(0.0, 1.0, 0.0)
		if tracked.has(int(enemy["index"])):
			_line(eye, enemy_eye, COLOR_TRACKED)
		else:
			_dashed_line(eye, enemy_eye, COLOR_HIDDEN)
		if int(enemy["index"]) == target_index:
			_circle(enemy_position + Vector3(0.0, 0.05, 0.0), 1.1, COLOR_TARGET, 20)

	var forward: Vector3 = real_world["agent_forward"]
	_line(eye, eye + forward.normalized() * 3.0, COLOR_FORWARD)
	_circle(
		agent_position + Vector3(0.0, 0.05, 0.0),
		float(real_world["weapon_range"]),
		COLOR_RANGE,
		RANGE_SEGMENTS
	)
	_draw_perception_state(perception.get("perception_state", {}), agent_position, eye)
	_mesh.surface_end()


## Draws everything sourced from the optional perception hooks. Each block
## is skipped when the corresponding hook is unavailable, so this degrades
## to the original overlay on an environment without perception.
func _draw_perception_state(state: Dictionary, agent_position: Vector3, eye: Vector3) -> void:
	if state.is_empty():
		return

	var fov: Dictionary = state.get("field_of_view", {})
	if not fov.is_empty() and bool(fov.get("enabled", false)):
		_fov_cone(
			agent_position,
			fov.get("forward", Vector3.FORWARD),
			float(fov.get("fov_deg", 100.0)),
			float(fov.get("range", 20.0))
		)

	for obstacle_value in state.get("obstacles", []) as Array:
		_obstacle_footprint(obstacle_value)

	for track_value in state.get("memory", []) as Array:
		var track: Dictionary = track_value
		var remembered: Vector3 = track["position"]
		var confidence: float = clampf(float(track["confidence"]), 0.0, 1.0)
		_circle(remembered + Vector3(0.0, 0.05, 0.0), 0.35 + confidence * 0.9, COLOR_MEMORY, 16)
		_dashed_line(eye, remembered + Vector3(0.0, 1.0, 0.0), COLOR_MEMORY)

	for sound_value in state.get("sounds", []) as Array:
		var sound: Dictionary = sound_value
		var origin: Vector3 = (
			agent_position + (sound["direction"] as Vector3) * float(sound["distance"])
		)
		origin.y = agent_position.y
		_circle(
			origin + Vector3(0.0, 0.05, 0.0), 0.4 + float(sound["loudness"]) * 1.4, COLOR_SOUND, 14
		)

	for corpse_value in state.get("corpses", []) as Array:
		var corpse_position: Vector3 = (corpse_value as Dictionary)["position"]
		var centre: Vector3 = corpse_position + Vector3(0.0, 0.05, 0.0)
		_line(centre + Vector3(-0.5, 0.0, -0.5), centre + Vector3(0.5, 0.0, 0.5), COLOR_CORPSE)
		_line(centre + Vector3(-0.5, 0.0, 0.5), centre + Vector3(0.5, 0.0, -0.5), COLOR_CORPSE)


## Two straight edges plus an arc, all at ankle height so the cone reads as
## a floor plan rather than obscuring the fight.
func _fov_cone(origin: Vector3, forward: Vector3, fov_deg: float, cone_range: float) -> void:
	var flat := Vector3(forward.x, 0.0, forward.z)
	if flat.is_zero_approx():
		return
	flat = flat.normalized()
	var base: Vector3 = origin + Vector3(0.0, 0.08, 0.0)
	var half: float = deg_to_rad(fov_deg * 0.5)
	var left: Vector3 = base + flat.rotated(Vector3.UP, half) * cone_range
	var right: Vector3 = base + flat.rotated(Vector3.UP, -half) * cone_range
	_line(base, left, COLOR_FOV)
	_line(base, right, COLOR_FOV)
	var previous: Vector3 = right
	for step in range(1, FOV_EDGE_STEPS + 1):
		var angle: float = -half + 2.0 * half * float(step) / float(FOV_EDGE_STEPS)
		var point: Vector3 = base + flat.rotated(Vector3.UP, angle) * cone_range
		_line(previous, point, COLOR_FOV)
		previous = point


## Top-face rectangle of one obstacle box, so cover geometry is visible
## without occluding the agents.
func _obstacle_footprint(obstacle_value) -> void:
	var obstacle: Dictionary = obstacle_value
	var centre: Vector3 = obstacle["center"]
	var extents: Vector3 = obstacle["half_extents"]
	var top: float = centre.y + extents.y
	var corners: Array = [
		Vector3(centre.x - extents.x, top, centre.z - extents.z),
		Vector3(centre.x + extents.x, top, centre.z - extents.z),
		Vector3(centre.x + extents.x, top, centre.z + extents.z),
		Vector3(centre.x - extents.x, top, centre.z + extents.z),
	]
	for index in range(corners.size()):
		_line(corners[index], corners[(index + 1) % corners.size()], COLOR_OBSTACLE)


func _line(from: Vector3, to: Vector3, color: Color) -> void:
	_mesh.surface_set_color(color)
	_mesh.surface_add_vertex(from)
	_mesh.surface_set_color(color)
	_mesh.surface_add_vertex(to)


func _dashed_line(from: Vector3, to: Vector3, color: Color) -> void:
	for segment in range(DASH_SEGMENTS):
		if segment % 2 == 1:
			continue
		var start_ratio: float = float(segment) / float(DASH_SEGMENTS)
		var end_ratio: float = float(segment + 1) / float(DASH_SEGMENTS)
		_line(from.lerp(to, start_ratio), from.lerp(to, end_ratio), color)


func _circle(centre: Vector3, radius: float, color: Color, segments: int) -> void:
	var previous: Vector3 = centre + Vector3(radius, 0.0, 0.0)
	for step in range(1, segments + 1):
		var angle: float = TAU * float(step) / float(segments)
		var point: Vector3 = centre + Vector3(cos(angle) * radius, 0.0, sin(angle) * radius)
		_line(previous, point, color)
		previous = point

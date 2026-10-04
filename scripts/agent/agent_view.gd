## AgentView
##
## Purely visual/presentation layer for one AgentState. Holds no gameplay
## logic — it only mirrors AgentState into a Node3D transform and exposes
## the FPS camera a human plays through. Deleting this class (e.g. to run
## fully headless) would not change simulation behavior at all.
class_name AgentView
extends Node3D

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const AgentState = preload("res://scripts/agent/agent_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

var state: AgentState
var camera: Camera3D
var body_mesh: MeshInstance3D


func setup(p_state: AgentState) -> void:
	state = p_state

	body_mesh = MeshInstance3D.new()
	var capsule := CapsuleMesh.new()
	capsule.radius = state.radius
	capsule.height = 1.8
	body_mesh.mesh = capsule
	body_mesh.position = Vector3(0.0, 0.9, 0.0)
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.2, 0.55, 1.0)
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	body_mesh.material_override = mat
	add_child(body_mesh)

	camera = Camera3D.new()
	camera.position = Vector3(0.0, state.eye_height, 0.0)
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	camera.fov = vertical_fov_deg()
	add_child(camera)

	sync()


## The vertical field-of-view angle for the play camera, derived from the
## contract's cone instead of being typed in.
##
## The two live on different axes: `SandboxConfig.AGENT_FOV_DEG` is the
## HORIZONTAL cone - the one `PerceptionSystem.target_screen_box` and the
## observation's screen box are built on - while Godot's `Camera3D.fov` is the
## VERTICAL angle whenever `keep_aspect` is `KEEP_HEIGHT`, which is the
## default. A hardcoded `80.0` here was therefore not "a bit off": at the
## project's 16:9 window it is ~112° horizontal, so the human was shown
## noticeably more than the agent can perceive, and the crosshair could sit
## over an enemy the observation reports as outside the cone.
static func vertical_fov_deg() -> float:
	var half_horizontal: float = deg_to_rad(SandboxConfig.AGENT_FOV_DEG) * 0.5
	return rad_to_deg(2.0 * atan(tan(half_horizontal) / SandboxConfig.AGENT_VIEW_ASPECT))


## Mirrors the current AgentState onto this node's transform. Called once
## per rendered/ticked frame by EnvironmentView; never drives simulation.
func sync() -> void:
	if state == null:
		return
	position = state.position
	# Godot's default Y-rotation direction is opposite our yaw convention
	# (see AgentState.get_forward_horizontal), hence the negation.
	rotation.y = -deg_to_rad(state.yaw_deg)
	if camera != null:
		camera.rotation.x = deg_to_rad(state.pitch_deg)
	visible = true
	if body_mesh != null:
		body_mesh.visible = state.alive

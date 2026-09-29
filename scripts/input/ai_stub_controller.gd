## AIStubController
##
## A trivial, fully deterministic heuristic policy: turn toward the nearest
## alive enemy, close the distance, and fire once roughly aimed and in
## range. This is NOT a trained RL policy — it only exists so the simulator
## can demonstrate autonomous multi-environment operation before an actual
## trainer is connected (the next milestone). It implements the exact same
## `ControllerBase` interface a real RL policy adapter will use.
class_name AIStubController
extends ControllerBase

## ControllerBase provides the shared Action and EnvironmentCore dependencies.
## Keep the controller-specific dependencies explicit for standalone/headless parsing.
const AgentState = preload("res://scripts/agent/agent_state.gd")
const ControllerBase = preload("res://scripts/input/controller_base.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")


const AIM_TOLERANCE_DEG: float = 5.0
const APPROACH_DISTANCE_FACTOR: float = 0.6


func get_action(env: EnvironmentCore) -> Action:
	if env.is_done():
		return Action.idle()

	var agent: AgentState = env.agent
	var enemy: EnemyState = env.get_primary_enemy()
	if enemy == null or not enemy.alive:
		return Action.idle()

	var to_enemy: Vector3 = enemy.position - agent.position
	to_enemy.y = 0.0
	var distance: float = to_enemy.length()
	if distance < 0.0001:
		return Action.idle()

	var desired_yaw_deg: float = rad_to_deg(atan2(to_enemy.x, -to_enemy.z))
	var yaw_diff: float = wrapf(desired_yaw_deg - agent.yaw_deg, -180.0, 180.0)

	var look_yaw_axis: int = 0
	if yaw_diff > AIM_TOLERANCE_DEG:
		look_yaw_axis = 1
	elif yaw_diff < -AIM_TOLERANCE_DEG:
		look_yaw_axis = -1

	var move_axis: int = 0
	if distance > agent.weapon.range_m * APPROACH_DISTANCE_FACTOR:
		move_axis = 1

	var shoot: bool = absf(yaw_diff) <= AIM_TOLERANCE_DEG and distance <= agent.weapon.range_m

	return Action.new(move_axis, 0, look_yaw_axis, 0, shoot)

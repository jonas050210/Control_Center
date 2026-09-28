## NavigationAgent
##
## Per-character path-following state that sits between a tactical decision
## ("go to that cover spot") and the movement integration
## (`CharacterMotor`). One instance lives on each `EnemyState`.
##
## The design rule is *navigation is an exception path, not the default*.
## Steering straight at the destination is one vector subtraction; planning
## is a grid A*. So this object watches the character actually move and only
## engages the `NavigationGraph` once the character is demonstrably blocked
## (or is already following a plan). In an open arena it therefore costs two
## float comparisons per tick and nothing else, which is what keeps
## headless throughput on the perception-heavy levels unchanged.
##
## Everything here is deterministic: no RandomNumberGenerator, no wall
## clock, and the only tie-breaking happens inside NavigationGraph's A*.
class_name NavigationAgent
extends RefCounted

const NavigationGraph = preload("res://scripts/world/navigation_graph.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SELF_PATH: String = "res://scripts/world/navigation_agent.gd"

## What the agent did this tick. Debug/telemetry only — never observed by a
## policy.
const STATUS_DIRECT: String = "direct"
const STATUS_PATH: String = "path"
const STATUS_RECOVERY: String = "recovery"
const STATUS_UNREACHABLE: String = "unreachable"

var path: Array = []  # Array[Vector3]
var path_index: int = 0
var path_destination: Vector3 = Vector3.ZERO
var repath_timer: float = 0.0
## Seconds the character has been asking to move while barely moving.
var stuck_time: float = 0.0
var recovery_timer: float = 0.0
var recovery_vector: Vector3 = Vector3.ZERO
var last_position: Vector3 = Vector3.ZERO
var has_last_position: bool = false
## Cumulative count of destinations that turned out to be unreachable.
## Surfaced as a research metric ("navigation failures"), not a reward.
var failures: int = 0
var status: String = STATUS_DIRECT


static func create() -> NavigationAgent:
	return (load(SELF_PATH) as GDScript).new() as NavigationAgent


func reset(position: Vector3 = Vector3.ZERO) -> void:
	path = []
	path_index = 0
	path_destination = Vector3.ZERO
	repath_timer = 0.0
	stuck_time = 0.0
	recovery_timer = 0.0
	recovery_vector = Vector3.ZERO
	last_position = position
	has_last_position = true
	failures = 0
	status = STATUS_DIRECT


func is_following_path() -> bool:
	return path_index < path.size()


## Returns the position the caller should steer at this tick.
##
## `graph` may be null (no geometry / navigation disabled), in which case
## the destination is returned unchanged and this object degrades to a
## stuck-time counter.
func update(
	graph, world, position: Vector3, destination: Vector3, wants_move: bool, dt: float
) -> Vector3:
	_track_progress(position, wants_move, dt)

	if graph == null or not (graph as NavigationGraph).is_ready():
		status = STATUS_DIRECT
		return destination

	var navigation: NavigationGraph = graph
	if recovery_timer > 0.0:
		recovery_timer = maxf(0.0, recovery_timer - dt)
		if not recovery_vector.is_zero_approx():
			status = STATUS_RECOVERY
			return position + recovery_vector * maxf(navigation.cell_size, 1.0)

	repath_timer = maxf(0.0, repath_timer - dt)
	if not (is_following_path() or stuck_time >= SandboxConfig.NAV_STUCK_TIME):
		status = STATUS_DIRECT
		return destination
	return _navigate(navigation, world, position, destination)


## Path-following half of `update()`, split out so each half stays within
## the project's return-count lint budget and can be reasoned about alone.
func _navigate(
	navigation: NavigationGraph, world, position: Vector3, destination: Vector3
) -> Vector3:
	# The obstacle that forced the detour may no longer be between us (the
	# destination moves, and so do we). Dropping the plan the moment the
	# straight line is walkable again keeps the enemy from marching a long
	# way around a doorway it is now standing in front of.
	if is_following_path() and stuck_time < SandboxConfig.NAV_STUCK_TIME and world != null:
		if not navigation.direct_route_blocked(world, position, destination):
			abandon()
			return destination

	if _needs_replan(destination):
		_replan(navigation, position, destination)
	if path.is_empty():
		return _begin_recovery(navigation, position, destination)

	_consume_reached_waypoints(position)
	if not is_following_path():
		abandon()
		return destination

	status = STATUS_PATH
	return path[path_index]


## Clears any plan. Called when the tactical state changes so a stale route
## to the previous destination cannot survive into the new behavior.
func abandon() -> void:
	path = []
	path_index = 0
	repath_timer = 0.0
	status = STATUS_DIRECT


func _track_progress(position: Vector3, wants_move: bool, dt: float) -> void:
	if not has_last_position:
		last_position = position
		has_last_position = true
		return
	var travelled: float = Vector2(
		position.x - last_position.x, position.z - last_position.z
	).length()
	last_position = position
	if dt <= 0.0:
		return
	var speed: float = travelled / dt
	if wants_move and speed < SandboxConfig.NAV_STUCK_SPEED:
		stuck_time += dt
	else:
		stuck_time = 0.0


func _needs_replan(destination: Vector3) -> bool:
	if path.is_empty() or not is_following_path():
		return true
	if repath_timer > 0.0:
		return false
	# The destination is usually a moving target (a believed enemy position),
	# so re-plan when it has drifted meaningfully rather than every tick.
	return Vector2(
		destination.x - path_destination.x, destination.z - path_destination.z
	).length() > 1.5


func _replan(navigation: NavigationGraph, position: Vector3, destination: Vector3) -> void:
	repath_timer = SandboxConfig.NAV_REPATH_INTERVAL
	path_destination = destination
	path_index = 0
	var planned: Array = navigation.find_path(position, destination)
	if planned.size() > SandboxConfig.NAV_MAX_WAYPOINTS:
		planned = planned.slice(0, SandboxConfig.NAV_MAX_WAYPOINTS)
	path = planned


func _consume_reached_waypoints(position: Vector3) -> void:
	while path_index < path.size():
		var waypoint: Vector3 = path[path_index]
		var distance: float = Vector2(
			waypoint.x - position.x, waypoint.z - position.z
		).length()
		if distance > SandboxConfig.NAV_WAYPOINT_TOLERANCE:
			return
		path_index += 1


func _begin_recovery(
	navigation: NavigationGraph, position: Vector3, destination: Vector3
) -> Vector3:
	failures += 1
	recovery_vector = navigation.recovery_direction(position, destination)
	recovery_timer = SandboxConfig.NAV_RECOVERY_TIME
	if recovery_vector.is_zero_approx():
		status = STATUS_UNREACHABLE
		return destination
	status = STATUS_RECOVERY
	return position + recovery_vector * maxf(navigation.cell_size, 1.0)


func to_dict() -> Dictionary:
	return {
		"status": status,
		"waypoints_remaining": maxi(0, path.size() - path_index),
		"stuck_time": stuck_time,
		"recovery_timer": recovery_timer,
		"failures": failures,
	}

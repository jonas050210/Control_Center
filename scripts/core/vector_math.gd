## VectorMath
##
## The angle and vector helpers more than one subsystem needs, kept in one
## place so each convention has exactly ONE definition. It exists because of a
## real bug: the signed bearing had two implementations with opposite signs.
## `Observation` measured it as a yaw difference (positive = the agent's
## right, the way `Action.look_yaw_axis = +1` turns) while the world, sound
## and memory queries took the sign of `forward.cross(direction).y`, which is
## the opposite. Eight of the eleven bearing fields in the observation vector
## therefore pointed the wrong way, and the Stats page drew a contact and a
## crate on the same side of the agent on opposite sides.
##
## Three conventions live here, and nowhere else:
##   * the signed bearing (positive = right) - `signed_bearing_deg` and
##     `signed_bearing_between`,
##   * the yaw of a horizontal direction - `yaw_deg_from_direction`, the
##     inverse of `AgentState.get_forward_horizontal`,
##   * the elevation of a point as seen from an eye - `elevation_deg`.
##
## No dependencies on purpose: every subsystem, including `ArenaWorld`, may
## preload this without creating a cycle.
class_name VectorMath
extends RefCounted


## Signed horizontal angle (degrees) from `forward` to the direction
## `from_position -> to_position`.
##
## **Sign convention: positive is to the agent's RIGHT**, i.e. the way a
## positive `Action.look_yaw_axis` turns; negative is left. This is the
## convention the contract documents for `enemy_bearing_norm` (index 17), and
## every bearing field in the observation vector uses it - a policy (and the
## operator reading the Stats page) must be able to compare a contact's
## bearing with a cover's bearing without a per-field sign flip.
##
## Only the horizontal plane is considered: a target's height never changes
## its bearing. Returns `0.0` for a target dead ahead (or a degenerate input)
## and `+180.0` for one exactly behind, where left and right are the same
## decision.
static func signed_bearing_deg(
	forward: Vector3, from_position: Vector3, to_position: Vector3
) -> float:
	var flat_forward := Vector3(forward.x, 0.0, forward.z)
	var delta := Vector3(to_position.x - from_position.x, 0.0, to_position.z - from_position.z)
	if flat_forward.is_zero_approx() or delta.is_zero_approx():
		return 0.0
	return signed_bearing_between(flat_forward, delta)


## `signed_bearing_deg` for two direction vectors that already originate at
## the same point (a facing and a direction to a target).
##
## The y-component of `forward.cross(direction)` is NEGATIVE for a target on
## the right in Godot's right-handed frame, which is why the returned sign
## looks inverted next to the cross product - see the class comment.
static func signed_bearing_between(flat_forward: Vector3, direction: Vector3) -> float:
	var forward := Vector3(flat_forward.x, 0.0, flat_forward.z)
	var flat_direction := Vector3(direction.x, 0.0, direction.z)
	if forward.is_zero_approx() or flat_direction.is_zero_approx():
		return 0.0
	forward = forward.normalized()
	flat_direction = flat_direction.normalized()
	var dot: float = clampf(forward.dot(flat_direction), -1.0, 1.0)
	var angle: float = rad_to_deg(acos(dot))
	var side: float = forward.cross(flat_direction).y
	if is_zero_approx(side):
		# Dead ahead (angle 0) or dead astern (angle 180), where there is no
		# left/right to distinguish.
		return angle
	return angle if side < 0.0 else -angle


## Yaw (degrees) that points along a horizontal direction, in the convention
## of `AgentState.yaw_deg`: 0 looks along `(0, 0, -1)`, 90 along `+x`, 180
## along `+z` (the inverse of `AgentState.get_forward_horizontal`).
##
## The agent, the enemies, the stub controller, the scenario and world
## generators and the environment reset all used to spell out
## `rad_to_deg(atan2(x, -z))` themselves; this is that formula's one home.
## A direction without a horizontal component returns 0.0, so callers that
## need a project-specific default (a spawn yaw, say) must test for it first.
static func yaw_deg_from_direction(direction: Vector3) -> float:
	var flat := Vector3(direction.x, 0.0, direction.z)
	if flat.is_zero_approx():
		return 0.0
	return rad_to_deg(atan2(flat.x, -flat.z))


## Vertical angle (degrees) from an eye to a point. Positive means the point
## is above the eye, negative below; a point straight up or down is ±90
## degrees, where the horizontal component vanishes.
##
## The observation vector normalizes this by 90 degrees, so the eye geometry
## is the only thing that differs between callers - never the sign.
static func elevation_deg(from_eye: Vector3, to_position: Vector3) -> float:
	var delta: Vector3 = to_position - from_eye
	var horizontal: float = Vector2(delta.x, delta.z).length()
	if horizontal < 0.000001:
		return 90.0 if delta.y >= 0.0 else -90.0
	return rad_to_deg(atan2(delta.y, horizontal))

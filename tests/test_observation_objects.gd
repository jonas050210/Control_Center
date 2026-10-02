## Contract v4: the world objects the policy can see.
##
## `ArenaWorld.visible_object_infos()` is the single visibility query behind
## the three object slots, so these tests pin its semantics - nearest first,
## FOV cone, vision range, occlusion, the fence excluded, boxes that block
## nothing skipped - and then check that `Observation.build()` writes exactly
## those numbers into the vector and keeps empty slots neutral.
##
## Everything here is deliberately deterministic: fixed boxes, no layout
## generation, no randomness, so a failure names a real behaviour change
## instead of a seed.
class_name TestObservationObjects
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const AgentState = preload("res://scripts/agent/agent_state.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const Observation = preload("res://scripts/core/observation.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")

const FORWARD: Vector3 = Vector3(0.0, 0.0, -1.0)
## A crate that sits on the ground: 2 x 1 x 2 meters, top below the eye line.
const CRATE_HALF: Vector3 = Vector3(1.0, 0.5, 1.0)


## The query itself: the nearest visible object comes first, distance and
## bearing are measured to the closest surface point (not the center), and a
## wall further away is reported after the crate.
func test_object_query_orders_nearest_visible_first() -> SandboxTest:
	var t := SandboxTest.new("object_query_orders_nearest_visible_first")
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	world.add_box(Vector3(2.0, 1.5, -10.0), Vector3(0.25, 1.5, 1.0), Obstacle.Kind.WALL)
	world.add_box(Vector3(0.0, 0.5, -5.0), CRATE_HALF, Obstacle.Kind.CRATE)
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var eye: Vector3 = agent.get_eye_position()

	var entries: Array = world.visible_object_infos(eye, FORWARD, 100.0, 28.0, 8)
	t.assert_eq(entries.size(), 2, "both boxes are inside the cone and in range")

	var near: Dictionary = entries[0]
	var far: Dictionary = entries[1]
	t.assert_almost_eq(float(near["distance"]), 4.0, 0.01, "closest point of the crate is 4 m off")
	t.assert_almost_eq(float(near["bearing_deg"]), 0.0, 0.01, "the crate is dead ahead")
	t.assert_eq(int(near["kind"]), Obstacle.Kind.CRATE)
	t.assert_almost_eq(float(far["distance"]), 9.17, 0.05, "wall front face, x offset included")
	t.assert_gt(
		float(far["bearing_deg"]),
		0.0,
		"the wall is to the right, and right is a positive bearing (same sign as index 17)"
	)
	t.assert_almost_eq(float(far["bearing_deg"]), 11.0, 0.1, "atan(1.75 / 9) off the forward axis")
	t.assert_lt(
		float(near["distance"]), float(far["distance"]), "nearest first, not insertion order"
	)
	# The relative position points at the surface, so its y is the crate top
	# minus the eye - the same value the observation normalizes by wall height.
	var relative: Vector3 = near["relative_position"]
	t.assert_almost_eq(relative.y, 1.0 - eye.y, 0.01, "y is the closest surface point")
	t.assert_almost_eq(relative.z, -4.0, 0.01)
	return t


## Fence boxes are never reported, whatever else is true about them, and the
## FOV cone and the distance limit filter the rest.
func test_object_query_applies_fov_range_and_fence_rules() -> SandboxTest:
	var t := SandboxTest.new("object_query_applies_fov_range_and_fence_rules")
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	# The fence stands off to the right, inside the cone: it is real geometry
	# (it blocks sight like any wall, which is why it must NOT sit in front of
	# the crate below) but it is never an object the policy can use.
	world.add_box(Vector3(8.0, 1.5, -6.0), Vector3(2.0, 1.5, 0.5), Obstacle.Kind.BOUNDARY)
	world.add_box(Vector3(0.0, 0.5, 3.0), CRATE_HALF, Obstacle.Kind.CRATE)  # behind
	world.add_box(Vector3(0.0, 0.5, -40.0), CRATE_HALF, Obstacle.Kind.CRATE)  # out of range
	world.add_box(Vector3(0.0, 0.5, -6.0), CRATE_HALF, Obstacle.Kind.CRATE)
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var eye: Vector3 = agent.get_eye_position()

	var entries: Array = world.visible_object_infos(eye, FORWARD, 100.0, 28.0, 8)
	t.assert_eq(entries.size(), 1, "only the crate in front and in range is visible")
	t.assert_eq(int((entries[0] as Dictionary)["kind"]), Obstacle.Kind.CRATE)

	# The same world with a longer reach reports the far crate too, which
	# proves the previous result came from the distance limit.
	var wider: Array = world.visible_object_infos(eye, FORWARD, 100.0, 60.0, 8)
	t.assert_eq(wider.size(), 2, "a longer vision range must reach the far crate")

	# A narrow cone excludes an object that is visible to the wide one.
	var crate_right: Obstacle = world.add_box(
		Vector3(4.0, 0.5, -6.0), CRATE_HALF, Obstacle.Kind.CRATE
	)
	var narrow: Array = world.visible_object_infos(eye, FORWARD, 20.0, 28.0, 8)
	t.assert_eq(narrow.size(), 1, "the right-hand crate is outside a 20 degree cone")
	t.assert_false(
		_contains_kind_and_position(narrow, Obstacle.Kind.CRATE, crate_right.center),
		"the excluded crate must not appear anywhere in the result"
	)
	return t


## Cover hides what stands behind it, and a box that directly faces the agent
## must still be reported (the occlusion probe must not count a box as its own
## blocker).
func test_object_query_hides_objects_behind_cover() -> SandboxTest:
	var t := SandboxTest.new("object_query_hides_objects_behind_cover")
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	var wall: Obstacle = world.add_box(
		Vector3(0.0, 1.5, -5.0), Vector3(4.0, 1.5, 0.5), Obstacle.Kind.WALL
	)
	world.add_box(Vector3(0.0, 0.5, -10.0), CRATE_HALF, Obstacle.Kind.CRATE)
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var eye: Vector3 = agent.get_eye_position()

	var entries: Array = world.visible_object_infos(eye, FORWARD, 100.0, 28.0, 8)
	t.assert_eq(entries.size(), 1, "the crate behind the wall is hidden")
	var reported: Dictionary = entries[0]
	t.assert_eq(int(reported["kind"]), Obstacle.Kind.WALL)
	t.assert_almost_eq(
		float(reported["distance"]),
		4.5,
		0.01,
		"the wall itself is visible: its own surface is not its occluder (OCCLUSION_EPSILON)"
	)
	t.assert_almost_eq(
		float(reported["relative_position"].z),
		wall.max_corner().z,
		0.01,
		"the reported point is the wall's front face, not its center"
	)
	return t


## A box that neither blocks sight nor movement is not an object the agent
## can use, so the query ignores it even when it is right in front.
func test_object_query_skips_boxes_that_block_nothing() -> SandboxTest:
	var t := SandboxTest.new("object_query_skips_boxes_that_block_nothing")
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	var ghost: Obstacle = world.add_box(Vector3(0.0, 0.5, -4.0), CRATE_HALF, Obstacle.Kind.CRATE)
	ghost.blocks_sight = false
	ghost.blocks_movement = false
	world.add_box(Vector3(0.0, 0.5, -6.0), CRATE_HALF, Obstacle.Kind.CRATE)
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)

	var entries: Array = world.visible_object_infos(
		agent.get_eye_position(), FORWARD, 100.0, 28.0, 8
	)
	t.assert_eq(entries.size(), 1, "the non-blocking box is skipped")
	t.assert_almost_eq(float((entries[0] as Dictionary)["distance"]), 5.0, 0.01)
	return t


## The observation slots are the query result: normalized position, distance,
## bearing, kind ordinal, per-slot visibility, and a count that is NOT cut off
## at the three-slot budget.
func test_observation_object_slots_follow_the_query() -> SandboxTest:
	var t := SandboxTest.new("observation_object_slots_follow_the_query")
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	world.add_box(Vector3(0.0, 0.5, -6.0), CRATE_HALF, Obstacle.Kind.CRATE)
	world.add_box(Vector3(2.0, 1.5, -9.0), Vector3(0.25, 1.5, 1.0), Obstacle.Kind.PILLAR)
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var eye: Vector3 = agent.get_eye_position()

	var obs := Observation.build(agent, [], SandboxConfig.ARENA_HALF_EXTENT, {"world": world})
	t.assert_almost_eq(obs.object_1_visible, 1.0, 0.001, "slot 1 holds a sighting")
	t.assert_almost_eq(obs.object_1_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE, 5.0, 0.02)
	t.assert_almost_eq(obs.object_1_bearing_norm, 0.0, 0.001)
	t.assert_almost_eq(
		obs.object_1_kind_norm,
		float(Obstacle.Kind.CRATE) / float(SandboxConfig.OBJECT_KIND_COUNT - 1),
		0.001,
		"the kind is an ordinal over OBJECT_KIND_COUNT - 1, never a raw enum value"
	)
	t.assert_almost_eq(
		obs.object_1_relative_position_norm.z,
		-5.0 / SandboxConfig.ARENA_MAX_DISTANCE,
		0.01,
		"position is normalized exactly like every other relative position"
	)
	t.assert_almost_eq(
		obs.object_1_relative_position_norm.y, (1.0 - eye.y) / SandboxConfig.ARENA_WALL_HEIGHT, 0.01
	)
	t.assert_almost_eq(obs.object_2_visible, 1.0, 0.001, "the pillar is the second object")
	t.assert_almost_eq(obs.object_3_visible, 0.0, 0.001, "there is no third object")
	t.assert_almost_eq(obs.object_3_distance_norm, 1.0, 0.001, "an empty slot reads 1.0, not 0")
	t.assert_almost_eq(obs.object_3_kind_norm, 0.0, 0.001)
	t.assert_almost_eq(obs.visible_object_count_norm, 2.0 / Observation.COUNT_NORMALIZER, 0.001)

	# The vector must carry the same numbers as the named fields.
	var values: PackedFloat32Array = obs.to_array()
	t.assert_almost_eq(values[Observation.field_names().find("object_1_visible")], 1.0, 0.001)
	t.assert_almost_eq(
		values[Observation.field_names().find("visible_object_count_norm")],
		2.0 / Observation.COUNT_NORMALIZER,
		0.001
	)
	return t


## More visible objects than slots: the three slots fill up and the count
## reports the real total, because the builder asks the query for more
## entries than it will store.
func test_visible_object_count_is_not_truncated_by_the_slot_budget() -> SandboxTest:
	var t := SandboxTest.new("visible_object_count_is_not_truncated_by_the_slot_budget")
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	# Five small crates on a ring at 8 m, 20 degrees apart. Small boxes and a
	# wide ring keep them from occluding each other; the ring is well inside
	# the 100 degree cone.
	var radius: float = 8.0
	var half: Vector3 = Vector3(0.5, 0.5, 0.5)
	for bearing_deg in [0.0, 20.0, -20.0, 40.0, -40.0]:
		var radians: float = deg_to_rad(float(bearing_deg))
		var position := Vector3(sin(radians) * radius, 0.5, -cos(radians) * radius)
		world.add_box(position, half, Obstacle.Kind.CRATE)
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)

	var entries: Array = world.visible_object_infos(
		agent.get_eye_position(), FORWARD, SandboxConfig.AGENT_FOV_DEG, 28.0, 8
	)
	t.assert_eq(entries.size(), 5, "all five ring crates are inside the cone")

	var obs := Observation.build(agent, [], SandboxConfig.ARENA_HALF_EXTENT, {"world": world})
	t.assert_eq(Observation.MAX_TRACKED_OBJECTS, 3)
	t.assert_almost_eq(obs.object_1_visible, 1.0, 0.001)
	t.assert_almost_eq(obs.object_2_visible, 1.0, 0.001)
	t.assert_almost_eq(obs.object_3_visible, 1.0, 0.001)
	t.assert_almost_eq(
		obs.visible_object_count_norm,
		5.0 / Observation.COUNT_NORMALIZER,
		0.001,
		"the count reports all five, not the three that fit in slots"
	)
	return t


## Without a world in the context the object block keeps its neutral encoding:
## zero position, distance 1.0, kind 0, visible 0, count 0 - never a fake
## reading at the origin and never a gate on anything else in the vector.
func test_object_slots_are_neutral_without_a_world() -> SandboxTest:
	var t := SandboxTest.new("object_slots_are_neutral_without_a_world")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	var obs := Observation.build(agent, [], SandboxConfig.ARENA_HALF_EXTENT)
	t.assert_almost_eq(obs.object_1_visible, 0.0, 0.001)
	t.assert_almost_eq(obs.object_1_distance_norm, 1.0, 0.001)
	t.assert_almost_eq(obs.object_1_kind_norm, 0.0, 0.001)
	t.assert_almost_eq(obs.object_2_distance_norm, 1.0, 0.001)
	t.assert_almost_eq(obs.object_3_distance_norm, 1.0, 0.001)
	t.assert_almost_eq(obs.visible_object_count_norm, 0.0, 0.001)

	# A context that names a world but configures a narrow cone and a short
	# reach must change the result: the builder forwards fov_deg and
	# vision_range to the query.
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	world.add_box(Vector3(0.0, 0.5, -20.0), CRATE_HALF, Obstacle.Kind.CRATE)
	var short_reach := Observation.build(
		agent,
		[],
		SandboxConfig.ARENA_HALF_EXTENT,
		{"world": world, "fov_deg": 20.0, "vision_range": 5.0}
	)
	t.assert_almost_eq(short_reach.object_1_visible, 0.0, 0.001)
	t.assert_almost_eq(short_reach.visible_object_count_norm, 0.0, 0.001)
	return t


## One convention for every bearing field. An enemy and an object on the
## same side of the agent must produce the same sign (and, on the same ray,
## the same normalized value), because the policy and the operator reading
## the Stats page compare them directly.
##
## This is the regression test for a real bug: the enemy bearing was a yaw
## difference (positive = right, matching `look_yaw_axis`) while the world,
## sound and memory queries took the opposite sign of `cross().y`, so a
## contact and the cover next to it were reported on opposite sides.
func test_every_bearing_field_agrees_on_which_side_is_positive() -> SandboxTest:
	var t := SandboxTest.new("every_bearing_field_agrees_on_which_side_is_positive")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	# Everything on the 45 degree ray to the right: the enemy at (5, -5), and
	# a crate whose closest corner lies on that same ray (x from 2 to 4, z from
	# -4 to -2, so the clamp lands on the (2, -2) corner - 45 degrees off axis).
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	world.add_box(Vector3(3.0, 0.5, -3.0), Vector3(1.0, 0.5, 1.0), Obstacle.Kind.CRATE)
	var enemy := EnemyState.new()
	enemy.reset(Vector3(5.0, 0.0, -5.0))

	var obs := Observation.build(agent, [enemy], SandboxConfig.ARENA_HALF_EXTENT, {"world": world})
	t.assert_gt(obs.enemy_bearing_norm, 0.0, "an enemy to the right is a positive bearing")
	t.assert_gt(obs.object_1_bearing_norm, 0.0, "an object to the right is positive too")
	t.assert_gt(obs.nearest_obstacle_bearing_norm, 0.0, "and so is the nearest obstacle")
	t.assert_almost_eq(obs.enemy_bearing_norm, 45.0 / 180.0, 0.01)
	t.assert_almost_eq(
		obs.object_1_bearing_norm,
		obs.enemy_bearing_norm,
		0.01,
		"same ray, same angle, same normalization: the fields must agree"
	)

	# Mirror everything to the left; the signs must mirror with it.
	var left_world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	left_world.add_box(Vector3(-3.0, 0.5, -3.0), Vector3(1.0, 0.5, 1.0), Obstacle.Kind.CRATE)
	var left_enemy := EnemyState.new()
	left_enemy.reset(Vector3(-5.0, 0.0, -5.0))
	var left_obs := Observation.build(
		agent, [left_enemy], SandboxConfig.ARENA_HALF_EXTENT, {"world": left_world}
	)
	t.assert_lt(left_obs.enemy_bearing_norm, 0.0, "an enemy to the left is negative")
	t.assert_lt(left_obs.object_1_bearing_norm, 0.0, "an object to the left is negative too")
	t.assert_almost_eq(left_obs.object_1_bearing_norm, left_obs.enemy_bearing_norm, 0.01)
	return t


## Helper: does any entry describe a box of this kind at this center?
func _contains_kind_and_position(entries: Array, kind: int, center: Vector3) -> bool:
	for entry_value in entries:
		var entry: Dictionary = entry_value
		if int(entry["kind"]) != kind:
			continue
		var relative: Vector3 = entry["relative_position"]
		if relative.distance_to(center) < 3.0:
			return true
	return false

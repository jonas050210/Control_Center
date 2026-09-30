## Tests for the navigation layer: deterministic graph baking, pathfinding,
## unreachable destinations, corner recovery, obstacle changes and the
## per-character NavigationAgent path follower.
##
## Everything here is pure data: no scene tree, no renderer, no RNG, so it
## runs identically under `godot --headless`.
class_name TestNavigation
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const NavigationAgent = preload("res://scripts/world/navigation_agent.gd")
const NavigationGraph = preload("res://scripts/world/navigation_graph.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")

const RADIUS: float = SandboxConfig.ENEMY_RADIUS
const HEIGHT: float = SandboxConfig.AGENT_HEIGHT


## A room split in two by a full-height wall with a single doorway gap.
## Walking straight from one side to the other is impossible; a path must
## route through the gap.
static func _walled_world(gap_center_z: float = 6.0) -> ArenaWorld:
	var world: ArenaWorld = ArenaWorld.create(10.0, 3.0)
	# Wall along x = 0 from z = -10 to z = gap_center_z - 1.
	var lower_length: float = (gap_center_z - 1.0) + 10.0
	world.add_box(
		Vector3(0.0, 1.5, -10.0 + lower_length * 0.5),
		Vector3(0.35, 1.5, lower_length * 0.5),
		Obstacle.Kind.WALL
	)
	# Wall along x = 0 from z = gap_center_z + 1 to z = 10.
	var upper_length: float = 10.0 - (gap_center_z + 1.0)
	if upper_length > 0.1:
		world.add_box(
			Vector3(0.0, 1.5, (gap_center_z + 1.0) + upper_length * 0.5),
			Vector3(0.35, 1.5, upper_length * 0.5),
			Obstacle.Kind.WALL
		)
	return world


## A world where a 3x3 box completely seals the corner around (-8, -8).
static func _sealed_corner_world() -> ArenaWorld:
	var world: ArenaWorld = ArenaWorld.create(10.0, 3.0)
	world.add_box(Vector3(-5.5, 1.5, -10.0), Vector3(0.35, 1.5, 4.5), Obstacle.Kind.WALL)
	world.add_box(Vector3(-10.0, 1.5, -5.5), Vector3(4.5, 1.5, 0.35), Obstacle.Kind.WALL)
	return world


func test_graph_bakes_walkable_nodes_from_geometry() -> SandboxTest:
	var t := SandboxTest.new("graph_bakes_walkable_nodes_from_geometry")
	var empty: ArenaWorld = ArenaWorld.create(10.0, 3.0)
	var open_graph: NavigationGraph = NavigationGraph.build(empty, RADIUS, HEIGHT)
	t.assert_true(open_graph.is_ready(), "an empty arena must still produce nodes")
	t.assert_gt(float(open_graph.node_count()), 100.0, "a 20x20 m arena should bake many cells")
	t.assert_eq(open_graph.component_count, 1, "an empty arena is one connected region")

	var blocked: ArenaWorld = _walled_world()
	var blocked_graph: NavigationGraph = NavigationGraph.build(blocked, RADIUS, HEIGHT)
	t.assert_lt(
		float(blocked_graph.node_count()),
		float(open_graph.node_count()),
		"a wall must remove walkable cells"
	)
	return t


func test_graph_generation_is_deterministic() -> SandboxTest:
	var t := SandboxTest.new("graph_generation_is_deterministic")
	for layout_value in WorldGenerator.LAYOUT_IDS:
		var layout_id: String = str(layout_value)
		var world_a: ArenaWorld = WorldGenerator.build(layout_id, 991)
		var world_b: ArenaWorld = WorldGenerator.build(layout_id, 991)
		var graph_a: NavigationGraph = NavigationGraph.build(world_a, RADIUS, HEIGHT)
		var graph_b: NavigationGraph = NavigationGraph.build(world_b, RADIUS, HEIGHT)
		t.assert_eq(graph_a.node_count(), graph_b.node_count(), "%s node count differs" % layout_id)
		t.assert_eq(
			graph_a.component_count,
			graph_b.component_count,
			"%s component count differs" % layout_id
		)
		for index in range(graph_a.node_count()):
			t.assert_vec_almost_eq(
				graph_a.node_positions[index],
				graph_b.node_positions[index],
				0.0001,
				"%s node %d differs" % [layout_id, index]
			)
	return t


func test_paths_are_deterministic_and_reach_the_goal() -> SandboxTest:
	var t := SandboxTest.new("paths_are_deterministic_and_reach_the_goal")
	var world: ArenaWorld = _walled_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	var start := Vector3(-6.0, 0.0, 0.0)
	var goal := Vector3(6.0, 0.0, 0.0)

	t.assert_true(graph.is_reachable(start, goal), "the doorway must connect both halves")
	var first: Array = graph.find_path(start, goal)
	t.assert_gt(float(first.size()), 1.0, "a detour through the doorway needs several waypoints")
	t.assert_vec_almost_eq(first[first.size() - 1], goal, 0.001, "path must end at the goal")

	var second: Array = graph.find_path(start, goal)
	t.assert_eq(first.size(), second.size(), "repeated planning must return the same path")
	for index in range(first.size()):
		t.assert_vec_almost_eq(first[index], second[index], 0.0001)

	# The straight line crosses the wall, so the route must not.
	t.assert_true(
		graph.direct_route_blocked(world, start, goal),
		"the straight line between the halves must be blocked"
	)
	var detoured: bool = false
	for waypoint_value in first:
		var waypoint: Vector3 = waypoint_value
		if absf(waypoint.z) > 2.0:
			detoured = true
	t.assert_true(detoured, "the path must travel toward the doorway rather than straight through")
	return t


func test_unreachable_destination_returns_no_path() -> SandboxTest:
	var t := SandboxTest.new("unreachable_destination_returns_no_path")
	var world: ArenaWorld = _sealed_corner_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	var inside := Vector3(-8.0, 0.0, -8.0)
	var outside := Vector3(6.0, 0.0, 6.0)
	t.assert_gt(float(graph.component_count), 1.0, "the sealed corner must be its own component")
	t.assert_false(graph.is_reachable(inside, outside), "a sealed corner must be unreachable")
	t.assert_eq(graph.find_path(inside, outside).size(), 0, "no path may be invented")
	# And the reverse direction must agree.
	t.assert_false(graph.is_reachable(outside, inside))
	return t


func test_obstacle_changes_change_the_graph() -> SandboxTest:
	var t := SandboxTest.new("obstacle_changes_change_the_graph")
	var open_world: ArenaWorld = _walled_world()
	var open_graph: NavigationGraph = NavigationGraph.build(open_world, RADIUS, HEIGHT)
	t.assert_true(open_graph.is_reachable(Vector3(-6.0, 0.0, 0.0), Vector3(6.0, 0.0, 0.0)))

	# Plug the doorway and re-bake: the two halves must now be separate.
	open_world.add_box(Vector3(0.0, 1.5, 6.0), Vector3(0.35, 1.5, 1.6), Obstacle.Kind.WALL)
	var closed_graph: NavigationGraph = NavigationGraph.build(open_world, RADIUS, HEIGHT)
	t.assert_false(
		closed_graph.is_reachable(Vector3(-6.0, 0.0, 0.0), Vector3(6.0, 0.0, 0.0)),
		"sealing the doorway must make the far half unreachable"
	)
	t.assert_gt(
		float(closed_graph.component_count),
		float(open_graph.component_count),
		"sealing a doorway must increase the component count"
	)
	return t


func test_recovery_direction_makes_progress_around_a_corner() -> SandboxTest:
	var t := SandboxTest.new("recovery_direction_makes_progress_around_a_corner")
	var world: ArenaWorld = _walled_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	# Standing against the wall, wanting to go straight through it.
	var pinned := Vector3(-1.2, 0.0, 0.0)
	var desired := Vector3(6.0, 0.0, 0.0)
	var direction: Vector3 = graph.recovery_direction(pinned, desired)
	t.assert_false(direction.is_zero_approx(), "a pinned character must get an escape direction")
	t.assert_almost_eq(direction.length(), 1.0, 0.001, "recovery direction must be normalized")
	var nudged: Vector3 = pinned + direction * graph.cell_size
	t.assert_true(
		world.is_position_free(
			Vector3(nudged.x, world.ground_height(nudged, RADIUS, 0.0), nudged.z), RADIUS, HEIGHT
		),
		"the recovery nudge must land somewhere walkable"
	)
	# Deterministic: the same query twice gives the same vector.
	t.assert_vec_almost_eq(direction, graph.recovery_direction(pinned, desired), 0.0001)
	return t


func test_steering_target_prefers_the_direct_line() -> SandboxTest:
	var t := SandboxTest.new("steering_target_prefers_the_direct_line")
	var world: ArenaWorld = _walled_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	# Same side of the wall, nothing in between: no detour.
	var from_position := Vector3(-6.0, 0.0, -4.0)
	var to_position := Vector3(-6.0, 0.0, 4.0)
	t.assert_vec_almost_eq(
		graph.steering_target(world, from_position, to_position),
		to_position,
		0.001,
		"an unobstructed route must be followed directly"
	)
	# Across the wall: the immediate steering target must differ from the goal.
	var across: Vector3 = graph.steering_target(
		world, Vector3(-6.0, 0.0, 0.0), Vector3(6.0, 0.0, 0.0)
	)
	t.assert_gt(
		across.distance_to(Vector3(6.0, 0.0, 0.0)),
		0.5,
		"a blocked route must steer at a waypoint, not the goal"
	)
	return t


func test_navigation_agent_stays_direct_until_it_is_actually_stuck() -> SandboxTest:
	var t := SandboxTest.new("navigation_agent_stays_direct_until_it_is_actually_stuck")
	var world: ArenaWorld = _walled_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	var agent: NavigationAgent = NavigationAgent.create()
	var position := Vector3(-6.0, 0.0, 0.0)
	var goal := Vector3(6.0, 0.0, 0.0)
	agent.reset(position)

	# Moving freely: navigation must not engage, so the goal passes through.
	for step in range(10):
		var target: Vector3 = agent.update(graph, world, position, goal, true, 0.1)
		t.assert_vec_almost_eq(target, goal, 0.001, "step %d should still steer directly" % step)
		t.assert_eq(agent.status, NavigationAgent.STATUS_DIRECT)
		position += Vector3(0.4, 0.0, 0.0)
	return t


func test_navigation_agent_engages_when_blocked() -> SandboxTest:
	var t := SandboxTest.new("navigation_agent_engages_when_blocked")
	var world: ArenaWorld = _walled_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	var agent: NavigationAgent = NavigationAgent.create()
	var pinned := Vector3(-1.0, 0.0, 0.0)
	var goal := Vector3(6.0, 0.0, 0.0)
	agent.reset(pinned)

	var engaged: bool = false
	var target: Vector3 = pinned
	# The character asks to move but never actually moves: after
	# NAV_STUCK_TIME the agent must stop returning the raw goal.
	for _step in range(20):
		target = agent.update(graph, world, pinned, goal, true, 0.1)
		if agent.status != NavigationAgent.STATUS_DIRECT:
			engaged = true
			break
	t.assert_true(engaged, "a character that cannot move must engage navigation")
	t.assert_gt(agent.stuck_time, 0.0, "stuck time must accumulate while blocked")
	t.assert_gt(
		target.distance_to(goal), 0.5, "an engaged agent must steer somewhere other than the goal"
	)
	return t


func test_navigation_agent_reports_unreachable_without_inventing_a_path() -> SandboxTest:
	var t := SandboxTest.new("navigation_agent_reports_unreachable_without_inventing_a_path")
	var world: ArenaWorld = _sealed_corner_world()
	var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
	var agent: NavigationAgent = NavigationAgent.create()
	var inside := Vector3(-8.0, 0.0, -8.0)
	var goal := Vector3(6.0, 0.0, 6.0)
	agent.reset(inside)
	for _step in range(20):
		agent.update(graph, world, inside, goal, true, 0.1)
	t.assert_gt(float(agent.failures), 0.0, "an unreachable goal must be counted as a failure")
	t.assert_eq(agent.path.size(), 0, "no path may be fabricated for an unreachable goal")
	return t


func test_navigation_agent_without_a_graph_is_a_no_op() -> SandboxTest:
	var t := SandboxTest.new("navigation_agent_without_a_graph_is_a_no_op")
	var agent: NavigationAgent = NavigationAgent.create()
	var position := Vector3(1.0, 0.0, 2.0)
	var goal := Vector3(5.0, 0.0, -3.0)
	agent.reset(position)
	for _step in range(5):
		var target: Vector3 = agent.update(null, null, position, goal, true, 0.1)
		t.assert_vec_almost_eq(target, goal, 0.001, "no graph must mean unchanged steering")
	t.assert_eq(agent.status, NavigationAgent.STATUS_DIRECT)
	t.assert_eq(agent.failures, 0)
	return t


func test_graph_handles_every_generated_layout_headlessly() -> SandboxTest:
	var t := SandboxTest.new("graph_handles_every_generated_layout_headlessly")
	for layout_value in WorldGenerator.LAYOUT_IDS:
		var layout_id: String = str(layout_value)
		for seed_value in [1, 77, 4096]:
			var world: ArenaWorld = WorldGenerator.build(layout_id, int(seed_value))
			var graph: NavigationGraph = NavigationGraph.build(world, RADIUS, HEIGHT)
			t.assert_true(
				graph.is_ready(), "%s@%d produced an empty graph" % [layout_id, seed_value]
			)
			# Every baked node must itself be a legal standing position.
			for index in range(graph.node_count()):
				var node: Vector3 = graph.node_positions[index]
				if not world.is_position_free(node, RADIUS, HEIGHT):
					t.fail("%s@%d node %d is inside geometry" % [layout_id, seed_value, index])
					break
	return t

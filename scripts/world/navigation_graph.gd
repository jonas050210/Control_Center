## NavigationGraph
##
## A deterministic, allocation-light navigation graph derived from the
## walkable surface of an `ArenaWorld`. It exists because the enemies used
## to steer straight at their destination and slide along walls, which made
## them get permanently pinned in an inside corner of a corridor or room
## layout (see the "no navmesh, no path planner" entry that used to be in
## docs/ARCHITECTURE.md).
##
## Design constraints, in priority order:
##
## 1. **Deterministic.** The graph is a pure function of the world geometry
##    and the character footprint. No RandomNumberGenerator is consulted, no
##    dictionary iteration order is relied on, and A* ties break on the
##    node index. Two graphs built from the same `WorldGenerator.build(id,
##    seed)` world are identical edge-for-edge, which is what makes an
##    episode replayable from its seed.
## 2. **Cheap headless.** It is a flat uniform grid, not a navmesh: building
##    it is O(cells * obstacles) with no recursion and no scene tree, and it
##    is built LAZILY (see `EnvironmentCore._ensure_navigation()`) so the
##    obstacle-free curriculum levels never pay for it at all.
## 3. **No new dependencies.** Godot's own NavigationServer3D would drag in
##    the physics/navigation servers, a navigation map per environment and
##    asynchronous baking — none of which is acceptable for dozens of
##    parallel headless environments.
##
## Representation: one node per walkable grid cell, positioned at the cell
## center and raised to the highest standable surface under it, so a
## platform in a "vertical" layout becomes its own set of nodes. Edges are
## 8-connected, rejected when the height difference exceeds the world's step
## tolerance, when the midpoint is solid, or (for diagonals) when either
## orthogonal neighbour is solid — that last rule is what stops a path from
## squeezing through the diagonal gap between two wall corners.
##
## Connectivity is precomputed as component ids by a flood fill, so
## `is_reachable()` answers "can I get there at all" in O(1) instead of
## running a doomed A* every tick against an enemy standing in a sealed
## room.
class_name NavigationGraph
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SELF_PATH: String = "res://scripts/world/navigation_graph.gd"

## Returned by `nearest_node()` / `find_path()` when nothing is usable.
const INVALID_NODE: int = -1

## Neighbour offsets, orthogonals first so a straight edge is always
## discovered before the diagonal that shortcuts it. Order is fixed (not
## derived from a Dictionary) to keep expansion order deterministic.
const NEIGHBOR_OFFSETS: Array = [
	Vector2i(1, 0),
	Vector2i(-1, 0),
	Vector2i(0, 1),
	Vector2i(0, -1),
	Vector2i(1, 1),
	Vector2i(1, -1),
	Vector2i(-1, 1),
	Vector2i(-1, -1),
]

## Grid geometry.
var cell_size: float = SandboxConfig.NAV_CELL_SIZE
var origin_x: float = 0.0
var origin_z: float = 0.0
var columns: int = 0
var rows: int = 0

## Character footprint the graph was baked for. A graph baked for the agent
## radius is NOT valid for a wider enemy, so callers must not share one.
var agent_radius: float = SandboxConfig.ENEMY_RADIUS
var agent_height: float = SandboxConfig.AGENT_HEIGHT

## node index -> world position (feet, on the standable surface).
var node_positions: Array = []  # Array[Vector3]
## grid cell (row * columns + column) -> node index, or INVALID_NODE.
var cell_to_node: PackedInt32Array = PackedInt32Array()
## node index -> grid cell index. Inverse of `cell_to_node`.
var node_cells: PackedInt32Array = PackedInt32Array()
## node index -> PackedInt32Array of neighbour node indices.
var neighbors: Array = []
## node index -> connected component id.
var components: PackedInt32Array = PackedInt32Array()
var component_count: int = 0

## Provenance, for telemetry / the Control Center only. Never observed.
var layout_id: String = "none"
var layout_seed: int = -1

## Scratch buffers reused by `find_path()`. Allocated once at bake time so
## the hot path (an enemy re-planning every ~0.5 s) allocates only the
## returned waypoint array.
var _g_score: PackedFloat32Array = PackedFloat32Array()
var _f_score: PackedFloat32Array = PackedFloat32Array()
var _came_from: PackedInt32Array = PackedInt32Array()
var _visit_stamp: PackedInt32Array = PackedInt32Array()
var _closed_stamp: PackedInt32Array = PackedInt32Array()
var _search_id: int = 0
var _heap: PackedInt32Array = PackedInt32Array()
var _heap_size: int = 0


## Bakes a graph for `world` and a character of the given footprint.
##
## Passing a null world yields an empty graph rather than failing, so a
## caller on an obstacle-free curriculum level can hold a graph reference
## unconditionally and simply get `node_count() == 0`.
static func build(
	world,
	p_radius: float = SandboxConfig.ENEMY_RADIUS,
	p_height: float = SandboxConfig.AGENT_HEIGHT,
	p_cell_size: float = SandboxConfig.NAV_CELL_SIZE
) -> NavigationGraph:
	var graph: NavigationGraph = (load(SELF_PATH) as GDScript).new() as NavigationGraph
	graph.agent_radius = maxf(0.05, p_radius)
	graph.agent_height = maxf(0.2, p_height)
	graph.cell_size = maxf(0.25, p_cell_size)
	if world == null:
		return graph
	graph._bake(world as ArenaWorld)
	return graph


func node_count() -> int:
	return node_positions.size()


func is_ready() -> bool:
	return node_positions.size() > 0


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------


func _bake(world: ArenaWorld) -> void:
	layout_id = world.layout_id
	layout_seed = world.layout_seed
	var usable: float = maxf(1.0, world.half_extent - agent_radius)
	columns = maxi(2, int(floor((usable * 2.0) / cell_size)))
	rows = columns
	# Center the grid: leftover space is split evenly so the layout is
	# symmetric and independent of floating point accumulation order.
	var span: float = float(columns) * cell_size
	origin_x = -span * 0.5
	origin_z = -span * 0.5

	var cell_total: int = columns * rows
	cell_to_node = PackedInt32Array()
	cell_to_node.resize(cell_total)
	for index in range(cell_total):
		cell_to_node[index] = INVALID_NODE

	node_positions = []
	node_cells = PackedInt32Array()
	for row in range(rows):
		for column in range(columns):
			var candidate: Vector3 = _cell_center(column, row)
			candidate.y = world.ground_height(candidate, agent_radius, 0.0)
			if not world.is_position_free(candidate, agent_radius, agent_height):
				continue
			var node_index: int = node_positions.size()
			node_positions.append(candidate)
			node_cells.append(row * columns + column)
			cell_to_node[row * columns + column] = node_index

	_bake_edges(world)
	_bake_components()
	_allocate_scratch()


func _bake_edges(world: ArenaWorld) -> void:
	neighbors = []
	neighbors.resize(node_positions.size())
	for node_index in range(node_positions.size()):
		var cell: int = node_cells[node_index]
		var column: int = cell % columns
		var row: int = cell / columns
		var list: PackedInt32Array = PackedInt32Array()
		for offset_value in NEIGHBOR_OFFSETS:
			var offset: Vector2i = offset_value
			var neighbor_index: int = _node_at(column + offset.x, row + offset.y)
			if neighbor_index == INVALID_NODE:
				continue
			if not _edge_is_walkable(world, node_index, neighbor_index, column, row, offset):
				continue
			list.append(neighbor_index)
		neighbors[node_index] = list


func _edge_is_walkable(
	world: ArenaWorld, from_node: int, to_node: int, column: int, row: int, offset: Vector2i
) -> bool:
	var from_position: Vector3 = node_positions[from_node]
	var to_position: Vector3 = node_positions[to_node]
	if absf(to_position.y - from_position.y) > ArenaWorld.STEP_UP_TOLERANCE:
		return false
	if offset.x != 0 and offset.y != 0:
		# Diagonal: both orthogonal cells must exist, otherwise the path
		# would cut the corner of a wall the character cannot pass.
		if _node_at(column + offset.x, row) == INVALID_NODE:
			return false
		if _node_at(column, row + offset.y) == INVALID_NODE:
			return false
	var midpoint: Vector3 = (from_position + to_position) * 0.5
	midpoint.y = maxf(from_position.y, to_position.y)
	return world.is_position_free(midpoint, agent_radius, agent_height)


## Flood fill over the undirected edge set. Uses an explicit stack (no
## recursion) and iterates node indices in order, so component ids are
## stable for a given world.
func _bake_components() -> void:
	components = PackedInt32Array()
	components.resize(node_positions.size())
	for index in range(components.size()):
		components[index] = INVALID_NODE
	component_count = 0
	var stack: PackedInt32Array = PackedInt32Array()
	for start in range(node_positions.size()):
		if components[start] != INVALID_NODE:
			continue
		var component: int = component_count
		component_count += 1
		stack.clear()
		stack.append(start)
		components[start] = component
		while stack.size() > 0:
			var current: int = stack[stack.size() - 1]
			stack.remove_at(stack.size() - 1)
			for neighbor_value in (neighbors[current] as PackedInt32Array):
				var neighbor: int = neighbor_value
				if components[neighbor] == INVALID_NODE:
					components[neighbor] = component
					stack.append(neighbor)


func _allocate_scratch() -> void:
	var count: int = node_positions.size()
	_g_score = PackedFloat32Array()
	_g_score.resize(count)
	_f_score = PackedFloat32Array()
	_f_score.resize(count)
	_came_from = PackedInt32Array()
	_came_from.resize(count)
	_visit_stamp = PackedInt32Array()
	_visit_stamp.resize(count)
	_closed_stamp = PackedInt32Array()
	_closed_stamp.resize(count)
	for index in range(count):
		_visit_stamp[index] = 0
		_closed_stamp[index] = 0
	_search_id = 0


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


func _cell_center(column: int, row: int) -> Vector3:
	return Vector3(
		origin_x + (float(column) + 0.5) * cell_size,
		0.0,
		origin_z + (float(row) + 0.5) * cell_size
	)


func _node_at(column: int, row: int) -> int:
	if column < 0 or row < 0 or column >= columns or row >= rows:
		return INVALID_NODE
	return cell_to_node[row * columns + column]


func _column_of(position: Vector3) -> int:
	return int(floor((position.x - origin_x) / cell_size))


func _row_of(position: Vector3) -> int:
	return int(floor((position.z - origin_z) / cell_size))


## Nearest walkable node to `position`, searched as expanding square rings
## so the result is the closest in Chebyshev order and ties break on the
## lower node index. Returns INVALID_NODE for an empty graph or when
## nothing walkable exists within `max_rings` cells.
func nearest_node(position: Vector3, max_rings: int = 6) -> int:
	if node_positions.is_empty():
		return INVALID_NODE
	var column: int = _column_of(position)
	var row: int = _row_of(position)
	var direct: int = _node_at(column, row)
	if direct != INVALID_NODE:
		return direct
	for ring in range(1, maxi(1, max_rings) + 1):
		var best: int = INVALID_NODE
		var best_distance: float = INF
		for d in range(-ring, ring + 1):
			var candidates: Array = [
				Vector2i(column + d, row - ring),
				Vector2i(column + d, row + ring),
				Vector2i(column - ring, row + d),
				Vector2i(column + ring, row + d),
			]
			for candidate_value in candidates:
				var candidate: Vector2i = candidate_value
				var index: int = _node_at(candidate.x, candidate.y)
				if index == INVALID_NODE:
					continue
				var distance: float = _flat_distance(node_positions[index], position)
				if distance < best_distance - 0.0001:
					best_distance = distance
					best = index
		if best != INVALID_NODE:
			return best
	return INVALID_NODE


static func _flat_distance(a: Vector3, b: Vector3) -> float:
	return Vector2(a.x - b.x, a.z - b.z).length()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


## True when a walkable route exists between the two positions. O(1) after
## baking (component comparison), so it is safe to call every tick.
func is_reachable(from_position: Vector3, to_position: Vector3) -> bool:
	var start: int = nearest_node(from_position)
	var goal: int = nearest_node(to_position)
	if start == INVALID_NODE or goal == INVALID_NODE:
		return false
	return components[start] == components[goal]


## A* between two world positions. Returns the waypoint list EXCLUDING the
## start position and ending with `to_position` itself, or an empty array
## when no route exists.
##
## Ties in the open set break on the lower node index, so the returned path
## is identical for identical inputs on every run and on every platform.
func find_path(from_position: Vector3, to_position: Vector3) -> Array:
	var result: Array = []
	var start: int = nearest_node(from_position)
	var goal: int = nearest_node(to_position)
	if start == INVALID_NODE or goal == INVALID_NODE:
		return result
	if components[start] != components[goal]:
		return result
	if start == goal:
		result.append(to_position)
		return result

	_search_id += 1
	var stamp: int = _search_id
	# The heap buffer is reused across searches: only the logical size is
	# reset, so a long training run stops reallocating after the first few
	# plans.
	_heap_size = 0
	_g_score[start] = 0.0
	_f_score[start] = _flat_distance(node_positions[start], node_positions[goal])
	_came_from[start] = INVALID_NODE
	_visit_stamp[start] = stamp
	_heap_push(start)

	var found: bool = false
	while _heap_size > 0:
		var current: int = _heap_pop()
		if _closed_stamp[current] == stamp:
			continue
		_closed_stamp[current] = stamp
		if current == goal:
			found = true
			break
		var current_g: float = _g_score[current]
		for neighbor_value in (neighbors[current] as PackedInt32Array):
			var neighbor: int = neighbor_value
			if _closed_stamp[neighbor] == stamp:
				continue
			var step_cost: float = _flat_distance(
				node_positions[current], node_positions[neighbor]
			)
			var tentative: float = current_g + step_cost
			if _visit_stamp[neighbor] == stamp and tentative >= _g_score[neighbor] - 0.00001:
				continue
			_visit_stamp[neighbor] = stamp
			_g_score[neighbor] = tentative
			_f_score[neighbor] = tentative + _flat_distance(
				node_positions[neighbor], node_positions[goal]
			)
			_came_from[neighbor] = current
			_heap_push(neighbor)

	if not found:
		return result

	var reversed: Array = []
	var cursor: int = goal
	var guard: int = 0
	var limit: int = node_positions.size() + 2
	while cursor != INVALID_NODE and guard < limit:
		reversed.append(node_positions[cursor])
		cursor = _came_from[cursor]
		guard += 1
	reversed.reverse()
	# Drop the start node: the caller is already standing there, and
	# steering back to its cell center would look like a stutter.
	for index in range(1, reversed.size()):
		result.append(reversed[index])
	result.append(to_position)
	return result


## The point a mover at `from_position` should steer at right now in order
## to eventually reach `to_position`.
##
## When the straight line is walkable this returns `to_position` unchanged,
## which keeps the cheap direct-steering behavior for the common open-field
## case. Only when the direct line is blocked does it plan and return the
## first meaningful waypoint. Returns `from_position` when the destination
## is unreachable, which the caller can detect with `==`.
func steering_target(world, from_position: Vector3, to_position: Vector3) -> Vector3:
	if world != null and not direct_route_blocked(world as ArenaWorld, from_position, to_position):
		return to_position
	var path: Array = find_path(from_position, to_position)
	if path.is_empty():
		return from_position
	return first_meaningful_waypoint(path, from_position)


## First waypoint at least one cell away from the mover, so a character
## standing almost on top of waypoint 0 does not oscillate between cells.
func first_meaningful_waypoint(path: Array, from_position: Vector3) -> Vector3:
	var tolerance: float = cell_size * 0.5
	for waypoint_value in path:
		var waypoint: Vector3 = waypoint_value
		if _flat_distance(waypoint, from_position) > tolerance:
			return waypoint
	return path[path.size() - 1]


## Whether walking straight at the destination would collide. Sampled along
## the segment at half-cell resolution — cheaper and more forgiving than a
## swept volume, and it only has to be good enough to decide "plan a path".
func direct_route_blocked(world: ArenaWorld, from_position: Vector3, to_position: Vector3) -> bool:
	var delta := Vector3(to_position.x - from_position.x, 0.0, to_position.z - from_position.z)
	var distance: float = delta.length()
	if distance < 0.0001:
		return false
	var steps: int = maxi(1, int(ceil(distance / maxf(cell_size * 0.5, 0.1))))
	var direction: Vector3 = delta / distance
	for step in range(1, steps + 1):
		var travelled: float = minf(distance, float(step) * (distance / float(steps)))
		var probe: Vector3 = from_position + direction * travelled
		probe.y = world.ground_height(probe, agent_radius, from_position.y)
		if not world.is_position_free(probe, agent_radius, agent_height):
			return true
	return false


## Deterministic escape direction for a character wedged against geometry.
##
## Tries every neighbour of the node it is standing on and picks the one
## that both is walkable and makes the most progress toward `desired`,
## strictly preferring a node the mover is not already on. Returns
## `Vector3.ZERO` when even that fails (a fully sealed cell), which tells
## the caller to fall back on its own behavior rather than freezing.
func recovery_direction(from_position: Vector3, desired: Vector3) -> Vector3:
	var start: int = nearest_node(from_position)
	if start == INVALID_NODE:
		return Vector3.ZERO
	var goal_distance: float = _flat_distance(node_positions[start], desired)
	var best: Vector3 = Vector3.ZERO
	var best_score: float = -INF
	for neighbor_value in (neighbors[start] as PackedInt32Array):
		var neighbor: int = neighbor_value
		var candidate: Vector3 = node_positions[neighbor]
		var progress: float = goal_distance - _flat_distance(candidate, desired)
		if progress > best_score:
			best_score = progress
			best = candidate
	if best == Vector3.ZERO and best_score == -INF:
		return Vector3.ZERO
	var direction := Vector3(best.x - from_position.x, 0.0, best.z - from_position.z)
	if direction.is_zero_approx():
		return Vector3.ZERO
	return direction.normalized()


# ---------------------------------------------------------------------------
# Binary heap (min on _f_score, tie-break on node index)
# ---------------------------------------------------------------------------


func _heap_push(node_index: int) -> void:
	if _heap_size < _heap.size():
		_heap[_heap_size] = node_index
	else:
		_heap.append(node_index)
	_heap_size += 1
	var child: int = _heap_size - 1
	while child > 0:
		var parent: int = (child - 1) / 2
		if _heap_less(_heap[child], _heap[parent]):
			var swap: int = _heap[child]
			_heap[child] = _heap[parent]
			_heap[parent] = swap
			child = parent
		else:
			break


func _heap_pop() -> int:
	var top: int = _heap[0]
	_heap_size -= 1
	_heap[0] = _heap[_heap_size]
	var parent: int = 0
	while true:
		var left: int = parent * 2 + 1
		var right: int = left + 1
		var smallest: int = parent
		if left < _heap_size and _heap_less(_heap[left], _heap[smallest]):
			smallest = left
		if right < _heap_size and _heap_less(_heap[right], _heap[smallest]):
			smallest = right
		if smallest == parent:
			break
		var swap: int = _heap[parent]
		_heap[parent] = _heap[smallest]
		_heap[smallest] = swap
		parent = smallest
	return top


func _heap_less(a: int, b: int) -> bool:
	var fa: float = _f_score[a]
	var fb: float = _f_score[b]
	if absf(fa - fb) > 0.000001:
		return fa < fb
	return a < b


func to_dict() -> Dictionary:
	return {
		"layout_id": layout_id,
		"layout_seed": layout_seed,
		"cell_size": cell_size,
		"columns": columns,
		"rows": rows,
		"node_count": node_positions.size(),
		"component_count": component_count,
		"agent_radius": agent_radius,
	}

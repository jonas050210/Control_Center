## SpatialMemory
##
## The agent's long-term, structured memory OF THE MAP, as opposed to
## `EnemyMemory`, which is short-term memory of individual contacts.
##
## Hard rule, enforced by `tests/test_spatial_memory.gd`: a cell only ever
## becomes "known" because the agent actually observed it. There is no bulk
## import of the world geometry, no privileged seeding at reset, and
## `observe_cells()` takes the same field-of-view / line-of-sight / lighting
## queries the observation vector is built from. An unvisited room stays
## `unknown` even though the simulation obviously knows what is in it.
##
## Representation: a coarse uniform grid (one cell is roughly a corner of a
## room, not a texel) with per-cell parallel arrays. Parallel `Packed*Array`
## columns rather than an Array of Dictionaries because the update runs
## inside the simulation loop and must not allocate per cell.
##
## Per cell the agent remembers only things it could have perceived:
##   * whether it has ever been seen, and when
##   * whether it has been stood in, how often, and when
##   * how bright it looked
##   * how enclosed it looked (cover) and how far you can see from it
##     (openness)
##   * whether something dangerous happened there
##   * a confidence that decays with age, so old beliefs become uncertain
##     instead of silently staying authoritative
class_name SpatialMemory
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SELF_PATH: String = "res://scripts/exploration/spatial_memory.gd"

## Sentinel for "this cell has never been observed/visited".
const NEVER: float = -1.0

var cell_size: float = SandboxConfig.EXPLORATION_CELL_SIZE
var half_extent: float = SandboxConfig.ARENA_HALF_EXTENT
var columns: int = 0
var rows: int = 0
## Episode-local clock, advanced by `tick()`.
var time_seconds: float = 0.0

## Cells the agent could in principle ever observe (inside the arena).
var cell_count: int = 0
var observed_time: PackedFloat32Array = PackedFloat32Array()
var visited_time: PackedFloat32Array = PackedFloat32Array()
var visit_count: PackedInt32Array = PackedInt32Array()
var illumination: PackedFloat32Array = PackedFloat32Array()
var cover_score: PackedFloat32Array = PackedFloat32Array()
var openness: PackedFloat32Array = PackedFloat32Array()
var danger: PackedFloat32Array = PackedFloat32Array()
var confidence: PackedFloat32Array = PackedFloat32Array()

## Ordered list of distinct cells the agent has stood in, i.e. the route it
## actually walked. Capped so a long episode cannot grow without bound.
var route: PackedInt32Array = PackedInt32Array()


static func create(
	p_half_extent: float = SandboxConfig.ARENA_HALF_EXTENT,
	p_cell_size: float = SandboxConfig.EXPLORATION_CELL_SIZE
) -> SpatialMemory:
	var memory: SpatialMemory = (load(SELF_PATH) as GDScript).new() as SpatialMemory
	memory.configure(p_half_extent, p_cell_size)
	return memory


func configure(p_half_extent: float, p_cell_size: float) -> void:
	half_extent = maxf(2.0, p_half_extent)
	cell_size = maxf(0.5, p_cell_size)
	columns = maxi(2, int(ceil((half_extent * 2.0) / cell_size)))
	rows = columns
	cell_count = columns * rows
	observed_time = _new_float_column(NEVER)
	visited_time = _new_float_column(NEVER)
	illumination = _new_float_column(0.0)
	cover_score = _new_float_column(0.0)
	openness = _new_float_column(0.0)
	danger = _new_float_column(0.0)
	confidence = _new_float_column(0.0)
	visit_count = PackedInt32Array()
	visit_count.resize(cell_count)
	for index in range(cell_count):
		visit_count[index] = 0
	route = PackedInt32Array()
	time_seconds = 0.0


func _new_float_column(fill: float) -> PackedFloat32Array:
	var column: PackedFloat32Array = PackedFloat32Array()
	column.resize(cell_count)
	for index in range(cell_count):
		column[index] = fill
	return column


## Wipes every belief. Called on `reset()`: episode memory must not leak
## across episodes, which `tests/test_spatial_memory.gd` asserts directly.
func clear() -> void:
	configure(half_extent, cell_size)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


func cell_index(position: Vector3) -> int:
	var column: int = int(floor((position.x + half_extent) / cell_size))
	var row: int = int(floor((position.z + half_extent) / cell_size))
	if column < 0 or row < 0 or column >= columns or row >= rows:
		return -1
	return row * columns + column


func cell_center(index: int) -> Vector3:
	if index < 0 or index >= cell_count:
		return Vector3.ZERO
	var column: int = index % columns
	var row: int = int(floor(float(index) / float(columns)))
	return Vector3(
		-half_extent + (float(column) + 0.5) * cell_size,
		0.0,
		-half_extent + (float(row) + 0.5) * cell_size
	)


# ---------------------------------------------------------------------------
# Updating (perception-driven only)
# ---------------------------------------------------------------------------


## Advances the episode clock and decays confidence.
##
## Confidence uses the same exponential half-life shape as `EnemyMemory`,
## but a much longer one: terrain does not move, so an old observation of a
## corridor stays more useful than an old observation of a person. It still
## decays, because the agent cannot know whether something changed there.
func tick(dt: float) -> void:
	time_seconds += dt
	if dt <= 0.0:
		return
	var factor: float = pow(0.5, dt / maxf(SandboxConfig.EXPLORATION_HALF_LIFE, 0.0001))
	for index in range(cell_count):
		if observed_time[index] == NEVER:
			continue
		confidence[index] = maxf(
			SandboxConfig.EXPLORATION_MIN_CONFIDENCE, confidence[index] * factor
		)


## Records that the agent is standing here right now.
func observe_self(position: Vector3, local_illumination: float) -> void:
	var index: int = cell_index(position)
	if index < 0:
		return
	if visited_time[index] == NEVER:
		if route.size() < SandboxConfig.EXPLORATION_MAX_ROUTE:
			route.append(index)
	elif route.size() == 0 or route[route.size() - 1] != index:
		if route.size() < SandboxConfig.EXPLORATION_MAX_ROUTE:
			route.append(index)
	visited_time[index] = time_seconds
	visit_count[index] += 1
	_mark_observed(index, local_illumination)


## Sweeps the cells the agent can actually see and marks those as known.
##
## `lighting` may be null (uniform daylight). The sweep is bounded to the
## cells within the effective vision range, and each candidate must pass
## the same FOV + LOS test the observation vector uses, so a cell behind a
## wall is not learned by walking past the wall.
func observe_cells(
	eye: Vector3,
	feet: Vector3,
	forward: Vector3,
	world,
	lighting,
	fov_deg: float,
	base_range: float
) -> int:
	var learned: int = 0
	var reach: int = maxi(1, int(ceil(base_range / cell_size)))
	var center: int = cell_index(feet)
	if center < 0:
		return 0
	var center_column: int = center % columns
	var center_row: int = int(floor(float(center) / float(columns)))
	for row_offset in range(-reach, reach + 1):
		for column_offset in range(-reach, reach + 1):
			var column: int = center_column + column_offset
			var row: int = center_row + row_offset
			if column < 0 or row < 0 or column >= columns or row >= rows:
				continue
			var index: int = row * columns + column
			var point: Vector3 = cell_center(index)
			if not _can_see_cell(eye, feet, forward, point, world, lighting, fov_deg, base_range):
				continue
			var was_unknown: bool = observed_time[index] == NEVER
			var light: float = 1.0
			if lighting != null:
				light = lighting.illumination_at(point)
			_mark_observed(index, light)
			_measure_cell(index, point, world, base_range)
			if was_unknown:
				learned += 1
	return learned


func _can_see_cell(
	eye: Vector3,
	feet: Vector3,
	forward: Vector3,
	point: Vector3,
	world,
	lighting,
	fov_deg: float,
	base_range: float
) -> bool:
	var distance: float = Vector2(point.x - feet.x, point.z - feet.z).length()
	var reach: float = base_range
	if lighting != null:
		reach = lighting.detection_range(base_range, point)
	if distance > reach:
		return false
	if distance > 0.001 and not PerceptionSystem.in_field_of_view(forward, feet, point, fov_deg):
		return false
	if world == null:
		return true
	# Sample at knee height: a cell behind low cover is still "seen", a cell
	# behind a full wall is not.
	return not (world as ArenaWorld).segment_blocked(eye, point + Vector3(0.0, 0.5, 0.0))


func _mark_observed(index: int, light: float) -> void:
	observed_time[index] = time_seconds
	confidence[index] = 1.0
	illumination[index] = light


## Derives the two structural properties the agent could genuinely judge by
## looking: how much cover is nearby, and how far it can see from there.
func _measure_cell(index: int, point: Vector3, world, base_range: float) -> void:
	if world == null:
		cover_score[index] = 0.0
		openness[index] = 1.0
		return
	var arena: ArenaWorld = world
	var nearest: Dictionary = arena.nearest_obstacle_info(
		point + Vector3(0.0, 0.5, 0.0), Vector3.FORWARD, base_range
	)
	var distance: float = float(nearest["distance"])
	cover_score[index] = clampf(1.0 - distance / maxf(cell_size * 2.0, 0.001), 0.0, 1.0)
	# Openness: the mean unobstructed distance along four fixed bearings.
	var total: float = 0.0
	var directions: Array = [Vector3.FORWARD, Vector3.BACK, Vector3.LEFT, Vector3.RIGHT]
	for direction_value in directions:
		var hit: float = arena.ray_hit_distance(
			point + Vector3(0.0, 1.2, 0.0), direction_value, base_range
		)
		total += base_range if hit < 0.0 else hit
	openness[index] = clampf(total / (4.0 * maxf(base_range, 0.001)), 0.0, 1.0)


## Records that something dangerous happened at a position (the agent took
## damage, or was shot at from there). Only ever called with positions the
## agent itself was involved with.
func mark_danger(position: Vector3, amount: float = 1.0) -> void:
	var index: int = cell_index(position)
	if index < 0:
		return
	danger[index] = clampf(danger[index] + maxf(0.0, amount), 0.0, 1.0)
	_mark_observed(index, illumination[index])


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


func is_known(position: Vector3) -> bool:
	var index: int = cell_index(position)
	return index >= 0 and observed_time[index] != NEVER


func is_visited(position: Vector3) -> bool:
	var index: int = cell_index(position)
	return index >= 0 and visited_time[index] != NEVER


func known_cell_count() -> int:
	var count: int = 0
	for index in range(cell_count):
		if observed_time[index] != NEVER:
			count += 1
	return count


func visited_cell_count() -> int:
	var count: int = 0
	for index in range(cell_count):
		if visited_time[index] != NEVER:
			count += 1
	return count


## Fraction of the arena the agent has ever observed, in [0, 1].
func coverage_fraction() -> float:
	if cell_count <= 0:
		return 0.0
	return float(known_cell_count()) / float(cell_count)


## Mean (1 - confidence) over the cells the agent believes it knows. 0 when
## nothing is known yet, which reads as "no stale beliefs", not "certain".
func mean_uncertainty() -> float:
	var total: float = 0.0
	var count: int = 0
	for index in range(cell_count):
		if observed_time[index] == NEVER:
			continue
		total += 1.0 - confidence[index]
		count += 1
	if count == 0:
		return 0.0
	return total / float(count)


## Seconds since the agent last stood in this position's cell, or -1 when
## it never has.
func time_since_visit(position: Vector3) -> float:
	var index: int = cell_index(position)
	if index < 0 or visited_time[index] == NEVER:
		return NEVER
	return time_seconds - visited_time[index]


## Nearest remembered cell whose observed cover score exceeds
## `min_score`. Returns {} when nothing qualifying is remembered.
func nearest_remembered_cover(from_position: Vector3, min_score: float = 0.5) -> Dictionary:
	return _nearest_matching(from_position, cover_score, min_score)


## Nearest remembered cell where something dangerous happened.
func nearest_remembered_danger(from_position: Vector3, min_score: float = 0.2) -> Dictionary:
	return _nearest_matching(from_position, danger, min_score)


## Nearest cell that has never been observed. This is the exploration
## frontier and the one thing an explorer genuinely wants to know.
func nearest_unknown(from_position: Vector3) -> Dictionary:
	var best: int = -1
	var best_distance: float = INF
	for index in range(cell_count):
		if observed_time[index] != NEVER:
			continue
		var distance: float = (
			Vector2(cell_center(index).x - from_position.x, cell_center(index).z - from_position.z)
			. length()
		)
		if distance < best_distance - 0.0001:
			best_distance = distance
			best = index
	if best < 0:
		return {}
	return {"index": best, "position": cell_center(best), "distance": best_distance}


func _nearest_matching(
	from_position: Vector3, column: PackedFloat32Array, minimum: float
) -> Dictionary:
	var best: int = -1
	var best_distance: float = INF
	for index in range(cell_count):
		if observed_time[index] == NEVER or column[index] < minimum:
			continue
		var center: Vector3 = cell_center(index)
		var distance: float = (
			Vector2(center.x - from_position.x, center.z - from_position.z).length()
		)
		if distance < best_distance - 0.0001:
			best_distance = distance
			best = index
	if best < 0:
		return {}
	return {
		"index": best,
		"position": cell_center(best),
		"distance": best_distance,
		"score": column[best],
		"confidence": confidence[best],
	}


## Full dump for the Control Center Map Analyzer view. Presentation only:
## nothing in the simulation reads this, and it is never fed to a policy.
func to_dict() -> Dictionary:
	var cells: Array = []
	for index in range(cell_count):
		if observed_time[index] == NEVER:
			continue
		(
			cells
			. append(
				{
					"index": index,
					"position": cell_center(index),
					"observed_age": time_seconds - observed_time[index],
					"visited": visited_time[index] != NEVER,
					"visit_count": visit_count[index],
					"illumination": illumination[index],
					"cover": cover_score[index],
					"openness": openness[index],
					"danger": danger[index],
					"confidence": confidence[index],
				}
			)
		)
	return {
		"cell_size": cell_size,
		"columns": columns,
		"rows": rows,
		"cell_count": cell_count,
		"known_cells": cells.size(),
		"visited_cells": visited_cell_count(),
		"coverage": coverage_fraction(),
		"uncertainty": mean_uncertainty(),
		"route": route,
		"time": time_seconds,
		"cells": cells,
	}

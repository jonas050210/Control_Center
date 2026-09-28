## ExplorationReport
##
## Map Analyzer 2.0: the analysis layer on top of SpatialMemory.
##
## Everything in here is derived **exclusively** from what the agent
## perceived — the cells SpatialMemory recorded through FOV, line of sight,
## lighting and sound. No world geometry, no enemy positions, no hidden
## regions. The analyzer is deliberately constructed with a SpatialMemory
## and nothing else, so there is no handle through which ground truth
## could arrive: if a region was never observed, it appears here as
## unknown, which is the honest answer.
##
## Output is data (Dictionaries and packed arrays), not drawing calls, so
## the Control Center renders it, tests assert on it, and a headless run
## can write it to disk without a viewport.
class_name ExplorationReport
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SpatialMemory = preload("res://scripts/exploration/spatial_memory.gd")

const SELF_PATH: String = "res://scripts/exploration/exploration_report.gd"

## Heatmap layers the Control Center can display. Every layer is a
## per-cell float in [0, 1]; unknown cells are reported as -1.0 so
## "unknown" can be drawn differently from "known and zero", which is the
## distinction the whole subsystem exists to preserve.
const LAYERS: Array = [
	"coverage",
	"recency",
	"visits",
	"illumination",
	"cover",
	"danger",
	"confidence",
]

## Value written for a cell the agent has never observed.
const UNKNOWN: float = -1.0

## A cell counts as well-lit above this illumination.
const LIT_THRESHOLD: float = 0.45
## A cell counts as cover above this cover score.
const COVER_THRESHOLD: float = 0.5
## A cell counts as dangerous above this danger score.
const DANGER_THRESHOLD: float = 0.25

var memory: SpatialMemory = null
## Coverage samples over time, appended by `sample()`. The exploration
## curve is what distinguishes "explored steadily" from "stood in the
## doorway and then sprinted".
var coverage_samples: Array = []


static func create(p_memory: SpatialMemory) -> ExplorationReport:
	var report: ExplorationReport = load(SELF_PATH).new()
	report.memory = p_memory
	return report


## Records one point on the exploration curve. Cheap enough to call every
## analyzer sweep.
func sample(time_seconds: float) -> void:
	if memory == null:
		return
	coverage_samples.append(
		{
			"time": time_seconds,
			"coverage": memory.coverage_fraction(),
			"known_cells": memory.known_cell_count(),
			"visited_cells": memory.visited_cell_count(),
		}
	)


func clear_samples() -> void:
	coverage_samples.clear()


## Per-cell values for one layer, row-major, length cell_count.
func heatmap(layer: String) -> PackedFloat32Array:
	var values := PackedFloat32Array()
	if memory == null or not LAYERS.has(layer):
		return values
	values.resize(memory.cell_count)
	for index in range(memory.cell_count):
		values[index] = _layer_value(layer, index)
	return values


func _layer_value(layer: String, index: int) -> float:
	if memory.observed_time[index] == SpatialMemory.NEVER:
		return UNKNOWN
	var value: float = UNKNOWN
	match layer:
		"coverage":
			value = 1.0
		"recency":
			# Squashed rather than clipped, so a 10-second-old observation
			# and a 60-second-old one still look different.
			var age: float = memory.time_seconds - memory.observed_time[index]
			value = 1.0 / (1.0 + age * 0.1)
		"visits":
			value = float(memory.visit_count[index]) / 8.0
		"illumination":
			value = memory.illumination[index]
		"cover":
			value = memory.cover_score[index]
		"danger":
			value = memory.danger[index]
		"confidence":
			value = memory.confidence[index]
	return clampf(value, 0.0, 1.0) if value != UNKNOWN else UNKNOWN


## Cell indices the agent has never observed. These are the honest holes in
## its map; the report never fills them in.
func unknown_cells() -> PackedInt32Array:
	var out := PackedInt32Array()
	if memory == null:
		return out
	for index in range(memory.cell_count):
		if memory.observed_time[index] == SpatialMemory.NEVER:
			out.append(index)
	return out


## Known but never walked through: seen from a distance, not verified.
func observed_but_unvisited_cells() -> PackedInt32Array:
	var out := PackedInt32Array()
	if memory == null:
		return out
	for index in range(memory.cell_count):
		if (
			memory.observed_time[index] != SpatialMemory.NEVER
			and memory.visited_time[index] == SpatialMemory.NEVER
		):
			out.append(index)
	return out


## Counts of the qualitative region classes, all perception-derived.
func regions() -> Dictionary:
	var counts: Dictionary = {
		"unknown": 0,
		"observed": 0,
		"visited": 0,
		"lit": 0,
		"dark": 0,
		"cover": 0,
		"open": 0,
		"dangerous": 0,
	}
	if memory == null:
		return counts
	for index in range(memory.cell_count):
		if memory.observed_time[index] == SpatialMemory.NEVER:
			counts["unknown"] += 1
			continue
		counts["observed"] += 1
		if memory.visited_time[index] != SpatialMemory.NEVER:
			counts["visited"] += 1
		if memory.illumination[index] >= LIT_THRESHOLD:
			counts["lit"] += 1
		else:
			counts["dark"] += 1
		if memory.cover_score[index] >= COVER_THRESHOLD:
			counts["cover"] += 1
		else:
			counts["open"] += 1
		if memory.danger[index] >= DANGER_THRESHOLD:
			counts["dangerous"] += 1
	return counts


## The route actually walked, as world positions. SpatialMemory stores it
## as cell indices, which is compact but unreadable to a renderer.
func route_positions() -> Array:
	var out: Array = []
	if memory == null:
		return out
	for index in memory.route:
		out.append(memory.cell_center(int(index)))
	return out


## Straight-line path length of the walked route, in meters.
func route_length() -> float:
	var positions: Array = route_positions()
	var total: float = 0.0
	for index in range(1, positions.size()):
		total += (positions[index] as Vector3).distance_to(positions[index - 1] as Vector3)
	return total


## How much new ground each step of the route bought. A low value means the
## agent walked a lot and learned little — backtracking, or circling.
func exploration_efficiency() -> float:
	var length: float = route_length()
	if memory == null or length <= 0.001:
		return 0.0
	return float(memory.known_cell_count()) / length


## Best remembered vantage points: observed, lit enough to see from, open
## enough to see across, and not remembered as dangerous. Sorted by score,
## limited to `limit` entries.
func sightlines(limit: int = 5) -> Array:
	var candidates: Array = []
	if memory == null:
		return candidates
	for index in range(memory.cell_count):
		if memory.observed_time[index] == SpatialMemory.NEVER:
			continue
		var openness: float = memory.openness[index]
		var light: float = memory.illumination[index]
		var risk: float = memory.danger[index]
		var score: float = openness * 0.6 + light * 0.25 - risk * 0.5
		if score <= 0.0:
			continue
		candidates.append(
			{
				"index": index,
				"position": memory.cell_center(index),
				"score": score,
				"openness": openness,
				"illumination": light,
				"danger": risk,
				"confidence": memory.confidence[index],
			}
		)
	candidates.sort_custom(func(a, b): return float(a["score"]) > float(b["score"]))
	return candidates.slice(0, maxi(0, limit))


## Best remembered cover positions, by the same honest rules.
func cover_positions(limit: int = 5) -> Array:
	var candidates: Array = []
	if memory == null:
		return candidates
	for index in range(memory.cell_count):
		if memory.observed_time[index] == SpatialMemory.NEVER:
			continue
		if memory.cover_score[index] < COVER_THRESHOLD:
			continue
		candidates.append(
			{
				"index": index,
				"position": memory.cell_center(index),
				"cover": memory.cover_score[index],
				"danger": memory.danger[index],
				"confidence": memory.confidence[index],
			}
		)
	candidates.sort_custom(func(a, b): return float(a["cover"]) > float(b["cover"]))
	return candidates.slice(0, maxi(0, limit))


## Diagnostic exploration metrics. Diagnostic is the operative word: none
## of these is wired into a reward, and the report has no way to be.
func metrics() -> Dictionary:
	if memory == null:
		return {}
	var counts: Dictionary = regions()
	var total: int = maxi(1, memory.cell_count)
	return {
		"coverage": memory.coverage_fraction(),
		"visited_fraction": float(counts["visited"]) / float(total),
		"unknown_fraction": float(counts["unknown"]) / float(total),
		"dark_fraction": float(counts["dark"]) / float(maxi(1, int(counts["observed"]))),
		"cover_fraction": float(counts["cover"]) / float(maxi(1, int(counts["observed"]))),
		"danger_fraction": float(counts["dangerous"]) / float(maxi(1, int(counts["observed"]))),
		"mean_uncertainty": memory.mean_uncertainty(),
		"route_length": route_length(),
		"route_cells": memory.route.size(),
		"exploration_efficiency": exploration_efficiency(),
		"samples": coverage_samples.size(),
	}


## Full report payload: metrics, regions, landmarks, the exploration curve
## and the grid geometry a renderer needs. Presentation only.
func to_dict(heatmap_layers: Array = []) -> Dictionary:
	var payload: Dictionary = {
		"metrics": metrics(),
		"regions": regions(),
		"sightlines": sightlines(),
		"cover": cover_positions(),
		"route": route_positions(),
		"coverage_curve": coverage_samples.duplicate(true),
		"unknown_cells": unknown_cells().size(),
		"observed_unvisited_cells": observed_but_unvisited_cells().size(),
	}
	if memory != null:
		payload["grid"] = {
			"columns": memory.columns,
			"rows": memory.rows,
			"cell_size": memory.cell_size,
			"cell_count": memory.cell_count,
		}
	var layers: Array = LAYERS if heatmap_layers.is_empty() else heatmap_layers
	var maps: Dictionary = {}
	for layer in layers:
		var name: String = str(layer)
		if LAYERS.has(name):
			maps[name] = heatmap(name)
	payload["heatmaps"] = maps
	return payload


## Human-readable summary for logs and the Control Center's text pane.
func format_summary() -> String:
	if memory == null:
		return "exploration report: no spatial memory"
	var m: Dictionary = metrics()
	var counts: Dictionary = regions()
	return (
		(
			"exploration %.1f%% (%d/%d cells) | visited %.1f%% | unknown %d | "
			+ "cover %d | dangerous %d | route %.1fm | efficiency %.2f cells/m"
		)
		% [
			float(m["coverage"]) * 100.0,
			int(counts["observed"]),
			memory.cell_count,
			float(m["visited_fraction"]) * 100.0,
			int(counts["unknown"]),
			int(counts["cover"]),
			int(counts["dangerous"]),
			float(m["route_length"]),
			float(m["exploration_efficiency"]),
		]
	)

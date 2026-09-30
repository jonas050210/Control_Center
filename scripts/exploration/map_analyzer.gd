## MapAnalyzer
##
## Drives the agent's persistent map knowledge from its ordinary perception
## and owns the "Map Analyzer" exploration mode.
##
## Two responsibilities, deliberately kept in one small object because they
## share the same update cadence:
##
##  1. Maintain a `SpatialMemory` from what the agent can see and hear. This
##     runs in EVERY mode once perception is enabled, so a combat policy can
##     later be given exploration-derived observation fields without a
##     separate pass.
##  2. Score exploration progress (newly discovered cells, frontier
##     distance, completion) so the sandbox can run a dedicated exploration
##     episode with no combat objective at all.
##
## Determinism: the analyzer holds no RNG. Its entire state is a function of
## the positions and perception results handed to it, on a fixed
## `EXPLORATION_UPDATE_INTERVAL` cadence measured on the simulation clock.
## Replaying the same seed replays the same map knowledge, which
## `tests/test_map_analyzer.gd` asserts.
class_name MapAnalyzer
extends RefCounted

const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SpatialMemory = preload("res://scripts/exploration/spatial_memory.gd")

const SELF_PATH: String = "res://scripts/exploration/map_analyzer.gd"

var memory: SpatialMemory = null
## Simulation-clock timestamp of the next visibility sweep.
var next_sweep_at: float = 0.0
var elapsed: float = 0.0
## Cells discovered this episode (equals memory.known_cell_count()).
var discovered_cells: int = 0
## Cells discovered by the most recent update, used for reward shaping.
var last_discovered: int = 0
var sweeps: int = 0
var completed: bool = false


static func create(
	half_extent: float = SandboxConfig.ARENA_HALF_EXTENT,
	cell_size: float = SandboxConfig.EXPLORATION_CELL_SIZE
) -> MapAnalyzer:
	var analyzer: MapAnalyzer = (load(SELF_PATH) as GDScript).new() as MapAnalyzer
	analyzer.memory = SpatialMemory.create(half_extent, cell_size)
	return analyzer


func configure(half_extent: float, cell_size: float) -> void:
	if memory == null:
		memory = SpatialMemory.create(half_extent, cell_size)
	else:
		memory.configure(half_extent, cell_size)
	reset()


func reset() -> void:
	if memory != null:
		memory.clear()
	next_sweep_at = 0.0
	elapsed = 0.0
	discovered_cells = 0
	last_discovered = 0
	sweeps = 0
	completed = false


## Advances map knowledge by one simulation step.
##
## `context` keys (all optional except position/forward):
##   position, forward, eye_height, world, lighting, fov_deg, vision_range,
##   local_illumination, sounds, damage_taken_from.
##
## Returns a small summary dictionary; callers that do not need it can
## ignore the result without cost.
func update(dt: float, context: Dictionary) -> Dictionary:
	last_discovered = 0
	if memory == null:
		return _summary()
	elapsed += dt
	memory.tick(dt)

	var position: Vector3 = context.get("position", Vector3.ZERO)
	var illumination: float = float(context.get("local_illumination", 1.0))
	memory.observe_self(position, illumination)

	# Sounds are perceived information too: hearing a shot from a bearing
	# tells the agent something happened over there, but NOT what is there,
	# so it only raises danger, never marks the cell as observed geometry.
	# A heard event carries no source position (hearing is directional), so
	# the danger is placed at the agent's own estimate from the perceived
	# direction and distance — the same estimate AgentPerception builds for
	# its memory track. Reading a "position" key here was dead code: the
	# heard events never carry one.
	var sounds: Array = context.get("sounds", [])
	for sound_value in sounds:
		var sound: Dictionary = sound_value
		var category: int = int(sound.get("category", -1))
		if category != 3 and category != 4 and category != 5:  # SHOT / IMPACT / DEATH
			continue
		if not sound.has("direction") or not sound.has("distance"):
			continue
		var estimate: Vector3 = (
			position + (sound["direction"] as Vector3) * float(sound["distance"])
		)
		estimate.y = 0.0
		memory.mark_danger(estimate, 0.35 * float(sound.get("loudness", 1.0)))

	var damage_from = context.get("damage_taken_from", null)
	if damage_from != null:
		memory.mark_danger(damage_from, 1.0)

	if elapsed + 0.000001 >= next_sweep_at:
		next_sweep_at = elapsed + SandboxConfig.EXPLORATION_UPDATE_INTERVAL
		sweeps += 1
		var eye_height: float = float(context.get("eye_height", SandboxConfig.AGENT_EYE_HEIGHT))
		last_discovered = memory.observe_cells(
			position + Vector3(0.0, eye_height, 0.0),
			position,
			context.get("forward", Vector3.FORWARD),
			context.get("world", null),
			context.get("lighting", null),
			float(context.get("fov_deg", SandboxConfig.AGENT_FOV_DEG)),
			float(context.get("vision_range", SandboxConfig.VISION_RANGE))
		)
	discovered_cells = memory.known_cell_count()
	if not completed and coverage() >= SandboxConfig.EXPLORATION_TARGET_COVERAGE:
		completed = true
	return _summary()


func coverage() -> float:
	if memory == null:
		return 0.0
	return memory.coverage_fraction()


## Reward for the most recent step in Map Analyzer mode. Normalized by the
## grid size so a large map does not simply pay more.
func exploration_reward() -> float:
	if memory == null or memory.cell_count <= 0 or last_discovered <= 0:
		return 0.0
	var share: float = float(last_discovered) / float(memory.cell_count)
	return share * SandboxConfig.REWARD_EXPLORATION_COVERAGE


## Direction the agent would head to keep exploring, as a unit vector, or
## Vector3.ZERO when the map is fully observed. Offered as a scripted
## BASELINE for exploration episodes; it is never mixed into a learned
## policy's action and never appears in the observation vector.
func frontier_direction(from_position: Vector3) -> Vector3:
	if memory == null:
		return Vector3.ZERO
	var frontier: Dictionary = memory.nearest_unknown(from_position)
	if frontier.is_empty():
		return Vector3.ZERO
	var delta: Vector3 = (frontier["position"] as Vector3) - from_position
	delta.y = 0.0
	if delta.length() < 0.001:
		return Vector3.ZERO
	return delta.normalized()


func _summary() -> Dictionary:
	return {
		"coverage": coverage(),
		"discovered_cells": discovered_cells,
		"new_cells": last_discovered,
		"completed": completed,
	}


## Control Center payload. Presentation only.
func to_dict() -> Dictionary:
	var payload: Dictionary = _summary()
	payload["sweeps"] = sweeps
	payload["elapsed"] = elapsed
	if memory != null:
		payload["memory"] = memory.to_dict()
	return payload

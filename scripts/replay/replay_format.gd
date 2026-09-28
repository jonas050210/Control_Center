## ReplayFormat
##
## The on-disk contract for deterministic episode replays, shared by the
## Godot recorder (ReplayRecorder) and the Python reader
## (python/sandboxai/replay.py). Both sides are kept byte-compatible on
## purpose: an episode recorded during headless training must be loadable
## by the Control Center, and an episode recorded in the Control Center
## must be loadable by the Python analysis tools.
##
## Format (JSON Lines, one JSON object per line):
##   line 1        {"header": {...}}
##   lines 2..n    {"tick":   {"t", "a", "r", "d"?, "o"?, "i"?}}
##   then          {"event":  {"t", "e", "l"?, "v"?}}
##   last line     {"result": {...}}
##
## Short keys are deliberate: a 60 Hz episode of 3600 ticks writes 3600
## tick records, and `"observation"` spelled out 3600 times is pure waste.
## Observations are only stored at DETAIL_DETAILED, because a replay is
## reproducible from (seed, map, scenario, conditions, actions) alone —
## storing the observation stream as well is redundant by default and is
## offered only as an analysis aid.
class_name ReplayFormat
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const Observation = preload("res://scripts/core/observation.gd")

## Must match REPLAY_MAGIC in python/sandboxai/replay.py.
const MAGIC: String = "sandboxai.replay"

## Bumped whenever an existing field changes meaning or disappears.
## Purely additive changes do NOT bump it; readers ignore unknown keys.
const FORMAT_VERSION: int = 1

## Versions this build can read.
const READABLE_VERSIONS: Array = [1]

const DETAIL_LIGHT: String = "light"
const DETAIL_DETAILED: String = "detailed"
const DETAIL_LEVELS: Array = [DETAIL_LIGHT, DETAIL_DETAILED]

## Mirrors EVENT_KINDS in python/sandboxai/replay.py.
const EVENT_KINDS: Array = [
	"episode_start",
	"episode_end",
	"target_change",
	"perception",
	"memory",
	"sound",
	"combat",
	"death",
	"curriculum",
	"note",
]

## Events worth putting on a scrubbable timeline by default.
const IMPORTANT_EVENT_KINDS: Array = [
	"episode_start",
	"episode_end",
	"target_change",
	"combat",
	"death",
]


## A header with every field present, so the JSON shape never depends on
## which optional values a caller happened to supply.
static func default_header() -> Dictionary:
	return {
		"magic": MAGIC,
		"version": FORMAT_VERSION,
		"detail": DETAIL_LIGHT,
		"seed": 0,
		"map_id": "",
		"scenario": "",
		"lighting": "",
		"enemy_count": 1,
		"curriculum_level": 1,
		"policy_id": "",
		"checkpoint": "",
		"environment_index": 0,
		"agent_slot": 0,
		"team_id": 0,
		"observation_dim": Observation.FIELD_COUNT,
		"action_nvec": Action.MULTI_DISCRETE_NVECS.duplicate(),
		"simulation_dt": 1.0 / 60.0,
		"notes": {},
	}


static func is_known_event(kind: String) -> bool:
	return EVENT_KINDS.has(kind)


## Returns a list of problems; empty means the header is playable here.
## `strict_contract` false keeps an old recording inspectable (timeline,
## events, rewards) while still refusing to pretend it matches this
## build's observation/action contract.
static func validate_header(header: Dictionary, strict_contract: bool = true) -> Array:
	var problems: Array = []
	if str(header.get("magic", "")) != MAGIC:
		problems.append("not a SandboxAI replay (magic=%s)" % str(header.get("magic", "")))
	var version: int = int(header.get("version", -1))
	if not READABLE_VERSIONS.has(version):
		problems.append(
			"replay format version %d is not readable by this build (supported: %s)"
			% [version, str(READABLE_VERSIONS)]
		)
	var detail: String = str(header.get("detail", DETAIL_LIGHT))
	if not DETAIL_LEVELS.has(detail):
		problems.append("unknown detail level: %s" % detail)
	if not strict_contract:
		return problems
	var observation_dim: int = int(header.get("observation_dim", -1))
	if observation_dim != Observation.FIELD_COUNT:
		problems.append(
			(
				"replay was recorded with a %d-float observation contract; this build uses %d"
				% [observation_dim, Observation.FIELD_COUNT]
			)
		)
	var nvec: Array = header.get("action_nvec", [])
	if not _int_arrays_equal(nvec, Action.MULTI_DISCRETE_NVECS):
		problems.append(
			"replay action space %s != contract %s" % [str(nvec), str(Action.MULTI_DISCRETE_NVECS)]
		)
	return problems


## Structural validation of a whole parsed replay (see ReplayRecorder.parse).
static func validate_episode(episode: Dictionary, strict_contract: bool = true) -> Array:
	var header: Dictionary = episode.get("header", {})
	var problems: Array = validate_header(header, strict_contract)
	var ticks: Array = episode.get("ticks", [])
	var expected_width: int = int(Action.MULTI_DISCRETE_NVECS.size())
	for index in range(ticks.size()):
		var tick: Dictionary = ticks[index]
		if int(tick.get("tick", -1)) != index:
			problems.append(
				"tick %d is out of order (recorded index %d)" % [index, int(tick.get("tick", -1))]
			)
		var action: Array = tick.get("action", [])
		if action.size() != expected_width:
			problems.append(
				"tick %d has %d action values, expected %d" % [index, action.size(), expected_width]
			)
	for event_value in episode.get("events", []):
		var event: Dictionary = event_value
		if not is_known_event(str(event.get("kind", ""))):
			problems.append("unknown event kind: %s" % str(event.get("kind", "")))
	return problems


static func _int_arrays_equal(left: Array, right: Array) -> bool:
	if left.size() != right.size():
		return false
	for index in range(left.size()):
		if int(left[index]) != int(right[index]):
			return false
	return true

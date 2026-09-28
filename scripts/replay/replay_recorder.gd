## ReplayRecorder
##
## Records and reads deterministic episode replays in the JSON Lines format
## defined by ReplayFormat. The Python counterpart is
## python/sandboxai/replay.py; the two must stay byte-compatible, which is
## why every key written here is spelled out in ReplayFormat rather than
## inline.
##
## What is stored, and what deliberately is not:
##
## * Stored once, in the header: seed, map, scenario, lighting, enemy
##   count, curriculum level, policy id, checkpoint, env index, agent slot,
##   team, the observation/action contract fingerprint and the simulation
##   timestep. Together with the action stream this is enough to reproduce
##   the episode by re-simulating it.
## * Stored per tick: the action, the reward and the done flag. Nothing
##   else — the per-tick observation is reconstructable and is only written
##   at DETAIL_DETAILED, as an analysis aid.
## * Stored as events: the discrete things a human or a metric cares about
##   (target changes, perception/memory/sound transitions, combat, deaths,
##   curriculum changes). Events carry a tick index, so the Control Center
##   can jump to them without scanning the tick stream.
##
## The recorder never touches the simulation. It is fed by the caller
## after a step has already been computed, so switching recording on or
## off cannot change an episode's outcome.
class_name ReplayRecorder
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const Observation = preload("res://scripts/core/observation.gd")
const ReplayFormat = preload("res://scripts/replay/replay_format.gd")

var header: Dictionary = {}
var ticks: Array = []
var events: Array = []
var result: Dictionary = {}

var _recording: bool = false
var _tick: int = 0


func _init(p_header: Dictionary = {}, p_detail: String = ReplayFormat.DETAIL_LIGHT) -> void:
	header = ReplayFormat.default_header()
	for key in p_header:
		header[key] = p_header[key]
	header["detail"] = (
		p_detail if ReplayFormat.DETAIL_LEVELS.has(p_detail) else ReplayFormat.DETAIL_LIGHT
	)
	# The magic, version and contract fingerprint are ours to state, never
	# the caller's: a recording that lies about its own format is worse
	# than no recording.
	header["magic"] = ReplayFormat.MAGIC
	header["version"] = ReplayFormat.FORMAT_VERSION
	header["observation_dim"] = Observation.FIELD_COUNT
	header["action_nvec"] = Action.MULTI_DISCRETE_NVECS.duplicate()


func is_recording() -> bool:
	return _recording


func is_detailed() -> bool:
	return str(header.get("detail", "")) == ReplayFormat.DETAIL_DETAILED


func tick_count() -> int:
	return ticks.size()


func current_tick() -> int:
	return _tick


## Begins a new episode. Any previously recorded episode is discarded, so
## call `to_dictionary()` or `save()` first if it is still wanted.
func start(p_seed: int = -1, header_updates: Dictionary = {}) -> void:
	ticks.clear()
	events.clear()
	result.clear()
	_tick = 0
	_recording = true
	if p_seed >= 0:
		header["seed"] = p_seed
	for key in header_updates:
		header[key] = header_updates[key]
	add_event("episode_start", str(header.get("map_id", "")), {"seed": int(header.get("seed", 0))})


## Records one already-computed simulation step.
func record_step(
	action: Action,
	reward: float = 0.0,
	observation: Variant = null,
	done: bool = false,
	info: Dictionary = {}
) -> int:
	if not _recording:
		return -1
	var record: Dictionary = {
		"tick": _tick,
		"action": action.to_multidiscrete() if action != null else Action.idle().to_multidiscrete(),
		"reward": reward,
		"done": done,
	}
	if is_detailed() and observation != null:
		record["observation"] = _observation_to_array(observation)
	if not info.is_empty():
		record["info"] = info
	ticks.append(record)
	var index: int = _tick
	_tick += 1
	return index


## Tick an event recorded right now belongs to.
##
## Callers record a step and THEN translate that step's events, by which
## point `_tick` already points at the next, not yet recorded tick. Anchoring
## events to `_tick` therefore filed every event one tick after the thing it
## describes, so `events_at(n)` never found them and "jump to next event"
## landed one tick late. Before the first step (episode_start) the answer is
## tick 0.
func event_tick() -> int:
	return maxi(0, _tick - 1) if not ticks.is_empty() else 0


## Records a discrete event at the tick it describes. Unknown kinds are
## rejected rather than silently written, so a typo cannot create an event
## category the Python reader will later refuse.
func add_event(kind: String, label: String = "", data: Dictionary = {}) -> bool:
	if not ReplayFormat.is_known_event(kind):
		push_warning("ReplayRecorder: unknown event kind '%s'" % kind)
		return false
	events.append(
		{"tick": event_tick(), "kind": kind, "label": label, "data": data.duplicate(true)}
	)
	return true


## Translates one EnvironmentCore step `info["events"]` dictionary into
## replay events. Only things that *happened* become events; per-tick
## scalars already live on the tick record and are not duplicated.
func record_step_events(step_events: Dictionary) -> int:
	var emitted: int = 0
	if bool(step_events.get("shot_fired", false)):
		var shot_data: Dictionary = {
			"hit": bool(step_events.get("hit", false)),
			"damage": float(step_events.get("damage_dealt", 0.0)),
			"useless": bool(step_events.get("useless_shot", false)),
		}
		if add_event("combat", "shot", shot_data):
			emitted += 1
	if float(step_events.get("damage_taken", 0.0)) > 0.0:
		if (
			add_event(
				"combat", "damage_taken", {"amount": float(step_events.get("damage_taken", 0.0))}
			)
		):
			emitted += 1
	if bool(step_events.get("kill", false)):
		if add_event("death", "enemy_killed", {}):
			emitted += 1
	if bool(step_events.get("died", false)):
		if add_event("death", "agent_died", {}):
			emitted += 1
	return emitted


## Closes the episode. The result dictionary is also emitted as the final
## `episode_end` event so a timeline always ends with one.
func finish(p_result: Dictionary = {}) -> Dictionary:
	result = p_result.duplicate(true)
	add_event("episode_end", str(result.get("done_reason", "")), result)
	_recording = false
	return to_dictionary()


func to_dictionary() -> Dictionary:
	return {
		"header": header.duplicate(true),
		"ticks": ticks.duplicate(true),
		"events": events.duplicate(true),
		"result": result.duplicate(true),
	}


## -- serialization ---------------------------------------------------------


func to_lines() -> PackedStringArray:
	var lines := PackedStringArray()
	lines.append(JSON.stringify({"header": header}))
	for tick_value in ticks:
		lines.append(JSON.stringify({"tick": _tick_record(tick_value)}))
	for event_value in events:
		lines.append(JSON.stringify({"event": _event_record(event_value)}))
	lines.append(JSON.stringify({"result": result}))
	return lines


func save(path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("ReplayRecorder: cannot write %s" % path)
		return false
	for line in to_lines():
		file.store_line(line)
	file.close()
	return true


## Parses replay lines into {header, ticks, events, result, problems}.
## Parsing never throws: a corrupt file yields problems, which the caller
## is expected to check before playing anything back.
static func parse(lines: PackedStringArray, strict_contract: bool = true) -> Dictionary:
	var episode: Dictionary = {
		"header": {},
		"ticks": [],
		"events": [],
		"result": {},
		"problems": [],
	}
	var problems: Array = episode["problems"]
	var header_seen: bool = false
	for raw_line in lines:
		var line: String = str(raw_line).strip_edges()
		if line.is_empty():
			continue
		var parsed: Variant = JSON.parse_string(line)
		if typeof(parsed) != TYPE_DICTIONARY:
			problems.append("malformed replay line: %s" % line.substr(0, 60))
			continue
		var record: Dictionary = parsed
		if record.has("header"):
			episode["header"] = record["header"]
			header_seen = true
		elif record.has("tick"):
			episode["ticks"].append(_tick_from_record(record["tick"]))
		elif record.has("event"):
			episode["events"].append(_event_from_record(record["event"]))
		elif record.has("result"):
			episode["result"] = record["result"]
		else:
			problems.append("unknown replay record type: %s" % str(record.keys()))
	if not header_seen:
		problems.append("replay has no header record")
		return episode
	for problem in ReplayFormat.validate_episode(episode, strict_contract):
		problems.append(problem)
	return episode


static func load_replay(path: String, strict_contract: bool = true) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {
			"header": {},
			"ticks": [],
			"events": [],
			"result": {},
			"problems": ["replay file not found: %s" % path],
		}
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return {
			"header": {},
			"ticks": [],
			"events": [],
			"result": {},
			"problems": ["cannot read replay file: %s" % path],
		}
	var lines := PackedStringArray()
	while not file.eof_reached():
		lines.append(file.get_line())
	file.close()
	return parse(lines, strict_contract)


## Rebuilds the action stream of a parsed replay, ready to be fed back into
## EnvironmentCore.step() for a deterministic re-simulation.
static func actions_of(episode: Dictionary) -> Array:
	var out: Array = []
	for tick_value in episode.get("ticks", []):
		var tick: Dictionary = tick_value
		out.append(Action.from_multidiscrete(tick.get("action", [])))
	return out


## Events of the given kinds, in tick order. An empty `kinds` means the
## default "worth showing on a timeline" set.
static func timeline_of(episode: Dictionary, kinds: Array = []) -> Array:
	var wanted: Array = ReplayFormat.IMPORTANT_EVENT_KINDS if kinds.is_empty() else kinds
	var out: Array = []
	for event_value in episode.get("events", []):
		var event: Dictionary = event_value
		if wanted.has(str(event.get("kind", ""))):
			out.append(event)
	return out


static func total_reward(episode: Dictionary) -> float:
	var total: float = 0.0
	for tick_value in episode.get("ticks", []):
		total += float((tick_value as Dictionary).get("reward", 0.0))
	return total


## -- internals -------------------------------------------------------------


func _tick_record(tick_value: Dictionary) -> Dictionary:
	var record: Dictionary = {
		"t": int(tick_value.get("tick", 0)),
		"a": tick_value.get("action", []),
		"r": snappedf(float(tick_value.get("reward", 0.0)), 0.000001),
	}
	if bool(tick_value.get("done", false)):
		record["d"] = true
	if tick_value.has("observation"):
		record["o"] = tick_value["observation"]
	if tick_value.has("info"):
		record["i"] = tick_value["info"]
	return record


func _event_record(event_value: Dictionary) -> Dictionary:
	var record: Dictionary = {
		"t": int(event_value.get("tick", 0)),
		"e": str(event_value.get("kind", "note")),
	}
	var label: String = str(event_value.get("label", ""))
	if not label.is_empty():
		record["l"] = label
	var data: Dictionary = event_value.get("data", {})
	if not data.is_empty():
		record["v"] = data
	return record


static func _tick_from_record(record_value: Variant) -> Dictionary:
	var record: Dictionary = record_value if typeof(record_value) == TYPE_DICTIONARY else {}
	var tick: Dictionary = {
		"tick": int(record.get("t", 0)),
		"action": record.get("a", []),
		"reward": float(record.get("r", 0.0)),
		"done": bool(record.get("d", false)),
	}
	if record.has("o"):
		tick["observation"] = record["o"]
	if record.has("i"):
		tick["info"] = record["i"]
	return tick


static func _event_from_record(record_value: Variant) -> Dictionary:
	var record: Dictionary = record_value if typeof(record_value) == TYPE_DICTIONARY else {}
	return {
		"tick": int(record.get("t", 0)),
		"kind": str(record.get("e", "note")),
		"label": str(record.get("l", "")),
		"data": record.get("v", {}),
	}


func _observation_to_array(observation: Variant) -> Array:
	if observation is Observation:
		return Array((observation as Observation).to_array())
	if observation is PackedFloat32Array:
		return Array(observation as PackedFloat32Array)
	if observation is Array:
		return (observation as Array).duplicate()
	return []

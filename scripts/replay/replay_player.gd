## ReplayPlayer
##
## Playback cursor over a parsed replay (see ReplayRecorder.parse). It owns
## no simulation and mutates nothing outside itself: it answers "which tick
## should be shown now", which the Control Center's REPLAY tab renders and
## which a deterministic re-simulation harness feeds into EnvironmentCore.
##
## Keeping the cursor separate from both the recorder and the renderer is
## what lets the same playback logic drive an on-screen scrubber and a
## headless verification run.
class_name ReplayPlayer
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ReplayFormat = preload("res://scripts/replay/replay_format.gd")
const ReplayRecorder = preload("res://scripts/replay/replay_recorder.gd")

## Playback speeds the UI offers. 0.25x is slow enough to read a single
## engagement; 4x is fast enough to skim a 60-second episode.
const SPEED_PRESETS: Array = [0.25, 0.5, 1.0, 2.0, 4.0]

var episode: Dictionary = {}
var playing: bool = false
var speed: float = 1.0
var cursor: int = 0

var _accumulator: float = 0.0


func _init(p_episode: Dictionary = {}) -> void:
	episode = p_episode


func tick_count() -> int:
	return (episode.get("ticks", []) as Array).size()


func simulation_dt() -> float:
	var dt: float = float((episode.get("header", {}) as Dictionary).get("simulation_dt", 1.0 / 60.0))
	return dt if dt > 0.0 else 1.0 / 60.0


func duration() -> float:
	return float(tick_count()) * simulation_dt()


func time_at(tick_index: int) -> float:
	return float(tick_index) * simulation_dt()


func is_finished() -> bool:
	return cursor >= tick_count()


func play() -> void:
	playing = true


func pause() -> void:
	playing = false


func toggle() -> bool:
	playing = not playing
	return playing


func set_speed(value: float) -> float:
	speed = clampf(value, 0.05, 16.0)
	return speed


func reset() -> void:
	cursor = 0
	_accumulator = 0.0
	playing = false


## Advances by whole ticks regardless of play state (the "frame step"
## button). Returns the ticks that were crossed.
func step(count: int = 1) -> Array:
	var crossed: Array = []
	var ticks: Array = episode.get("ticks", [])
	for _index in range(maxi(0, count)):
		if cursor >= ticks.size():
			break
		crossed.append(ticks[cursor])
		cursor += 1
	return crossed


## Advances wall-clock time. Returns the ticks crossed during `delta`.
## Does nothing while paused, which is what makes pause exact rather than
## approximate.
func advance(delta: float) -> Array:
	if not playing or is_finished():
		return []
	_accumulator += maxf(0.0, delta) * speed
	var dt: float = simulation_dt()
	var steps: int = int(floor(_accumulator / dt))
	if steps <= 0:
		return []
	_accumulator -= float(steps) * dt
	return step(steps)


## Moves the cursor to an absolute tick. Returns the clamped position.
func seek(tick_index: int) -> int:
	cursor = clampi(tick_index, 0, tick_count())
	_accumulator = 0.0
	return cursor


## The epsilon is not cosmetic: `simulation_dt` is 1/60, which is not
## representable in binary, so 0.5 / (1.0 / 60.0) evaluates to
## 29.999999999999996 and a plain floor() seeks one tick too early. The
## tolerance is far smaller than a tick, so it only absorbs that error.
func seek_time(seconds: float) -> int:
	var ticks: float = maxf(0.0, seconds) / simulation_dt()
	return seek(int(floor(ticks + 1e-6)))


## Jumps to the next event at or after the cursor. Returns the event, or
## an empty Dictionary when there is none.
func next_event(kinds: Array = []) -> Dictionary:
	var wanted: Array = ReplayFormat.IMPORTANT_EVENT_KINDS if kinds.is_empty() else kinds
	for event_value in episode.get("events", []):
		var event: Dictionary = event_value
		if int(event.get("tick", 0)) >= cursor and wanted.has(str(event.get("kind", ""))):
			seek(int(event.get("tick", 0)))
			return event
	return {}


func previous_event(kinds: Array = []) -> Dictionary:
	var wanted: Array = ReplayFormat.IMPORTANT_EVENT_KINDS if kinds.is_empty() else kinds
	var found: Dictionary = {}
	for event_value in episode.get("events", []):
		var event: Dictionary = event_value
		if int(event.get("tick", 0)) < cursor and wanted.has(str(event.get("kind", ""))):
			found = event
	if not found.is_empty():
		seek(int(found.get("tick", 0)))
	return found


## Events anchored to a specific tick, for the per-frame event readout.
func events_at(tick_index: int) -> Array:
	var out: Array = []
	for event_value in episode.get("events", []):
		var event: Dictionary = event_value
		if int(event.get("tick", 0)) == tick_index:
			out.append(event)
	return out


func current_tick_record() -> Dictionary:
	var ticks: Array = episode.get("ticks", [])
	var index: int = clampi(cursor, 0, maxi(0, ticks.size() - 1))
	if ticks.is_empty():
		return {}
	return ticks[index]


## Everything the REPLAY tab shows in its status line. Presentation only:
## no field here is ever fed back into a policy observation.
func status() -> Dictionary:
	var header: Dictionary = episode.get("header", {})
	var tick: Dictionary = current_tick_record()
	return {
		"tick": cursor,
		"tick_count": tick_count(),
		"time": time_at(cursor),
		"duration": duration(),
		"playing": playing,
		"speed": speed,
		"finished": is_finished(),
		"map_id": str(header.get("map_id", "")),
		"scenario": str(header.get("scenario", "")),
		"lighting": str(header.get("lighting", "")),
		"seed": int(header.get("seed", 0)),
		"curriculum_level": int(header.get("curriculum_level", 0)),
		"policy_id": str(header.get("policy_id", "")),
		"checkpoint": str(header.get("checkpoint", "")),
		"detail": str(header.get("detail", ReplayFormat.DETAIL_LIGHT)),
		"action": tick.get("action", []),
		"reward": float(tick.get("reward", 0.0)),
		"total_reward": ReplayRecorder.total_reward(episode),
		"done_reason": str((episode.get("result", {}) as Dictionary).get("done_reason", "")),
	}

## ControlCenterEventLog
##
## Bounded, throttled, filterable in-memory event log for the Control
## Center. It is a plain RefCounted data structure with no Node, UI or
## simulation dependency, so it is cheap to unit test and impossible to
## accidentally pull into the training path.
##
## Performance rules baked into the type itself (Phase 15):
##   * `enabled` is switched off entirely in TRAINING mode, making every
##     `log_event()` call an immediate early-out.
##   * a ring buffer of `capacity` entries bounds memory; old entries are
##     discarded instead of growing forever.
##   * `throttle_key` collapses repeated events (e.g. "took damage") to at
##     most one entry per `throttle_seconds`.
##   * `max_events_per_second` is a global budget so a pathological event
##     storm cannot stall the frame; dropped events are counted and
##     reported instead of silently lost.
##
## Callers should log DISCRETE events (a shot, a kill, an episode boundary),
## never per-tick state.
class_name ControlCenterEventLog
extends RefCounted

enum Category {
	SYSTEM = 0,
	COMBAT = 1,
	PERCEPTION = 2,
	REWARD = 3,
	ERROR = 4,
}

## Filter value meaning "do not filter by category".
const FILTER_ALL: int = -1
const DEFAULT_CAPACITY: int = 400
const DEFAULT_THROTTLE_SECONDS: float = 0.15
const DEFAULT_MAX_EVENTS_PER_SECOND: float = 60.0

var enabled: bool = true
var capacity: int = DEFAULT_CAPACITY
var throttle_seconds: float = DEFAULT_THROTTLE_SECONDS
var max_events_per_second: float = DEFAULT_MAX_EVENTS_PER_SECOND
var dropped_count: int = 0
var total_accepted: int = 0

var _entries: Array = []
var _last_key_time: Dictionary = {}
var _window_start: float = -1.0
var _window_count: int = 0


static func category_name(value: int) -> String:
	match value:
		Category.SYSTEM:
			return "SYSTEM"
		Category.COMBAT:
			return "COMBAT"
		Category.PERCEPTION:
			return "PERCEPTION"
		Category.REWARD:
			return "REWARD"
		Category.ERROR:
			return "ERROR"
		_:
			return "ALL" if value == FILTER_ALL else "UNKNOWN"


static func category_from_name(value: String) -> int:
	match value.strip_edges().to_upper():
		"SYSTEM":
			return Category.SYSTEM
		"COMBAT":
			return Category.COMBAT
		"PERCEPTION":
			return Category.PERCEPTION
		"REWARD":
			return Category.REWARD
		"ERROR":
			return Category.ERROR
		_:
			return FILTER_ALL


## Category values in the order the UI offers them, with ALL first.
static func filter_options() -> Array:
	return [
		FILTER_ALL,
		Category.COMBAT,
		Category.PERCEPTION,
		Category.SYSTEM,
		Category.REWARD,
		Category.ERROR,
	]


## Appends one event. Returns true when it was stored, false when it was
## dropped (disabled, throttled or over the per-second budget).
##
## `now_seconds` is injectable so tests are deterministic; the default uses
## the engine clock.
func log_event(
	category: int,
	message: String,
	data: Dictionary = {},
	throttle_key: String = "",
	now_seconds: float = -1.0
) -> bool:
	if not enabled:
		return false
	var now: float = now_seconds if now_seconds >= 0.0 else _engine_seconds()

	if _window_start < 0.0 or now - _window_start >= 1.0:
		_window_start = now
		_window_count = 0
	if max_events_per_second > 0.0 and float(_window_count) >= max_events_per_second:
		dropped_count += 1
		return false

	if not throttle_key.is_empty() and throttle_seconds > 0.0:
		var previous: float = float(_last_key_time.get(throttle_key, -1.0e20))
		if now - previous < throttle_seconds:
			dropped_count += 1
			return false
		_last_key_time[throttle_key] = now

	(
		_entries
		. append(
			{
				"time": now,
				"category": category,
				"category_name": category_name(category),
				"message": message,
				"data": data,
			}
		)
	)
	_window_count += 1
	total_accepted += 1
	if _entries.size() > capacity:
		_entries = _entries.slice(_entries.size() - capacity)
	return true


## Entries matching `category_filter`, oldest first. `limit` > 0 returns
## only the newest `limit` matching entries.
func entries(category_filter: int = FILTER_ALL, limit: int = 0) -> Array:
	var selected: Array = []
	for entry_value in _entries:
		var entry: Dictionary = entry_value
		if category_filter == FILTER_ALL or int(entry["category"]) == category_filter:
			selected.append(entry)
	if limit > 0 and selected.size() > limit:
		selected = selected.slice(selected.size() - limit)
	return selected


func size() -> int:
	return _entries.size()


func clear() -> void:
	_entries.clear()
	_last_key_time.clear()
	_window_start = -1.0
	_window_count = 0
	dropped_count = 0
	total_accepted = 0


## Human-readable "[ 12.34s] CATEGORY  message" lines for the log panel.
func format_lines(category_filter: int = FILTER_ALL, limit: int = 0) -> PackedStringArray:
	var lines := PackedStringArray()
	for entry_value in entries(category_filter, limit):
		var entry: Dictionary = entry_value
		lines.append(
			(
				"[%7.2fs] %-10s %s"
				% [float(entry["time"]), str(entry["category_name"]), str(entry["message"])]
			)
		)
	return lines


func _engine_seconds() -> float:
	return float(Time.get_ticks_msec()) / 1000.0

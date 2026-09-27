extends Node
class_name SandboxEpisodeLogger

var events: Array[Dictionary] = []
var frame_index: int = 0
var started_at_usec: int = 0
@export var capture_observations: bool = true
var observation_count: int = 0

func begin(seed_value: int, map_name: String, scenario: String) -> void:
	events.clear(); frame_index = 0; observation_count = 0; started_at_usec = Time.get_ticks_usec()
	record("episode_start", {"seed": seed_value, "map": map_name, "scenario": scenario})

func tick() -> void: frame_index += 1

func record_observation(image: Image) -> void:
	if not capture_observations or image == null: return
	observation_count += 1
	var encoded: String = Marshalls.raw_to_base64(image.save_png_to_buffer())
	record("observation", {"observation_png": encoded})

func record(kind: String, payload: Dictionary = {}) -> void:
	var event: Dictionary = {"frame": frame_index, "time_ms": float(Time.get_ticks_usec() - started_at_usec) / 1000.0, "type": kind}
	event.merge(payload); events.append(event)
	if events.size() > 256: events.pop_front()

func snapshot() -> Array[Dictionary]:
	return events.duplicate(true)

func save_json(path: String = "user://replays") -> String:
	var directory: String = path
	if not directory.ends_with(".json"):
		DirAccess.make_dir_recursive_absolute(directory)
		directory = directory.path_join("episode_%d.json" % started_at_usec)
	var file: FileAccess = FileAccess.open(directory, FileAccess.WRITE)
	if file == null: return ""
	file.store_string(JSON.stringify({"version": 1, "events": events}, "\t"))
	file.close()
	return directory

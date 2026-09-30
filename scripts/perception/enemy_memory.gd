## EnemyMemory
##
## Limited, decaying belief about where opponents are. This is the piece
## that makes corner fights meaningful: once an observer loses sight of a
## target it does NOT keep receiving that target's live position — it keeps
## a `MemoryTrack` holding the last known position, the direction the
## target was last moving, how the information was acquired (visual or
## sound), how old it is, and a confidence that decays exponentially until
## the track is forgotten entirely.
##
## Both the learning agent's observation and the scripted enemy brains read
## from this same structure, so "the AI must not magically know where an
## enemy is" is enforced by construction rather than by convention.
class_name EnemyMemory
extends RefCounted

enum Source { NONE = 0, VISUAL = 1, SOUND = 2 }

const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SELF_PATH: String = "res://scripts/perception/enemy_memory.gd"

## target id -> track Dictionary:
##   {position, direction, source, age, confidence, last_seen_position,
##    seen_at_least_once, investigated}
var tracks: Dictionary = {}


static func create() -> EnemyMemory:
	return (load(SELF_PATH) as GDScript).new() as EnemyMemory


func clear() -> void:
	tracks.clear()


func has(target_id: int) -> bool:
	return tracks.has(target_id)


func get_track(target_id: int) -> Dictionary:
	return tracks.get(target_id, {})


func size() -> int:
	return tracks.size()


## Direct visual contact: the strongest possible evidence. Resets age and
## confidence to full.
func observe_visual(
	target_id: int, position: Vector3, direction: Vector3, health_norm: float = 1.0
) -> void:
	var track: Dictionary = tracks.get(target_id, _new_track())
	track["position"] = position
	track["health_norm"] = clampf(health_norm, 0.0, 1.0)
	track["last_seen_position"] = position
	track["direction"] = direction
	track["source"] = Source.VISUAL
	track["age"] = 0.0
	track["confidence"] = 1.0
	track["seen_at_least_once"] = true
	track["investigated"] = false
	tracks[target_id] = track
	_enforce_capacity()


## Audio contact: weaker evidence. The position is an ESTIMATE derived from
## the perceived direction and distance, and confidence is capped below a
## visual sighting's so target selection prefers something actually seen.
func observe_sound(
	target_id: int, estimated_position: Vector3, direction: Vector3, strength: float
) -> void:
	var track: Dictionary = tracks.get(target_id, _new_track())
	var existing_confidence: float = float(track.get("confidence", 0.0))
	var sound_confidence: float = clampf(strength, 0.0, 1.0) * 0.6
	if int(track.get("source", Source.NONE)) == Source.VISUAL and existing_confidence >= 0.6:
		# A fresh visual fix outranks a vague noise; keep it.
		return
	track["position"] = estimated_position
	track["direction"] = direction
	track["source"] = Source.SOUND
	track["age"] = 0.0
	track["confidence"] = maxf(existing_confidence, sound_confidence)
	track["investigated"] = false
	tracks[target_id] = track
	_enforce_capacity()


## Ages every track and drops the ones that fell below the forget
## threshold. Exponential decay with MEMORY_HALF_LIFE.
func tick(dt: float) -> void:
	if tracks.is_empty():
		return
	var decay: float = pow(0.5, dt / maxf(SandboxConfig.MEMORY_HALF_LIFE, 0.0001))
	var expired: Array = []
	for key in tracks.keys():
		var track: Dictionary = tracks[key]
		track["age"] = float(track["age"]) + dt
		track["confidence"] = float(track["confidence"]) * decay
		if float(track["confidence"]) < SandboxConfig.MEMORY_FORGET_CONFIDENCE:
			expired.append(key)
		else:
			tracks[key] = track
	for key in expired:
		tracks.erase(key)


## Permanently drops a track. Called when a target dies: a corpse must not
## remain a remembered threat.
func forget(target_id: int) -> void:
	tracks.erase(target_id)


## Marks a track as already searched so a brain does not loop forever on
## the same stale position.
func mark_investigated(target_id: int) -> void:
	if tracks.has(target_id):
		var track: Dictionary = tracks[target_id]
		track["investigated"] = true
		tracks[target_id] = track


## Tracks ordered by confidence (most trusted first). Each entry is the
## track Dictionary plus its "id".
func ranked() -> Array:
	var out: Array = []
	for key in tracks.keys():
		var track: Dictionary = (tracks[key] as Dictionary).duplicate()
		track["id"] = key
		out.append(track)
	out.sort_custom(func(a, b): return float(a["confidence"]) > float(b["confidence"]))
	return out


func to_dict() -> Array:
	var out: Array = []
	for entry_value in ranked():
		var entry: Dictionary = entry_value
		(
			out
			. append(
				{
					"id": int(entry["id"]),
					"position": entry["position"],
					"direction": entry["direction"],
					"source": int(entry["source"]),
					"source_name": source_name(int(entry["source"])),
					"age": float(entry["age"]),
					"confidence": float(entry["confidence"]),
					"health_norm": float(entry.get("health_norm", 1.0)),
					"investigated": bool(entry["investigated"]),
				}
			)
		)
	return out


static func source_name(value: int) -> String:
	match value:
		Source.VISUAL:
			return "visual"
		Source.SOUND:
			return "sound"
		_:
			return "none"


func _new_track() -> Dictionary:
	return {
		"position": Vector3.ZERO,
		"last_seen_position": Vector3.ZERO,
		"health_norm": 1.0,
		"direction": Vector3.ZERO,
		"source": Source.NONE,
		"age": 0.0,
		"confidence": 0.0,
		"seen_at_least_once": false,
		"investigated": false,
	}


## Keeps at most MEMORY_MAX_TRACKS entries, dropping the least confident.
func _enforce_capacity() -> void:
	if tracks.size() <= SandboxConfig.MEMORY_MAX_TRACKS:
		return
	var ordered: Array = ranked()
	for index in range(SandboxConfig.MEMORY_MAX_TRACKS, ordered.size()):
		tracks.erase(int((ordered[index] as Dictionary)["id"]))

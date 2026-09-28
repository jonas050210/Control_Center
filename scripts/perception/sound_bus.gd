## SoundBus
##
## Per-environment registry of transient combat sounds. Characters EMIT
## events (footstep, jump, land, shot, impact, death); listeners SAMPLE the
## bus and receive only what their position and the geometry between them
## allow — direction, distance, loudness, category and age.
##
## Deliberate information limits (Phase 3): a heard event never carries the
## emitter's identity or its exact coordinates. The perceived direction is
## perturbed by a deterministic, position-derived error so that sound gives
## an approximate bearing, not a free aimbot. Occlusion attenuates rather
## than silences: a shot behind a wall is quieter but still audible.
##
## Events are stored in a plain Array capped at SOUND_MAX_ACTIVE and are
## dropped oldest-first, so the bus never grows without bound on a long
## episode and costs no allocation per tick beyond the events themselves.
class_name SoundBus
extends RefCounted

enum Category {
	FOOTSTEP = 0,
	JUMP = 1,
	LAND = 2,
	SHOT = 3,
	IMPACT = 4,
	DEATH = 5,
	## Environmental noise with no character behind it. A listener can hear
	## it and must learn that it means nothing tactically.
	ENVIRONMENT = 6,
}

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const CATEGORY_COUNT: int = 7

const SELF_PATH: String = "res://scripts/perception/sound_bus.gd"

## Active events. Each entry is a Dictionary:
##   {position, category, loudness, age, source_id}
## `source_id` is used ONLY to let an emitter ignore its own sounds; it is
## never exposed to a listener.
var events: Array = []
## Monotonic simulation time this bus has seen, in seconds.
var time_seconds: float = 0.0

## Fixed positions that emit environmental noise, in round-robin order.
## Set once per episode from the map; empty means a silent map.
var ambient_sources: Array = []
var ambient_interval: float = SandboxConfig.SOUND_AMBIENCE_INTERVAL
var _ambient_timer: float = 0.0
var _ambient_cursor: int = 0


static func create() -> SoundBus:
	return (load(SELF_PATH) as GDScript).new() as SoundBus


func clear() -> void:
	events.clear()
	time_seconds = 0.0
	ambient_sources = []
	_ambient_timer = 0.0
	_ambient_cursor = 0


## Installs the map's environmental noise emitters. Round-robin and
## interval-driven, so ambience is fully determined by the episode clock and
## consumes no random numbers.
func configure_ambience(
	sources: Array, interval: float = SandboxConfig.SOUND_AMBIENCE_INTERVAL
) -> void:
	ambient_sources = sources
	ambient_interval = maxf(0.1, interval)
	_ambient_timer = 0.0
	_ambient_cursor = 0


## Base audible radius (in meters, before occlusion) for a category.
static func base_loudness(category: int) -> float:
	match category:
		Category.FOOTSTEP:
			return SandboxConfig.SOUND_FOOTSTEP_RADIUS
		Category.JUMP:
			return SandboxConfig.SOUND_JUMP_RADIUS
		Category.LAND:
			return SandboxConfig.SOUND_LAND_RADIUS
		Category.SHOT:
			return SandboxConfig.SOUND_SHOT_RADIUS
		Category.IMPACT:
			return SandboxConfig.SOUND_IMPACT_RADIUS
		Category.DEATH:
			return SandboxConfig.SOUND_DEATH_RADIUS
		Category.ENVIRONMENT:
			return SandboxConfig.SOUND_ENVIRONMENT_RADIUS
		_:
			return SandboxConfig.SOUND_FOOTSTEP_RADIUS


static func category_name(value: int) -> String:
	match value:
		Category.FOOTSTEP:
			return "footstep"
		Category.JUMP:
			return "jump"
		Category.LAND:
			return "land"
		Category.SHOT:
			return "shot"
		Category.IMPACT:
			return "impact"
		Category.DEATH:
			return "death"
		Category.ENVIRONMENT:
			return "environment"
		_:
			return "unknown"


## Records a sound. `source_id` identifies the emitter (-1 = the agent,
## >= 0 = enemy index) so listeners can skip their own noise.
func emit_sound(category: int, position: Vector3, source_id: int, loudness: float = -1.0) -> void:
	var resolved_loudness: float = loudness if loudness > 0.0 else base_loudness(category)
	if events.size() >= SandboxConfig.SOUND_MAX_ACTIVE:
		events.remove_at(0)
	events.append(
		{
			"position": position,
			"category": category,
			"loudness": resolved_loudness,
			"age": 0.0,
			"source_id": source_id,
		}
	)


## Ages every event and discards expired ones. Called once per simulation
## tick by EnvironmentCore.
func tick(dt: float) -> void:
	time_seconds += dt
	_tick_ambience(dt)
	if events.is_empty():
		return
	var surviving: Array = []
	for event_value in events:
		var event: Dictionary = event_value
		var age: float = float(event["age"]) + dt
		if age <= SandboxConfig.SOUND_EVENT_LIFETIME:
			event["age"] = age
			surviving.append(event)
	events = surviving


func _tick_ambience(dt: float) -> void:
	if ambient_sources.is_empty():
		return
	_ambient_timer += dt
	while _ambient_timer >= ambient_interval:
		_ambient_timer -= ambient_interval
		var position: Vector3 = ambient_sources[_ambient_cursor % ambient_sources.size()]
		_ambient_cursor += 1
		emit_sound(Category.ENVIRONMENT, position, SandboxConfig.SOUND_UNKNOWN_SOURCE_ID)


## What a listener at `listener_position` can currently hear.
##
## Returns an Array of Dictionaries sorted loudest-first:
##   {category, category_name, direction, distance, loudness, age,
##    occluders, bearing_deg, direction_error_deg, confidence, masked}
## `direction` is a unit vector carrying the PERCEIVED (error-perturbed)
## bearing, not the true one, and `direction_error_deg` is the listener's
## own estimate of how wrong that bearing may be: a muffled, distant sound
## is placed far less precisely than a close, clear one. `confidence`
## combines loudness, occlusion and that angular error into one [0, 1]
## trust value, and `masked` marks an event that was partly drowned out by
## a louder simultaneous one.
##
## Nothing here consumes random numbers: every perturbation is derived from
## the source position, so an episode replays identically.
func sample(
	listener_position: Vector3,
	listener_forward: Vector3,
	world,
	ignore_source_id: int = -9999,
	min_age: float = 0.0
) -> Array:
	var heard: Array = []
	if events.is_empty():
		return heard
	for event_value in events:
		var event: Dictionary = event_value
		if int(event["source_id"]) == ignore_source_id:
			continue
		var age: float = float(event["age"])
		if age < min_age:
			# Detection latency: the event exists but has not been
			# consciously registered by this listener yet.
			continue
		var source: Vector3 = event["position"]
		var delta: Vector3 = source - listener_position
		var distance: float = delta.length()
		var occluders: int = 0
		if world != null:
			occluders = (world as ArenaWorld).occluder_count(listener_position, source)
		var radius: float = float(event["loudness"]) * pow(
			SandboxConfig.SOUND_OCCLUSION_ATTENUATION, float(occluders)
		)
		if distance > radius or radius <= 0.0:
			continue
		var perceived: float = clampf(1.0 - distance / maxf(radius, 0.0001), 0.0, 1.0)
		# Fade with age so a two-second-old footstep is weaker than a fresh one.
		perceived *= clampf(1.0 - age / maxf(SandboxConfig.SOUND_EVENT_LIFETIME, 0.0001), 0.0, 1.0)
		if perceived <= 0.0:
			continue
		var error_deg: float = _direction_error_deg(occluders, distance, radius)
		var direction: Vector3 = _perceived_direction(delta, distance, source, error_deg)
		heard.append(
			{
				"category": int(event["category"]),
				"category_name": category_name(int(event["category"])),
				"direction": direction,
				"distance": distance,
				"loudness": perceived,
				"age": age,
				"occluders": occluders,
				"bearing_deg": _bearing(listener_forward, direction),
				"direction_error_deg": error_deg,
				"confidence": _confidence(perceived, error_deg),
				"masked": false,
			}
		)
	heard.sort_custom(func(a, b): return float(a["loudness"]) > float(b["loudness"]))
	return _apply_masking(heard)


## The loudest currently audible event, or {} when nothing is heard.
func loudest(
	listener_position: Vector3,
	listener_forward: Vector3,
	world,
	ignore_source_id: int = -9999,
	min_age: float = 0.0
) -> Dictionary:
	var heard: Array = sample(listener_position, listener_forward, world, ignore_source_id, min_age)
	return heard[0] if heard.size() > 0 else {}


## How badly this listener can place the sound, in degrees. Grows with the
## number of walls in the way and with how close the event is to the edge
## of its audible radius.
static func _direction_error_deg(occluders: int, distance: float, radius: float) -> float:
	var range_ratio: float = clampf(distance / maxf(radius, 0.0001), 0.0, 1.0)
	var scale: float = (
		1.0
		+ float(occluders) * SandboxConfig.SOUND_OCCLUSION_ERROR_GAIN
		+ range_ratio * SandboxConfig.SOUND_RANGE_ERROR_GAIN
	)
	return SandboxConfig.SOUND_DIRECTION_ERROR_DEG * scale


## Trust in a heard event: loud and well-placed is trustworthy, faint and
## smeared across 40 degrees is not.
static func _confidence(perceived: float, error_deg: float) -> float:
	var angular: float = clampf(1.0 - error_deg / 90.0, 0.05, 1.0)
	return clampf(perceived * angular, 0.0, 1.0)


## Competing sounds mask each other. Each event is attenuated by the
## loudest OTHER event heard at the same moment; what falls under
## SOUND_MASK_FLOOR was drowned out and is never reported. This is why a
## careful enemy can move under covering fire.
static func _apply_masking(heard: Array) -> Array:
	if heard.size() < 2:
		return heard
	var dominant: float = float((heard[0] as Dictionary)["loudness"])
	var runner_up: float = float((heard[1] as Dictionary)["loudness"])
	var surviving: Array = []
	for index in range(heard.size()):
		var event: Dictionary = heard[index]
		var masker: float = runner_up if index == 0 else dominant
		var factor: float = clampf(1.0 - SandboxConfig.SOUND_MASKING_STRENGTH * masker, 0.0, 1.0)
		var loudness: float = float(event["loudness"]) * factor
		if loudness < SandboxConfig.SOUND_MASK_FLOOR:
			continue
		event["masked"] = factor < 0.999
		event["loudness"] = loudness
		event["confidence"] = _confidence(loudness, float(event["direction_error_deg"]))
		surviving.append(event)
	surviving.sort_custom(func(a, b): return float(a["loudness"]) > float(b["loudness"]))
	return surviving


## Aggregate view of everything heard this tick, for the observation vector
## and the Control Center. Only derived quantities: no positions, no ids.
##
##   count                 -- events registered
##   distinct_sources      -- clusters of bearings at least
##                            SOUND_DISTINCT_SOURCE_ANGLE_DEG apart, i.e. how
##                            many different places noise is coming from
##   dominant_bearing_deg  -- bearing of the loudest event
##   second_bearing_deg    -- bearing of the second loudest (0 when alone)
##   second_loudness       -- loudness of the second loudest
##   mean_confidence       -- average trust across the events
##   direction_error_deg   -- angular error of the loudest event
##   masked_count          -- events that were partly drowned out
static func summarize(heard: Array) -> Dictionary:
	var summary: Dictionary = {
		"count": heard.size(),
		"distinct_sources": 0,
		"dominant_bearing_deg": 0.0,
		"second_bearing_deg": 0.0,
		"second_loudness": 0.0,
		"mean_confidence": 0.0,
		"direction_error_deg": 0.0,
		"masked_count": 0,
	}
	if heard.is_empty():
		return summary
	var bearings: Array = []
	var confidence_total: float = 0.0
	var masked: int = 0
	for event_value in heard:
		var event: Dictionary = event_value
		var bearing: float = float(event.get("bearing_deg", 0.0))
		confidence_total += float(event.get("confidence", 0.0))
		if bool(event.get("masked", false)):
			masked += 1
		var distinct: bool = true
		for known_value in bearings:
			if absf(bearing - float(known_value)) < SandboxConfig.SOUND_DISTINCT_SOURCE_ANGLE_DEG:
				distinct = false
				break
		if distinct:
			bearings.append(bearing)
	var loudest_event: Dictionary = heard[0]
	summary["distinct_sources"] = bearings.size()
	summary["dominant_bearing_deg"] = float(loudest_event.get("bearing_deg", 0.0))
	summary["direction_error_deg"] = float(loudest_event.get("direction_error_deg", 0.0))
	summary["mean_confidence"] = confidence_total / float(heard.size())
	summary["masked_count"] = masked
	if heard.size() > 1:
		var second: Dictionary = heard[1]
		summary["second_bearing_deg"] = float(second.get("bearing_deg", 0.0))
		summary["second_loudness"] = float(second.get("loudness", 0.0))
	return summary


## Deterministic directional error. Derived from the source position (not
## from an RNG stream) so replaying the same episode produces identical
## perceived bearings without consuming random numbers.
func _perceived_direction(
	delta: Vector3, distance: float, source: Vector3, error_deg: float
) -> Vector3:
	if distance < 0.000001:
		return Vector3.ZERO
	var direction: Vector3 = delta / distance
	var hash_input: float = source.x * 12.9898 + source.z * 78.233 + source.y * 37.719
	var pseudo: float = sin(hash_input) * 43758.5453
	pseudo = pseudo - floor(pseudo)  # fract() in [0,1)
	return direction.rotated(Vector3.UP, deg_to_rad((pseudo * 2.0 - 1.0) * error_deg)).normalized()


func _bearing(forward: Vector3, direction: Vector3) -> float:
	var flat_forward := Vector3(forward.x, 0.0, forward.z)
	var flat_direction := Vector3(direction.x, 0.0, direction.z)
	if flat_forward.is_zero_approx() or flat_direction.is_zero_approx():
		return 0.0
	flat_forward = flat_forward.normalized()
	flat_direction = flat_direction.normalized()
	var dot: float = clampf(flat_forward.dot(flat_direction), -1.0, 1.0)
	var angle: float = rad_to_deg(acos(dot))
	return angle if flat_forward.cross(flat_direction).y >= 0.0 else -angle


func to_dict() -> Array:
	var out: Array = []
	for event_value in events:
		var event: Dictionary = event_value
		out.append(
			{
				"category": int(event["category"]),
				"category_name": category_name(int(event["category"])),
				"position": event["position"],
				"loudness": float(event["loudness"]),
				"age": float(event["age"]),
			}
		)
	return out

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
}

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const CATEGORY_COUNT: int = 6

const SELF_PATH: String = "res://scripts/perception/sound_bus.gd"

## Active events. Each entry is a Dictionary:
##   {position, category, loudness, age, source_id}
## `source_id` is used ONLY to let an emitter ignore its own sounds; it is
## never exposed to a listener.
var events: Array = []
## Monotonic simulation time this bus has seen, in seconds.
var time_seconds: float = 0.0


static func create() -> SoundBus:
	return (load(SELF_PATH) as GDScript).new() as SoundBus


func clear() -> void:
	events.clear()
	time_seconds = 0.0


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


## What a listener at `listener_position` can currently hear.
##
## Returns an Array of Dictionaries sorted loudest-first:
##   {category, category_name, direction, distance, loudness, age,
##    occluders, bearing_deg}
## `direction` is a unit vector carrying the PERCEIVED (error-perturbed)
## bearing, not the true one.
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
		var direction: Vector3 = _perceived_direction(delta, distance, source)
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
			}
		)
	heard.sort_custom(func(a, b): return float(a["loudness"]) > float(b["loudness"]))
	return heard


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


## Deterministic directional error. Derived from the source position (not
## from an RNG stream) so replaying the same episode produces identical
## perceived bearings without consuming random numbers.
func _perceived_direction(delta: Vector3, distance: float, source: Vector3) -> Vector3:
	if distance < 0.000001:
		return Vector3.ZERO
	var direction: Vector3 = delta / distance
	var hash_input: float = source.x * 12.9898 + source.z * 78.233 + source.y * 37.719
	var pseudo: float = sin(hash_input) * 43758.5453
	pseudo = pseudo - floor(pseudo)  # fract() in [0,1)
	var error_deg: float = (pseudo * 2.0 - 1.0) * SandboxConfig.SOUND_DIRECTION_ERROR_DEG
	return direction.rotated(Vector3.UP, deg_to_rad(error_deg)).normalized()


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

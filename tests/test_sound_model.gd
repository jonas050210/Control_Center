## Tests for the advanced sound model.
##
## Hearing is the second perception channel, so the properties tested here
## are the ones that decide whether it is a real sense or a free radar:
## attenuation with distance and walls, approximate (never exact) direction
## with an honest error estimate, timing, masking between simultaneous
## sounds, environmental noise that means nothing, and full determinism.
class_name TestSoundModel
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")

const LISTENER := Vector3.ZERO
const FORWARD := Vector3(0.0, 0.0, -1.0)


func _walled_world() -> ArenaWorld:
	var world: ArenaWorld = ArenaWorld.create(20.0, 3.0)
	world.add_obstacle(
		Obstacle.make(Vector3(0.0, 1.5, -4.0), Vector3(6.0, 1.5, 0.4), Obstacle.Kind.WALL)
	)
	return world


func _heard_after_delay(bus: SoundBus, world) -> Array:
	bus.tick(SandboxConfig.SOUND_DETECTION_DELAY + 0.001)
	return bus.sample(LISTENER, FORWARD, world, -9999, SandboxConfig.SOUND_DETECTION_DELAY)


func test_every_category_has_a_radius_and_a_name() -> SandboxTest:
	var t := SandboxTest.new("every_category_has_a_radius_and_a_name")
	t.assert_eq(SoundBus.CATEGORY_COUNT, 7, "footstep, jump, land, shot, impact, death, ambient")
	for category in range(SoundBus.CATEGORY_COUNT):
		t.assert_gt(SoundBus.base_loudness(category), 0.0)
		t.assert_ne(SoundBus.category_name(category), "unknown")
	return t


func test_loudness_falls_off_with_distance() -> SandboxTest:
	var t := SandboxTest.new("loudness_falls_off_with_distance")
	var near_bus: SoundBus = SoundBus.create()
	near_bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(0.0, 0.0, -2.0), 1)
	var far_bus: SoundBus = SoundBus.create()
	far_bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(0.0, 0.0, -8.0), 1)
	var near_heard: Array = _heard_after_delay(near_bus, null)
	var far_heard: Array = _heard_after_delay(far_bus, null)
	t.assert_eq(near_heard.size(), 1)
	t.assert_eq(far_heard.size(), 1)
	t.assert_gt(
		float((near_heard[0] as Dictionary)["loudness"]),
		float((far_heard[0] as Dictionary)["loudness"])
	)
	return t


func test_out_of_range_sounds_are_not_heard() -> SandboxTest:
	var t := SandboxTest.new("out_of_range_sounds_are_not_heard")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(
		SoundBus.Category.FOOTSTEP,
		Vector3(0.0, 0.0, -(SandboxConfig.SOUND_FOOTSTEP_RADIUS + 2.0)),
		1
	)
	t.assert_eq(_heard_after_delay(bus, null).size(), 0)
	return t


func test_walls_muffle_but_do_not_silence() -> SandboxTest:
	var t := SandboxTest.new("walls_muffle_but_do_not_silence")
	var world: ArenaWorld = _walled_world()
	var open_bus: SoundBus = SoundBus.create()
	open_bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -8.0), 1)
	var walled_bus: SoundBus = SoundBus.create()
	walled_bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -8.0), 1)
	var open_heard: Array = _heard_after_delay(open_bus, null)
	var walled_heard: Array = _heard_after_delay(walled_bus, world)
	t.assert_eq(open_heard.size(), 1)
	t.assert_eq(walled_heard.size(), 1, "a shot behind a wall is quieter, not inaudible")
	var occluded: Dictionary = walled_heard[0]
	t.assert_gt(float(occluded["occluders"]), 0.0)
	t.assert_lt(float(occluded["loudness"]), float((open_heard[0] as Dictionary)["loudness"]))
	t.assert_gt(
		float(occluded["direction_error_deg"]),
		float((open_heard[0] as Dictionary)["direction_error_deg"]),
		"a muffled sound must be harder to place"
	)
	return t


func test_direction_is_approximate_but_never_random() -> SandboxTest:
	var t := SandboxTest.new("direction_is_approximate_but_never_random")
	var source := Vector3(5.0, 0.0, -5.0)
	var first: SoundBus = SoundBus.create()
	first.emit_sound(SoundBus.Category.SHOT, source, 1)
	var second: SoundBus = SoundBus.create()
	second.emit_sound(SoundBus.Category.SHOT, source, 1)
	var a: Dictionary = (_heard_after_delay(first, null))[0]
	var b: Dictionary = (_heard_after_delay(second, null))[0]
	t.assert_almost_eq(float(a["bearing_deg"]), float(b["bearing_deg"]), 0.000001)

	var true_direction: Vector3 = (source - LISTENER).normalized()
	var perceived: Vector3 = a["direction"]
	var error_deg: float = rad_to_deg(acos(clampf(true_direction.dot(perceived), -1.0, 1.0)))
	t.assert_gt(error_deg, 0.0, "hearing must not be a pinpoint sensor")
	t.assert_lte(
		error_deg,
		float(a["direction_error_deg"]) + 0.001,
		"the reported uncertainty must bound the real error"
	)
	return t


func test_detection_delay_hides_a_brand_new_sound() -> SandboxTest:
	var t := SandboxTest.new("detection_delay_hides_a_brand_new_sound")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -5.0), 1)
	var immediate: Array = bus.sample(
		LISTENER, FORWARD, null, -9999, SandboxConfig.SOUND_DETECTION_DELAY
	)
	t.assert_eq(immediate.size(), 0, "a sound is not registered on the tick it happens")
	t.assert_eq(_heard_after_delay(bus, null).size(), 1)
	return t


func test_sounds_expire_and_fade_with_age() -> SandboxTest:
	var t := SandboxTest.new("sounds_expire_and_fade_with_age")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -5.0), 1)
	bus.tick(0.2)
	var fresh: float = float((bus.sample(LISTENER, FORWARD, null)[0] as Dictionary)["loudness"])
	bus.tick(1.0)
	var older: float = float((bus.sample(LISTENER, FORWARD, null)[0] as Dictionary)["loudness"])
	t.assert_lt(older, fresh)
	bus.tick(SandboxConfig.SOUND_EVENT_LIFETIME)
	t.assert_eq(bus.sample(LISTENER, FORWARD, null).size(), 0, "expired events are discarded")
	return t


func test_emitter_never_hears_itself_and_identity_is_not_exposed() -> SandboxTest:
	var t := SandboxTest.new("emitter_never_hears_itself_and_identity_is_not_exposed")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(0.0, 0.0, -3.0), 7)
	bus.tick(0.2)
	t.assert_eq(bus.sample(LISTENER, FORWARD, null, 7).size(), 0)
	var heard: Array = bus.sample(LISTENER, FORWARD, null, 99)
	t.assert_eq(heard.size(), 1)
	var event: Dictionary = heard[0]
	t.assert_false(event.has("source_id"), "a listener must never learn who made the noise")
	t.assert_false(event.has("position"), "a listener must never get exact coordinates")
	return t


func test_a_loud_sound_masks_a_quiet_simultaneous_one() -> SandboxTest:
	var t := SandboxTest.new("a_loud_sound_masks_a_quiet_simultaneous_one")
	var quiet_only: SoundBus = SoundBus.create()
	quiet_only.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(1.0, 0.0, -1.0), 1)
	var alone: Array = _heard_after_delay(quiet_only, null)
	t.assert_eq(alone.size(), 1)
	var alone_loudness: float = float((alone[0] as Dictionary)["loudness"])

	var both: SoundBus = SoundBus.create()
	both.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(1.0, 0.0, -1.0), 1)
	both.emit_sound(SoundBus.Category.SHOT, Vector3(-1.0, 0.0, -1.0), 2)
	var mixed: Array = _heard_after_delay(both, null)
	t.assert_gt(float(mixed.size()), 0.0)
	var footstep: Dictionary = {}
	for event_value in mixed:
		var event: Dictionary = event_value
		if int(event["category"]) == SoundBus.Category.FOOTSTEP:
			footstep = event
	if footstep.is_empty():
		# Fully drowned out is the strongest possible form of masking.
		t.assert_true(true)
	else:
		t.assert_lt(float(footstep["loudness"]), alone_loudness, "the shot must mask the footstep")
		t.assert_true(bool(footstep["masked"]))
	return t


func test_summary_counts_distinct_directions() -> SandboxTest:
	var t := SandboxTest.new("summary_counts_distinct_directions")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -6.0), 1)
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, 6.0), 2)
	var heard: Array = _heard_after_delay(bus, null)
	var summary: Dictionary = SoundBus.summarize(heard)
	t.assert_eq(int(summary["count"]), heard.size())
	t.assert_gte(float(summary["distinct_sources"]), 2.0, "front and back are different places")
	t.assert_gt(float(summary["mean_confidence"]), 0.0)
	var empty: Dictionary = SoundBus.summarize([])
	t.assert_eq(int(empty["count"]), 0)
	t.assert_eq(int(empty["distinct_sources"]), 0)
	t.assert_almost_eq(float(empty["mean_confidence"]), 0.0, 0.0001)
	return t


func test_environmental_noise_is_heard_but_carries_no_identity() -> SandboxTest:
	var t := SandboxTest.new("environmental_noise_is_heard_but_carries_no_identity")
	var bus: SoundBus = SoundBus.create()
	bus.configure_ambience([Vector3(0.0, 0.0, -4.0)], 1.0)
	bus.tick(0.5)
	t.assert_eq(bus.events.size(), 0, "ambience fires on its own schedule, not instantly")
	bus.tick(0.6)
	t.assert_eq(bus.events.size(), 1)
	var heard: Array = bus.sample(LISTENER, FORWARD, null, -1, 0.0)
	t.assert_eq(heard.size(), 1)
	var event: Dictionary = heard[0]
	t.assert_eq(int(event["category"]), SoundBus.Category.ENVIRONMENT)
	t.assert_eq(str(event["category_name"]), "environment")
	return t


func test_ambience_is_deterministic_and_bounded() -> SandboxTest:
	var t := SandboxTest.new("ambience_is_deterministic_and_bounded")
	var counts: Array = []
	for _run in range(2):
		var bus: SoundBus = SoundBus.create()
		bus.configure_ambience([Vector3(2.0, 0.0, 0.0), Vector3(-2.0, 0.0, 0.0)], 0.5)
		for _i in range(100):
			bus.tick(0.1)
		counts.append(bus.events.size())
	t.assert_eq(counts[0], counts[1])
	t.assert_lte(float(counts[0]), float(SandboxConfig.SOUND_MAX_ACTIVE))
	return t


func test_bus_never_grows_past_its_cap() -> SandboxTest:
	var t := SandboxTest.new("bus_never_grows_past_its_cap")
	var bus: SoundBus = SoundBus.create()
	for i in range(SandboxConfig.SOUND_MAX_ACTIVE * 3):
		bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(float(i % 5), 0.0, 0.0), i)
	t.assert_lte(float(bus.events.size()), float(SandboxConfig.SOUND_MAX_ACTIVE))
	return t


func test_clear_resets_the_bus_and_its_ambience() -> SandboxTest:
	var t := SandboxTest.new("clear_resets_the_bus_and_its_ambience")
	var bus: SoundBus = SoundBus.create()
	bus.configure_ambience([Vector3(1.0, 0.0, 1.0)], 0.5)
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -2.0), 1)
	bus.tick(1.0)
	bus.clear()
	t.assert_eq(bus.events.size(), 0)
	t.assert_eq(bus.ambient_sources.size(), 0)
	t.assert_almost_eq(bus.time_seconds, 0.0, 0.0001)
	bus.tick(5.0)
	t.assert_eq(bus.events.size(), 0, "a cleared bus must stay silent")
	return t

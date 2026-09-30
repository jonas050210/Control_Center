## Tests for the perception layer: FOV cones, line-of-sight occlusion,
## sound propagation/attenuation and reaction-latency profiles.
class_name TestPerception
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const ReactionProfile = preload("res://scripts/perception/reaction_profile.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")

## yaw = 0 forward, matching AgentState.get_forward_horizontal().
const FORWARD: Vector3 = Vector3(0.0, 0.0, -1.0)


func test_bearing_is_signed_and_zero_dead_ahead() -> SandboxTest:
	var t := SandboxTest.new("bearing_is_signed_and_zero_dead_ahead")
	t.assert_almost_eq(
		PerceptionSystem.bearing_deg(FORWARD, Vector3.ZERO, Vector3(0.0, 0.0, -5.0)), 0.0, 0.01
	)
	var right: float = PerceptionSystem.bearing_deg(FORWARD, Vector3.ZERO, Vector3(5.0, 0.0, -5.0))
	var left: float = PerceptionSystem.bearing_deg(FORWARD, Vector3.ZERO, Vector3(-5.0, 0.0, -5.0))
	t.assert_almost_eq(absf(right), 45.0, 0.01)
	t.assert_almost_eq(absf(left), 45.0, 0.01)
	t.assert_true(right * left < 0.0, "left and right must have opposite signs")
	t.assert_almost_eq(
		absf(PerceptionSystem.bearing_deg(FORWARD, Vector3.ZERO, Vector3(0.0, 0.0, 5.0))),
		180.0,
		0.01
	)
	return t


func test_field_of_view_cone_excludes_targets_outside_the_half_angle() -> SandboxTest:
	var t := SandboxTest.new("field_of_view_cone_excludes_targets_outside_the_half_angle")
	# 90 degree total FOV -> +/-45 degrees.
	t.assert_true(
		PerceptionSystem.in_field_of_view(FORWARD, Vector3.ZERO, Vector3(0.0, 0.0, -5.0), 90.0)
	)
	t.assert_true(
		PerceptionSystem.in_field_of_view(FORWARD, Vector3.ZERO, Vector3(4.0, 0.0, -5.0), 90.0)
	)
	t.assert_false(
		PerceptionSystem.in_field_of_view(FORWARD, Vector3.ZERO, Vector3(6.0, 0.0, -5.0), 90.0),
		"50 degrees off-axis must be outside a 90 degree cone"
	)
	t.assert_false(
		PerceptionSystem.in_field_of_view(FORWARD, Vector3.ZERO, Vector3(0.0, 0.0, 5.0), 90.0),
		"a target directly behind must never be in the FOV"
	)
	return t


func test_line_of_sight_is_blocked_by_high_cover_but_not_by_low_cover() -> SandboxTest:
	var t := SandboxTest.new("line_of_sight_is_blocked_by_high_cover_but_not_by_low_cover")
	var eye := Vector3(0.0, 1.6, 4.0)
	var target := Vector3(0.0, 0.0, -4.0)

	var high: ArenaWorld = ArenaWorld.create(10.0)
	high.add_box(Vector3(0.0, 1.4, 0.0), Vector3(3.0, 1.4, 0.5), Obstacle.Kind.HIGH_COVER)
	t.assert_false(
		PerceptionSystem.has_line_of_sight(high, eye, target, 1.8),
		"a 2.8 m tall wall must fully occlude a standing target"
	)

	var low: ArenaWorld = ArenaWorld.create(10.0)
	low.add_box(Vector3(0.0, 0.35, 0.0), Vector3(3.0, 0.35, 0.5), Obstacle.Kind.LOW_COVER)
	t.assert_true(
		PerceptionSystem.has_line_of_sight(low, eye, target, 1.8),
		"a 0.7 m crate must not hide a standing target's head"
	)

	t.assert_true(
		PerceptionSystem.has_line_of_sight(null, eye, target, 1.8), "an empty world never occludes"
	)
	return t


func test_evaluate_target_reports_range_fov_and_occlusion_separately() -> SandboxTest:
	var t := SandboxTest.new("evaluate_target_reports_range_fov_and_occlusion_separately")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 1.5, 0.0), Vector3(3.0, 1.5, 0.5), Obstacle.Kind.HIGH_COVER)

	var occluded: Dictionary = PerceptionSystem.evaluate_target(
		world, Vector3(0.0, 1.6, 4.0), Vector3(0.0, 0.0, 4.0), FORWARD, Vector3(0.0, 0.0, -4.0), 1.8
	)
	t.assert_true(bool(occluded["in_range"]))
	t.assert_true(bool(occluded["in_fov"]))
	t.assert_false(bool(occluded["los_clear"]))
	t.assert_false(bool(occluded["visible"]), "occlusion alone must make a target invisible")

	var behind: Dictionary = PerceptionSystem.evaluate_target(
		null, Vector3(0.0, 1.6, 0.0), Vector3.ZERO, FORWARD, Vector3(0.0, 0.0, 5.0), 1.8
	)
	t.assert_true(bool(behind["los_clear"]))
	t.assert_false(bool(behind["in_fov"]))
	t.assert_false(bool(behind["visible"]), "a target behind the observer must be invisible")

	var far: Dictionary = PerceptionSystem.evaluate_target(
		null, Vector3(0.0, 1.6, 0.0), Vector3.ZERO, FORWARD, Vector3(0.0, 0.0, -500.0), 1.8
	)
	t.assert_false(bool(far["in_range"]))
	t.assert_false(bool(far["visible"]))
	return t


func test_sound_attenuates_with_distance_and_expires() -> SandboxTest:
	var t := SandboxTest.new("sound_attenuates_with_distance_and_expires")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(0.0, 0.0, 0.0), 0)

	var near: Array = bus.sample(Vector3(1.0, 0.0, 0.0), FORWARD, null, -1)
	var far: Array = bus.sample(Vector3(8.0, 0.0, 0.0), FORWARD, null, -1)
	var silent: Array = bus.sample(Vector3(40.0, 0.0, 0.0), FORWARD, null, -1)
	t.assert_eq(near.size(), 1)
	t.assert_eq(far.size(), 1)
	t.assert_eq(silent.size(), 0, "beyond the audible radius nothing is heard")
	t.assert_gt(
		float((near[0] as Dictionary)["loudness"]),
		float((far[0] as Dictionary)["loudness"]),
		"a closer sound must be louder"
	)

	for _i in range(200):
		bus.tick(1.0 / 60.0)
	t.assert_eq(bus.events.size(), 0, "events must expire after SOUND_EVENT_LIFETIME")
	return t


func test_sound_is_quieter_through_a_wall() -> SandboxTest:
	var t := SandboxTest.new("sound_is_quieter_through_a_wall")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 1.5, 0.0), Vector3(4.0, 1.5, 0.5), Obstacle.Kind.HIGH_COVER)
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, -4.0), 0)

	var open: Array = bus.sample(Vector3(0.0, 0.0, 4.0), FORWARD, null, -1)
	var through_wall: Array = bus.sample(Vector3(0.0, 0.0, 4.0), FORWARD, world, -1)
	t.assert_eq(open.size(), 1)
	t.assert_eq(through_wall.size(), 1, "a wall attenuates a shot, it does not silence it")
	t.assert_gt(
		float((open[0] as Dictionary)["loudness"]),
		float((through_wall[0] as Dictionary)["loudness"])
	)
	t.assert_gte(float((through_wall[0] as Dictionary)["occluders"]), 1.0)
	return t


func test_listener_ignores_its_own_sounds_and_respects_detection_delay() -> SandboxTest:
	var t := SandboxTest.new("listener_ignores_its_own_sounds_and_respects_detection_delay")
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(1.0, 0.0, 0.0), 3)
	t.assert_eq(
		bus.sample(Vector3.ZERO, FORWARD, null, 3).size(), 0, "a source must not hear itself"
	)
	t.assert_eq(bus.sample(Vector3.ZERO, FORWARD, null, 7).size(), 1)
	t.assert_eq(
		bus.sample(Vector3.ZERO, FORWARD, null, 7, 0.5).size(),
		0,
		"a fresh event must not be reported before the detection delay elapses"
	)
	bus.tick(0.6)
	t.assert_eq(bus.sample(Vector3.ZERO, FORWARD, null, 7, 0.5).size(), 1)
	return t


func test_sound_direction_is_approximate_but_deterministic() -> SandboxTest:
	var t := SandboxTest.new("sound_direction_is_approximate_but_deterministic")
	var bus: SoundBus = SoundBus.create()
	var source := Vector3(0.0, 0.0, -5.0)
	bus.emit_sound(SoundBus.Category.SHOT, source, 0)
	var first: Dictionary = bus.sample(Vector3.ZERO, FORWARD, null, -1)[0]
	var second: Dictionary = bus.sample(Vector3.ZERO, FORWARD, null, -1)[0]
	t.assert_vec_almost_eq(
		first["direction"], second["direction"], 0.000001, "perceived direction must be stable"
	)
	var true_direction: Vector3 = (source - Vector3.ZERO).normalized()
	var error_deg: float = rad_to_deg(
		acos(clampf((first["direction"] as Vector3).dot(true_direction), -1.0, 1.0))
	)
	t.assert_lte(
		error_deg,
		SandboxConfig.SOUND_DIRECTION_ERROR_DEG + 0.5,
		"directional error must stay within the configured bound"
	)
	return t


func test_sound_bus_is_capped_and_drops_oldest_first() -> SandboxTest:
	var t := SandboxTest.new("sound_bus_is_capped_and_drops_oldest_first")
	var bus: SoundBus = SoundBus.create()
	for index in range(SandboxConfig.SOUND_MAX_ACTIVE + 12):
		bus.emit_sound(SoundBus.Category.FOOTSTEP, Vector3(float(index), 0.0, 0.0), 0)
	t.assert_eq(bus.events.size(), SandboxConfig.SOUND_MAX_ACTIVE)
	var first_event: Dictionary = bus.events[0]
	t.assert_almost_eq(float(first_event["position"].x), 12.0, 0.001)
	return t


func test_reaction_archetypes_are_ordered_and_never_instant_by_default() -> SandboxTest:
	var t := SandboxTest.new("reaction_archetypes_are_ordered_and_never_instant_by_default")
	var rookie: ReactionProfile = ReactionProfile.create(ReactionProfile.Archetype.ROOKIE)
	var regular: ReactionProfile = ReactionProfile.create(ReactionProfile.Archetype.REGULAR)
	var elite: ReactionProfile = ReactionProfile.create(ReactionProfile.Archetype.ELITE)
	t.assert_gt(rookie.total_engagement_latency(), regular.total_engagement_latency())
	t.assert_gt(regular.total_engagement_latency(), elite.total_engagement_latency())
	t.assert_gt(elite.total_engagement_latency(), 0.0, "even an elite enemy is not instant")
	t.assert_gt(elite.aim_speed_deg, rookie.aim_speed_deg)
	t.assert_lt(elite.aim_error_deg, rookie.aim_error_deg)

	var instant: ReactionProfile = ReactionProfile.create(ReactionProfile.Archetype.INSTANT)
	t.assert_almost_eq(instant.total_engagement_latency(), 0.0)
	t.assert_eq(ReactionProfile.archetype_name(ReactionProfile.Archetype.VETERAN), "veteran")
	return t

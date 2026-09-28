## Tests for the map abstraction (MapLibrary) and the environmental
## visibility model (LightingProfile).
##
## The two most important properties asserted here are information
## boundaries, not gameplay: a map must expose geometry and lighting to the
## SIMULATION and prose/tags only to HUMANS, and lighting must act on
## perception rather than becoming a label.
class_name TestMapsAndLighting
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const MapLibrary = preload("res://scripts/world/map_library.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")


func test_every_map_resolves_to_geometry_and_lighting() -> SandboxTest:
	var t := SandboxTest.new("every_map_resolves_to_geometry_and_lighting")
	for map_value in MapLibrary.ids():
		var map_id: String = str(map_value)
		var instance: Dictionary = MapLibrary.resolve(map_id, 31337)
		t.assert_eq(str(instance["map_id"]), map_id)
		var world: ArenaWorld = instance["world"]
		t.assert_not_null(world, "%s produced no world" % map_id)
		t.assert_gt(world.half_extent, 1.0)
		var lighting: LightingProfile = instance["lighting"]
		t.assert_not_null(lighting, "%s produced no lighting profile" % map_id)
		t.assert_true(
			WorldGenerator.LAYOUT_IDS.has(str(instance["layout_id"])),
			"%s chose an unknown layout %s" % [map_id, instance["layout_id"]]
		)
	return t


func test_map_resolution_is_deterministic_for_a_seed() -> SandboxTest:
	var t := SandboxTest.new("map_resolution_is_deterministic_for_a_seed")
	for map_value in MapLibrary.ids():
		var map_id: String = str(map_value)
		var a: Dictionary = MapLibrary.resolve(map_id, 7)
		var b: Dictionary = MapLibrary.resolve(map_id, 7)
		t.assert_eq(str(a["layout_id"]), str(b["layout_id"]), "%s layout differs" % map_id)
		t.assert_eq(str(a["lighting_id"]), str(b["lighting_id"]), "%s lighting differs" % map_id)
		var world_a: ArenaWorld = a["world"]
		var world_b: ArenaWorld = b["world"]
		t.assert_eq(world_a.obstacle_count(), world_b.obstacle_count())
		for index in range(world_a.obstacle_count()):
			var left: Obstacle = world_a.obstacles[index]
			var right: Obstacle = world_b.obstacles[index]
			t.assert_vec_almost_eq(left.center, right.center, 0.0001)
			t.assert_vec_almost_eq(left.half_extents, right.half_extents, 0.0001)
	return t


func test_different_seeds_vary_the_randomized_map() -> SandboxTest:
	var t := SandboxTest.new("different_seeds_vary_the_randomized_map")
	var seen_layouts: Dictionary = {}
	var seen_lighting: Dictionary = {}
	for seed_value in range(24):
		var instance: Dictionary = MapLibrary.resolve("random_ops", seed_value * 977 + 1)
		seen_layouts[str(instance["layout_id"])] = true
		seen_lighting[str(instance["lighting_id"])] = true
	t.assert_gt(
		float(seen_lighting.size()), 1.0, "random_ops must sample more than one lighting mode"
	)
	return t


func test_metadata_is_human_only_and_never_carries_simulation_state() -> SandboxTest:
	var t := SandboxTest.new("metadata_is_human_only_and_never_carries_simulation_state")
	var metadata: Dictionary = MapLibrary.metadata("compound")
	t.assert_true(bool(metadata["known"]))
	t.assert_ne(str(metadata["label"]), "")
	t.assert_ne(str(metadata["description"]), "")
	# Metadata must be pure description: no geometry, no spawns, no enemies.
	t.assert_false(metadata.has("world"), "metadata must not carry the world")
	t.assert_false(metadata.has("enemy_spawns"), "metadata must not carry spawn positions")
	t.assert_false(metadata.has("lighting"), "metadata must not carry a live LightingProfile")
	# An unknown map still returns a usable, clearly-marked descriptor.
	var unknown: Dictionary = MapLibrary.metadata("does_not_exist")
	t.assert_false(bool(unknown["known"]))
	return t


func test_unknown_map_degrades_instead_of_failing() -> SandboxTest:
	var t := SandboxTest.new("unknown_map_degrades_instead_of_failing")
	t.assert_false(MapLibrary.has_map("nope"))
	var instance: Dictionary = MapLibrary.resolve("nope", 5)
	t.assert_eq(str(instance["map_id"]), "open_field")
	return t


func test_tag_query_finds_conceptually_related_maps() -> SandboxTest:
	var t := SandboxTest.new("tag_query_finds_conceptually_related_maps")
	var cover_maps: PackedStringArray = MapLibrary.ids_with_tags(["cover"])
	t.assert_gt(float(cover_maps.size()), 1.0, "several maps should teach cover")
	var night_maps: PackedStringArray = MapLibrary.ids_with_tags(["night"])
	t.assert_gt(float(night_maps.size()), 0.0)
	for id_value in night_maps:
		t.assert_true(
			MapLibrary.metadata(str(id_value))["tags"].has("night"),
			"%s was returned for the night tag but does not carry it" % id_value
		)
	return t


func test_lighting_modes_reduce_detection_range_monotonically() -> SandboxTest:
	var t := SandboxTest.new("lighting_modes_reduce_detection_range_monotonically")
	var point := Vector3(3.0, 0.0, -2.0)
	var normal: float = LightingProfile.create(LightingProfile.Mode.NORMAL).detection_range(
		SandboxConfig.VISION_RANGE, point
	)
	var low: float = LightingProfile.create(LightingProfile.Mode.LOW_LIGHT).detection_range(
		SandboxConfig.VISION_RANGE, point
	)
	var night: float = LightingProfile.create(LightingProfile.Mode.NIGHT).detection_range(
		SandboxConfig.VISION_RANGE, point
	)
	t.assert_almost_eq(normal, SandboxConfig.VISION_RANGE, 0.001)
	t.assert_lt(low, normal, "low light must shorten acquisition range")
	t.assert_lt(night, low, "night must shorten it further")
	t.assert_gt(night, 0.0, "range must never collapse to zero")
	return t


func test_fog_imposes_a_distance_horizon() -> SandboxTest:
	var t := SandboxTest.new("fog_imposes_a_distance_horizon")
	var fog: LightingProfile = LightingProfile.create(LightingProfile.Mode.FOG)
	t.assert_lt(fog.transmittance(20.0), fog.transmittance(5.0), "fog must attenuate with range")
	t.assert_almost_eq(
		LightingProfile.create(LightingProfile.Mode.NORMAL).transmittance(20.0), 1.0, 0.0001
	)
	var horizon: float = fog.detection_range(SandboxConfig.VISION_RANGE, Vector3.ZERO)
	t.assert_lt(horizon, SandboxConfig.VISION_RANGE, "fog must cap the visible range")
	return t


func test_detection_delay_grows_in_the_dark() -> SandboxTest:
	var t := SandboxTest.new("detection_delay_grows_in_the_dark")
	var point := Vector3(1.0, 0.0, 1.0)
	var normal: float = LightingProfile.create(LightingProfile.Mode.NORMAL).detection_delay_scale(
		point
	)
	var night: float = LightingProfile.create(LightingProfile.Mode.NIGHT).detection_delay_scale(
		point
	)
	t.assert_almost_eq(normal, 1.0, 0.0001, "normal light must not change reaction time")
	t.assert_gt(night, normal, "darkness must cost reaction time")
	t.assert_lt(
		LightingProfile.create(LightingProfile.Mode.NIGHT).loss_grace_scale(point),
		1.0,
		"darkness must shorten how long a lost contact is still reported"
	)
	return t


func test_spatially_varying_lighting_is_deterministic_and_varied() -> SandboxTest:
	var t := SandboxTest.new("spatially_varying_lighting_is_deterministic_and_varied")
	var mixed: LightingProfile = LightingProfile.create(LightingProfile.Mode.MIXED, 4242)
	var same: LightingProfile = LightingProfile.create(LightingProfile.Mode.MIXED, 4242)
	var brightest: float = -1.0
	var darkest: float = 2.0
	for step in range(40):
		var point := Vector3(float(step) * 0.9 - 18.0, 0.0, float(step % 7) * 2.3 - 8.0)
		var value: float = mixed.illumination_at(point)
		t.assert_almost_eq(
			value, same.illumination_at(point), 0.000001, "same seed must give same light"
		)
		t.assert_gte(value, 0.0)
		t.assert_lte(value, 1.0)
		brightest = maxf(brightest, value)
		darkest = minf(darkest, value)
	t.assert_gt(brightest - darkest, 0.05, "MIXED lighting must actually vary across the map")

	# A uniform mode must be flat everywhere.
	var night: LightingProfile = LightingProfile.create(LightingProfile.Mode.NIGHT, 99)
	t.assert_almost_eq(
		night.illumination_at(Vector3(-9.0, 0.0, 9.0)),
		night.illumination_at(Vector3(8.0, 0.0, -3.0)),
		0.000001
	)
	return t


func test_lighting_mode_ids_round_trip() -> SandboxTest:
	var t := SandboxTest.new("lighting_mode_ids_round_trip")
	for mode_value in LightingProfile.MODE_IDS:
		var mode_id: String = str(mode_value)
		var profile: LightingProfile = LightingProfile.from_id(mode_id, 1)
		t.assert_eq(LightingProfile.mode_id(profile.mode), mode_id)
	# An unknown id must degrade to NORMAL rather than failing.
	t.assert_eq(LightingProfile.from_id("moonless_eclipse", 0).mode, LightingProfile.Mode.NORMAL)
	return t


func test_new_layouts_are_generated_and_deterministic() -> SandboxTest:
	var t := SandboxTest.new("new_layouts_are_generated_and_deterministic")
	for layout_value in ["multi_room", "ambush", "sound_maze"]:
		var layout_id: String = str(layout_value)
		t.assert_true(WorldGenerator.LAYOUT_IDS.has(layout_id))
		var a: ArenaWorld = WorldGenerator.build(layout_id, 808)
		var b: ArenaWorld = WorldGenerator.build(layout_id, 808)
		t.assert_gt(float(a.obstacle_count()), 1.0, "%s must place geometry" % layout_id)
		t.assert_eq(a.obstacle_count(), b.obstacle_count())
		for index in range(a.obstacle_count()):
			t.assert_vec_almost_eq(
				(a.obstacles[index] as Obstacle).center,
				(b.obstacles[index] as Obstacle).center,
				0.0001
			)
	return t

## Tests for ControlCenterConfig: the Control Center's plain data model.
## Covers mode naming/parsing, clamping, scenario presets, the
## "requires reset" classification and dictionary round-tripping.
class_name TestControlCenterConfig
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_mode_names_round_trip() -> SandboxTest:
	var t := SandboxTest.new("control_center_config_mode_names_round_trip")
	for mode in [
		ControlCenterConfig.Mode.TRAINING,
		ControlCenterConfig.Mode.WATCH,
		ControlCenterConfig.Mode.HUMAN
	]:
		var name_value: String = ControlCenterConfig.mode_name(mode)
		t.assert_eq(
			ControlCenterConfig.mode_from_name(name_value), mode, "round trip for %s" % name_value
		)
	t.assert_eq(
		ControlCenterConfig.mode_from_name("play"),
		ControlCenterConfig.Mode.HUMAN,
		"'play' is an accepted alias for HUMAN"
	)
	t.assert_eq(
		ControlCenterConfig.mode_from_name("nonsense"),
		ControlCenterConfig.Mode.WATCH,
		"unknown mode names fall back to WATCH instead of crashing"
	)
	return t


func test_sanitize_clamps_every_numeric_field() -> SandboxTest:
	var t := SandboxTest.new("control_center_config_sanitize_clamps")
	var config := ControlCenterConfig.new()
	config.mode = 99
	config.environment_count = 0
	config.enemy_count = 10000
	config.curriculum_level = 42
	config.seed = -5
	config.simulation_speed = 1000.0
	config.selected_environment = 77
	config.camera_mode = -3
	config.policy_source = 9
	config.sanitize()

	t.assert_eq(config.mode, ControlCenterConfig.Mode.HUMAN, "mode clamped into the enum range")
	t.assert_eq(config.environment_count, 1, "at least one environment")
	t.assert_eq(config.enemy_count, ControlCenterConfig.MAX_ENEMY_COUNT, "enemy count capped")
	t.assert_eq(
		config.curriculum_level,
		CurriculumConfig.Level.AGENT_VS_AGENT,
		"curriculum level clamped to the highest defined level"
	)
	t.assert_eq(config.seed, 0, "negative seeds are not passed to the simulation")
	t.assert_almost_eq(config.simulation_speed, ControlCenterConfig.MAX_SPEED, 0.001)
	t.assert_eq(config.selected_environment, 0, "selection clamped to the environment count")
	t.assert_eq(config.camera_mode, ControlCenterConfig.CameraMode.FIRST_PERSON)
	t.assert_eq(config.policy_source, ControlCenterConfig.PolicySource.EXTERNAL_POLICY)
	return t


func test_rebuild_settings_are_explicitly_classified() -> SandboxTest:
	var t := SandboxTest.new("control_center_config_rebuild_settings")
	for key in ["environment_count", "enemy_count", "seed"]:
		t.assert_true(
			ControlCenterConfig.requires_rebuild(key), "%s must be marked as requiring reset" % key
		)
	for key in ["curriculum_level", "simulation_speed", "camera_mode"]:
		t.assert_false(
			ControlCenterConfig.requires_rebuild(key),
			"%s applies live and must not be pending" % key
		)
	return t


func test_scenarios_only_bundle_supported_settings() -> SandboxTest:
	var t := SandboxTest.new("control_center_config_scenarios")
	var config := ControlCenterConfig.new()
	var changed: PackedStringArray = config.apply_scenario("three_way")
	t.assert_eq(config.curriculum_level, CurriculumConfig.Level.MULTIPLE_ENEMIES)
	t.assert_eq(config.enemy_count, 3)
	t.assert_true(changed.has("enemy_count"), "the enemy-count change is reported to the caller")
	t.assert_eq(config.scenario_id, "three_way")

	# Every scenario may only contain keys the simulation genuinely has.
	var allowed: PackedStringArray = PackedStringArray(
		["id", "label", "description", "curriculum_level", "enemy_count"]
	)
	for entry_value in ControlCenterConfig.SCENARIOS:
		var entry: Dictionary = entry_value
		for key in entry.keys():
			t.assert_true(allowed.has(str(key)), "unexpected scenario key %s" % str(key))

	t.assert_true(
		config.apply_scenario("does_not_exist").is_empty(), "an unknown scenario id changes nothing"
	)
	return t


func test_policy_sources_report_availability_honestly() -> SandboxTest:
	var t := SandboxTest.new("control_center_config_policy_sources")
	t.assert_true(
		ControlCenterConfig.policy_source_available(ControlCenterConfig.PolicySource.HEURISTIC)
	)
	t.assert_true(
		ControlCenterConfig.policy_source_available(ControlCenterConfig.PolicySource.IDLE)
	)
	t.assert_false(
		ControlCenterConfig.policy_source_available(
			ControlCenterConfig.PolicySource.EXTERNAL_POLICY
		),
		"in-engine inference does not exist and must not be offered as working"
	)
	return t


func test_dictionary_round_trip_and_duplicate_are_independent() -> SandboxTest:
	var t := SandboxTest.new("control_center_config_round_trip")
	var config := ControlCenterConfig.new(ControlCenterConfig.Mode.HUMAN)
	config.environment_count = 3
	config.enemy_count = 4
	config.seed = 99
	config.simulation_speed = 2.0
	config.show_bottom_panel = false

	var copy: ControlCenterConfig = config.duplicate_config()
	t.assert_eq(copy.mode, config.mode)
	t.assert_eq(copy.environment_count, 3)
	t.assert_eq(copy.enemy_count, 4)
	t.assert_eq(copy.seed, 99)
	t.assert_almost_eq(copy.simulation_speed, 2.0, 0.001)
	t.assert_false(copy.show_bottom_panel)

	copy.enemy_count = 1
	t.assert_eq(config.enemy_count, 4, "the copy must not alias the original")

	var restored := ControlCenterConfig.new()
	restored.apply_dict(config.to_dict())
	t.assert_eq(restored.to_dict()["enemy_count"], 4)
	t.assert_eq(restored.to_dict()["mode_name"], "HUMAN")
	return t

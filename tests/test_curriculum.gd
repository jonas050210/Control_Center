## Tests for curriculum progression, level scaling, and difficulty configuration.
class_name TestCurriculum
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_curriculum_level_constants_and_names() -> SandboxTest:
	var t := SandboxTest.new("curriculum_level_constants_and_names")
	var l1: int = CurriculumConfig.Level.STATIONARY_TARGET
	var l2: int = CurriculumConfig.Level.MOVING_TARGET
	var l3: int = CurriculumConfig.Level.ENEMY_ATTACKS
	var l4: int = CurriculumConfig.Level.MULTIPLE_ENEMIES
	var l5: int = CurriculumConfig.Level.AGENT_VS_AGENT
	t.assert_eq(CurriculumConfig.level_name(l1), "stationary_target")
	t.assert_eq(CurriculumConfig.level_name(l2), "moving_target")
	t.assert_eq(CurriculumConfig.level_name(l3), "enemy_attacks")
	t.assert_eq(CurriculumConfig.level_name(l4), "multiple_enemies")
	t.assert_eq(CurriculumConfig.level_name(l5), "agent_vs_agent")
	return t


func test_curriculum_flags_match_design() -> SandboxTest:
	var t := SandboxTest.new("curriculum_flags_match_design")
	var c1 := CurriculumConfig.new(CurriculumConfig.Level.STATIONARY_TARGET)
	t.assert_false(c1.enemy_movement_enabled())
	t.assert_false(c1.enemy_attacks_enabled())
	t.assert_almost_eq(c1.target_radius_scale(), 1.5)

	var c2 := CurriculumConfig.new(CurriculumConfig.Level.MOVING_TARGET)
	t.assert_true(c2.enemy_movement_enabled())
	t.assert_false(c2.enemy_attacks_enabled())
	t.assert_almost_eq(c2.target_radius_scale(), 1.0)

	var c3 := CurriculumConfig.new(CurriculumConfig.Level.ENEMY_ATTACKS)
	t.assert_true(c3.enemy_movement_enabled())
	t.assert_true(c3.enemy_attacks_enabled())

	var c4 := CurriculumConfig.new(CurriculumConfig.Level.MULTIPLE_ENEMIES, 1)
	t.assert_gte(float(c4.effective_enemy_count()), 2.0)
	return t


func test_spawn_variety_and_strafing_flags_progress_with_level() -> SandboxTest:
	var t := SandboxTest.new("spawn_variety_and_strafing_flags_progress_with_level")
	var c1 := CurriculumConfig.new(CurriculumConfig.Level.STATIONARY_TARGET)
	t.assert_false(c1.spawn_variety_enabled(), "level 1 keeps the fixed legacy spawn layout")
	t.assert_false(c1.strafing_enabled(), "level 1 enemies must not strafe")
	t.assert_almost_eq(c1.spawn_angle_spread_deg(), 0.0)

	var c2 := CurriculumConfig.new(CurriculumConfig.Level.MOVING_TARGET)
	t.assert_true(c2.spawn_variety_enabled(), "level 2 should vary spawn position")
	t.assert_false(c2.strafing_enabled(), "level 2 enemies chase but do not strafe")
	t.assert_gt(c2.spawn_angle_spread_deg(), 0.0)

	var c3 := CurriculumConfig.new(CurriculumConfig.Level.ENEMY_ATTACKS)
	t.assert_true(c3.spawn_variety_enabled())
	t.assert_true(c3.strafing_enabled(), "level 3 enemies should strafe while engaging")

	var c5 := CurriculumConfig.new(CurriculumConfig.Level.AGENT_VS_AGENT)
	t.assert_false(
		c5.strafing_enabled(), "the self-play hook level should not force EnemyState strafing"
	)
	return t


func test_multiple_enemies_level_requires_at_least_three() -> SandboxTest:
	var t := SandboxTest.new("multiple_enemies_level_requires_at_least_three")
	var c := CurriculumConfig.new(CurriculumConfig.Level.MULTIPLE_ENEMIES, 1)
	t.assert_eq(c.effective_enemy_count(), 3, "level 4 should default up to 3 enemies")
	var c_more := CurriculumConfig.new(CurriculumConfig.Level.MULTIPLE_ENEMIES, 5)
	t.assert_eq(c_more.effective_enemy_count(), 5, "an explicit larger count should be respected")
	return t


func test_curriculum_to_dict_reports_new_fields() -> SandboxTest:
	var t := SandboxTest.new("curriculum_to_dict_reports_new_fields")
	var c := CurriculumConfig.new(CurriculumConfig.Level.ENEMY_ATTACKS)
	var data: Dictionary = c.to_dict()
	t.assert_true(data.has("spawn_variety"))
	t.assert_true(data.has("strafing"))
	t.assert_true(data.has("spawn_angle_spread_deg"))
	t.assert_true(data.has("spawn_distance_range"))
	return t

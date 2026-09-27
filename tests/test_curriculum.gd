## Tests for curriculum progression, level scaling, and difficulty configuration.
class_name TestCurriculum
extends RefCounted

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

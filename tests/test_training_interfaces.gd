## Regression tests for the Python-facing action/observation contract and
## terminal metrics introduced by the real training pipeline.
class_name TestTrainingInterfaces
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_action_multidiscrete_round_trip_has_fixed_dimension() -> SandboxTest:
	var t := SandboxTest.new("action_multidiscrete_round_trip")
	var original := Action.new(-1, 1, 0, -1, true)
	var encoded: Array = original.to_multidiscrete()
	var decoded := Action.from_multidiscrete(encoded)
	t.assert_eq(encoded.size(), Action.MULTI_DISCRETE_SIZE)
	t.assert_eq(decoded.move_axis, original.move_axis)
	t.assert_eq(decoded.strafe_axis, original.strafe_axis)
	t.assert_eq(decoded.look_yaw_axis, original.look_yaw_axis)
	t.assert_eq(decoded.look_pitch_axis, original.look_pitch_axis)
	t.assert_eq(decoded.shoot, original.shoot)
	return t


func test_environment_step_exposes_metrics_and_terminal_reason() -> SandboxTest:
	var t := SandboxTest.new("environment_step_exposes_metrics")
	var env := EnvironmentCore.new(0, 1)
	env.max_steps = 1
	env.reset(44)
	var result: Dictionary = env.step(Action.idle())
	t.assert_true(result.info.has("metrics"))
	t.assert_eq(result.info.metrics.episode_length, 1)
	t.assert_eq(result.info.metrics.done_reason, "timeout")
	t.assert_true(result.info.metrics.has("shots_fired"))
	return t


func test_curriculum_level_changes_enemy_behavior_without_new_environment() -> SandboxTest:
	var t := SandboxTest.new("curriculum_level_changes_enemy_behavior")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	env.reset(2)
	# `env.enemies` is an untyped Array, so `env.enemies[0].position` is a
	# Variant and cannot drive `:=` type inference under Godot 4.7. Declare the
	# type explicitly.
	var start: Vector3 = env.enemies[0].position
	env.step(Action.idle())
	t.assert_vec_almost_eq(env.enemies[0].position, start, 0.0001)
	env.set_curriculum_level(CurriculumConfig.Level.MOVING_TARGET)
	env.reset(2)
	start = env.enemies[0].position
	env.step(Action.idle())
	t.assert_true(env.enemies[0].position.distance_to(start) > 0.0)
	return t

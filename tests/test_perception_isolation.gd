## Tests that the simulation's debug/introspection flag can OBSERVE
## perception without ever CHANGING what the policy sees, and that the
## headless training path does not pay for the debug machinery.
##
## This is the invariant that keeps any future observer honest: the
## perception context may be evaluated for inspection at any curriculum
## level, but the observation vector must be bit-identical whether or not
## anyone is watching. (The in-simulator Control Center that used this flag
## was removed with the move to the headless-only desktop Control Center;
## the simulation-side guarantee it relied on stays pinned here.)
class_name TestPerceptionIsolation
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")


func _run(level: int, seed_value: int, debug: bool) -> Array:
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(level)
	env.debug_perception = debug
	env.reset(seed_value)
	var trace: Array = []
	for step_index in range(150):
		var action: Action = Action.new(
			1, (step_index % 3) - 1, 1, 0, step_index % 4 == 0, Vector2.ZERO, step_index % 37 == 0
		)
		env.step(action)
		trace.append(env.get_observations().to_array())
	return trace


func test_debug_perception_does_not_change_the_observation() -> SandboxTest:
	var t := SandboxTest.new("debug_perception_does_not_change_the_observation")
	for level in [
		CurriculumConfig.Level.ENEMY_ATTACKS,
		CurriculumConfig.Level.OBSTACLES_COVER,
		CurriculumConfig.Level.MEMORY_LOST_TARGETS
	]:
		var plain: Array = _run(level, 6161, false)
		var watched: Array = _run(level, 6161, true)
		t.assert_eq(plain.size(), watched.size())
		for step_index in range(plain.size()):
			var left: PackedFloat32Array = plain[step_index]
			var right: PackedFloat32Array = watched[step_index]
			for field in range(left.size()):
				t.assert_almost_eq(
					left[field],
					right[field],
					0.000001,
					(
						"level %d step %d field %d changed when debug_perception was on"
						% [level, step_index, field]
					)
				)
	return t


func test_headless_environment_defaults_to_no_debug_perception() -> SandboxTest:
	var t := SandboxTest.new("headless_environment_defaults_to_no_debug_perception")
	var env := EnvironmentCore.new(0, 1)
	t.assert_false(env.debug_perception, "debug perception must be opt-in")
	env.set_curriculum_level(CurriculumConfig.Level.ENEMY_ATTACKS)
	env.reset(1)
	env.step(Action.idle())
	# With perception neither gated nor debugged, no belief list is built at
	# all — that is the cost saving the headless hot path relies on.
	t.assert_eq((env.get_target_memory()["beliefs"] as Array).size(), 0)
	return t

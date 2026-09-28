## Tests that the debug/visualization layer can OBSERVE perception without
## ever CHANGING what the policy sees, and that headless training does not
## pay for the debug machinery.
##
## This is the invariant that keeps the Control Center honest: it may draw
## FOV cones, LOS rays, memory markers and sound pings at any curriculum
## level, but the observation vector must be bit-identical whether or not
## anyone is watching.
class_name TestPerceptionIsolation
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const PerceptionOverlay3D = preload("res://scripts/control_center/perception_overlay_3d.gd")
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
					"level %d step %d field %d changed when debug_perception was on"
					% [level, step_index, field]
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


func test_perception_model_exposes_the_new_state_for_the_overlay() -> SandboxTest:
	var t := SandboxTest.new("perception_model_exposes_the_new_state_for_the_overlay")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.MEMORY_LOST_TARGETS)
	env.reset(808)
	for step_index in range(120):
		env.step(Action.new(1, 0, 1, 0, step_index % 6 == 0))

	var perception: Dictionary = PerceptionModel.build(env)
	t.assert_true(perception.has("perception_state"))
	var state: Dictionary = perception["perception_state"]
	t.assert_true(bool(state["gated"]), "level 8 gates the observation")
	t.assert_gt(float((state["obstacles"] as Array).size()), 0.0)
	t.assert_true((state["field_of_view"] as Dictionary).has("fov_deg"))
	t.assert_true(state.has("memory"))
	t.assert_true(state.has("sounds"))
	t.assert_true(state.has("corpses"))

	var lines: PackedStringArray = PerceptionModel.format_lines(perception)
	var joined: String = "\n".join(lines)
	t.assert_true(joined.contains("REAL WORLD"))
	t.assert_true(joined.contains("AI PERCEPTION"))
	t.assert_true(joined.contains("AI MEMORY"))
	t.assert_true(joined.contains("SOUND"))
	return t


func test_overlay_accepts_the_new_perception_state_without_touching_the_env() -> SandboxTest:
	var t := SandboxTest.new("overlay_accepts_the_new_perception_state")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.VERTICAL_COMBAT)
	env.reset(414)
	for _i in range(60):
		env.step(Action.new(1, 0, 0, 0, false))
	var before: PackedFloat32Array = env.get_observations().to_array()

	var overlay := PerceptionOverlay3D.new()
	overlay.visible = true
	overlay.update_from_perception(PerceptionModel.build(env))
	overlay.update_from_perception({})
	overlay.free()

	var after: PackedFloat32Array = env.get_observations().to_array()
	for index in range(before.size()):
		t.assert_almost_eq(before[index], after[index], 0.000001, "the overlay mutated the env")
	return t


func test_environment_without_perception_still_reports_empty_hook_payloads() -> SandboxTest:
	var t := SandboxTest.new("environment_without_perception_reports_empty_payloads")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	env.reset(2)
	t.assert_eq(env.get_obstacles().size(), 0)
	t.assert_eq(env.get_sound_events().size(), 0)
	t.assert_eq(env.get_dead_bodies().size(), 0)
	t.assert_false(bool(env.get_agent_field_of_view()["enabled"]))
	t.assert_true(
		env.has_line_of_sight(Vector3.ZERO, Vector3(0.0, 0.0, -10.0)),
		"an empty arena never occludes"
	)
	t.assert_eq(str(env.get_navigation_state()["layout_id"]), "none")
	return t

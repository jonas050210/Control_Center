## Tests for explicit target selection (TargetSelector) and for how the
## observation scales with the number of enemies.
##
## Two questions decide whether this part of the system is honest:
##   * is the choice of target explainable in terms of named factors the
##     agent could actually perceive, and stable enough not to oscillate?
##   * does the observation keep the same shape and the same meaning with
##     1, 2, 3, 5 or 8 enemies, while still telling the agent that it is
##     outnumbered?
class_name TestTargetAndScaling
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentPerception = preload("res://scripts/perception/agent_perception.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const TargetSelector = preload("res://scripts/perception/target_selector.gd")


func _belief(id: int, distance: float, visible: bool, confidence: float = 1.0) -> Dictionary:
	return {
		"id": id,
		"visible": visible,
		"distance": distance,
		"confidence": confidence,
		"age": 0.0,
		"health_norm": 1.0,
		"threatening": false,
		"position": Vector3(distance, 0.0, 0.0),
		"source": 1,
	}


func test_visible_beats_remembered_and_near_beats_far() -> SandboxTest:
	var t := SandboxTest.new("visible_beats_remembered_and_near_beats_far")
	var ranked: Array = TargetSelector.rank(
		[_belief(1, 2.0, false, 0.5), _belief(0, 18.0, true)], {}
	)
	t.assert_eq(int((ranked[0] as Dictionary)["id"]), 0, "a visible contact outranks a memory")

	var by_distance: Array = TargetSelector.rank(
		[_belief(1, 9.0, true), _belief(0, 4.0, true)], {}
	)
	t.assert_eq(int((by_distance[0] as Dictionary)["id"]), 0, "the nearer visible contact wins")
	return t


func test_recent_damage_and_threat_raise_priority() -> SandboxTest:
	var t := SandboxTest.new("recent_damage_and_threat_raise_priority")
	var near: Dictionary = _belief(0, 4.0, true)
	var far: Dictionary = _belief(1, 9.0, true)
	var by_damage: Array = TargetSelector.rank([near, far], {"damage_source": 1})
	t.assert_eq(int((by_damage[0] as Dictionary)["id"]), 1, "whoever just hurt me comes first")

	var threatening: Dictionary = _belief(2, 12.0, true)
	threatening["threatening"] = true
	var by_threat: Array = TargetSelector.rank([near, threatening], {})
	t.assert_eq(
		int((by_threat[0] as Dictionary)["id"]),
		2,
		"a contact that can see me outranks a closer one that cannot"
	)
	return t


func test_unreachable_contacts_are_deprioritised_but_not_hidden() -> SandboxTest:
	var t := SandboxTest.new("unreachable_contacts_are_deprioritised_but_not_hidden")
	var near_blocked: Dictionary = _belief(0, 3.0, false, 0.9)
	var far_open: Dictionary = _belief(1, 12.0, false, 0.9)
	var ranked: Array = TargetSelector.rank([near_blocked, far_open], {"unreachable": [0]})
	t.assert_eq(int((ranked[0] as Dictionary)["id"]), 1)
	t.assert_eq(ranked.size(), 2, "an unreachable contact is still reported, just ranked lower")
	t.assert_true(
		TargetSelector.reason(near_blocked, {"unreachable": [0]}).contains("unreachable")
	)
	return t


func test_continuity_prevents_oscillation_between_equal_threats() -> SandboxTest:
	var t := SandboxTest.new("continuity_prevents_oscillation_between_equal_threats")
	var a: Dictionary = _belief(0, 6.0, true)
	var b: Dictionary = _belief(1, 6.0, true)
	var without: Dictionary = TargetSelector.select([a, b], {})
	var first_id: int = int((without["target"] as Dictionary)["id"])
	var with_history: Dictionary = TargetSelector.select(
		[a, b], {"previous_target_id": 1 - first_id}
	)
	t.assert_eq(
		int((with_history["target"] as Dictionary)["id"]),
		1 - first_id,
		"an equally good current target must be kept"
	)
	t.assert_false(bool(with_history["switched"]))
	return t


func test_selection_reports_priority_and_a_reason() -> SandboxTest:
	var t := SandboxTest.new("selection_reports_priority_and_a_reason")
	var result: Dictionary = TargetSelector.select([_belief(0, 3.0, true)], {"damage_source": 0})
	t.assert_gt(float(result["priority_norm"]), 0.0)
	t.assert_lte(float(result["priority_norm"]), 1.0)
	t.assert_true(str(result["reason"]).contains("damaged"))
	var factors: Dictionary = result["factors"]
	for factor_value in TargetSelector.FACTORS:
		t.assert_true(factors.has(factor_value), "missing factor %s" % factor_value)

	var empty: Dictionary = TargetSelector.select([], {})
	t.assert_true((empty["target"] as Dictionary).is_empty())
	t.assert_eq(str(empty["reason"]), "no target")
	t.assert_almost_eq(float(empty["priority_norm"]), 0.0, 0.0001)
	return t


func test_contact_summary_describes_the_overflow() -> SandboxTest:
	var t := SandboxTest.new("contact_summary_describes_the_overflow")
	var beliefs: Array = [
		_belief(0, 2.0, true),
		_belief(1, 4.0, true),
		_belief(2, 6.0, true),
		_belief(3, 8.0, true),
		_belief(4, 20.0, false, 0.4),
	]
	var summary: Dictionary = AgentPerception.summarize_contacts(beliefs, 3)
	t.assert_eq(int(summary["contact_count"]), 5)
	t.assert_eq(int(summary["visible_count"]), 4)
	t.assert_eq(int(summary["remembered_count"]), 1)
	t.assert_eq(int(summary["overflow_count"]), 2, "two contacts beyond the three slots")
	t.assert_eq(int(summary["overflow_visible"]), 1)
	t.assert_almost_eq(float(summary["overflow_min_distance"]), 8.0, 0.0001)
	t.assert_almost_eq(float(summary["overflow_mean_distance"]), 14.0, 0.0001)
	t.assert_almost_eq(float(summary["memory_uncertainty"]), 0.6, 0.0001)
	return t


func test_empty_contact_summary_is_all_zero() -> SandboxTest:
	var t := SandboxTest.new("empty_contact_summary_is_all_zero")
	var summary: Dictionary = AgentPerception.summarize_contacts([], 3)
	t.assert_eq(int(summary["contact_count"]), 0)
	t.assert_eq(int(summary["overflow_count"]), 0)
	t.assert_almost_eq(float(summary["overflow_mean_distance"]), 0.0, 0.0001)
	t.assert_almost_eq(float(summary["memory_uncertainty"]), 0.0, 0.0001)
	return t


func test_observation_shape_is_independent_of_enemy_count() -> SandboxTest:
	var t := SandboxTest.new("observation_shape_is_independent_of_enemy_count")
	for count_value in [1, 2, 3, 5, 8]:
		var count: int = int(count_value)
		var env := EnvironmentCore.new(0, count)
		env.set_curriculum_level(CurriculumConfig.Level.FOV_LOS)
		env.reset(1234)
		t.assert_eq(env.enemies.size(), maxi(count, 3), "level 8+ enforces a minimum count")
		var observation: Observation = env.get_observations()
		var array: PackedFloat32Array = observation.to_array()
		t.assert_eq(array.size(), Observation.FIELD_COUNT, "the vector must never change shape")
		for _step in range(20):
			env.step(Action.from_multidiscrete([2, 1, 2, 1, 0, 0]), SandboxConfig.SIMULATION_DT)
		t.assert_eq(env.get_observations().to_array().size(), Observation.FIELD_COUNT)
	return t


func test_more_enemies_produce_more_overflow_signal() -> SandboxTest:
	var t := SandboxTest.new("more_enemies_produce_more_overflow_signal")
	var few: Array = [_belief(0, 3.0, true), _belief(1, 5.0, true)]
	var many: Array = [
		_belief(0, 3.0, true),
		_belief(1, 5.0, true),
		_belief(2, 6.0, true),
		_belief(3, 7.0, true),
		_belief(4, 9.0, true),
		_belief(5, 11.0, true),
	]
	var few_summary: Dictionary = AgentPerception.summarize_contacts(few, 3)
	var many_summary: Dictionary = AgentPerception.summarize_contacts(many, 3)
	t.assert_eq(int(few_summary["overflow_count"]), 0, "two contacts fit in three slots")
	t.assert_eq(int(many_summary["overflow_count"]), 3)
	t.assert_gt(
		float(many_summary["visible_count"]),
		float(few_summary["visible_count"]),
		"being outnumbered must be visible in the observation"
	)
	return t


func test_v3_fields_are_neutral_without_perception_context() -> SandboxTest:
	var t := SandboxTest.new("v3_fields_are_neutral_without_perception_context")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	env.reset(7)
	var array: PackedFloat32Array = env.get_observations().to_array()
	t.assert_eq(array.size(), Observation.FIELD_COUNT)
	# Levels 1-4 supply no context, so every v3 field must read as
	# "no information" rather than as a plausible-looking guess.
	for index in range(66, 78):
		if index == 71 or index == 77:
			continue
		t.assert_almost_eq(array[index], 0.0, 0.0001, "field %d must be neutral" % index)
	t.assert_almost_eq(array[65], 1.0, 0.0001, "unlit episodes report full daylight")
	return t

## Tests for ObservationInspector: the Observation Inspector must be driven
## by the EXISTING contract definitions, never by a second hand-maintained
## list of field names. These tests fail if the inspector ever drifts from
## Observation.FIELD_SPEC / RLAdapter.action_space_info().
class_name TestObservationInspector
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const ObservationInspector = preload("res://scripts/control_center/observation_inspector.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_inspector_covers_the_full_contract() -> SandboxTest:
	var t := SandboxTest.new("inspector_covers_full_contract")
	t.assert_true(
		ObservationInspector.covers_full_contract(),
		"field_names() must describe exactly FIELD_COUNT entries"
	)
	t.assert_eq(Observation.field_names().size(), Observation.FIELD_COUNT)

	var spec_width: int = 0
	for entry_value in Observation.FIELD_SPEC:
		var entry: Dictionary = entry_value
		spec_width += int(entry["width"])
	t.assert_eq(spec_width, Observation.FIELD_COUNT, "FIELD_SPEC widths must sum to FIELD_COUNT")
	return t


func test_rows_mirror_the_observation_vector_exactly() -> SandboxTest:
	var t := SandboxTest.new("inspector_rows_mirror_vector")
	var env := EnvironmentCore.new(0, 2)
	env.reset(4242)
	for _step in range(12):
		env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))
	var observation: Observation = env.get_observations()
	var vector: PackedFloat32Array = observation.to_array()

	var rows: Array = ObservationInspector.build_rows(observation)
	t.assert_eq(rows.size(), vector.size(), "one row per observation index")
	var seen_names: Dictionary = {}
	for index in range(rows.size()):
		var row: Dictionary = rows[index]
		t.assert_eq(int(row["index"]), index, "rows stay in contract order")
		t.assert_almost_eq(
			float(row["value"]), float(vector[index]), 0.00001, "row %d value" % index
		)
		t.assert_false(str(row["name"]).is_empty(), "row %d has a name" % index)
		t.assert_false(str(row["group"]).is_empty(), "row %d has a group" % index)
		t.assert_false(seen_names.has(str(row["name"])), "duplicate field name %s" % row["name"])
		seen_names[str(row["name"])] = true
	return t


func test_rows_accept_a_raw_vector_too() -> SandboxTest:
	var t := SandboxTest.new("inspector_accepts_raw_vector")
	var values := PackedFloat32Array()
	values.resize(Observation.FIELD_COUNT)
	for index in range(Observation.FIELD_COUNT):
		values[index] = float(index) / 100.0
	var rows: Array = ObservationInspector.build_rows(values)
	t.assert_eq(rows.size(), Observation.FIELD_COUNT)
	t.assert_almost_eq(float((rows[7] as Dictionary)["value"]), 0.07, 0.00001)
	t.assert_eq(ObservationInspector.build_rows(null).size(), Observation.FIELD_COUNT)
	return t


func test_grouped_rows_partition_every_index_once() -> SandboxTest:
	var t := SandboxTest.new("inspector_grouped_rows_partition")
	var env := EnvironmentCore.new(0, 3)
	env.reset(7)
	var grouped: Array = ObservationInspector.build_grouped_rows(env.get_observations())
	var covered: Dictionary = {}
	for bucket_value in grouped:
		var bucket: Dictionary = bucket_value
		t.assert_false(str(bucket["group"]).is_empty())
		for row_value in (bucket["rows"] as Array):
			var row: Dictionary = row_value
			t.assert_false(covered.has(int(row["index"])), "index %d appears twice" % row["index"])
			covered[int(row["index"])] = true
	t.assert_eq(covered.size(), Observation.FIELD_COUNT, "every field belongs to exactly one group")
	return t


func test_action_rows_follow_the_adapter_action_space() -> SandboxTest:
	var t := SandboxTest.new("inspector_action_rows_follow_adapter")
	var info: Dictionary = RLAdapter.action_space_info()
	var fields: Array = info["fields"]
	var action := Action.new(1, -1, 0, 1, true)

	var rows: Array = ObservationInspector.build_action_rows(action)
	t.assert_eq(rows.size(), fields.size(), "one row per action-space field")
	var multidiscrete: Array = action.to_multidiscrete()
	for index in range(rows.size()):
		var row: Dictionary = rows[index]
		t.assert_eq(str(row["name"]), str(fields[index]))
		t.assert_eq(int(row["multidiscrete"]), int(multidiscrete[index]))
		t.assert_eq(int(row["cardinality"]), int(Action.MULTI_DISCRETE_NVECS[index]))
	t.assert_eq(
		ObservationInspector.build_action_rows(null).size(),
		0,
		"no action means no invented rows"
	)
	return t


func test_continuous_rows_are_reported_separately() -> SandboxTest:
	var t := SandboxTest.new("inspector_continuous_rows_separate")
	var action := Action.new(0, 0, 0, 0, false, Vector2(3.0, -2.0))
	var rows: Array = ObservationInspector.build_continuous_action_rows(action)
	var reserved: Array = RLAdapter.action_space_info()["continuous_reserved"]
	t.assert_eq(rows.size(), reserved.size())
	if rows.size() >= 2:
		t.assert_almost_eq(float((rows[0] as Dictionary)["value"]), 3.0, 0.001)
		t.assert_almost_eq(float((rows[1] as Dictionary)["value"]), -2.0, 0.001)
	return t


func test_reward_rows_come_from_the_episode_breakdown() -> SandboxTest:
	var t := SandboxTest.new("inspector_reward_rows_from_breakdown")
	var env := EnvironmentCore.new(0, 1)
	env.reset(11)
	for _step in range(20):
		env.step(Action.new(0, 0, 0, 0, true))
	var breakdown: Dictionary = env.episode.get_reward_breakdown()
	var rows: Array = ObservationInspector.build_reward_rows(breakdown)
	t.assert_gt(float(rows.size()), 0.0, "a stepped episode has reward components")

	var names: PackedStringArray = PackedStringArray()
	for row_value in rows:
		names.append(str((row_value as Dictionary)["name"]))
	for key in breakdown.keys():
		t.assert_true(names.has(str(key)), "component %s must not be dropped" % str(key))
	if breakdown.has("total"):
		t.assert_eq(names[names.size() - 1], "total", "the total is rendered last")
	t.assert_eq(ObservationInspector.build_reward_rows({}).size(), 0)
	return t


func test_formatting_helpers_are_stable() -> SandboxTest:
	var t := SandboxTest.new("inspector_formatting_helpers")
	var env := EnvironmentCore.new(0, 1)
	env.reset(3)
	var lines: PackedStringArray = ObservationInspector.format_observation_lines(
		ObservationInspector.build_rows(env.get_observations())
	)
	t.assert_eq(lines.size(), Observation.FIELD_COUNT)
	t.assert_true(lines[0].contains(Observation.field_names()[0]))

	var action_lines: PackedStringArray = ObservationInspector.format_action_lines(
		ObservationInspector.build_action_rows(Action.idle())
	)
	t.assert_eq(action_lines.size(), (RLAdapter.action_space_info()["fields"] as Array).size())
	return t

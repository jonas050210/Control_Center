## Tests for ControlCenterResults: episode history, per-source aggregates
## and the HUMAN vs AI comparison foundation. Unmeasured metrics must stay
## unmeasured (-1) instead of being invented.
class_name TestControlCenterResults
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterResults = preload("res://scripts/control_center/control_center_results.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


static func _episode(source: String, reward: float, win: bool, reaction: float) -> Dictionary:
	return {
		"source": source,
		"reward": reward,
		"kills": 1 if win else 0,
		"deaths": 0 if win else 1,
		"accuracy": 0.5,
		"survival_time": 4.0,
		"episode_length": 240,
		"win": win,
		"loss": not win,
		"reaction_time": reaction,
		"done_reason": "all_enemies_eliminated" if win else "agent_died",
	}


func test_record_normalizes_missing_metrics() -> SandboxTest:
	var t := SandboxTest.new("results_record_normalizes")
	var results := ControlCenterResults.new()
	var entry: Dictionary = results.record({"source": "human"})
	t.assert_eq(int(entry["index"]), 0)
	t.assert_eq(str(entry["source"]), "human")
	t.assert_almost_eq(float(entry["reward"]), 0.0, 0.0001)
	t.assert_almost_eq(
		float(entry["reaction_time"]), -1.0, 0.0001, "unmeasured reaction time stays -1"
	)
	t.assert_eq(int(entry["kills"]), 0)
	t.assert_eq(results.size(), 1)
	return t


func test_aggregate_reports_no_data_instead_of_dividing_by_zero() -> SandboxTest:
	var t := SandboxTest.new("results_aggregate_empty")
	var results := ControlCenterResults.new()
	var summary: Dictionary = results.aggregate()
	t.assert_eq(int(summary["episodes"]), 0)
	t.assert_false(summary.has("reward"), "no averages are reported without data")
	return t


func test_aggregate_averages_and_rates() -> SandboxTest:
	var t := SandboxTest.new("results_aggregate_averages")
	var results := ControlCenterResults.new()
	results.record(_episode("ai", 10.0, true, 0.4))
	results.record(_episode("ai", 20.0, false, -1.0))

	var summary: Dictionary = results.aggregate("ai")
	t.assert_eq(int(summary["episodes"]), 2)
	t.assert_almost_eq(float(summary["reward"]), 15.0, 0.001)
	t.assert_almost_eq(float(summary["win_rate"]), 0.5, 0.001)
	t.assert_almost_eq(float(summary["loss_rate"]), 0.5, 0.001)
	t.assert_almost_eq(float(summary["best_reward"]), 20.0, 0.001)
	t.assert_almost_eq(float(summary["worst_reward"]), 10.0, 0.001)
	t.assert_almost_eq(
		float(summary["reaction_time"]),
		0.4,
		0.001,
		"only measured reaction times enter the average"
	)
	t.assert_eq(int(summary["reaction_time_samples"]), 1)
	return t


func test_comparison_requires_both_populations() -> SandboxTest:
	var t := SandboxTest.new("results_comparison")
	var results := ControlCenterResults.new()
	results.record(_episode("ai", 10.0, true, 0.4))
	var only_ai: Dictionary = results.comparison()
	t.assert_false(bool(only_ai["comparable"]), "one-sided data is not a comparison")
	t.assert_true((only_ai["deltas"] as Dictionary).is_empty())

	results.record(_episode("human", 16.0, true, 0.9))
	var both: Dictionary = results.comparison()
	t.assert_true(bool(both["comparable"]))
	var deltas: Dictionary = both["deltas"]
	t.assert_almost_eq(float(deltas["reward"]), 6.0, 0.001, "delta is human - ai")
	t.assert_almost_eq(float(deltas["win_rate"]), 0.0, 0.001)
	t.assert_eq(int((both["ai"] as Dictionary)["episodes"]), 1)
	t.assert_eq(int((both["human"] as Dictionary)["episodes"]), 1)
	return t


func test_history_filters_orders_and_limits() -> SandboxTest:
	var t := SandboxTest.new("results_history")
	var results := ControlCenterResults.new()
	for index in range(5):
		results.record(_episode("ai" if index % 2 == 0 else "human", float(index), true, -1.0))

	var newest: Array = results.history("", 2, true)
	t.assert_eq(newest.size(), 2)
	t.assert_almost_eq(float((newest[0] as Dictionary)["reward"]), 4.0, 0.001)
	var oldest_first: Array = results.history("", 0, false)
	t.assert_almost_eq(float((oldest_first[0] as Dictionary)["reward"]), 0.0, 0.001)
	t.assert_eq(results.history("human", 0, false).size(), 2)
	t.assert_eq(results.size("ai"), 3)
	return t


func test_capacity_bounds_history() -> SandboxTest:
	var t := SandboxTest.new("results_capacity")
	var results := ControlCenterResults.new()
	results.capacity = 4
	for index in range(20):
		results.record(_episode("ai", float(index), true, -1.0))
	t.assert_eq(results.size(), 4)
	var newest: Array = results.history("", 1, true)
	t.assert_almost_eq(float((newest[0] as Dictionary)["reward"]), 19.0, 0.001)
	return t


func test_export_writes_parseable_json() -> SandboxTest:
	var t := SandboxTest.new("results_export_json")
	var results := ControlCenterResults.new()
	results.record(_episode("human", 3.0, true, 0.25))
	var path: String = "user://test_control_center_results.json"
	t.assert_true(results.save_json(path), "export must report success honestly")

	var file: FileAccess = FileAccess.open(path, FileAccess.READ)
	t.assert_not_null(file)
	if file != null:
		var parsed = JSON.parse_string(file.get_as_text())
		file.close()
		t.assert_true(parsed is Dictionary, "the export is a JSON object")
		if parsed is Dictionary:
			t.assert_eq(str(parsed["schema"]), "sandboxai.control_center.results")
			t.assert_eq((parsed["episodes"] as Array).size(), 1)
		DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	return t


## The record index must stay unique once the ring buffer starts trimming;
## UI code uses it to detect "is this the same newest episode?".
func test_record_index_is_monotonic_across_capacity_trimming() -> SandboxTest:
	var t := SandboxTest.new("results_record_index_monotonic")
	var results := ControlCenterResults.new()
	results.capacity = 4
	var last_index: int = -1
	for i in range(10):
		var entry: Dictionary = results.record(_episode("ai", float(i), true, -1.0))
		t.assert_eq(int(entry["index"]), i, "index should keep counting past the capacity")
		t.assert_gt(float(int(entry["index"])), float(last_index), "indices must be increasing")
		last_index = int(entry["index"])
	t.assert_eq(results.episodes.size(), 4, "capacity still bounds the buffer")
	t.assert_eq(results.total_recorded, 10, "total_recorded counts every episode")
	results.clear()
	t.assert_eq(results.total_recorded, 0, "clear() resets the counter")
	return t

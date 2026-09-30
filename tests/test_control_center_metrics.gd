## Tests the METRICS and REPLAY tabs, and the diagnostic SkillMetrics they
## render.
##
## The assertions that matter here are the boundaries: AI-available metrics
## must never contain a ground-truth key, the ground-truth block must be
## returned separately, and neither panel may mutate the snapshot it is
## given or the environment behind it.
class_name TestControlCenterMetrics
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterMetricsPanel = preload("res://scripts/control_center/ui/metrics_panel.gd")
const ControlCenterReplayPanel = preload("res://scripts/control_center/ui/replay_panel.gd")
const ReplayRecorder = preload("res://scripts/replay/replay_recorder.gd")
const SkillMetrics = preload("res://scripts/metrics/skill_metrics.gd")
const Action = preload("res://scripts/core/action.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func _snapshot() -> Dictionary:
	return {
		"episode":
		{
			"step": 120,
			"time_seconds": 2.0,
			"reward": 3.5,
			"kills": 1,
			"deaths": 0,
			"damage_dealt": 60.0,
			"damage_received": 20.0,
			"shots_fired": 8,
			"shots_hit": 3,
			"alive_enemies": 1,
			"total_enemies": 2,
		},
		"agent":
		{
			"health": 80.0,
			"max_health": 100.0,
			"alive": true,
			"velocity": Vector3(3.0, 0.0, 4.0),
		},
		"perception":
		{
			"visible_count": 1,
			"remembered_count": 2,
			"sound_count": 3,
			"exploration_coverage": 0.42,
		},
		"action": {"moving": true, "turning": false, "shooting": true},
		"target": {"has_target": true, "distance_m": 12.5, "health": 40.0, "position": Vector3.ONE},
	}


func test_all_eight_categories_are_reported() -> SandboxTest:
	var t := SandboxTest.new("skill_metrics_categories")
	var categories: Dictionary = SkillMetrics.from_snapshot(_snapshot())
	t.assert_eq(categories.size(), SkillMetrics.CATEGORIES.size())
	for category in SkillMetrics.CATEGORIES:
		t.assert_true(categories.has(category), "missing category %s" % str(category))
	return t


func test_metrics_are_computed_not_guessed() -> SandboxTest:
	var t := SandboxTest.new("skill_metrics_values")
	var categories: Dictionary = SkillMetrics.from_snapshot(_snapshot())
	t.assert_almost_eq(float(categories["aim"]["accuracy"]), 3.0 / 8.0)
	t.assert_almost_eq(float(categories["aim"]["damage_per_shot"]), 60.0 / 8.0)
	t.assert_almost_eq(float(categories["positioning"]["health_fraction"]), 0.8)
	t.assert_almost_eq(float(categories["positioning"]["damage_ratio"]), 3.0)
	t.assert_almost_eq(float(categories["movement"]["speed"]), 5.0)
	t.assert_almost_eq(float(categories["reaction"]["shots_per_second"]), 4.0)
	t.assert_almost_eq(float(categories["exploration"]["coverage"]), 0.42)
	t.assert_eq(int(categories["combat"]["kills"]), 1)
	return t


func test_empty_snapshot_degrades_to_zeros_not_errors() -> SandboxTest:
	var t := SandboxTest.new("skill_metrics_empty_snapshot")
	var categories: Dictionary = SkillMetrics.from_snapshot({})
	t.assert_eq(categories.size(), SkillMetrics.CATEGORIES.size())
	t.assert_almost_eq(float(categories["aim"]["accuracy"]), 0.0)
	t.assert_eq(int(categories["combat"]["kills"]), 0)
	return t


func test_ground_truth_is_kept_separate() -> SandboxTest:
	var t := SandboxTest.new("skill_metrics_ground_truth_separation")
	var snapshot: Dictionary = _snapshot()
	var available: Dictionary = SkillMetrics.ai_available(snapshot)
	for category in available:
		for key in available[category] as Dictionary:
			t.assert_false(
				SkillMetrics.GROUND_TRUTH_KEYS.has(str(key)),
				"AI-available metrics leaked ground truth key %s" % str(key)
			)
	var truth: Dictionary = SkillMetrics.ground_truth(snapshot)
	t.assert_almost_eq(float(truth["true_enemy_distance_m"]), 12.5)
	t.assert_eq(int(truth["alive_enemies"]), 1)
	return t


func test_ground_truth_without_a_target() -> SandboxTest:
	var t := SandboxTest.new("skill_metrics_ground_truth_no_target")
	var snapshot: Dictionary = _snapshot()
	snapshot["target"] = {"has_target": false}
	var truth: Dictionary = SkillMetrics.ground_truth(snapshot)
	t.assert_false(truth.has("true_enemy_distance_m"))
	t.assert_true(truth.has("alive_enemies"))
	return t


func test_rows_and_formatting() -> SandboxTest:
	var t := SandboxTest.new("skill_metrics_rows")
	var categories: Dictionary = SkillMetrics.ai_available(_snapshot())
	var rows: Array = SkillMetrics.rows(categories)
	t.assert_true(rows.size() > 10)
	t.assert_true((rows[0] as Dictionary).has("category"))
	t.assert_true((rows[0] as Dictionary).has("metric"))
	t.assert_true((rows[0] as Dictionary).has("value"))
	t.assert_eq(str((rows[0] as Dictionary)["category"]), "aim")
	var text: String = SkillMetrics.format_categories(categories)
	t.assert_true(text.contains("AIM"))
	t.assert_true(text.contains("EXPLORATION"))
	return t


func test_metrics_panel_does_not_mutate_the_snapshot() -> SandboxTest:
	var t := SandboxTest.new("metrics_panel_is_read_only")
	var panel := ControlCenterMetricsPanel.new()
	panel.setup(null)
	var snapshot: Dictionary = _snapshot()
	var before: Dictionary = snapshot.duplicate(true)
	panel.refresh(snapshot)
	t.assert_eq(str(snapshot), str(before), "the panel must not modify telemetry")
	var described: Dictionary = panel.describe(snapshot)
	t.assert_true(described.has("ai_available"))
	t.assert_true(described.has("ground_truth"))
	panel.refresh({})
	panel.free()
	return t


func test_replay_panel_rejects_a_missing_file() -> SandboxTest:
	var t := SandboxTest.new("replay_panel_rejects_missing_file")
	var panel := ControlCenterReplayPanel.new()
	panel.setup(null)
	t.assert_false(panel.load_path("user://definitely_not_a_replay.jsonl"))
	t.assert_false(panel.has_replay())
	t.assert_true(panel.problems.size() > 0)
	t.assert_false(bool(panel.describe()["loaded"]))
	panel.free()
	return t


func test_replay_panel_loads_and_scrubs() -> SandboxTest:
	var t := SandboxTest.new("replay_panel_transport")
	var recorder := ReplayRecorder.new({"map_id": "compound", "policy_id": "brain_a"})
	recorder.start(42)
	for index in range(30):
		recorder.record_step(Action.idle(), 0.1, null, index == 29)
	recorder.add_event("combat", "shot", {})
	recorder.finish({"done_reason": "timeout"})
	var path := "user://sandboxai_test_panel_replay.jsonl"
	t.assert_true(recorder.save(path))

	var panel := ControlCenterReplayPanel.new()
	panel.setup(null)
	t.assert_true(panel.load_path(path), str(panel.problems))
	t.assert_true(panel.has_replay())
	var described: Dictionary = panel.describe()
	t.assert_eq(int(described["tick_count"]), 30)
	t.assert_eq(str(described["map_id"]), "compound")
	t.assert_eq(str(described["policy_id"]), "brain_a")
	t.assert_eq(int(described["tick"]), 0)

	# Paused by default: refreshing the panel must not move playback.
	panel.refresh({})
	t.assert_eq(int(panel.describe()["tick"]), 0)

	panel.player.play()
	panel.advance(0.25)
	t.assert_true(int(panel.describe()["tick"]) > 0)
	panel.free()
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	return t

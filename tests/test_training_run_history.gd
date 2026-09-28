## Tests for the persisted training-run history reader. It may only relay
## what python/sandboxai/run_control.py actually published — no invented
## durations, metrics or states.
class_name TestTrainingRunHistory
extends RefCounted

const SandboxTest = preload("res://tests/sandbox_test.gd")
const TrainingRunHistory = preload("res://scripts/control_center/training_run_history.gd")


func test_parse_run_extracts_published_fields_only() -> SandboxTest:
	var t := SandboxTest.new("run_history_parse_published_fields")
	var row: Dictionary = TrainingRunHistory.parse_run(
		"control_center_ppo_1700000000_123",
		{
			"state": "Finished",
			"updated_at": 1700000600.0,
			"timesteps": 50000,
			"total_training_steps": 50000,
			"episodes": 321,
			"mean_episode_reward": 1.25,
			"mean_accuracy": 0.4,
			"current_checkpoint": "training/runs/x/checkpoints",
			"device": "cpu",
		}
	)
	t.assert_eq(row["run_id"], "control_center_ppo_1700000000_123")
	t.assert_eq(row["algorithm"], "PPO")
	t.assert_eq(row["state"], "Finished")
	t.assert_almost_eq(float(row["started_unix"]), 1700000000.0, 0.5)
	t.assert_almost_eq(float(row["duration_seconds"]), 600.0, 0.5)
	t.assert_true(bool(row["duration_final"]), "a Finished run has a final duration")
	t.assert_eq(int(row["timesteps"]), 50000)
	t.assert_eq(int(row["episodes"]), 321)
	t.assert_almost_eq(float(row["mean_episode_reward"]), 1.25)
	t.assert_eq(row["current_checkpoint"], "training/runs/x/checkpoints")
	return t


func test_parse_run_without_status_invents_nothing() -> SandboxTest:
	var t := SandboxTest.new("run_history_no_invented_values")
	var row: Dictionary = TrainingRunHistory.parse_run("control_center_bc_1700000000_9", {})
	t.assert_eq(row["algorithm"], "BC")
	t.assert_eq(row["state"], "Unknown")
	t.assert_false(row.has("duration_seconds"), "no updated_at -> no duration")
	t.assert_false(row.has("timesteps"))
	t.assert_false(row.has("episodes"))
	t.assert_false(row.has("mean_episode_reward"))

	var odd: Dictionary = TrainingRunHistory.parse_run("weird-directory-name", {})
	t.assert_eq(odd["algorithm"], "unknown")
	t.assert_false(odd.has("started_unix"), "an unparsable name yields no start time")
	return t


func test_running_run_duration_is_marked_non_final() -> SandboxTest:
	var t := SandboxTest.new("run_history_running_duration_non_final")
	var row: Dictionary = TrainingRunHistory.parse_run(
		"control_center_ppo_1700000000_1",
		{"state": "Running", "updated_at": 1700000100.0}
	)
	t.assert_true(row.has("duration_seconds"))
	t.assert_false(bool(row["duration_final"]), "a live run's duration is only a last update")
	return t


func test_algorithm_and_start_time_from_run_id() -> SandboxTest:
	var t := SandboxTest.new("run_history_run_id_parsing")
	t.assert_eq(TrainingRunHistory.algorithm_from_run_id("control_center_ppo_17_1"), "PPO")
	t.assert_eq(TrainingRunHistory.algorithm_from_run_id("control_center_bc_17_1"), "BC")
	t.assert_eq(TrainingRunHistory.algorithm_from_run_id("something_else"), "unknown")
	t.assert_almost_eq(
		TrainingRunHistory.started_unix_from_run_id("control_center_ppo_1700000000_5"),
		1700000000.0,
		0.5
	)
	t.assert_eq(
		TrainingRunHistory.started_unix_from_run_id("control_center_ppo_99_5"),
		0.0,
		"timestamps outside the sanity window are rejected, not trusted"
	)
	return t


func test_load_runs_reads_persisted_status_files_newest_first() -> SandboxTest:
	var t := SandboxTest.new("run_history_load_runs")
	var root: String = "user://sandboxai_test_history_runs"
	var global_root: String = ProjectSettings.globalize_path(root)
	_remove_dir_recursive(global_root)
	_write_status(
		root.path_join("control_center_ppo_1700000000_1"),
		{"state": "Finished", "updated_at": 1700000500.0, "timesteps": 100}
	)
	_write_status(
		root.path_join("control_center_bc_1700001000_2"),
		{"state": "Error", "updated_at": 1700001200.0, "error": "dataset missing"}
	)
	# A directory without status.json (crashed before publishing anything).
	DirAccess.make_dir_recursive_absolute(
		ProjectSettings.globalize_path(root.path_join("control_center_ppo_1700002000_3"))
	)
	# The preview directory must never appear as a run.
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(root.path_join("preview")))

	var rows: Array = TrainingRunHistory.load_runs(root)
	t.assert_eq(rows.size(), 3, "three runs, preview excluded")
	t.assert_eq((rows[0] as Dictionary)["run_id"], "control_center_ppo_1700002000_3")
	t.assert_eq((rows[0] as Dictionary)["state"], "Unknown")
	t.assert_eq((rows[1] as Dictionary)["run_id"], "control_center_bc_1700001000_2")
	t.assert_eq((rows[1] as Dictionary)["error"], "dataset missing")
	t.assert_eq((rows[2] as Dictionary)["run_id"], "control_center_ppo_1700000000_1")
	t.assert_eq(int((rows[2] as Dictionary)["timesteps"]), 100)
	_remove_dir_recursive(global_root)
	return t


static func _write_status(directory: String, status: Dictionary) -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(directory))
	var file := FileAccess.open(directory.path_join("status.json"), FileAccess.WRITE)
	file.store_string(JSON.stringify(status))
	file.close()


static func _remove_dir_recursive(global_path: String) -> void:
	var dir := DirAccess.open(global_path)
	if dir == null:
		return
	dir.list_dir_begin()
	var name: String = dir.get_next()
	while name != "":
		var child: String = global_path.path_join(name)
		if dir.current_is_dir():
			_remove_dir_recursive(child)
		else:
			DirAccess.remove_absolute(child)
		name = dir.get_next()
	dir.list_dir_end()
	DirAccess.remove_absolute(global_path)

## TrainingRunHistory
##
## Read-only view over the run directories the managed trainers already
## persist (`user://control_center_runs/<run_id>/status.json` written
## atomically by python/sandboxai/run_control.py). Nothing is recomputed
## and nothing is invented: a row contains exactly the fields the backend
## published, and consumers render every absent field as "n/a".
class_name TrainingRunHistory
extends RefCounted

const RUNS_ROOT: String = "user://control_center_runs"

## Metric keys copied verbatim from a published status file into a history
## row when (and only when) the backend wrote them.
const OPTIONAL_METRIC_KEYS: Array = [
	"timesteps",
	"total_training_steps",
	"epoch",
	"total_epochs",
	"episodes",
	"progress",
	"mean_episode_reward",
	"mean_accuracy",
	"mean_kills",
	"mean_deaths",
	"steps_per_second",
	"exact_accuracy",
	"component_accuracy",
	"train_loss",
	"validation_loss",
	"win_rate",
	"current_checkpoint",
	"device",
	"error",
	"stopped",
]


## All persisted runs, newest first. `root` is overridable for tests.
static func load_runs(root: String = RUNS_ROOT) -> Array:
	var rows: Array = []
	var dir := DirAccess.open(root)
	if dir == null:
		return rows
	dir.list_dir_begin()
	var name: String = dir.get_next()
	while name != "":
		if dir.current_is_dir() and not name.begins_with(".") and name != "preview":
			var row: Dictionary = _load_run(root.path_join(name), name)
			if not row.is_empty():
				rows.append(row)
		name = dir.get_next()
	dir.list_dir_end()
	rows.sort_custom(_newest_first)
	return rows


## Pure translation of one published status dictionary into a history row.
## Public and static so tests can cover it without touching the filesystem.
static func parse_run(run_id: String, status: Dictionary) -> Dictionary:
	var row: Dictionary = {
		"run_id": run_id,
		"algorithm": algorithm_from_run_id(run_id),
		"state": str(status.get("state", "Unknown")),
	}
	var started: float = started_unix_from_run_id(run_id)
	if started > 0.0:
		row["started_unix"] = started
	var updated = status.get("updated_at")
	if updated != null and str(updated).is_valid_float():
		row["updated_unix"] = float(updated)
		# Duration is derivable only when both endpoints were recorded, and
		# it is only final for a terminal run.
		if started > 0.0 and float(updated) >= started:
			row["duration_seconds"] = float(updated) - started
			row["duration_final"] = row["state"] in ["Finished", "Error"]
	for key in OPTIONAL_METRIC_KEYS:
		if status.has(key):
			row[key] = status[key]
	if status.has("training_type") and str(status["training_type"]) != "":
		row["algorithm"] = _algorithm_display(str(status["training_type"]))
	return row


## Run ids created by TrainingRunController look like
## `control_center_<ppo|bc>_<unix>_<ticks>`; anything else is "unknown".
static func algorithm_from_run_id(run_id: String) -> String:
	var parts: PackedStringArray = run_id.split("_")
	for index in range(parts.size()):
		if parts[index] == "ppo":
			return "PPO"
		if parts[index] == "bc":
			return "BC"
	return "unknown"


static func started_unix_from_run_id(run_id: String) -> float:
	var parts: PackedStringArray = run_id.split("_")
	# The unix timestamp is the second-to-last component; validate rather
	# than trust the shape.
	if parts.size() < 2:
		return 0.0
	var candidate: String = parts[parts.size() - 2]
	if not candidate.is_valid_int():
		return 0.0
	var value: int = candidate.to_int()
	# Sanity window: after 2020-01-01, before year ~2100.
	if value < 1_577_836_800 or value > 4_102_444_800:
		return 0.0
	return float(value)


static func _load_run(directory: String, run_id: String) -> Dictionary:
	var status_path: String = directory.path_join("status.json")
	if not FileAccess.file_exists(status_path):
		# A directory without a published status is a run that never got
		# far enough to report anything; show it honestly as such.
		return parse_run(run_id, {})
	var file := FileAccess.open(status_path, FileAccess.READ)
	if file == null:
		return parse_run(run_id, {})
	var parsed = JSON.parse_string(file.get_as_text())
	if parsed is Dictionary:
		return parse_run(run_id, parsed)
	return parse_run(run_id, {})


static func _newest_first(a: Dictionary, b: Dictionary) -> bool:
	var a_time: float = float(a.get("updated_unix", a.get("started_unix", 0.0)))
	var b_time: float = float(b.get("updated_unix", b.get("started_unix", 0.0)))
	return a_time > b_time


static func _algorithm_display(training_type: String) -> String:
	match training_type.to_lower():
		"ppo":
			return "PPO"
		"behavior cloning", "behavior_cloning", "bc":
			return "BC"
		_:
			return training_type

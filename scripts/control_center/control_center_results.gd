## ControlCenterResults
##
## Bounded episode-result history plus per-source aggregates. It is the
## foundation for the HUMAN vs AI comparison (Phase 12): every finished
## episode is recorded with the SAME metric keys regardless of who produced
## the actions, so the two populations are directly comparable.
##
## Metrics that the simulation genuinely produces (reward, kills, deaths,
## damage, shots, accuracy, survival, win/loss) come straight from
## `EpisodeState.to_metrics()`. Derived observation-layer metrics
## (reaction time, useless/missed shots, target switches) are measured by
## ControlCenterSession for the SELECTED environment only and are reported
## as -1 / 0 when they were not measured, never invented.
class_name ControlCenterResults
extends RefCounted

const DEFAULT_CAPACITY: int = 200
## Metric keys averaged by `aggregate()`. Anything not in this list is kept
## on the individual record but not summarised.
## Declared as a plain Array literal: `PackedStringArray([...])` is not a
## constant expression in Godot 4.7 and fails to compile.
const AVERAGED_KEYS: Array = [
	"reward",
	"kills",
	"deaths",
	"damage_dealt",
	"damage_received",
	"shots_fired",
	"shots_hit",
	"accuracy",
	"survival_time",
	"episode_length",
	"useless_shots",
	"missed_shots",
	"target_switches",
	"distance_travelled",
]

var capacity: int = DEFAULT_CAPACITY
var episodes: Array = []  # Array[Dictionary], oldest first
## Monotonic episode counter. `episodes.size()` cannot be used as the record
## index because the ring buffer trims old entries, which would make indices
## repeat once `capacity` is reached.
var total_recorded: int = 0


## Normalizes and stores one finished episode. `source` is "ai", "human" or
## "idle" — whoever produced the actions. Returns the stored record.
func record(values: Dictionary) -> Dictionary:
	var entry: Dictionary = {
		"index": total_recorded,
		"source": str(values.get("source", "unknown")),
		"mode": str(values.get("mode", "")),
		"env_index": int(values.get("env_index", 0)),
		"episode": int(values.get("episode", 0)),
		"seed": int(values.get("seed", -1)),
		"curriculum_level": int(values.get("curriculum_level", 0)),
		"enemy_count": int(values.get("enemy_count", 0)),
		"reward": float(values.get("reward", 0.0)),
		"episode_length": int(values.get("episode_length", 0)),
		"kills": int(values.get("kills", 0)),
		"deaths": int(values.get("deaths", 0)),
		"damage_dealt": float(values.get("damage_dealt", 0.0)),
		"damage_received": float(values.get("damage_received", 0.0)),
		"shots_fired": int(values.get("shots_fired", 0)),
		"shots_hit": int(values.get("shots_hit", 0)),
		"accuracy": float(values.get("accuracy", 0.0)),
		"survival_time": float(values.get("survival_time", 0.0)),
		"win": bool(values.get("win", false)),
		"loss": bool(values.get("loss", false)),
		"done_reason": str(values.get("done_reason", "")),
		# -1.0 means "not measured" (only the selected environment is
		# instrumented for reaction time).
		"reaction_time": float(values.get("reaction_time", -1.0)),
		"useless_shots": int(values.get("useless_shots", 0)),
		"missed_shots": int(values.get("missed_shots", 0)),
		"target_switches": int(values.get("target_switches", 0)),
		"distance_travelled": float(values.get("distance_travelled", 0.0)),
		"reward_breakdown": values.get("reward_breakdown", {}),
		"wall_time": float(values.get("wall_time", 0.0)),
	}
	episodes.append(entry)
	total_recorded += 1
	if episodes.size() > capacity:
		episodes = episodes.slice(episodes.size() - capacity)
	return entry


## Episode history, newest first when `newest_first` is true. An empty
## `source` returns every source.
func history(source: String = "", limit: int = 0, newest_first: bool = true) -> Array:
	var selected: Array = []
	for entry_value in episodes:
		var entry: Dictionary = entry_value
		if source.is_empty() or str(entry["source"]) == source:
			selected.append(entry)
	if newest_first:
		selected.reverse()
	if limit > 0 and selected.size() > limit:
		selected = selected.slice(0, limit)
	return selected


func size(source: String = "") -> int:
	return history(source, 0, false).size()


## Mean of every AVERAGED_KEYS metric plus win/loss rates. Returns
## {"episodes": 0} when nothing matches, so callers can render "no data"
## instead of dividing by zero.
func aggregate(source: String = "") -> Dictionary:
	var selected: Array = history(source, 0, false)
	var summary: Dictionary = {"source": source, "episodes": selected.size()}
	if selected.is_empty():
		return summary

	for key in AVERAGED_KEYS:
		var total: float = 0.0
		for entry_value in selected:
			var entry: Dictionary = entry_value
			total += float(entry.get(key, 0.0))
		summary[key] = total / float(selected.size())

	var wins: int = 0
	var losses: int = 0
	var best_reward: float = -1.0e20
	var worst_reward: float = 1.0e20
	var measured_reaction_total: float = 0.0
	var measured_reaction_count: int = 0
	for entry_value in selected:
		var entry: Dictionary = entry_value
		if bool(entry.get("win", false)):
			wins += 1
		if bool(entry.get("loss", false)):
			losses += 1
		var reward: float = float(entry.get("reward", 0.0))
		best_reward = maxf(best_reward, reward)
		worst_reward = minf(worst_reward, reward)
		var reaction: float = float(entry.get("reaction_time", -1.0))
		if reaction >= 0.0:
			measured_reaction_total += reaction
			measured_reaction_count += 1

	summary["win_rate"] = float(wins) / float(selected.size())
	summary["loss_rate"] = float(losses) / float(selected.size())
	summary["best_reward"] = best_reward
	summary["worst_reward"] = worst_reward
	summary["reaction_time"] = (
		measured_reaction_total / float(measured_reaction_count)
		if measured_reaction_count > 0
		else -1.0
	)
	summary["reaction_time_samples"] = measured_reaction_count
	return summary


## Side-by-side HUMAN vs AI aggregates plus per-metric deltas
## (human - ai). Metrics without data on either side are omitted from
## "deltas" rather than reported as zero.
func comparison() -> Dictionary:
	var ai: Dictionary = aggregate("ai")
	var human: Dictionary = aggregate("human")
	var deltas: Dictionary = {}
	if int(ai.get("episodes", 0)) > 0 and int(human.get("episodes", 0)) > 0:
		for key in AVERAGED_KEYS:
			if ai.has(key) and human.has(key):
				deltas[key] = float(human[key]) - float(ai[key])
		if ai.has("win_rate") and human.has("win_rate"):
			deltas["win_rate"] = float(human["win_rate"]) - float(ai["win_rate"])
	return {
		"ai": ai,
		"human": human,
		"deltas": deltas,
		"comparable": int(ai.get("episodes", 0)) > 0 and int(human.get("episodes", 0)) > 0,
	}


func clear() -> void:
	episodes.clear()
	total_recorded = 0


func to_dict() -> Dictionary:
	return {
		"schema": "sandboxai.control_center.results",
		"schema_version": 1,
		"episodes": episodes.duplicate(true),
		"comparison": comparison(),
	}


## Writes the full history as JSON. Returns true on success; the caller is
## responsible for choosing a writable path (the UI uses `user://`).
func save_json(path: String) -> bool:
	var file: FileAccess = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		return false
	file.store_string(JSON.stringify(to_dict(), "\t"))
	file.close()
	return true

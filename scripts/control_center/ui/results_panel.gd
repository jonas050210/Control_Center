## ControlCenterResultsPanel
##
## Results tab (Phase 8 + Phase 12): the current episode, accumulated
## aggregates per action source, the HUMAN vs AI comparison, and a
## selectable list of finished episodes.
##
## Metrics come from EpisodeState/ControlCenterResults. Values that were
## not measured (for example reaction time outside the selected
## environment) are shown as "n/a" instead of a fabricated number.
class_name ControlCenterResultsPanel
extends VBoxContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var session
var _current_label: Label
var _aggregate_label: Label
var _comparison_label: Label
var _episode_list: ItemList
var _detail_label: Label
var _export_button: Button
var _export_status: Label
var _history_cache: Array = []
var _newest_history_index: int = -1


func setup(p_session) -> void:
	session = p_session
	add_theme_constant_override("separation", 6)

	add_child(
		ControlCenterTheme.make_label(
			"CURRENT EPISODE", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_current_label = ControlCenterTheme.make_value_label("-")
	_current_label.clip_text = false
	add_child(_current_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"ACCUMULATED", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_aggregate_label = ControlCenterTheme.make_value_label("-")
	_aggregate_label.clip_text = false
	add_child(_aggregate_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"HUMAN vs AI", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_comparison_label = ControlCenterTheme.make_value_label("-")
	_comparison_label.clip_text = false
	_comparison_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_comparison_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"EPISODE HISTORY", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_episode_list = ItemList.new()
	_episode_list.custom_minimum_size = Vector2(0.0, 120.0)
	_episode_list.add_theme_font_size_override("font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	_episode_list.item_selected.connect(_on_episode_selected)
	add_child(_episode_list)

	_detail_label = ControlCenterTheme.make_value_label("select an episode for details")
	_detail_label.clip_text = false
	_detail_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_detail_label)

	var row := ControlCenterTheme.make_row()
	add_child(row)
	_export_button = ControlCenterTheme.make_button(
		"Export results JSON", "Writes the full episode history to user://"
	)
	_export_button.pressed.connect(_on_export_pressed)
	row.add_child(_export_button)
	_export_status = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	row.add_child(_export_status)


func refresh(snapshot: Dictionary) -> void:
	if not snapshot.is_empty():
		var episode: Dictionary = snapshot["episode"]
		_current_label.text = (
			(
				"reward %.2f   kills %d   deaths %d\n"
				+ "damage %.0f dealt / %.0f taken\n"
				+ "shots %d, hits %d, accuracy %.0f%%\n"
				+ "near %d   useless %d   cooldown %d\n"
				+ "step %d   time %.2fs   enemies %d/%d"
			)
			% [
				float(episode["reward"]),
				int(episode["kills"]),
				int(episode["deaths"]),
				float(episode["damage_dealt"]),
				float(episode["damage_received"]),
				int(episode["shots_fired"]),
				int(episode["shots_hit"]),
				float(episode["accuracy"]) * 100.0,
				int(episode.get("near_miss_shots", 0)),
				int(episode.get("useless_shots", 0)),
				int(episode.get("cooldown_shots", 0)),
				int(episode["step"]),
				float(episode["time_seconds"]),
				int(episode["alive_enemies"]),
				int(episode["total_enemies"]),
			]
		)

	var overall: Dictionary = session.results.aggregate()
	if int(overall["episodes"]) == 0:
		_aggregate_label.text = "no finished episodes yet"
	else:
		_aggregate_label.text = _format_aggregate("all", overall)

	var comparison: Dictionary = session.results.comparison()
	var lines: PackedStringArray = PackedStringArray()
	lines.append(_format_aggregate("ai", comparison["ai"]))
	lines.append(_format_aggregate("human", comparison["human"]))
	if bool(comparison["comparable"]):
		var deltas: Dictionary = comparison["deltas"]
		lines.append(
			(
				"delta (human - ai): reward %+.2f   accuracy %+.1f%%   kills %+.2f   win %+.0f%%"
				% [
					float(deltas.get("reward", 0.0)),
					float(deltas.get("accuracy", 0.0)) * 100.0,
					float(deltas.get("kills", 0.0)),
					float(deltas.get("win_rate", 0.0)) * 100.0,
				]
			)
		)
	else:
		lines.append(
			"Play the same scenario in HUMAN mode and watch it in WATCH mode to compare."
		)
	_comparison_label.text = "\n".join(lines)

	_refresh_history()


func _format_aggregate(label: String, summary: Dictionary) -> String:
	var episodes: int = int(summary.get("episodes", 0))
	if episodes == 0:
		return "%-6s no episodes" % label
	var reaction: float = float(summary.get("reaction_time", -1.0))
	return (
		(
			"%-6s n=%d  reward %.2f  kills %.2f  acc %.0f%%  win %.0f%%  "
			+ "useless %.2f  reaction %s"
		)
		% [
			label,
			episodes,
			float(summary.get("reward", 0.0)),
			float(summary.get("kills", 0.0)),
			float(summary.get("accuracy", 0.0)) * 100.0,
			float(summary.get("win_rate", 0.0)) * 100.0,
			float(summary.get("useless_shots", 0.0)),
			"n/a" if reaction < 0.0 else "%.2fs" % reaction,
		]
	)


func _refresh_history() -> void:
	var history: Array = session.results.history("", 40, true)
	var newest: int = -1 if history.is_empty() else int((history[0] as Dictionary)["index"])
	# Rebuild only when something actually changed; the newest episode index
	# also catches the case where an old entry was trimmed at the same time.
	if history.size() == _history_cache.size() and newest == _newest_history_index:
		return
	_history_cache = history
	_newest_history_index = newest
	var selected_index: int = -1
	var selected_items: PackedInt32Array = _episode_list.get_selected_items()
	if selected_items.size() > 0:
		selected_index = selected_items[0]
	_episode_list.clear()
	for entry_value in history:
		var entry: Dictionary = entry_value
		_episode_list.add_item(
			(
				"#%d %-5s env%d  r %.1f  k%d  acc %.0f%%  %s"
				% [
					int(entry["index"]),
					str(entry["source"]),
					int(entry["env_index"]),
					float(entry["reward"]),
					int(entry["kills"]),
					float(entry["accuracy"]) * 100.0,
					str(entry["done_reason"]),
				]
			)
		)
	if selected_index >= 0 and selected_index < _episode_list.item_count:
		_episode_list.select(selected_index)


func _on_episode_selected(index: int) -> void:
	if index < 0 or index >= _history_cache.size():
		return
	var entry: Dictionary = _history_cache[index]
	var reaction: float = float(entry.get("reaction_time", -1.0))
	_detail_label.text = (
		(
			"episode #%d (%s, env %d, seed %d, level %d, %s)\n"
			+ "reward %.2f   length %d   survival %.2fs   %s\n"
			+ "kills %d   deaths %d   damage %.0f/%.0f\n"
			+ "shots %d   hits %d   accuracy %.0f%%\n"
			+ "near %d   useless %d   cooldown %d   last %s\n"
			+ "target switches %d   reaction %s"
		)
		% [
			int(entry["index"]),
			str(entry["source"]),
			int(entry["env_index"]),
			int(entry["seed"]),
			int(entry["curriculum_level"]),
			str(entry.get("weapon_profile", "rifle")),
			float(entry["reward"]),
			int(entry["episode_length"]),
			float(entry["survival_time"]),
			str(entry["done_reason"]),
			int(entry["kills"]),
			int(entry["deaths"]),
			float(entry["damage_dealt"]),
			float(entry["damage_received"]),
			int(entry["shots_fired"]),
			int(entry["shots_hit"]),
			float(entry["accuracy"]) * 100.0,
			int(entry.get("near_miss_shots", entry.get("missed_shots", 0))),
			int(entry["useless_shots"]),
			int(entry.get("cooldown_shots", 0)),
			str(entry.get("last_shot_result", "none")),
			int(entry["target_switches"]),
			"n/a" if reaction < 0.0 else "%.2fs" % reaction,
		]
	)


func _on_export_pressed() -> void:
	var path: String = "user://control_center_results.json"
	if session.results.save_json(path):
		_export_status.text = "saved to %s" % ProjectSettings.globalize_path(path)
	else:
		_export_status.text = "export failed (could not open %s)" % path

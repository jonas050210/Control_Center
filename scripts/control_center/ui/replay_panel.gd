## ControlCenterReplayPanel
##
## REPLAY tab. Loads a recorded episode (ReplayRecorder JSON Lines), shows
## its header, a scrubbable timeline and the per-tick values, and offers
## transport controls: play/pause, single step, speed, jump to the next or
## previous event.
##
## Presentation only, and deliberately so: the panel drives a ReplayPlayer
## cursor over already-recorded data. It never steps the live simulation,
## never writes to an environment and never touches a policy observation.
## Everything shown here is data that was recorded, labelled with where it
## came from.
class_name ControlCenterReplayPanel
extends VBoxContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const ReplayFormat = preload("res://scripts/replay/replay_format.gd")
const ReplayPlayer = preload("res://scripts/replay/replay_player.gd")
const ReplayRecorder = preload("res://scripts/replay/replay_recorder.gd")

## Default location the record/eval tooling writes replays to.
const DEFAULT_PATH: String = "user://replays/latest.jsonl"

var session
var player: ReplayPlayer = null
var problems: Array = []

var _path_edit: LineEdit
var _load_button: Button
var _status_label: Label
var _header_label: Label
var _tick_label: Label
var _timeline: HSlider
var _event_list: ItemList
var _play_button: Button
var _speed_option: OptionButton
var _syncing_timeline: bool = false


func setup(p_session = null) -> void:
	session = p_session
	add_theme_constant_override("separation", 6)

	add_child(
		ControlCenterTheme.make_label(
			"Replay", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	add_child(
		ControlCenterTheme.make_label(
			"Recorded episode playback. Nothing here affects the live simulation.",
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)

	var load_row := HBoxContainer.new()
	_path_edit = LineEdit.new()
	_path_edit.text = DEFAULT_PATH
	_path_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	load_row.add_child(_path_edit)
	_load_button = Button.new()
	_load_button.text = "Load"
	_load_button.pressed.connect(_on_load_pressed)
	load_row.add_child(_load_button)
	add_child(load_row)

	_status_label = ControlCenterTheme.make_value_label("no replay loaded")
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_status_label)

	add_child(ControlCenterTheme.make_separator())
	_header_label = ControlCenterTheme.make_value_label("-")
	_header_label.clip_text = false
	_header_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_header_label)

	add_child(ControlCenterTheme.make_separator())
	var transport := HBoxContainer.new()
	_play_button = Button.new()
	_play_button.text = "Play"
	_play_button.pressed.connect(_on_play_pressed)
	transport.add_child(_play_button)
	transport.add_child(_make_button("Step", _on_step_pressed))
	transport.add_child(_make_button("<< Event", _on_previous_event_pressed))
	transport.add_child(_make_button("Event >>", _on_next_event_pressed))
	transport.add_child(_make_button("Restart", _on_restart_pressed))
	_speed_option = OptionButton.new()
	for speed_value in ReplayPlayer.SPEED_PRESETS:
		_speed_option.add_item("%sx" % str(speed_value))
	_speed_option.selected = ReplayPlayer.SPEED_PRESETS.find(1.0)
	_speed_option.item_selected.connect(_on_speed_selected)
	transport.add_child(_speed_option)
	add_child(transport)

	_timeline = HSlider.new()
	_timeline.min_value = 0.0
	_timeline.max_value = 0.0
	_timeline.step = 1.0
	_timeline.value_changed.connect(_on_timeline_changed)
	add_child(_timeline)

	_tick_label = ControlCenterTheme.make_value_label("-")
	_tick_label.clip_text = false
	add_child(_tick_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"Events", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_event_list = ItemList.new()
	_event_list.custom_minimum_size = Vector2(0.0, 150.0)
	_event_list.item_selected.connect(_on_event_selected)
	add_child(_event_list)


func _make_button(text: String, handler: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.pressed.connect(handler)
	return button


## Loads a replay from disk. Returns true when it is playable; a rejected
## file leaves the previous replay untouched and reports why.
func load_path(path: String) -> bool:
	var episode: Dictionary = ReplayRecorder.load_replay(path)
	problems = episode.get("problems", [])
	if not problems.is_empty():
		_set_status("rejected: %s" % str(problems[0]), ControlCenterTheme.COLOR_BAD)
		if session != null:
			session.log_warning("replay %s rejected: %s" % [path, str(problems[0])])
		return false
	set_episode(episode)
	_set_status("loaded %s" % path, ControlCenterTheme.COLOR_OK)
	return true


func set_episode(episode: Dictionary) -> void:
	player = ReplayPlayer.new(episode)
	problems = []
	_rebuild_event_list()
	_timeline.max_value = float(player.tick_count())
	_timeline.value = 0.0
	_refresh_labels()


func has_replay() -> bool:
	return player != null and player.tick_count() > 0


## Advances playback by wall-clock time. Called from the UI refresh loop,
## never from the physics step, so a paused simulation can still be
## scrubbed.
func advance(delta: float) -> void:
	if player == null:
		return
	if not player.advance(delta).is_empty():
		_refresh_labels()


func refresh(_snapshot: Dictionary = {}) -> void:
	_refresh_labels()


func _on_load_pressed() -> void:
	load_path(_path_edit.text.strip_edges())


func _on_play_pressed() -> void:
	if player == null:
		return
	_play_button.text = "Pause" if player.toggle() else "Play"
	_refresh_labels()


func _on_step_pressed() -> void:
	if player == null:
		return
	player.pause()
	_play_button.text = "Play"
	player.step(1)
	_refresh_labels()


func _on_restart_pressed() -> void:
	if player == null:
		return
	player.reset()
	_play_button.text = "Play"
	_refresh_labels()


func _on_next_event_pressed() -> void:
	if player == null:
		return
	player.step(1)
	player.next_event()
	_refresh_labels()


func _on_previous_event_pressed() -> void:
	if player == null:
		return
	player.previous_event()
	_refresh_labels()


func _on_speed_selected(index: int) -> void:
	if player == null or index < 0 or index >= ReplayPlayer.SPEED_PRESETS.size():
		return
	player.set_speed(float(ReplayPlayer.SPEED_PRESETS[index]))
	_refresh_labels()


func _on_timeline_changed(value: float) -> void:
	if player == null or _syncing_timeline:
		return
	player.seek(int(value))
	_refresh_labels()


func _on_event_selected(index: int) -> void:
	if player == null:
		return
	var events: Array = player.episode.get("events", [])
	if index < 0 or index >= events.size():
		return
	player.seek(int((events[index] as Dictionary).get("tick", 0)))
	_refresh_labels()


func _rebuild_event_list() -> void:
	_event_list.clear()
	if player == null:
		return
	for event_value in player.episode.get("events", []):
		var event: Dictionary = event_value
		var label: String = str(event.get("label", ""))
		_event_list.add_item(
			(
				"t%05d  %-14s %s"
				% [int(event.get("tick", 0)), str(event.get("kind", "")), label]
			)
		)


func _set_status(message: String, color: Color) -> void:
	_status_label.text = message
	_status_label.add_theme_color_override("font_color", color)


func _refresh_labels() -> void:
	if player == null:
		_header_label.text = "-"
		_tick_label.text = "-"
		return
	var status: Dictionary = player.status()
	_header_label.text = (
		(
			"map %s | scenario %s | lighting %s | seed %d | level %d\n"
			+ "policy %s | checkpoint %s | detail %s | result %s"
		)
		% [
			_or_dash(str(status["map_id"])),
			_or_dash(str(status["scenario"])),
			_or_dash(str(status["lighting"])),
			int(status["seed"]),
			int(status["curriculum_level"]),
			_or_dash(str(status["policy_id"])),
			_or_dash(str(status["checkpoint"])),
			str(status["detail"]),
			_or_dash(str(status["done_reason"])),
		]
	)
	_tick_label.text = (
		"tick %d / %d | t=%.2fs / %.2fs | speed %.2fx | action %s | reward %+.3f | total %+.2f"
		% [
			int(status["tick"]),
			int(status["tick_count"]),
			float(status["time"]),
			float(status["duration"]),
			float(status["speed"]),
			str(status["action"]),
			float(status["reward"]),
			float(status["total_reward"]),
		]
	)
	_play_button.text = "Pause" if bool(status["playing"]) else "Play"
	_syncing_timeline = true
	_timeline.value = float(int(status["tick"]))
	_syncing_timeline = false


func _or_dash(value: String) -> String:
	return value if not value.is_empty() else "-"


## Header/detail summary as data, for tests and logs.
func describe() -> Dictionary:
	if player == null:
		return {"loaded": false, "problems": problems.duplicate()}
	var status: Dictionary = player.status()
	status["loaded"] = true
	status["problems"] = problems.duplicate()
	status["events"] = (player.episode.get("events", []) as Array).size()
	status["detail_levels"] = ReplayFormat.DETAIL_LEVELS
	return status

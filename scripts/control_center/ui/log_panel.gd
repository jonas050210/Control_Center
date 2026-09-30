## ControlCenterLogPanel
##
## Live event log (Phase 7) with ALL / COMBAT / PERCEPTION / SYSTEM /
## REWARD / ERROR filters.
##
## Performance (Phase 15): the underlying ControlCenterEventLog already
## throttles and bounds events. This panel additionally appends only the
## entries that arrived since the last refresh, tracked through the log's
## monotonic `total_accepted` counter, so a steady event stream costs one
## `append_text` per new line instead of a full re-render. A complete
## rebuild happens only when the filter changes or the log is cleared.
class_name ControlCenterLogPanel
extends PanelContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterEventLog = preload("res://scripts/control_center/control_center_event_log.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

const CATEGORY_COLORS: Dictionary = {
	ControlCenterEventLog.Category.SYSTEM: Color(0.62, 0.68, 0.76, 1.0),
	ControlCenterEventLog.Category.COMBAT: Color(1.0, 0.72, 0.45, 1.0),
	ControlCenterEventLog.Category.PERCEPTION: Color(0.45, 0.85, 0.95, 1.0),
	ControlCenterEventLog.Category.REWARD: Color(0.60, 0.90, 0.62, 1.0),
	ControlCenterEventLog.Category.ERROR: Color(0.98, 0.45, 0.45, 1.0),
}

var session
var _text: RichTextLabel
var _filter_buttons: Array = []  # Array[Button], parallel to filter_options()
var _stats_label: Label
var _autoscroll: Button
var _filter: int = ControlCenterEventLog.FILTER_ALL
var _seen_total: int = 0


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 4)
	add_child(root)

	var header := ControlCenterTheme.make_row()
	root.add_child(header)
	header.add_child(
		ControlCenterTheme.make_label(
			"Event log", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	for option_value in ControlCenterEventLog.filter_options():
		var option: int = int(option_value)
		var button := ControlCenterTheme.make_toggle(
			ControlCenterEventLog.category_name(option),
			option == _filter,
			"Show only %s events" % ControlCenterEventLog.category_name(option)
		)
		button.pressed.connect(_on_filter_pressed.bind(option))
		header.add_child(button)
		_filter_buttons.append(button)

	_autoscroll = ControlCenterTheme.make_toggle(
		"Follow", true, "Keep scrolling to the newest entry"
	)
	_autoscroll.toggled.connect(_on_autoscroll_toggled)
	header.add_child(_autoscroll)

	var clear_button := ControlCenterTheme.make_button("Clear", "Discard all buffered events")
	clear_button.pressed.connect(_on_clear_pressed)
	header.add_child(clear_button)

	_stats_label = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	header.add_child(_stats_label)

	_text = RichTextLabel.new()
	_text.bbcode_enabled = true
	_text.scroll_following = true
	_text.selection_enabled = true
	_text.focus_mode = Control.FOCUS_NONE
	_text.custom_minimum_size = Vector2(0.0, 110.0)
	_text.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_text.add_theme_font_size_override("normal_font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	_text.add_theme_font_size_override("mono_font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	root.add_child(_text)

	rebuild()


## Appends everything logged since the previous call. Cheap enough to run
## at the panel refresh rate; a full rebuild only happens on filter change.
func refresh(_snapshot: Dictionary = {}) -> void:
	var log_ref = session.event_log
	if log_ref.total_accepted < _seen_total:
		# The log was cleared or replaced underneath us.
		rebuild()
		return

	var new_count: int = mini(log_ref.total_accepted - _seen_total, log_ref.size())
	if new_count > 0:
		var all_entries: Array = log_ref.entries(ControlCenterEventLog.FILTER_ALL)
		for entry_value in all_entries.slice(all_entries.size() - new_count):
			_append_entry(entry_value)
		_seen_total = log_ref.total_accepted

	_stats_label.text = (
		"%d shown / %d buffered / %d dropped"
		% [_line_count(), log_ref.size(), log_ref.dropped_count]
	)
	if not session.event_log.enabled:
		_stats_label.text = "logging disabled in TRAINING mode"


## Rebuilds the whole view from the buffered entries (filter change, clear).
func rebuild() -> void:
	_text.clear()
	for entry_value in session.event_log.entries(ControlCenterEventLog.FILTER_ALL):
		_append_entry(entry_value)
	_seen_total = session.event_log.total_accepted


func _append_entry(entry_value) -> void:
	var entry: Dictionary = entry_value
	var category: int = int(entry["category"])
	if _filter != ControlCenterEventLog.FILTER_ALL and category != _filter:
		return
	var color: Color = CATEGORY_COLORS.get(category, ControlCenterTheme.COLOR_TEXT)
	_text.push_color(ControlCenterTheme.COLOR_MUTED)
	_text.add_text("[%7.2fs] " % float(entry["time"]))
	_text.pop()
	_text.push_color(color)
	_text.add_text("%-10s " % str(entry["category_name"]))
	_text.pop()
	_text.add_text(str(entry["message"]))
	_text.newline()


func _line_count() -> int:
	return _text.get_line_count()


func _on_filter_pressed(option: int) -> void:
	_filter = option
	for index in range(_filter_buttons.size()):
		var button: Button = _filter_buttons[index]
		button.button_pressed = int(ControlCenterEventLog.filter_options()[index]) == option
	rebuild()


func _on_autoscroll_toggled(pressed: bool) -> void:
	_text.scroll_following = pressed
	if pressed:
		_text.scroll_to_line(maxi(_text.get_line_count() - 1, 0))


func _on_clear_pressed() -> void:
	session.event_log.clear()
	_seen_total = 0
	_text.clear()

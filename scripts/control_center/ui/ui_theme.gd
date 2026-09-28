## ControlCenterTheme
##
## Small factory helpers shared by every Control Center panel so the look
## is defined once instead of being re-invented per panel. Pure
## presentation: nothing here reads or writes simulation state.
class_name ControlCenterTheme
extends RefCounted

const COLOR_BACKGROUND: Color = Color(0.07, 0.08, 0.10, 0.92)
const COLOR_BACKGROUND_SOLID: Color = Color(0.07, 0.08, 0.10, 1.0)
const COLOR_BORDER: Color = Color(0.22, 0.25, 0.30, 1.0)
const COLOR_TEXT: Color = Color(0.86, 0.89, 0.93, 1.0)
const COLOR_MUTED: Color = Color(0.58, 0.63, 0.70, 1.0)
const COLOR_TITLE: Color = Color(0.55, 0.78, 1.0, 1.0)
const COLOR_OK: Color = Color(0.45, 0.85, 0.55, 1.0)
const COLOR_WARN: Color = Color(0.98, 0.76, 0.32, 1.0)
const COLOR_BAD: Color = Color(0.95, 0.42, 0.42, 1.0)
const COLOR_AI: Color = Color(0.40, 0.85, 0.95, 1.0)
const COLOR_HUMAN: Color = Color(0.85, 0.65, 1.0, 1.0)
const COLOR_HIDDEN: Color = Color(0.95, 0.45, 0.45, 1.0)

const FONT_SIZE_TITLE: int = 14
const FONT_SIZE_NORMAL: int = 12
const FONT_SIZE_SMALL: int = 11


static func panel_style(
	background: Color = COLOR_BACKGROUND, border: Color = COLOR_BORDER
) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = background
	style.border_color = border
	style.set_border_width_all(1)
	style.set_corner_radius_all(3)
	style.content_margin_left = 8.0
	style.content_margin_right = 8.0
	style.content_margin_top = 6.0
	style.content_margin_bottom = 6.0
	return style


## PanelContainer + VBoxContainer with an optional caption. Returns the
## panel; `content_container(panel)` gives the VBox to fill.
static func make_panel(title: String = "") -> PanelContainer:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", panel_style())
	var box := VBoxContainer.new()
	box.name = "Content"
	box.add_theme_constant_override("separation", 4)
	panel.add_child(box)
	if not title.is_empty():
		box.add_child(make_label(title, FONT_SIZE_TITLE, COLOR_TITLE))
	return panel


static func content_container(panel: PanelContainer) -> VBoxContainer:
	return panel.get_node("Content") as VBoxContainer


static func make_label(
	text: String, font_size: int = FONT_SIZE_NORMAL, color: Color = COLOR_TEXT
) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	return label


## Label intended for tabular/numeric output: smaller, muted, clipped so a
## long value cannot stretch the panel.
static func make_value_label(text: String = "") -> Label:
	var label := make_label(text, FONT_SIZE_SMALL, COLOR_TEXT)
	label.clip_text = true
	return label


static func make_button(text: String, tooltip: String = "") -> Button:
	var button := Button.new()
	button.text = text
	button.tooltip_text = tooltip
	button.add_theme_font_size_override("font_size", FONT_SIZE_NORMAL)
	button.focus_mode = Control.FOCUS_NONE
	return button


static func make_toggle(text: String, pressed: bool, tooltip: String = "") -> Button:
	var button := make_button(text, tooltip)
	button.toggle_mode = true
	button.button_pressed = pressed
	return button


static func make_separator() -> HSeparator:
	var separator := HSeparator.new()
	separator.add_theme_constant_override("separation", 6)
	return separator


static func make_row() -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	return row


## Two-column key/value grid used by most read-only panels.
static func make_grid(columns: int = 2) -> GridContainer:
	var grid := GridContainer.new()
	grid.columns = columns
	grid.add_theme_constant_override("h_separation", 10)
	grid.add_theme_constant_override("v_separation", 2)
	return grid


static func make_scroll(minimum_height: float = 0.0) -> ScrollContainer:
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	if minimum_height > 0.0:
		scroll.custom_minimum_size = Vector2(0.0, minimum_height)
	return scroll


static func make_option_button(tooltip: String = "") -> OptionButton:
	var option := OptionButton.new()
	option.tooltip_text = tooltip
	option.add_theme_font_size_override("font_size", FONT_SIZE_NORMAL)
	option.focus_mode = Control.FOCUS_NONE
	return option


static func make_spin_box(
	minimum: float, maximum: float, step: float, value: float, tooltip: String = ""
) -> SpinBox:
	var spin := SpinBox.new()
	spin.min_value = minimum
	spin.max_value = maximum
	spin.step = step
	spin.value = value
	spin.tooltip_text = tooltip
	spin.custom_minimum_size = Vector2(90.0, 0.0)
	return spin


## Colour for a managed-run lifecycle state name (TrainingRunController
## state names). One mapping shared by every card/panel/badge.
static func state_color(state_name: String) -> Color:
	match state_name.strip_edges().to_lower():
		"running":
			return COLOR_OK
		"starting", "paused", "stopping":
			return COLOR_WARN
		"error":
			return COLOR_BAD
		"finished":
			return COLOR_TITLE
		_:
			return COLOR_MUTED


## Value string for an optional backend metric: the real number when the
## backend published the key, "n/a" otherwise. Never fabricates.
static func optional_metric(
	status: Dictionary, key: String, decimals: int, scale: float = 1.0, suffix: String = ""
) -> String:
	if not status.has(key) or status[key] == null:
		return "n/a"
	return String.num(float(status[key]) * scale, maxi(0, decimals)) + suffix


## "HH:MM:SS" for a duration; "n/a" for null/negative input.
static func format_duration(seconds) -> String:
	if seconds == null or float(seconds) < 0.0:
		return "n/a"
	var total: int = int(seconds)
	return "%02d:%02d:%02d" % [total / 3600, (total % 3600) / 60, total % 60]


## "HH:MM:SS" wall-clock label for a unix timestamp (local time).
static func format_clock(unix_time: float) -> String:
	var bias_minutes: int = int(Time.get_time_zone_from_system().get("bias", 0))
	var parts: Dictionary = Time.get_datetime_dict_from_unix_time(
		int(unix_time) + bias_minutes * 60
	)
	return "%02d:%02d:%02d" % [parts["hour"], parts["minute"], parts["second"]]


## Colour for a 0..1 ratio: green when high, amber mid, red when low.
static func ratio_color(ratio: float) -> Color:
	if ratio >= 0.66:
		return COLOR_OK
	if ratio >= 0.33:
		return COLOR_WARN
	return COLOR_BAD


## Compact number for a label: "0.25", "1", "16" — trailing zeros removed.
##
## GDScript's `%` formatter does NOT implement C's `%g` (it knows %d, %f,
## %s, %x, %o, %c and %%). Using "%g" raised
## "String formatting error: unsupported format character." on every
## Control Center build and produced an unusable label, which is why speed
## labels go through this helper instead.
static func format_number(value: float, decimals: int = 2) -> String:
	var text: String = String.num(value, maxi(0, decimals))
	if text.contains("."):
		text = text.rstrip("0")
		text = text.rstrip(".")
	if text.is_empty() or text == "-0":
		return "0"
	return text

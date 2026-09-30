## ControlCenterTheme
##
## Small factory helpers shared by every Control Center panel so the look
## is defined once instead of being re-invented per panel. Pure
## presentation: nothing here reads or writes simulation state.
class_name ControlCenterTheme
extends RefCounted

const ControlCenterSurfaceLabel = preload("res://scripts/control_center/ui/surface_label.gd")

# Calm, slightly blue neutral palette shared by every in-simulator surface.
const COLOR_BACKGROUND: Color = Color("#111827e8")
const COLOR_BACKGROUND_SOLID: Color = Color("#0b1120")
const COLOR_SURFACE: Color = Color("#151f32")
const COLOR_SURFACE_HOVER: Color = Color("#1c2940")
const COLOR_BORDER: Color = Color("#2b3a52")
const COLOR_TEXT: Color = Color("#e7edf7")
const COLOR_MUTED: Color = Color("#94a3b8")
const COLOR_TITLE: Color = Color("#7dd3fc")
const COLOR_ACCENT: Color = Color("#38bdf8")
const COLOR_OK: Color = Color("#4ade80")
const COLOR_WARN: Color = Color("#fbbf24")
const COLOR_BAD: Color = Color("#fb7185")
const COLOR_AI: Color = Color("#67e8f9")
const COLOR_HUMAN: Color = Color("#c4b5fd")
const COLOR_HIDDEN: Color = Color("#fda4af")

const FONT_SIZE_TITLE: int = 17
const FONT_SIZE_NORMAL: int = 14
const FONT_SIZE_SMALL: int = 12
const RADIUS_PANEL: int = 12
const RADIUS_CONTROL: int = 9


static func panel_style(
	background: Color = COLOR_BACKGROUND, border: Color = COLOR_BORDER
) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = background
	style.border_color = border
	style.set_border_width_all(1)
	style.set_corner_radius_all(RADIUS_PANEL)
	style.content_margin_left = 16.0
	style.content_margin_right = 16.0
	style.content_margin_top = 12.0
	style.content_margin_bottom = 12.0
	return style


## PanelContainer + VBoxContainer with an optional caption. Returns the
## panel; `content_container(panel)` gives the VBox to fill.
static func make_panel(title: String = "") -> PanelContainer:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", panel_style())
	var box := VBoxContainer.new()
	box.name = "Content"
	box.add_theme_constant_override("separation", 8)
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


static func make_empty_label(text: String) -> Label:
	var label := ControlCenterSurfaceLabel.new()
	label.text = text
	label.add_theme_font_size_override("font_size", FONT_SIZE_NORMAL)
	label.add_theme_color_override("font_color", COLOR_MUTED)
	label.custom_minimum_size = Vector2(0.0, 64.0)
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.set_surface(Color("#0f172a"), Color("#243044"), RADIUS_CONTROL)
	return label


static func make_status_label(text: String = "") -> Label:
	var label := ControlCenterSurfaceLabel.new()
	label.text = text
	label.add_theme_font_size_override("font_size", FONT_SIZE_NORMAL)
	label.custom_minimum_size = Vector2(90.0, 28.0)
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	return label


static func apply_status_badge(label: Label, text: String, color: Color) -> void:
	label.text = text
	label.add_theme_color_override("font_color", color)
	if label is ControlCenterSurfaceLabel:
		var background := Color(color.r, color.g, color.b, 0.14)
		var border := Color(color.r, color.g, color.b, 0.42)
		(label as ControlCenterSurfaceLabel).set_surface(background, border, 99)


static func control_style(background: Color, border: Color = COLOR_BORDER) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = background
	style.border_color = border
	style.set_border_width_all(1)
	style.set_corner_radius_all(RADIUS_CONTROL)
	style.content_margin_left = 12.0
	style.content_margin_right = 12.0
	style.content_margin_top = 7.0
	style.content_margin_bottom = 7.0
	return style


static func make_button(text: String, tooltip: String = "") -> Button:
	var button := Button.new()
	button.text = text
	button.tooltip_text = tooltip
	button.custom_minimum_size = Vector2(0.0, 36.0)
	button.add_theme_font_size_override("font_size", FONT_SIZE_NORMAL)
	button.add_theme_color_override("font_color", COLOR_TEXT)
	button.add_theme_color_override("font_hover_color", Color.WHITE)
	button.add_theme_stylebox_override("normal", control_style(COLOR_SURFACE))
	button.add_theme_stylebox_override("hover", control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
	button.add_theme_stylebox_override("pressed", control_style(Color("#243b53"), COLOR_ACCENT))
	button.add_theme_stylebox_override("focus", control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
	button.add_theme_stylebox_override("disabled", control_style(Color("#111827"), Color("#243044")))
	button.focus_mode = Control.FOCUS_ALL
	return button


## A base Theme catches controls created directly by specialist panels, while
## the factory helpers below can still add component-specific refinements.
static func make_primary_button(text: String, tooltip: String = "") -> Button:
	var button := make_button(text, tooltip)
	button.add_theme_color_override("font_color", Color("#06131d"))
	button.add_theme_color_override("font_hover_color", Color("#06131d"))
	button.add_theme_stylebox_override("normal", control_style(COLOR_ACCENT, COLOR_ACCENT))
	button.add_theme_stylebox_override("hover", control_style(Color("#7dd3fc"), Color("#7dd3fc")))
	button.add_theme_stylebox_override("pressed", control_style(Color("#0ea5e9"), Color("#0ea5e9")))
	return button


static func make_danger_button(text: String, tooltip: String = "") -> Button:
	var button := make_button(text, tooltip)
	button.add_theme_color_override("font_color", COLOR_BAD)
	button.add_theme_color_override("font_hover_color", Color("#fff1f2"))
	button.add_theme_stylebox_override("hover", control_style(Color("#4c1d2a"), COLOR_BAD))
	button.add_theme_stylebox_override("pressed", control_style(Color("#881337"), COLOR_BAD))
	return button


static func build_theme() -> Theme:
	var theme := Theme.new()
	for type_name in ["Button", "OptionButton"]:
		theme.set_font_size("font_size", type_name, FONT_SIZE_NORMAL)
		theme.set_color("font_color", type_name, COLOR_TEXT)
		theme.set_color("font_hover_color", type_name, Color.WHITE)
		theme.set_stylebox("normal", type_name, control_style(COLOR_SURFACE))
		theme.set_stylebox("hover", type_name, control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
		theme.set_stylebox("pressed", type_name, control_style(Color("#243b53"), COLOR_ACCENT))
		theme.set_stylebox("focus", type_name, control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
		theme.set_stylebox("disabled", type_name, control_style(Color("#111827"), Color("#243044")))
	for type_name in ["LineEdit", "SpinBox"]:
		theme.set_font_size("font_size", type_name, FONT_SIZE_NORMAL)
		theme.set_color("font_color", type_name, COLOR_TEXT)
		theme.set_color("font_placeholder_color", type_name, COLOR_MUTED)
		theme.set_stylebox("normal", type_name, control_style(COLOR_SURFACE))
		theme.set_stylebox("focus", type_name, control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
	theme.set_stylebox("panel", "PanelContainer", panel_style())
	theme.set_stylebox("panel", "TabContainer", panel_style())
	theme.set_stylebox("tab_unselected", "TabBar", control_style(COLOR_SURFACE))
	theme.set_stylebox("tab_hovered", "TabBar", control_style(COLOR_SURFACE_HOVER))
	theme.set_stylebox("tab_selected", "TabBar", control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
	theme.set_color("font_unselected_color", "TabBar", COLOR_MUTED)
	theme.set_color("font_selected_color", "TabBar", COLOR_ACCENT)
	theme.set_font_size("font_size", "TabBar", FONT_SIZE_SMALL)
	var progress_background := control_style(Color("#111827"), Color("#243044"))
	var progress_fill := control_style(COLOR_ACCENT, COLOR_ACCENT)
	theme.set_stylebox("background", "ProgressBar", progress_background)
	theme.set_stylebox("fill", "ProgressBar", progress_fill)
	return theme


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
	row.add_theme_constant_override("separation", 8)
	return row


## Two-column key/value grid used by most read-only panels.
static func make_grid(columns: int = 2) -> GridContainer:
	var grid := GridContainer.new()
	grid.columns = columns
	grid.add_theme_constant_override("h_separation", 16)
	grid.add_theme_constant_override("v_separation", 8)
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
	option.custom_minimum_size = Vector2(0.0, 36.0)
	option.add_theme_font_size_override("font_size", FONT_SIZE_NORMAL)
	option.add_theme_stylebox_override("normal", control_style(COLOR_SURFACE))
	option.add_theme_stylebox_override("hover", control_style(COLOR_SURFACE_HOVER, COLOR_ACCENT))
	option.add_theme_stylebox_override("pressed", control_style(Color("#243b53"), COLOR_ACCENT))
	option.focus_mode = Control.FOCUS_ALL
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

## Compact telemetry chart for a bounded series of REAL metric samples. It
## draws exactly the values it was given — no smoothing or extrapolation —
## and stays data-empty until two samples exist.
class_name ControlCenterMetricSparkline
extends Control

const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

const PADDING: float = 5.0
const GRID_LINES: int = 3

var _samples: Array = []


func _init() -> void:
	custom_minimum_size = Vector2(160.0, 42.0)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func set_samples(samples: Array) -> void:
	_samples = samples.duplicate()
	queue_redraw()


func sample_count() -> int:
	return _samples.size()


func _draw() -> void:
	var bounds := Rect2(Vector2.ZERO, size)
	if bounds.size.x < 2.0 or bounds.size.y < 2.0:
		return
	var surface := Color(
		ControlCenterTheme.COLOR_BACKGROUND_DEEP.r,
		ControlCenterTheme.COLOR_BACKGROUND_DEEP.g,
		ControlCenterTheme.COLOR_BACKGROUND_DEEP.b,
		0.64
	)
	var border := Color(
		ControlCenterTheme.COLOR_ACCENT.r,
		ControlCenterTheme.COLOR_ACCENT.g,
		ControlCenterTheme.COLOR_ACCENT.b,
		0.26
	)
	draw_rect(bounds, surface, true)
	draw_rect(bounds, border, false, 1.0)
	_draw_grid(bounds)
	if _samples.size() < 2:
		return
	_draw_series(bounds)


func _draw_grid(bounds: Rect2) -> void:
	var grid_color := Color(
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.r,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.g,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.b,
		0.11
	)
	for index in range(1, GRID_LINES + 1):
		var fraction: float = float(index) / float(GRID_LINES + 1)
		var y: float = lerpf(bounds.position.y, bounds.end.y, fraction)
		draw_line(Vector2(bounds.position.x, y), Vector2(bounds.end.x, y), grid_color, 1.0)


func _draw_series(bounds: Rect2) -> void:
	var low: float = float(_samples[0])
	var high: float = float(_samples[0])
	for value in _samples:
		low = minf(low, float(value))
		high = maxf(high, float(value))
	var flat: bool = is_equal_approx(low, high)
	var span: float = maxf(high - low, 0.000001)
	var count: int = _samples.size()
	var drawable_width: float = maxf(bounds.size.x - PADDING * 2.0, 1.0)
	var drawable_height: float = maxf(bounds.size.y - PADDING * 2.0, 1.0)
	var points := PackedVector2Array()
	for index in range(count):
		var x: float = (
			bounds.position.x + PADDING + drawable_width * float(index) / float(count - 1)
		)
		var normalized: float = 0.5 if flat else (float(_samples[index]) - low) / span
		var y: float = bounds.position.y + PADDING + (1.0 - normalized) * drawable_height
		points.append(Vector2(x, y))
	var glow := Color(
		ControlCenterTheme.COLOR_ACCENT.r,
		ControlCenterTheme.COLOR_ACCENT.g,
		ControlCenterTheme.COLOR_ACCENT.b,
		0.18
	)
	draw_polyline(points, glow, 4.0, true)
	draw_polyline(points, ControlCenterTheme.COLOR_AI, 1.5, true)
	var latest: Vector2 = points[points.size() - 1]
	draw_circle(latest, 3.0, glow)
	draw_circle(latest, 1.75, ControlCenterTheme.COLOR_ACCENT)

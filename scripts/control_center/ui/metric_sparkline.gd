## Minimal polyline sparkline for a bounded series of REAL metric samples.
## It draws exactly the values it was given — no smoothing, no
## extrapolation — and stays blank until two samples exist.
class_name ControlCenterMetricSparkline
extends Control

const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var _samples: Array = []


func _init() -> void:
	custom_minimum_size = Vector2(160.0, 36.0)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL


func set_samples(samples: Array) -> void:
	_samples = samples.duplicate()
	queue_redraw()


func sample_count() -> int:
	return _samples.size()


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color(1.0, 1.0, 1.0, 0.04))
	if _samples.size() < 2:
		return
	var low: float = float(_samples[0])
	var high: float = float(_samples[0])
	for value in _samples:
		low = minf(low, float(value))
		high = maxf(high, float(value))
	var span: float = maxf(high - low, 0.000001)
	var points := PackedVector2Array()
	var count: int = _samples.size()
	for index in range(count):
		var x: float = size.x * float(index) / float(count - 1)
		var normalized: float = (float(_samples[index]) - low) / span
		var y: float = size.y - 3.0 - normalized * maxf(size.y - 6.0, 1.0)
		points.append(Vector2(x, y))
	draw_polyline(points, ControlCenterTheme.COLOR_AI, 1.5, true)

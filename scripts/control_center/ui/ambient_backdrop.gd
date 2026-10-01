## ControlCenterAmbientBackdrop
##
## Presentation-only atmospheric layer for the Control Center. It deliberately
## draws no simulation data: the grid, scan line and corner brackets make the
## operator workspace feel like an instrument panel without implying that any
## of those graphics exist in TTK Testing's player HUD.
class_name ControlCenterAmbientBackdrop
extends Control

const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

const GRID_SPACING: float = 56.0
const GLOW_COUNT: int = 7
const GLOW_RADIUS_STEP: float = 90.0
const SCAN_PERIOD_SECONDS: float = 9.0

var _elapsed: float = 0.0


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	set_anchors_preset(Control.PRESET_FULL_RECT)
	queue_redraw()


func _process(delta: float) -> void:
	_elapsed = fmod(_elapsed + maxf(0.0, delta), SCAN_PERIOD_SECONDS)
	queue_redraw()


func _draw() -> void:
	var bounds := Rect2(Vector2.ZERO, size)
	if bounds.size.x < 2.0 or bounds.size.y < 2.0:
		return

	# A restrained navy base and several large translucent circles supply the
	# depth normally provided by an expensive shader, while remaining cheap
	# and deterministic for this operator-only UI.
	draw_rect(bounds, ControlCenterTheme.COLOR_BACKGROUND_DEEP, true)
	var glow_center := Vector2(bounds.size.x * 0.72, bounds.size.y * 0.18)
	for index in range(GLOW_COUNT, 0, -1):
		var radius: float = float(index) * GLOW_RADIUS_STEP
		var alpha: float = 0.008 + float(GLOW_COUNT - index) * 0.004
		draw_circle(
			glow_center,
			radius,
			Color(
				ControlCenterTheme.COLOR_ACCENT.r,
				ControlCenterTheme.COLOR_ACCENT.g,
				ControlCenterTheme.COLOR_ACCENT.b,
				alpha
			)
		)

	_draw_grid(bounds)
	_draw_scan_line(bounds)
	_draw_corner_brackets(bounds)


func _draw_grid(bounds: Rect2) -> void:
	var offset_x: float = fmod(_elapsed * 5.0, GRID_SPACING)
	var offset_y: float = fmod(_elapsed * 2.0, GRID_SPACING)
	var grid_color := Color(
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.r,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.g,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.b,
		0.045
	)
	var x: float = -GRID_SPACING + offset_x
	while x < bounds.size.x + GRID_SPACING:
		draw_line(Vector2(x, 0.0), Vector2(x, bounds.size.y), grid_color, 1.0)
		x += GRID_SPACING
	var y: float = -GRID_SPACING + offset_y
	while y < bounds.size.y + GRID_SPACING:
		draw_line(Vector2(0.0, y), Vector2(bounds.size.x, y), grid_color, 1.0)
		y += GRID_SPACING


func _draw_scan_line(bounds: Rect2) -> void:
	var progress: float = _elapsed / SCAN_PERIOD_SECONDS
	var y: float = lerpf(0.0, bounds.size.y, progress)
	var scan_color := Color(
		ControlCenterTheme.COLOR_ACCENT.r,
		ControlCenterTheme.COLOR_ACCENT.g,
		ControlCenterTheme.COLOR_ACCENT.b,
		0.12
	)
	draw_line(Vector2(0.0, y), Vector2(bounds.size.x, y), scan_color, 1.0)


func _draw_corner_brackets(bounds: Rect2) -> void:
	var color := Color(
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.r,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.g,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY.b,
		0.38
	)
	var inset: float = 6.0
	var length: float = 34.0
	for corner in [
		Vector2(inset, inset),
		Vector2(bounds.size.x - inset, inset),
		Vector2(inset, bounds.size.y - inset),
		Vector2(bounds.size.x - inset, bounds.size.y - inset),
	]:
		var sign_x: float = -1.0 if corner.x > bounds.size.x * 0.5 else 1.0
		var sign_y: float = -1.0 if corner.y > bounds.size.y * 0.5 else 1.0
		draw_line(corner, corner + Vector2(sign_x * length, 0.0), color, 1.0)
		draw_line(corner, corner + Vector2(0.0, sign_y * length), color, 1.0)

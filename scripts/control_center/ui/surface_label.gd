## Label with a lightweight rounded background. Godot's built-in Label has no
## StyleBox theme slot, so badges/empty states use this control instead of
## installing an override the engine would silently ignore.
class_name ControlCenterSurfaceLabel
extends Label

var surface_color: Color = Color.TRANSPARENT
var border_color: Color = Color.TRANSPARENT
var corner_radius: int = 9
var border_width: int = 1


func set_surface(background: Color, border: Color, radius: int = 9) -> void:
	surface_color = background
	border_color = border
	corner_radius = radius
	queue_redraw()


func _draw() -> void:
	if surface_color.a <= 0.0 and border_color.a <= 0.0:
		return
	var style := StyleBoxFlat.new()
	style.bg_color = surface_color
	style.border_color = border_color
	style.set_border_width_all(border_width)
	style.set_corner_radius_all(corner_radius)
	style.draw(get_canvas_item(), Rect2(Vector2.ZERO, size))

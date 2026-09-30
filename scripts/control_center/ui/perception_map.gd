## PerceptionMap
##
## Side-by-side top-down maps that make the Phase 5 question concrete:
##
##   LEFT  "REAL WORLD"    - every enemy the simulation actually contains.
##   RIGHT "AI PERCEPTION" - reconstructed ONLY from the observation vector
##                           (bearing + distance per tracked slot), i.e.
##                           exactly what the policy could know.
##
## Enemies present on the left but missing on the right are drawn in red on
## the left with a dashed outline: they exist, and the policy cannot see
## them. Nothing here computes perception — it renders the dictionary built
## by PerceptionModel.
class_name PerceptionMap
extends Control

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

const MAP_GAP: float = 8.0
const HEADER_HEIGHT: float = 14.0
const AGENT_RADIUS_PX: float = 4.0
const ENEMY_RADIUS_PX: float = 3.5

var perception: Dictionary = {}
var _font: Font


func _init() -> void:
	custom_minimum_size = Vector2(0.0, 170.0)
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func _ready() -> void:
	_font = get_theme_default_font()


## Stores the latest PerceptionModel.build() output and requests a redraw.
func set_perception(value: Dictionary) -> void:
	perception = value
	queue_redraw()


func _draw() -> void:
	var full: Vector2 = size
	var map_width: float = (full.x - MAP_GAP) * 0.5
	var map_height: float = full.y - HEADER_HEIGHT
	var left := Rect2(Vector2(0.0, HEADER_HEIGHT), Vector2(map_width, map_height))
	var right := Rect2(Vector2(map_width + MAP_GAP, HEADER_HEIGHT), Vector2(map_width, map_height))

	_draw_caption("REAL WORLD", Vector2(0.0, 0.0), ControlCenterTheme.COLOR_TEXT)
	_draw_caption("AI PERCEPTION", Vector2(map_width + MAP_GAP, 0.0), ControlCenterTheme.COLOR_AI)
	_draw_frame(left)
	_draw_frame(right)

	if perception.is_empty():
		_draw_text(left.position + Vector2(6.0, 18.0), "no data", ControlCenterTheme.COLOR_MUTED)
		return

	_draw_real_world(left)
	_draw_ai_perception(right)


func _draw_caption(text: String, position_offset: Vector2, color: Color) -> void:
	_draw_text(position_offset + Vector2(2.0, 11.0), text, color)


func _draw_text(at: Vector2, text: String, color: Color) -> void:
	if _font == null:
		_font = get_theme_default_font()
	if _font == null:
		return
	draw_string(
		_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, ControlCenterTheme.FONT_SIZE_SMALL, color
	)


func _draw_frame(rect: Rect2) -> void:
	draw_rect(rect, Color(0.10, 0.11, 0.14, 0.9), true)
	draw_rect(rect, ControlCenterTheme.COLOR_BORDER, false, 1.0)


func _draw_real_world(rect: Rect2) -> void:
	var real_world: Dictionary = perception.get("real_world", {})
	if real_world.is_empty():
		return
	var half_extent: float = maxf(1.0, float(real_world.get("arena_half_extent", 12.0)))
	var agent_position: Vector3 = real_world.get("agent_position", Vector3.ZERO)
	var forward: Vector3 = real_world.get("agent_forward", Vector3.FORWARD)
	var tracked: Array = perception.get("tracked_enemy_indices", [])

	var center: Vector2 = rect.position + rect.size * 0.5
	var px_per_m: float = (minf(rect.size.x, rect.size.y) * 0.5 - 6.0) / half_extent

	# Arena bounds.
	var bounds := Rect2(
		center - Vector2(half_extent, half_extent) * px_per_m,
		Vector2(half_extent, half_extent) * px_per_m * 2.0
	)
	draw_rect(bounds, Color(0.30, 0.33, 0.40, 1.0), false, 1.0)

	var agent_point: Vector2 = center + Vector2(agent_position.x, agent_position.z) * px_per_m
	# Facing indicator.
	draw_line(
		agent_point,
		agent_point + Vector2(forward.x, forward.z).normalized() * 14.0,
		ControlCenterTheme.COLOR_WARN,
		1.5
	)
	draw_circle(agent_point, AGENT_RADIUS_PX, ControlCenterTheme.COLOR_OK)

	for entry_value in real_world.get("enemies", []) as Array:
		var entry: Dictionary = entry_value
		var enemy_position: Vector3 = entry["position"]
		var point: Vector2 = center + Vector2(enemy_position.x, enemy_position.z) * px_per_m
		var alive: bool = bool(entry["alive"])
		var visible_to_ai: bool = tracked.has(int(entry["index"]))
		var color: Color = ControlCenterTheme.COLOR_MUTED
		if alive:
			color = (
				ControlCenterTheme.COLOR_AI if visible_to_ai else ControlCenterTheme.COLOR_HIDDEN
			)
		draw_circle(point, ENEMY_RADIUS_PX, color)
		if alive and not visible_to_ai:
			draw_arc(point, ENEMY_RADIUS_PX + 3.0, 0.0, TAU, 12, color, 1.0)
		if alive and visible_to_ai:
			draw_line(agent_point, point, Color(color.r, color.g, color.b, 0.45), 1.0)
		_draw_text(point + Vector2(5.0, -4.0), "#%d" % int(entry["index"]), color)


## Right map: only what the observation vector contains. Slot positions are
## reconstructed from (bearing, distance) relative to the agent's facing,
## which is why the agent is always at the centre pointing "up".
func _draw_ai_perception(rect: Rect2) -> void:
	var ai_perception: Dictionary = perception.get("ai_perception", {})
	var real_world: Dictionary = perception.get("real_world", {})
	if ai_perception.is_empty():
		return
	var half_extent: float = maxf(1.0, float(real_world.get("arena_half_extent", 12.0)))
	var center: Vector2 = rect.position + rect.size * 0.5
	var px_per_m: float = (minf(rect.size.x, rect.size.y) * 0.5 - 6.0) / half_extent

	draw_line(center, center + Vector2(0.0, -16.0), ControlCenterTheme.COLOR_WARN, 1.5)
	draw_circle(center, AGENT_RADIUS_PX, ControlCenterTheme.COLOR_OK)

	var weapon_range: float = float(real_world.get("weapon_range", 15.0))
	draw_arc(center, weapon_range * px_per_m, 0.0, TAU, 48, Color(0.35, 0.45, 0.55, 0.7), 1.0)

	var any_slot: bool = false
	for slot_value in ai_perception.get("slots", []) as Array:
		var slot: Dictionary = slot_value
		if not bool(slot["alive"]):
			continue
		any_slot = true
		var bearing_rad: float = deg_to_rad(float(slot["bearing_deg"]))
		var distance: float = float(slot["distance_m"])
		# Agent-relative frame: +bearing is to the right, "up" is forward.
		var point: Vector2 = (
			center + Vector2(sin(bearing_rad), -cos(bearing_rad)) * distance * px_per_m
		)
		draw_circle(point, ENEMY_RADIUS_PX, ControlCenterTheme.COLOR_AI)
		draw_line(center, point, Color(0.40, 0.85, 0.95, 0.45), 1.0)
		_draw_text(
			point + Vector2(5.0, -4.0), str(slot["slot"]).substr(0, 3), ControlCenterTheme.COLOR_AI
		)

	if not any_slot:
		_draw_text(
			rect.position + Vector2(6.0, rect.size.y - 8.0),
			"observation reports no live enemy",
			ControlCenterTheme.COLOR_MUTED
		)

	var hidden: Array = perception.get("hidden_from_ai", [])
	if not hidden.is_empty():
		_draw_text(
			rect.position + Vector2(6.0, 14.0),
			"%d enemy(s) not in observation" % hidden.size(),
			ControlCenterTheme.COLOR_HIDDEN
		)

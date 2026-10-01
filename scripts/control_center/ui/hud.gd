## ControlCenterHud
##
## Local simulation instrumentation drawn over the first-person viewport. It
## remains mouse-ignoring, so it never captures gameplay input. The visual
## language is intentionally an operator overlay, not a reconstruction of
## TTK Testing's unverified player HUD.
class_name ControlCenterHud
extends Control

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

const CROSSHAIR_SIZE: float = 10.0
const CROSSHAIR_DOT_RADIUS: float = 2.0
const HIT_MARKER_SECONDS: float = 0.35
const DAMAGE_FLASH_SECONDS: float = 0.45
const HUD_MARGIN: float = 16.0
const HEALTH_PANEL_SIZE := Vector2(274.0, 66.0)
const STATUS_PANEL_WIDTH: float = 304.0

var _snapshot: Dictionary = {}
var _mode: int = ControlCenterConfig.Mode.WATCH
var _human_armed: bool = false
var _hit_timer: float = 0.0
var _damage_timer: float = 0.0
var _last_shots_hit: int = -1
var _last_damage_taken: float = -1.0
var _font: Font


func setup() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	set_anchors_preset(Control.PRESET_FULL_RECT)
	_font = ThemeDB.fallback_font


func _process(delta: float) -> void:
	if _hit_timer > 0.0:
		_hit_timer = maxf(_hit_timer - delta, 0.0)
		queue_redraw()
	if _damage_timer > 0.0:
		_damage_timer = maxf(_damage_timer - delta, 0.0)
		queue_redraw()


func refresh(snapshot: Dictionary, mode: int, human_armed: bool) -> void:
	_snapshot = snapshot
	_mode = mode
	_human_armed = human_armed
	if not snapshot.is_empty():
		var episode: Dictionary = snapshot["episode"]
		var shots_hit: int = int(episode["shots_hit"])
		var damage_taken: float = float(episode["damage_received"])
		if _last_shots_hit >= 0 and shots_hit > _last_shots_hit:
			_hit_timer = HIT_MARKER_SECONDS
		if _last_damage_taken >= 0.0 and damage_taken > _last_damage_taken:
			_damage_timer = DAMAGE_FLASH_SECONDS
		_last_shots_hit = shots_hit
		_last_damage_taken = damage_taken
	queue_redraw()


func _draw() -> void:
	if _snapshot.is_empty():
		return
	var viewport := Rect2(Vector2.ZERO, size)
	if viewport.size.x < 40.0 or viewport.size.y < 40.0:
		return

	var agent: Dictionary = _snapshot["agent"]
	var episode: Dictionary = _snapshot["episode"]
	var target: Dictionary = _snapshot["target"]

	_draw_damage_feedback(viewport)
	_draw_crosshair(viewport, bool(target.get("in_range", false)), bool(agent["weapon_ready"]))
	_draw_health(agent)
	_draw_status(agent, episode, target)
	_draw_mode_banner()


func _draw_damage_feedback(viewport: Rect2) -> void:
	if _damage_timer <= 0.0:
		return
	var alpha: float = 0.42 * (_damage_timer / DAMAGE_FLASH_SECONDS)
	var color := Color(
		ControlCenterTheme.COLOR_BAD.r,
		ControlCenterTheme.COLOR_BAD.g,
		ControlCenterTheme.COLOR_BAD.b,
		alpha
	)
	var length: float = 80.0
	var inset: float = 10.0
	# Four compact brackets preserve a clear first-person view rather than
	# covering the screen with an opaque damage effect.
	for corner in [
		Vector2(inset, inset),
		Vector2(viewport.size.x - inset, inset),
		Vector2(inset, viewport.size.y - inset),
		Vector2(viewport.size.x - inset, viewport.size.y - inset),
	]:
		var sign_x: float = -1.0 if corner.x > viewport.size.x * 0.5 else 1.0
		var sign_y: float = -1.0 if corner.y > viewport.size.y * 0.5 else 1.0
		draw_line(corner, corner + Vector2(sign_x * length, 0.0), color, 2.0)
		draw_line(corner, corner + Vector2(0.0, sign_y * length), color, 2.0)


func _draw_crosshair(viewport: Rect2, in_range: bool, weapon_ready: bool) -> void:
	var center: Vector2 = viewport.size * 0.5
	# The dot tracks the simulation's forward hitscan direction. Its colour
	# reports only local simulator state, never a claimed game mechanic.
	var reticle_color := Color(1.0, 0.1, 0.1, 0.64)
	if in_range:
		reticle_color = (
			ControlCenterTheme.COLOR_OK if weapon_ready else ControlCenterTheme.COLOR_WARN
		)
	reticle_color.a = 0.72
	var gap: float = 5.0
	var outer: float = CROSSHAIR_SIZE
	draw_circle(center, CROSSHAIR_DOT_RADIUS + 1.0, Color(0.0, 0.03, 0.08, 0.72))
	draw_circle(center, CROSSHAIR_DOT_RADIUS, Color(1.0, 0.06, 0.04, 0.96))
	draw_arc(center, outer + 6.0, -0.55, 0.55, 12, reticle_color, 1.0)
	draw_arc(center, outer + 6.0, PI - 0.55, PI + 0.55, 12, reticle_color, 1.0)
	draw_line(center + Vector2(-outer, 0.0), center + Vector2(-gap, 0.0), reticle_color, 1.0)
	draw_line(center + Vector2(gap, 0.0), center + Vector2(outer, 0.0), reticle_color, 1.0)
	draw_line(center + Vector2(0.0, -outer), center + Vector2(0.0, -gap), reticle_color, 1.0)
	draw_line(center + Vector2(0.0, gap), center + Vector2(0.0, outer), reticle_color, 1.0)
	if _hit_timer > 0.0:
		var fade: float = _hit_timer / HIT_MARKER_SECONDS
		var marker := Color(1.0, 0.38, 0.4, fade)
		var offset: float = outer + 5.0
		draw_line(center + Vector2(-offset, -offset), center + Vector2(-gap, -gap), marker, 2.0)
		draw_line(center + Vector2(offset, -offset), center + Vector2(gap, -gap), marker, 2.0)
		draw_line(center + Vector2(-offset, offset), center + Vector2(-gap, gap), marker, 2.0)
		draw_line(center + Vector2(offset, offset), center + Vector2(gap, gap), marker, 2.0)


func _draw_health(agent: Dictionary) -> void:
	var health: float = float(agent["health"])
	var max_health: float = maxf(float(agent["max_health"]), 0.001)
	var ratio: float = clampf(health / max_health, 0.0, 1.0)
	var panel := Rect2(
		Vector2(HUD_MARGIN, size.y - HEALTH_PANEL_SIZE.y - HUD_MARGIN), HEALTH_PANEL_SIZE
	)
	_draw_telemetry_frame(panel, ControlCenterTheme.ratio_color(ratio))
	_draw_text(
		panel.position + Vector2(11.0, 16.0),
		"LOCAL AGENT // HEALTH",
		ControlCenterTheme.COLOR_MUTED,
		ControlCenterTheme.FONT_SIZE_SMALL
	)
	_draw_text(
		panel.position + Vector2(11.0, 37.0),
		"%03d" % roundi(health),
		ControlCenterTheme.ratio_color(ratio),
		20
	)
	_draw_text(
		panel.position + Vector2(58.0, 37.0),
		"/ %03d" % roundi(max_health),
		ControlCenterTheme.COLOR_TEXT,
		ControlCenterTheme.FONT_SIZE_SMALL
	)
	var bar := Rect2(panel.position + Vector2(11.0, 46.0), Vector2(panel.size.x - 22.0, 9.0))
	draw_rect(bar, Color(0.02, 0.07, 0.13, 0.82), true)
	draw_rect(
		Rect2(bar.position, Vector2(bar.size.x * ratio, bar.size.y)),
		ControlCenterTheme.ratio_color(ratio),
		true
	)
	for tick in range(1, 10):
		var tick_x: float = bar.position.x + bar.size.x * float(tick) / 10.0
		draw_line(
			Vector2(tick_x, bar.position.y),
			Vector2(tick_x, bar.end.y),
			Color(0.02, 0.07, 0.13, 0.8),
			1.0
		)

	var ready: bool = bool(agent["weapon_ready"])
	var cycle_seconds: float = float(agent["weapon_cooldown"])
	var weapon_label: String = str(agent.get("weapon_label", agent.get("weapon_profile", "weapon")))
	var readiness: String = "READY" if ready else "CYCLE %.2fs" % cycle_seconds
	var readiness_color: Color = (
		ControlCenterTheme.COLOR_OK if ready else ControlCenterTheme.COLOR_WARN
	)
	_draw_text(
		panel.position + Vector2(panel.size.x - 11.0, 18.0),
		"%s // %s" % [weapon_label.to_upper(), readiness],
		readiness_color,
		ControlCenterTheme.FONT_SIZE_SMALL,
		HORIZONTAL_ALIGNMENT_RIGHT
	)


func _draw_status(agent: Dictionary, episode: Dictionary, target: Dictionary) -> void:
	var lines: PackedStringArray = PackedStringArray()
	lines.append(
		"EPISODE %04d  //  %05.1fs" % [int(episode["episode"]), float(episode["time_seconds"])]
	)
	lines.append("PROGRESS %04d / %04d" % [int(episode["step"]), int(episode["max_steps"])])
	lines.append(
		(
			"SCORE %7.2f    ACCURACY %3.0f%%"
			% [float(episode["reward"]), float(episode["accuracy"]) * 100.0]
		)
	)
	lines.append(
		"HOSTILES %d / %d" % [int(episode["alive_enemies"]), int(episode["total_enemies"])]
	)
	if bool(target.get("has_target", false)):
		(
			lines
			. append(
				(
					"OBSERVED %4.1fm  //  %s"
					% [
						float(target["distance_m"]),
						"IN LOCAL RANGE" if bool(target["in_range"]) else "OUT OF RANGE",
					]
				)
			)
		)
	else:
		lines.append("OBSERVED // NO TARGET")
	if not bool(agent["alive"]):
		lines.append("LOCAL AGENT DOWN // RESET PENDING")

	var panel_height: float = 40.0 + float(lines.size()) * 17.0
	var panel := Rect2(
		Vector2(size.x - STATUS_PANEL_WIDTH - HUD_MARGIN, 52.0),
		Vector2(STATUS_PANEL_WIDTH, panel_height)
	)
	_draw_telemetry_frame(panel, ControlCenterTheme.COLOR_ACCENT)
	_draw_text(
		panel.position + Vector2(11.0, 17.0),
		"SIMULATION INSTRUMENTATION",
		ControlCenterTheme.COLOR_ACCENT,
		ControlCenterTheme.FONT_SIZE_SMALL
	)
	for index in range(lines.size()):
		var line: String = lines[index]
		_draw_text(
			panel.position + Vector2(11.0, 38.0 + float(index) * 17.0),
			line,
			_line_color(line),
			ControlCenterTheme.FONT_SIZE_SMALL
		)


func _line_color(line: String) -> Color:
	if line.begins_with("LOCAL AGENT DOWN"):
		return ControlCenterTheme.COLOR_BAD
	if line.contains("OUT OF RANGE"):
		return ControlCenterTheme.COLOR_WARN
	return ControlCenterTheme.COLOR_TEXT


func _draw_mode_banner() -> void:
	var text := "WATCH // AI CONTROLLER"
	var color := ControlCenterTheme.COLOR_AI
	if _mode == ControlCenterConfig.Mode.HUMAN:
		if _human_armed:
			text = "HUMAN // INPUT ARMED"
			color = ControlCenterTheme.COLOR_HUMAN
		else:
			text = "HUMAN // INPUT DISARMED"
			color = ControlCenterTheme.COLOR_WARN
	elif _mode == ControlCenterConfig.Mode.TRAINING:
		text = "TRAINING // VIEWPORT TELEMETRY OFF"
		color = ControlCenterTheme.COLOR_MUTED
	var panel := Rect2(Vector2(HUD_MARGIN, 52.0), Vector2(236.0, 32.0))
	_draw_telemetry_frame(panel, color)
	_draw_text(
		panel.position + Vector2(10.0, 20.0), text, color, ControlCenterTheme.FONT_SIZE_SMALL
	)


func _draw_telemetry_frame(rect: Rect2, accent: Color) -> void:
	var background := Color(
		ControlCenterTheme.COLOR_BACKGROUND_DEEP.r,
		ControlCenterTheme.COLOR_BACKGROUND_DEEP.g,
		ControlCenterTheme.COLOR_BACKGROUND_DEEP.b,
		0.74
	)
	draw_rect(rect, background, true)
	draw_rect(rect, Color(accent.r, accent.g, accent.b, 0.48), false, 1.0)
	var corner: float = 8.0
	var line_color := Color(accent.r, accent.g, accent.b, 0.92)
	draw_line(rect.position, rect.position + Vector2(corner, 0.0), line_color, 2.0)
	draw_line(rect.position, rect.position + Vector2(0.0, corner), line_color, 2.0)
	draw_line(rect.end, rect.end - Vector2(corner, 0.0), line_color, 2.0)
	draw_line(rect.end, rect.end - Vector2(0.0, corner), line_color, 2.0)


## `at` (not `position`) so the parameter never shadows `Control.position`.
func _draw_text(
	at: Vector2,
	text: String,
	color: Color,
	font_size: int = ControlCenterTheme.FONT_SIZE_NORMAL,
	alignment: HorizontalAlignment = HORIZONTAL_ALIGNMENT_LEFT
) -> void:
	if _font == null:
		return
	var draw_at := at
	if alignment == HORIZONTAL_ALIGNMENT_RIGHT:
		draw_at.x -= _font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size).x
	elif alignment == HORIZONTAL_ALIGNMENT_CENTER:
		draw_at.x -= _font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size).x * 0.5
	draw_string(
		_font,
		draw_at + Vector2(1.0, 1.0),
		text,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1.0,
		font_size,
		Color(0.0, 0.0, 0.0, 0.76)
	)
	draw_string(_font, draw_at, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, color)

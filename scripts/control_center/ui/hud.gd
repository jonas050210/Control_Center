## ControlCenterHud
##
## Transparent in-world HUD drawn over the 3D viewport (Phase 11): health
## bar, weapon readiness, crosshair with hit feedback, target distance,
## episode timer, live reward and a mode banner.
##
## It is deliberately drawn with `_draw()` on a mouse-ignoring Control so
## it never intercepts gameplay input and adds no layout cost.
class_name ControlCenterHud
extends Control

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const CROSSHAIR_SIZE: float = 9.0
const HIT_MARKER_SECONDS: float = 0.35
const DAMAGE_FLASH_SECONDS: float = 0.45

var _snapshot: Dictionary = {}
var _mode: int = ControlCenterConfig.Mode.WATCH
var _human_armed: bool = false
var _hit_timer: float = 0.0
var _damage_timer: float = 0.0
var _last_shots_hit: int = -1
var _last_damage_taken: float = -1.0
var _font: Font
var _font_size: int = ControlCenterTheme.FONT_SIZE_NORMAL


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
	var rect := Rect2(Vector2.ZERO, size)
	if rect.size.x < 40.0 or rect.size.y < 40.0:
		return

	var agent: Dictionary = _snapshot["agent"]
	var episode: Dictionary = _snapshot["episode"]
	var target: Dictionary = _snapshot["target"]

	if _damage_timer > 0.0:
		var alpha: float = 0.35 * (_damage_timer / DAMAGE_FLASH_SECONDS)
		draw_rect(rect, Color(0.85, 0.1, 0.1, alpha), true)

	_draw_crosshair(rect, bool(target.get("in_range", false)), bool(agent["weapon_ready"]))
	_draw_health(agent)
	_draw_status(agent, episode, target)
	_draw_mode_banner()


func _draw_crosshair(rect: Rect2, in_range: bool, weapon_ready: bool) -> void:
	var center: Vector2 = rect.size * 0.5
	var color: Color = ControlCenterTheme.COLOR_TEXT
	if in_range:
		color = ControlCenterTheme.COLOR_OK if weapon_ready else ControlCenterTheme.COLOR_WARN
	var gap: float = 3.0
	var length: float = CROSSHAIR_SIZE
	draw_line(center + Vector2(-length, 0.0), center + Vector2(-gap, 0.0), color, 1.5)
	draw_line(center + Vector2(gap, 0.0), center + Vector2(length, 0.0), color, 1.5)
	draw_line(center + Vector2(0.0, -length), center + Vector2(0.0, -gap), color, 1.5)
	draw_line(center + Vector2(0.0, gap), center + Vector2(0.0, length), color, 1.5)
	if _hit_timer > 0.0:
		var fade: float = _hit_timer / HIT_MARKER_SECONDS
		var marker := Color(1.0, 0.35, 0.35, fade)
		var offset: float = length + 3.0
		draw_line(center + Vector2(-offset, -offset), center + Vector2(-gap, -gap), marker, 2.0)
		draw_line(center + Vector2(offset, -offset), center + Vector2(gap, -gap), marker, 2.0)
		draw_line(center + Vector2(-offset, offset), center + Vector2(-gap, gap), marker, 2.0)
		draw_line(center + Vector2(offset, offset), center + Vector2(gap, gap), marker, 2.0)


func _draw_health(agent: Dictionary) -> void:
	var health: float = float(agent["health"])
	var max_health: float = maxf(float(agent["max_health"]), 0.001)
	var ratio: float = clampf(health / max_health, 0.0, 1.0)
	var bar := Rect2(Vector2(16.0, size.y - 46.0), Vector2(220.0, 14.0))
	draw_rect(bar, Color(0.0, 0.0, 0.0, 0.55), true)
	var fill := Rect2(bar.position, Vector2(bar.size.x * ratio, bar.size.y))
	draw_rect(fill, ControlCenterTheme.ratio_color(ratio), true)
	draw_rect(bar, ControlCenterTheme.COLOR_BORDER, false, 1.0)
	_draw_text(
		bar.position + Vector2(6.0, 11.0),
		"HP %3.0f / %3.0f" % [health, max_health],
		ControlCenterTheme.COLOR_TEXT
	)

	var ready: bool = bool(agent["weapon_ready"])
	var cooldown: float = float(agent["weapon_cooldown"])
	_draw_text(
		Vector2(16.0, size.y - 22.0),
		"WEAPON %s" % ("READY" if ready else "reload %.2fs" % cooldown),
		ControlCenterTheme.COLOR_OK if ready else ControlCenterTheme.COLOR_WARN
	)


func _draw_status(agent: Dictionary, episode: Dictionary, target: Dictionary) -> void:
	var lines: PackedStringArray = PackedStringArray()
	lines.append(
		"TIME %5.1fs   STEP %4d/%d"
		% [float(episode["time_seconds"]), int(episode["step"]), int(episode["max_steps"])]
	)
	lines.append(
		"REWARD %7.2f   KILLS %d   DEATHS %d"
		% [float(episode["reward"]), int(episode["kills"]), int(episode["deaths"])]
	)
	lines.append(
		"ENEMIES %d/%d   ACCURACY %3.0f%%"
		% [
			int(episode["alive_enemies"]),
			int(episode["total_enemies"]),
			float(episode["accuracy"]) * 100.0,
		]
	)
	if bool(target.get("has_target", false)):
		lines.append(
			"TARGET %4.1fm %s   HP %3.0f"
			% [
				float(target["distance_m"]),
				"IN RANGE" if bool(target["in_range"]) else "far (%.0fm)" % SandboxConfig.WEAPON_RANGE,
				float(target["health"]),
			]
		)
	else:
		lines.append("TARGET none")
	if not bool(agent["alive"]):
		lines.append("AGENT DOWN - waiting for reset")

	var origin := Vector2(size.x - 300.0, 20.0)
	for index in range(lines.size()):
		_draw_text(origin + Vector2(0.0, float(index) * 16.0), lines[index], _line_color(index))


func _line_color(index: int) -> Color:
	if index == 4:
		return ControlCenterTheme.COLOR_BAD
	return ControlCenterTheme.COLOR_TEXT


func _draw_mode_banner() -> void:
	var text: String = "WATCH - AI is driving"
	var color: Color = ControlCenterTheme.COLOR_AI
	if _mode == ControlCenterConfig.Mode.HUMAN:
		if _human_armed:
			text = "HUMAN - you are driving (Esc releases the mouse)"
			color = ControlCenterTheme.COLOR_HUMAN
		else:
			text = "HUMAN - input disarmed, agent is idle"
			color = ControlCenterTheme.COLOR_WARN
	elif _mode == ControlCenterConfig.Mode.TRAINING:
		text = "TRAINING - rendering and telemetry disabled"
		color = ControlCenterTheme.COLOR_MUTED
	_draw_text(Vector2(16.0, 22.0), text, color)


## `at` (not `position`) so the parameter never shadows `Control.position`.
func _draw_text(at: Vector2, text: String, color: Color) -> void:
	if _font == null:
		return
	draw_string(
		_font, at + Vector2(1.0, 1.0), text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, _font_size,
		Color(0.0, 0.0, 0.0, 0.7)
	)
	draw_string(_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, _font_size, color)

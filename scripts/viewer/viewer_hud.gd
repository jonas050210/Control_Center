## ViewerHud
##
## On-screen text for the checkpoint viewer: what is playing, how the
## current episode is going, what the policy is pressing, recent results and
## the key bindings. The text itself comes from pure static functions so it
## can be unit-tested without building any Control node (tests/test_viewer.gd).
class_name ViewerHud
extends CanvasLayer

const RESULT_HISTORY: int = 8
const BANNER_SECONDS: float = 2.5
const HELP_LINES: Array = [
	"Space pause   N step   R new episode   M / Shift+M map",
	"+ / - speed   C or 1-4 camera   drag/wheel orbit   H HUD   Esc quit",
]
const AXIS_WORDS: Dictionary = {
	"move": ["back", "-", "forward"],
	"strafe": ["left", "-", "right"],
	"yaw": ["turn L", "-", "turn R"],
	"pitch": ["look down", "-", "look up"],
}

var _status_label: Label
var _help_label: Label
var _banner_label: Label
var _banner_left: float = 0.0


func _ready() -> void:
	layer = 10
	_status_label = _make_label(Vector2(14.0, 12.0), 15)
	_help_label = _make_label(Vector2(14.0, 0.0), 13)
	_help_label.anchor_top = 1.0
	_help_label.anchor_bottom = 1.0
	_help_label.offset_top = -48.0
	_help_label.text = "\n".join(HELP_LINES)
	_banner_label = _make_label(Vector2.ZERO, 26)
	_banner_label.anchor_left = 0.5
	_banner_label.anchor_right = 0.5
	_banner_label.offset_left = -300.0
	_banner_label.offset_right = 300.0
	_banner_label.offset_top = 18.0
	_banner_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_banner_label.visible = false


func _process(delta: float) -> void:
	if _banner_left > 0.0:
		_banner_left -= delta
		if _banner_left <= 0.0 and _banner_label != null:
			_banner_label.visible = false


func set_status(state: Dictionary) -> void:
	if _status_label != null:
		_status_label.text = "\n".join(format_status_lines(state))


func show_banner(text: String, color: Color = Color(1.0, 1.0, 1.0)) -> void:
	if _banner_label == null:
		return
	_banner_label.text = text
	_banner_label.add_theme_color_override("font_color", color)
	_banner_label.visible = true
	_banner_left = BANNER_SECONDS


func toggle() -> void:
	visible = not visible


func _make_label(at: Vector2, font_size: int) -> Label:
	var label := Label.new()
	label.position = at
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", Color(1.0, 1.0, 1.0))
	label.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.9))
	label.add_theme_constant_override("shadow_offset_x", 1)
	label.add_theme_constant_override("shadow_offset_y", 1)
	add_child(label)
	return label


## Human-readable form of one MultiDiscrete action
## [move, strafe, yaw, pitch, shoot, jump] (each axis 0/1/2 = neg/none/pos).
static func describe_action(values: Array) -> String:
	if values.size() < 5:
		return "-"
	var parts: Array = []
	var axes: Array = ["move", "strafe", "yaw", "pitch"]
	for index in range(axes.size()):
		var value: int = clampi(int(values[index]), 0, 2)
		if value != 1:
			parts.append(str(AXIS_WORDS[axes[index]][value]))
	if int(values[4]) > 0:
		parts.append("SHOOT")
	if values.size() > 5 and int(values[5]) > 0:
		parts.append("JUMP")
	if parts.is_empty():
		return "idle"
	return " + ".join(parts)


## One-character-per-episode result strip, newest last: W win, L loss,
## T timeout.
static func format_results(results: Array) -> String:
	var symbols: Array = []
	for result in results.slice(maxi(0, results.size() - RESULT_HISTORY)):
		var outcome: String = str(result)
		if outcome == "win":
			symbols.append("W")
		elif outcome == "loss":
			symbols.append("L")
		else:
			symbols.append("T")
	return " ".join(symbols) if not symbols.is_empty() else "-"


static func format_status_lines(state: Dictionary) -> PackedStringArray:
	var lines := PackedStringArray()
	lines.append("SandboxAI Viewer - %s" % str(state.get("policy", "?")))
	if not bool(state.get("connected", true)):
		lines.append("!! policy disconnected: %s" % str(state.get("error", "")))
	var map_text: String = str(state.get("map", ""))
	(
		lines
		. append(
			(
				"Map %s | level %d | enemies %d | seed %d"
				% [
					map_text if not map_text.is_empty() else "(generated)",
					int(state.get("level", 0)),
					int(state.get("enemies", 0)),
					int(state.get("seed", -1)),
				]
			)
		)
	)
	(
		lines
		. append(
			(
				"Episode %d | step %d | reward %.2f | HP %d | kills %d/%d"
				% [
					int(state.get("episode", 1)),
					int(state.get("step", 0)),
					float(state.get("reward", 0.0)),
					int(round(float(state.get("health", 0.0)))),
					int(state.get("kills", 0)),
					int(state.get("enemies", 0)),
				]
			)
		)
	)
	lines.append("Action: %s" % describe_action(state.get("action", [])))
	var wins: int = int(state.get("wins", 0))
	var played: int = int(state.get("episodes_done", 0))
	lines.append(
		"Results: %s  (%d/%d won)" % [format_results(state.get("results", [])), wins, played]
	)
	var speed_text: String = (
		"PAUSED" if bool(state.get("paused", false)) else ("%sx" % str(state.get("speed", 1.0)))
	)
	lines.append("Speed %s | camera %s" % [speed_text, str(state.get("camera", ""))])
	return lines

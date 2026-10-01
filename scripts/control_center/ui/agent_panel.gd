## ControlCenterAgentPanel
##
## Left column: "which agent am I watching, and what is it doing right
## now?" (Phase 4). Pure read-out of the telemetry snapshot — selected
## agent, enemies, HP, current target, current action, movement/shooting
## state, episode time, reward, kills, deaths, damage and accuracy.
class_name ControlCenterAgentPanel
extends PanelContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const PerceptionMap = preload("res://scripts/control_center/ui/perception_map.gd")

var _rows: Dictionary = {}
var _title: Label
var _action_label: Label
var _enemy_list: Label
var _map: PerceptionMap


func setup() -> void:
	add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 4)
	add_child(box)

	_title = ControlCenterTheme.make_label(
		"Agent", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
	)
	box.add_child(_title)
	var scope_note := ControlCenterTheme.make_label(
		"LOCAL SIMULATION TELEMETRY",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_ACCENT_SECONDARY
	)
	scope_note.tooltip_text = (
		"These values describe the local calibration simulation. "
		+ "They are not a reconstructed TTK Testing player HUD."
	)
	box.add_child(scope_note)

	var grid := ControlCenterTheme.make_grid(2)
	box.add_child(grid)
	for key in [
		"mode",
		"alive",
		"position",
		"direction",
		"orientation",
		"velocity",
		"movement",
		"combat",
		"tactical",
		"health",
		"enemies",
		"target",
		"weapon",
		"episode",
		"time",
		"survival",
		"reward",
		"kills",
		"deaths",
		"damage",
		"accuracy",
	]:
		grid.add_child(
			ControlCenterTheme.make_label(
				key, ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
			)
		)
		var value := ControlCenterTheme.make_value_label("-")
		grid.add_child(value)
		_rows[key] = value

	box.add_child(ControlCenterTheme.make_separator())
	box.add_child(
		ControlCenterTheme.make_label(
			"Current action", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_action_label = ControlCenterTheme.make_value_label("-")
	_action_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(_action_label)

	box.add_child(ControlCenterTheme.make_separator())
	box.add_child(
		ControlCenterTheme.make_label(
			"ENEMIES (world)", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
		)
	)
	_enemy_list = ControlCenterTheme.make_value_label("-")
	_enemy_list.clip_text = false
	box.add_child(_enemy_list)

	box.add_child(ControlCenterTheme.make_separator())
	_map = PerceptionMap.new()
	_map.custom_minimum_size = Vector2(0.0, 150.0)
	box.add_child(_map)


func refresh(snapshot: Dictionary) -> void:
	if snapshot.is_empty():
		_title.text = "AGENT // TELEMETRY DISABLED"
		return
	var status: Dictionary = snapshot["status"]
	var agent: Dictionary = snapshot["agent"]
	var episode: Dictionary = snapshot["episode"]
	var target: Dictionary = snapshot["target"]

	_title.text = (
		"AGENT %d  ·  ENV %d  // LOCAL"
		% [int(snapshot["selected_agent_slot"]), int(snapshot["selected_environment"])]
	)

	_set_row("mode", "%s (%s)" % [str(status["mode_name"]), str(status["policy_source_name"])])
	_set_row("alive", "alive" if bool(agent["alive"]) else "dead")
	_set_row("position", _vector_text(agent["position"] as Vector3))
	_set_row("direction", _vector_text(agent["forward"] as Vector3))
	_set_row(
		"orientation",
		"yaw %.1f° / pitch %.1f°" % [float(agent["yaw_deg"]), float(agent["pitch_deg"])]
	)
	var velocity: Vector3 = agent["velocity"]
	_set_row("velocity", "%s  %.2f m/s" % [_vector_text(velocity), velocity.length()])
	_set_row("movement", "moving" if velocity.length() > 0.05 else "idle")
	_set_row("combat", "engaged" if bool(agent["in_combat"]) else "not engaged")
	_set_row("tactical", "n/a (agent tactical state is not modeled)")
	var health_ratio: float = float(agent["health"]) / maxf(1.0, float(agent["max_health"]))
	_set_row("health", "%.0f / %.0f" % [float(agent["health"]), float(agent["max_health"])])
	(_rows["health"] as Label).add_theme_color_override(
		"font_color", ControlCenterTheme.ratio_color(health_ratio)
	)
	_set_row(
		"enemies", "%d alive / %d" % [int(episode["alive_enemies"]), int(episode["total_enemies"])]
	)
	if bool(target.get("has_target", false)):
		_set_row(
			"target",
			(
				"#%d  %.1f m  %s"
				% [
					int(target["index"]),
					float(target["distance_m"]),
					"in range" if bool(target["in_range"]) else "out of range",
				]
			)
		)
	else:
		_set_row("target", "none")
	var weapon_state: String = "ready"
	if not bool(agent["weapon_ready"]):
		weapon_state = "cooldown %.2fs" % float(agent["weapon_cooldown"])
	_set_row(
		"weapon",
		(
			"%s  %s  %.0fdmg %.0fm x%d"
			% [
				str(agent.get("weapon_label", agent.get("weapon_profile", "weapon"))),
				weapon_state,
				float(agent.get("weapon_damage", 0.0)),
				float(agent.get("weapon_range", 0.0)),
				int(agent.get("weapon_projectile_count", 1)),
			]
		)
	)
	_set_row(
		"episode",
		(
			"#%d  step %d/%d"
			% [int(episode["episode"]), int(episode["step"]), int(episode["max_steps"])]
		)
	)
	_set_row("time", "%.2f s" % float(episode["time_seconds"]))
	_set_row("survival", "%.2f s" % float(episode["survival_time"]))
	_set_row(
		"reward", "%.2f  (last %+.3f)" % [float(episode["reward"]), float(episode["last_reward"])]
	)
	_set_row("kills", str(int(episode["kills"])))
	_set_row("deaths", str(int(episode["deaths"])))
	_set_row(
		"damage",
		(
			"dealt %.0f / taken %.0f"
			% [float(episode["damage_dealt"]), float(episode["damage_received"])]
		)
	)
	_set_row(
		"accuracy",
		(
			"%.0f%%  (%d/%d)"
			% [
				float(episode["accuracy"]) * 100.0,
				int(episode["shots_hit"]),
				int(episode["shots_fired"]),
			]
		)
	)

	if snapshot.has("action"):
		var action: Dictionary = snapshot["action"]
		var parts: PackedStringArray = PackedStringArray()
		for row_value in action["rows"] as Array:
			var row: Dictionary = row_value
			parts.append("%s=%s" % [str(row["name"]), str(row["canonical"])])
		var flags: PackedStringArray = PackedStringArray()
		if bool(action["moving"]):
			flags.append("MOVING")
		if bool(action["turning"]):
			flags.append("TURNING")
		if bool(action["shooting"]):
			flags.append("SHOOTING")
		if flags.is_empty():
			flags.append("IDLE")
		_action_label.text = "%s\n%s" % [" ".join(flags), "  ".join(parts)]

	if snapshot.has("perception"):
		var perception: Dictionary = snapshot["perception"]
		_map.set_perception(perception)
		var lines: PackedStringArray = PackedStringArray()
		for entry_value in perception["real_world"]["enemies"] as Array:
			var entry: Dictionary = entry_value
			var marker: String = "*" if int(entry["index"]) == int(target.get("index", -1)) else " "
			(
				lines
				. append(
					(
						"%s#%d %-6s %5.1fm %+6.0f° hp %3.0f"
						% [
							marker,
							int(entry["index"]),
							str(entry["ai_state"]),
							float(entry["distance_m"]),
							float(entry["bearing_deg"]),
							float(entry["health"]),
						]
					)
				)
			)
		_enemy_list.text = "\n".join(lines)


## Renamed from `_set`: that name is the engine's `Object::_set(StringName,
## Variant) -> bool` virtual. Godot 4.7 validates virtual-method signatures at
## compile time, so a private helper sharing the name failed to compile the
## whole panel (and with it the Control Center scene).
func _set_row(key: String, value: String) -> void:
	var label: Label = _rows.get(key)
	if label != null:
		label.text = value


static func _vector_text(value: Vector3) -> String:
	return "(%.2f, %.2f, %.2f)" % [value.x, value.y, value.z]

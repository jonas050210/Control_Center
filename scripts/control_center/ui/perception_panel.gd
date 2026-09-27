## ControlCenterPerceptionPanel
##
## "What does the AI see?" tab (Phase 5).
##
## Renders the two clearly separated sections built by PerceptionModel:
## REAL WORLD (ground truth, debug only) and AI PERCEPTION (decoded from
## the observation vector, i.e. everything the policy knows), plus the list
## of enemies that exist but are absent from the observation, plus the
## perception features the simulation does not implement yet.
##
## It performs no perception logic of its own.
class_name ControlCenterPerceptionPanel
extends VBoxContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const PerceptionMap = preload("res://scripts/control_center/ui/perception_map.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")

var session
var _map: PerceptionMap
var _summary: Label
var _hidden_label: Label
var _capability_label: Label
var _overlay_toggle: Button


func setup(p_session) -> void:
	session = p_session
	add_theme_constant_override("separation", 6)

	var header := ControlCenterTheme.make_row()
	add_child(header)
	header.add_child(
		ControlCenterTheme.make_label(
			"REAL WORLD vs AI PERCEPTION",
			ControlCenterTheme.FONT_SIZE_TITLE,
			ControlCenterTheme.COLOR_TITLE
		)
	)
	_overlay_toggle = ControlCenterTheme.make_toggle(
		"3D overlay",
		session.config.show_perception_overlay,
		"Draw sight lines, target marker and weapon range in the 3D view."
	)
	_overlay_toggle.toggled.connect(_on_overlay_toggled)
	header.add_child(_overlay_toggle)

	_map = PerceptionMap.new()
	add_child(_map)

	_summary = ControlCenterTheme.make_value_label("")
	_summary.clip_text = false
	add_child(_summary)

	add_child(ControlCenterTheme.make_separator())
	_hidden_label = ControlCenterTheme.make_value_label("")
	_hidden_label.clip_text = false
	_hidden_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_hidden_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"PERCEPTION FEATURES",
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)
	_capability_label = ControlCenterTheme.make_value_label("")
	_capability_label.clip_text = false
	_capability_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_capability_label)


func refresh(snapshot: Dictionary) -> void:
	if snapshot.is_empty() or not snapshot.has("perception"):
		_summary.text = "Perception view is disabled in TRAINING mode."
		_map.set_perception({})
		return
	var perception: Dictionary = snapshot["perception"]
	_map.set_perception(perception)
	_summary.text = "\n".join(PerceptionModel.format_lines(perception))

	var hidden: Array = perception["hidden_from_ai"]
	if hidden.is_empty():
		_hidden_label.add_theme_color_override("font_color", ControlCenterTheme.COLOR_MUTED)
		_hidden_label.text = (
			"All alive enemies are represented in the observation vector "
			+ "(contract tracks the %d nearest)." % int(perception["max_tracked_enemies"])
		)
	else:
		_hidden_label.add_theme_color_override("font_color", ControlCenterTheme.COLOR_HIDDEN)
		var names: PackedStringArray = PackedStringArray()
		for entry_value in hidden:
			var entry: Dictionary = entry_value
			names.append("#%d (%.1f m)" % [int(entry["index"]), float(entry["distance_m"])])
		_hidden_label.text = (
			"HIDDEN FROM AI: %s — alive in the world, absent from the observation (%s)."
			% [", ".join(names), str((hidden[0] as Dictionary)["detail"])]
		)

	var capability_lines: PackedStringArray = PackedStringArray()
	var capabilities: Dictionary = perception["capabilities"]
	for feature_id in capabilities.keys():
		var capability: Dictionary = capabilities[feature_id]
		if bool(capability["available"]):
			capability_lines.append("[available] %s" % str(capability["label"]))
		else:
			capability_lines.append(
				"[unavailable] %s — %s" % [str(capability["label"]), str(capability["note"])]
			)
	_capability_label.text = "\n".join(capability_lines)


func _on_overlay_toggled(pressed: bool) -> void:
	session.config.show_perception_overlay = pressed

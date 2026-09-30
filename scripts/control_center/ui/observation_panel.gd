## ControlCenterObservationPanel
##
## Observation Inspector tab (Phase 6): the exact vector the policy
## receives, field by field, plus the action it produced and the reward
## components that were credited.
##
## Every label comes from the contract itself via ObservationInspector
## (Observation.FIELD_SPEC / RLAdapter.action_space_info() /
## EpisodeState.get_reward_breakdown()), so the panel follows a contract
## change automatically instead of needing a parallel hand-written list.
class_name ControlCenterObservationPanel
extends VBoxContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const Observation = preload("res://scripts/core/observation.gd")

var _grid: GridContainer
var _value_labels: Array = []  # Array[Label], one per observation index
var _action_label: Label
var _reward_label: Label
var _header: Label


func setup() -> void:
	add_theme_constant_override("separation", 6)
	_header = ControlCenterTheme.make_label(
		"Observation vector", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
	)
	add_child(_header)
	add_child(
		ControlCenterTheme.make_label(
			(
				"source of truth: Observation.FIELD_SPEC (%d floats)"
				% Observation.FIELD_COUNT
			),
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)

	var scroll := ControlCenterTheme.make_scroll(220.0)
	add_child(scroll)
	_grid = ControlCenterTheme.make_grid(3)
	_grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_grid)

	var names: PackedStringArray = Observation.field_names()
	for index in range(names.size()):
		_grid.add_child(
			ControlCenterTheme.make_label(
				"%2d" % index, ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
			)
		)
		_grid.add_child(
			ControlCenterTheme.make_label(names[index], ControlCenterTheme.FONT_SIZE_SMALL)
		)
		var value := ControlCenterTheme.make_value_label("0.000")
		_grid.add_child(value)
		_value_labels.append(value)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"ACTION (MultiDiscrete)",
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)
	_action_label = ControlCenterTheme.make_value_label("-")
	_action_label.clip_text = false
	add_child(_action_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"REWARD COMPONENTS (this episode)",
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)
	_reward_label = ControlCenterTheme.make_value_label("-")
	_reward_label.clip_text = false
	add_child(_reward_label)


func refresh(snapshot: Dictionary) -> void:
	if snapshot.is_empty() or not snapshot.has("observation"):
		_header.text = "OBSERVATION VECTOR (disabled in TRAINING mode)"
		return
	_header.text = "Observation vector"
	var rows: Array = snapshot["observation"]["rows"]
	for row_value in rows:
		var row: Dictionary = row_value
		var index: int = int(row["index"])
		if index >= _value_labels.size():
			continue
		var label: Label = _value_labels[index]
		var value: float = float(row["value"])
		label.text = String.num(value, 3)
		label.add_theme_color_override(
			"font_color",
			(
				ControlCenterTheme.COLOR_MUTED
				if is_zero_approx(value)
				else ControlCenterTheme.COLOR_TEXT
			)
		)

	if snapshot.has("action"):
		var action_lines: PackedStringArray = PackedStringArray()
		for row_value in (snapshot["action"]["rows"] as Array):
			var row: Dictionary = row_value
			action_lines.append(
				(
					"%-16s md=%d  value=%s"
					% [str(row["name"]), int(row["multidiscrete"]), str(row["canonical"])]
				)
			)
		for row_value in (snapshot["action"]["continuous_rows"] as Array):
			var row: Dictionary = row_value
			action_lines.append(
				"%-16s %s  (logged, not in the PPO action space)"
				% [str(row["name"]), String.num(float(row["value"]), 2)]
			)
		_action_label.text = "\n".join(action_lines)

	if snapshot.has("reward"):
		var reward_lines: PackedStringArray = PackedStringArray()
		for row_value in (snapshot["reward"]["rows"] as Array):
			var row: Dictionary = row_value
			reward_lines.append(
				"%-22s %s" % [str(row["name"]), String.num(float(row["value"]), 3)]
			)
		_reward_label.text = "\n".join(reward_lines)

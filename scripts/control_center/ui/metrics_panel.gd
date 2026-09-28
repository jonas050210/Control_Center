## ControlCenterMetricsPanel
##
## METRICS tab. Shows the eight diagnostic skill categories for the
## selected environment, and — strictly separated below a red heading —
## the ground-truth values the agent cannot perceive.
##
## The separation is the feature. Everything above the divider is
## information the policy could have derived from its own observation;
## everything below is simulator state shown for debugging only. Nothing in
## this panel writes to the simulation or to an observation, and the
## metrics it renders are diagnostic: none of them is a reward term.
class_name ControlCenterMetricsPanel
extends VBoxContainer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const SkillMetrics = preload("res://scripts/metrics/skill_metrics.gd")

var session
var _categories_label: Label
var _ground_truth_label: Label
var _note_label: Label


func setup(p_session = null) -> void:
	session = p_session
	add_theme_constant_override("separation", 6)

	add_child(
		ControlCenterTheme.make_label(
			"RESEARCH METRICS (AI-AVAILABLE)",
			ControlCenterTheme.FONT_SIZE_TITLE,
			ControlCenterTheme.COLOR_AI
		)
	)
	_note_label = ControlCenterTheme.make_label(
		"Diagnostic only - none of these values is used as a reward.",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	add_child(_note_label)

	_categories_label = ControlCenterTheme.make_value_label("-")
	_categories_label.clip_text = false
	add_child(_categories_label)

	add_child(ControlCenterTheme.make_separator())
	add_child(
		ControlCenterTheme.make_label(
			"GROUND TRUTH (DEBUG ONLY - NOT VISIBLE TO THE POLICY)",
			ControlCenterTheme.FONT_SIZE_TITLE,
			ControlCenterTheme.COLOR_HIDDEN
		)
	)
	_ground_truth_label = ControlCenterTheme.make_value_label("-")
	_ground_truth_label.clip_text = false
	_ground_truth_label.add_theme_color_override("font_color", ControlCenterTheme.COLOR_HIDDEN)
	add_child(_ground_truth_label)


func refresh(snapshot: Dictionary) -> void:
	if snapshot.is_empty():
		_categories_label.text = "telemetry disabled"
		_ground_truth_label.text = "-"
		return
	_categories_label.text = SkillMetrics.format_categories(SkillMetrics.ai_available(snapshot))
	var truth: Dictionary = SkillMetrics.ground_truth(snapshot)
	var parts: Array = []
	var keys: Array = truth.keys()
	keys.sort()
	for key in keys:
		parts.append("%s=%s" % [str(key), str(truth[key])])
	_ground_truth_label.text = " ".join(parts) if not parts.is_empty() else "-"


## Panel content as data, for tests.
func describe(snapshot: Dictionary) -> Dictionary:
	return {
		"ai_available": SkillMetrics.ai_available(snapshot),
		"ground_truth": SkillMetrics.ground_truth(snapshot),
		"rows": SkillMetrics.rows(SkillMetrics.ai_available(snapshot)),
	}

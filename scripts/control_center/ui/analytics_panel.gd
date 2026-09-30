## ANALYTICS page: current backend metrics per agent plus a reward trend
## built only from status samples the backend actually published. Kept
## deliberately lightweight — the headless page is the live monitor; this
## page is a comparison table.
class_name ControlCenterAnalyticsPanel
extends PanelContainer

## Bounded per-agent reward series (samples, not wall time).
const MAX_TREND_SAMPLES: int = 240

const ControlCenterMetricSparkline = preload(
	"res://scripts/control_center/ui/metric_sparkline.gd"
)
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")

var session

var _list: VBoxContainer
var _empty: Label
var _rows: Dictionary = {}  # agent_id -> {panel, values_label, sparkline}
var _trends: Dictionary = {}  # agent_id -> {"samples": Array, "last_marker": Variant}


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var scroll := ControlCenterTheme.make_scroll()
	add_child(scroll)
	var root := VBoxContainer.new()
	root.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_theme_constant_override("separation", 8)
	scroll.add_child(root)
	root.add_child(
		ControlCenterTheme.make_label(
			"Analytics", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	root.add_child(
		ControlCenterTheme.make_label(
			"Backend-published metrics per agent. Reward trend uses only "
			+ "real progress samples from the current run.",
			ControlCenterTheme.FONT_SIZE_SMALL,
			ControlCenterTheme.COLOR_MUTED
		)
	)
	_list = VBoxContainer.new()
	_list.add_theme_constant_override("separation", 8)
	_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(_list)
	_empty = ControlCenterTheme.make_empty_label(
		"No analytics yet. Start an agent to collect live metrics."
	)
	root.add_child(_empty)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	if session == null or session.agent_manager == null:
		return
	var manager = session.agent_manager
	var seen: Array = []
	var count: int = 0
	for id_value in manager.agent_ids():
		var agent_id: int = int(id_value)
		var snapshot: Dictionary = manager.agent_snapshot(agent_id)
		var state_id: int = int(snapshot.get("state_id", TrainingRunController.State.IDLE))
		if state_id == TrainingRunController.State.IDLE:
			continue
		seen.append(agent_id)
		count += 1
		_record_trend(agent_id, snapshot)
		var row: Dictionary = _ensure_row(agent_id)
		(row["title"] as Label).text = "%s · %s · %s" % [
			str(snapshot.get("label", "Agent %d" % agent_id)),
			str(snapshot.get("algorithm", "")),
			str(snapshot.get("state", "Idle")),
		]
		(row["values"] as Label).text = _values_text(snapshot)
		var trend: Dictionary = _trends.get(agent_id, {})
		var samples: Array = trend.get("samples", [])
		(row["sparkline"] as ControlCenterMetricSparkline).set_samples(samples)
		(row["trend_label"] as Label).text = (
			"reward trend (%d samples)" % samples.size()
			if samples.size() >= 2
			else "reward trend: waiting for progress samples"
		)
	_prune_rows(seen)
	_empty.visible = count == 0


static func _values_text(snapshot: Dictionary) -> String:
	var parts: Array = [
		"reward %s" % ControlCenterTheme.optional_metric(snapshot, "mean_episode_reward", 3),
		"episodes %s" % ControlCenterTheme.optional_metric(snapshot, "episodes", 0),
		"kills %s" % ControlCenterTheme.optional_metric(snapshot, "mean_kills", 2),
		"deaths %s" % ControlCenterTheme.optional_metric(snapshot, "mean_deaths", 2),
		"accuracy %s" % ControlCenterTheme.optional_metric(snapshot, "mean_accuracy", 1, 100.0, "%"),
		"throughput %s" % ControlCenterTheme.optional_metric(
			snapshot, "steps_per_second", 1, 1.0, " sps"
		),
	]
	return " · ".join(parts)


## Appends a reward sample only when the backend published a NEW progress
## marker together with a reward value.
func _record_trend(agent_id: int, snapshot: Dictionary) -> void:
	if not snapshot.has("mean_episode_reward"):
		return
	var marker = snapshot.get("timesteps", snapshot.get("epoch"))
	if marker == null:
		return
	if not _trends.has(agent_id):
		_trends[agent_id] = {"samples": [], "last_marker": null}
	var trend: Dictionary = _trends[agent_id]
	if trend["last_marker"] != null and int(marker) <= int(trend["last_marker"]):
		return
	trend["last_marker"] = marker
	var samples: Array = trend["samples"]
	samples.append(float(snapshot["mean_episode_reward"]))
	while samples.size() > MAX_TREND_SAMPLES:
		samples.pop_front()


func trend_sample_count(agent_id: int) -> int:
	var trend: Dictionary = _trends.get(agent_id, {})
	return (trend.get("samples", []) as Array).size()


func _ensure_row(agent_id: int) -> Dictionary:
	if _rows.has(agent_id):
		return _rows[agent_id]
	var panel := ControlCenterTheme.make_panel()
	var box := ControlCenterTheme.content_container(panel)
	var title := ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_NORMAL, ControlCenterTheme.COLOR_TEXT
	)
	box.add_child(title)
	var values := ControlCenterTheme.make_value_label("")
	values.clip_text = false
	box.add_child(values)
	var trend_label := ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	box.add_child(trend_label)
	var sparkline := ControlCenterMetricSparkline.new()
	box.add_child(sparkline)
	_list.add_child(panel)
	var row: Dictionary = {
		"panel": panel,
		"title": title,
		"values": values,
		"trend_label": trend_label,
		"sparkline": sparkline,
	}
	_rows[agent_id] = row
	return row


func _prune_rows(seen: Array) -> void:
	for agent_id in _rows.keys():
		if seen.has(agent_id):
			continue
		var row: Dictionary = _rows[agent_id]
		var panel: Control = row["panel"]
		if panel.get_parent() != null:
			panel.get_parent().remove_child(panel)
		panel.queue_free()
		_rows.erase(agent_id)
		_trends.erase(agent_id)

## Headless/managed-training dashboard. Every value comes from the Python
## backend status channel; unavailable hardware counters stay "n/a".
class_name ControlCenterTrainingDashboardPanel
extends PanelContainer

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var session
var _title: Label
var _progress: ProgressBar
var _progress_text: Label
var _metrics: Label
var _resources: Label
var _logs: RichTextLabel
var _log_panel: PanelContainer


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)
	_title = ControlCenterTheme.make_label(
		"TRAINING PROGRESS", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
	)
	root.add_child(_title)
	_progress = ProgressBar.new()
	_progress.min_value = 0.0
	_progress.max_value = 100.0
	_progress.show_percentage = false
	root.add_child(_progress)
	_progress_text = ControlCenterTheme.make_value_label("Idle")
	_progress_text.clip_text = false
	root.add_child(_progress_text)

	var columns := HBoxContainer.new()
	columns.add_theme_constant_override("separation", 12)
	columns.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(columns)
	var metrics_panel := ControlCenterTheme.make_panel("AGENT / RL METRICS")
	metrics_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(metrics_panel)
	_metrics = ControlCenterTheme.make_value_label("n/a")
	_metrics.clip_text = false
	ControlCenterTheme.content_container(metrics_panel).add_child(_metrics)

	var resources_panel := ControlCenterTheme.make_panel("DEVICE / ENVIRONMENT")
	resources_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(resources_panel)
	_resources = ControlCenterTheme.make_value_label("n/a")
	_resources.clip_text = false
	ControlCenterTheme.content_container(resources_panel).add_child(_resources)

	_log_panel = ControlCenterTheme.make_panel("TRAINING LOG")
	_log_panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(_log_panel)
	_logs = RichTextLabel.new()
	_logs.bbcode_enabled = true
	_logs.selection_enabled = true
	_logs.scroll_following = true
	_logs.custom_minimum_size = Vector2(0.0, 160.0)
	_logs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_logs.add_theme_font_size_override("normal_font_size", ControlCenterTheme.FONT_SIZE_SMALL)
	ControlCenterTheme.content_container(_log_panel).add_child(_logs)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	var status: Dictionary = session.training_run.snapshot()
	var state_name: String = str(status.get("state", "Idle"))
	_title.text = "%s · %s" % [
		str(
			status.get(
				"training_type",
				ControlCenterConfig.training_type_name(session.config.training_type)
			)
		),
		state_name.to_upper(),
	]
	var progress: float = clampf(float(status.get("progress", 0.0)), 0.0, 1.0)
	_progress.value = progress * 100.0
	_progress_text.text = _progress_description(status, state_name, progress)
	_metrics.text = _metric_description(status)
	_resources.text = _resource_description(status)
	_log_panel.visible = session.config.is_tile_visible("logs")
	_render_events(status.get("recent_events", []))


func _progress_description(status: Dictionary, state_name: String, progress: float) -> String:
	var lines: Array = ["state              %s" % state_name]
	if status.has("timesteps") or status.has("total_training_steps"):
		lines.append(
			"step               %d / %d"
			% [int(status.get("timesteps", 0)), int(status.get("total_training_steps", 0))]
		)
	elif status.has("epoch") or status.has("total_epochs"):
		lines.append(
			"epoch              %d / %d"
			% [int(status.get("epoch", 0)), int(status.get("total_epochs", 0))]
		)
	lines.append("progress           %.1f%%" % (progress * 100.0))
	if status.has("episodes"):
		lines.append("episodes           %d" % int(status["episodes"]))
	if status.has("eta_seconds") and status["eta_seconds"] != null:
		lines.append("ETA                %s" % _duration(float(status["eta_seconds"])))
	else:
		lines.append("ETA                n/a")
	return "\n".join(lines)


func _metric_description(status: Dictionary) -> String:
	var rows: Array = []
	_append_metric(rows, status, "mean_episode_reward", "episode reward", 3)
	_append_metric(rows, status, "mean_kills", "kills", 3)
	_append_metric(rows, status, "mean_deaths", "deaths", 3)
	_append_metric(rows, status, "mean_shots_fired", "shots", 2)
	_append_metric(rows, status, "mean_shots_hit", "hits", 2)
	_append_metric(rows, status, "mean_near_miss_shots", "near misses", 2)
	_append_metric(rows, status, "mean_useless_shots", "useless shots", 2)
	_append_metric(rows, status, "mean_cooldown_shots", "cooldown pulls", 2)
	_append_metric(rows, status, "mean_accuracy", "accuracy", 3, 100.0, "%")
	_append_metric(rows, status, "mean_damage_dealt", "damage dealt", 2)
	_append_metric(rows, status, "mean_damage_received", "damage received", 2)
	_append_metric(rows, status, "mean_survival_time", "survival seconds", 2)
	_append_metric(rows, status, "win_rate", "win rate", 3, 100.0, "%")
	_append_metric(rows, status, "loss_rate", "loss rate", 3, 100.0, "%")
	_append_metric(rows, status, "train_loss", "train loss", 5)
	_append_metric(rows, status, "validation_loss", "validation loss", 5)
	_append_metric(rows, status, "component_accuracy", "component accuracy", 3, 100.0, "%")
	_append_metric(rows, status, "exact_accuracy", "exact accuracy", 3, 100.0, "%")
	if rows.is_empty():
		return "No backend metric sample yet."
	return "\n".join(rows)


func _resource_description(status: Dictionary) -> String:
	var device_name: String = str(
		status.get("gpu_name", status.get("cuda_device", status.get("device", "n/a")))
	)
	var lines: Array = [
		"device             %s" % device_name,
		"environments       %s" % str(status.get("environment_count", "n/a")),
		"steps/sec          %s" % _optional_number(status, "steps_per_second", 1),
		"CPU utilization    %s" % _optional_suffix(status, "cpu_percent", 1, "%"),
		"process memory     %s" % _optional_suffix(status, "process_rss_mb", 1, " MB"),
		"GPU utilization    %s" % _optional_suffix(status, "gpu_utilization_percent", 0, "%"),
		(
			"GPU VRAM           %s / %s"
			% [
				_optional_suffix(status, "gpu_vram_used_mb", 0, " MB"),
				_optional_suffix(status, "gpu_vram_total_mb", 0, " MB"),
			]
		),
		"GPU temperature    %s" % _optional_suffix(status, "gpu_temperature_c", 0, " °C"),
		"CUDA allocated     %s" % _optional_suffix(status, "cuda_allocated_mb", 1, " MB"),
		"CUDA reserved      %s" % _optional_suffix(status, "cuda_reserved_mb", 1, " MB"),
	]
	return "\n".join(lines)


func _render_events(events_value) -> void:
	_logs.clear()
	if not (events_value is Array) or events_value.is_empty():
		_logs.add_text("No training events yet.")
		return
	for event_value in events_value:
		var entry: Dictionary = event_value
		var category: String = str(entry.get("category", "system")).to_upper()
		var color: Color = (
			ControlCenterTheme.COLOR_BAD
			if category == "ERROR"
			else ControlCenterTheme.COLOR_MUTED
		)
		_logs.push_color(color)
		_logs.add_text("%-8s " % category)
		_logs.pop()
		_logs.add_text(str(entry.get("message", "")))
		_logs.newline()


static func _append_metric(
	rows: Array,
	status: Dictionary,
	key: String,
	label: String,
	decimals: int,
	scale: float = 1.0,
	suffix: String = ""
) -> void:
	if not status.has(key):
		return
	rows.append(
		"%-20s %s%s"
		% [label, String.num(float(status[key]) * scale, decimals), suffix]
	)


static func _optional_number(status: Dictionary, key: String, decimals: int) -> String:
	return String.num(float(status[key]), decimals) if status.has(key) else "n/a"


static func _optional_suffix(
	status: Dictionary, key: String, decimals: int, suffix: String
) -> String:
	return _optional_number(status, key, decimals) + suffix if status.has(key) else "n/a"


static func _duration(seconds: float) -> String:
	var total: int = maxi(0, int(seconds))
	return "%02d:%02d:%02d" % [total / 3600, (total % 3600) / 60, total % 60]

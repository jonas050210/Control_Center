## PC status strip: GPU / CPU / RAM cards fed exclusively by the measured
## ControlCenterSystemMonitor snapshot. Metrics the machine cannot provide
## are rendered as "N/A" — this panel never estimates anything.
class_name ControlCenterSystemStatusPanel
extends PanelContainer

const ControlCenterSystemMonitor = preload("res://scripts/control_center/system_monitor.gd")
const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")

var session
var _gpu_name: Label
var _gpu_values: Label
var _cpu_values: Label
var _ram_values: Label
var _sampled_label: Label


func setup(p_session) -> void:
	session = p_session
	add_theme_stylebox_override("panel", ControlCenterTheme.panel_style())
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 6)
	add_child(root)

	var header := ControlCenterTheme.make_row()
	root.add_child(header)
	header.add_child(
		ControlCenterTheme.make_label(
			"SYSTEM", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_sampled_label = ControlCenterTheme.make_label(
		"no sample yet", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_sampled_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sampled_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	header.add_child(_sampled_label)

	var columns := HBoxContainer.new()
	columns.add_theme_constant_override("separation", 10)
	root.add_child(columns)

	var gpu_panel := ControlCenterTheme.make_panel("GPU")
	gpu_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(gpu_panel)
	_gpu_name = ControlCenterTheme.make_value_label("N/A")
	ControlCenterTheme.content_container(gpu_panel).add_child(_gpu_name)
	_gpu_values = ControlCenterTheme.make_value_label("")
	_gpu_values.clip_text = false
	ControlCenterTheme.content_container(gpu_panel).add_child(_gpu_values)

	var cpu_panel := ControlCenterTheme.make_panel("CPU")
	cpu_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(cpu_panel)
	_cpu_values = ControlCenterTheme.make_value_label("")
	_cpu_values.clip_text = false
	ControlCenterTheme.content_container(cpu_panel).add_child(_cpu_values)

	var ram_panel := ControlCenterTheme.make_panel("RAM")
	ram_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(ram_panel)
	_ram_values = ControlCenterTheme.make_value_label("")
	_ram_values.clip_text = false
	ControlCenterTheme.content_container(ram_panel).add_child(_ram_values)
	refresh()


func refresh(_snapshot: Dictionary = {}) -> void:
	if session == null or session.system_monitor == null:
		return
	render(session.system_monitor.snapshot())


## Pure rendering of a monitor snapshot; public so tests can feed shaped
## payloads without a live monitor.
func render(sample: Dictionary) -> void:
	var gpu: Dictionary = sample.get("gpu", {})
	var cpu: Dictionary = sample.get("cpu", {})
	var ram: Dictionary = sample.get("ram", {})
	var gpu_name: String = str(gpu.get("name", ""))
	_gpu_name.text = gpu_name if not gpu_name.is_empty() else "N/A"
	_gpu_values.text = "\n".join(
		[
			"utilization   %s" % _metric(gpu, "utilization_percent", 0, "%"),
			(
				"VRAM          %s / %s"
				% [_metric(gpu, "vram_used_mb", 0, " MB"), _metric(gpu, "vram_total_mb", 0, " MB")]
			),
			"temperature   %s" % _metric(gpu, "temperature_c", 0, " °C"),
		]
	)
	_cpu_values.text = "\n".join(
		[
			"utilization   %s" % _metric(cpu, "utilization_percent", 0, "%"),
			"temperature   %s" % _metric(cpu, "temperature_c", 0, " °C"),
		]
	)
	_ram_values.text = "\n".join(
		[
			(
				"used          %s / %s"
				% [_metric(ram, "used_mb", 0, " MB"), _metric(ram, "total_mb", 0, " MB")]
			),
			"utilization   %s" % _metric(ram, "utilization_percent", 0, "%"),
		]
	)
	var sampled = sample.get("sampled_at")
	if sampled == null:
		_sampled_label.text = "no sample yet"
	else:
		_sampled_label.text = "sampled %s" % ControlCenterTheme.format_clock(float(sampled))


## Currently displayed strings, for focused tests.
func rendered_texts() -> Dictionary:
	return {
		"gpu_name": _gpu_name.text,
		"gpu": _gpu_values.text,
		"cpu": _cpu_values.text,
		"ram": _ram_values.text,
		"sampled": _sampled_label.text,
	}


static func _metric(section: Dictionary, key: String, decimals: int, suffix: String) -> String:
	return ControlCenterSystemMonitor.format_metric(section.get(key), decimals, suffix)

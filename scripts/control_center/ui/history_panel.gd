## HISTORY page: previous managed training runs, read from the run
## directories the trainers already persist. Absent fields render as
## "n/a"; nothing here recomputes or invents historical values.
class_name ControlCenterHistoryPanel
extends PanelContainer

const ControlCenterTheme = preload("res://scripts/control_center/ui/ui_theme.gd")
const TrainingRunHistory = preload("res://scripts/control_center/training_run_history.gd")

const COLUMN_TITLES: Array = [
	"run id", "algo", "status", "duration", "steps", "episodes", "reward", "accuracy"
]

var _tree: Tree
var _detail: Label
var _count_label: Label
var _rows: Array = []


func setup(_p_session = null) -> void:
	add_theme_stylebox_override(
		"panel", ControlCenterTheme.panel_style(ControlCenterTheme.COLOR_BACKGROUND_SOLID)
	)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	var header := ControlCenterTheme.make_row()
	root.add_child(header)
	header.add_child(
		ControlCenterTheme.make_label(
			"Training history", ControlCenterTheme.FONT_SIZE_TITLE, ControlCenterTheme.COLOR_TITLE
		)
	)
	_count_label = ControlCenterTheme.make_label(
		"", ControlCenterTheme.FONT_SIZE_SMALL, ControlCenterTheme.COLOR_MUTED
	)
	_count_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(_count_label)
	var reload := ControlCenterTheme.make_button("Reload", "Re-scan the persisted run directories")
	reload.pressed.connect(reload_runs)
	header.add_child(reload)

	_tree = Tree.new()
	_tree.columns = COLUMN_TITLES.size()
	_tree.column_titles_visible = true
	_tree.hide_root = true
	_tree.select_mode = Tree.SELECT_ROW
	_tree.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_tree.custom_minimum_size = Vector2(0.0, 200.0)
	for index in range(COLUMN_TITLES.size()):
		_tree.set_column_title(index, str(COLUMN_TITLES[index]))
		_tree.set_column_expand(index, index == 0)
	_tree.item_selected.connect(_on_row_selected)
	root.add_child(_tree)

	_detail = ControlCenterTheme.make_label(
		"Select a run for details.",
		ControlCenterTheme.FONT_SIZE_SMALL,
		ControlCenterTheme.COLOR_MUTED
	)
	_detail.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(_detail)
	reload_runs()


## History is disk state, not live telemetry: it reloads on demand and on
## page entry rather than on the 10 Hz refresh tick.
func reload_runs() -> void:
	render_rows(TrainingRunHistory.load_runs())


func refresh(_snapshot: Dictionary = {}) -> void:
	# Intentionally does NOT rescan the disk every UI tick.
	pass


## Renders prepared history rows; public so tests can feed fixed data.
func render_rows(rows: Array) -> void:
	_rows = rows
	_tree.clear()
	var tree_root: TreeItem = _tree.create_item()
	for row_value in rows:
		var row: Dictionary = row_value
		var item: TreeItem = _tree.create_item(tree_root)
		item.set_text(0, str(row.get("run_id", "")))
		item.set_text(1, str(row.get("algorithm", "unknown")))
		var state: String = str(row.get("state", "Unknown"))
		item.set_text(2, state)
		item.set_custom_color(2, ControlCenterTheme.state_color(state))
		item.set_text(3, _duration_text(row))
		item.set_text(4, _steps_text(row))
		item.set_text(5, ControlCenterTheme.optional_metric(row, "episodes", 0))
		item.set_text(6, ControlCenterTheme.optional_metric(row, "mean_episode_reward", 3))
		item.set_text(7, _accuracy_text(row))
	_count_label.text = "%d persisted runs" % rows.size()


func row_count() -> int:
	return _rows.size()


static func _duration_text(row: Dictionary) -> String:
	if not row.has("duration_seconds"):
		return "n/a"
	var text: String = ControlCenterTheme.format_duration(row["duration_seconds"])
	if not bool(row.get("duration_final", false)):
		text += " (last update)"
	return text


static func _steps_text(row: Dictionary) -> String:
	if row.has("timesteps"):
		return String.num(float(row["timesteps"]), 0)
	if row.has("epoch"):
		return "epoch %d" % int(row["epoch"])
	return "n/a"


static func _accuracy_text(row: Dictionary) -> String:
	if row.has("mean_accuracy"):
		return ControlCenterTheme.optional_metric(row, "mean_accuracy", 1, 100.0, "%")
	if row.has("exact_accuracy"):
		return ControlCenterTheme.optional_metric(row, "exact_accuracy", 1, 100.0, "%")
	return "n/a"


func _on_row_selected() -> void:
	var selected: TreeItem = _tree.get_selected()
	if selected == null:
		return
	var index: int = selected.get_index()
	if index < 0 or index >= _rows.size():
		return
	var row: Dictionary = _rows[index]
	var lines: Array = []
	lines.append("run        %s" % str(row.get("run_id", "")))
	if row.has("started_unix"):
		lines.append("started    %s" % ControlCenterTheme.format_clock(float(row["started_unix"])))
	if row.has("current_checkpoint"):
		lines.append("checkpoint %s" % str(row["current_checkpoint"]))
	if row.has("device"):
		lines.append("device     %s" % str(row["device"]))
	if row.has("stopped"):
		lines.append("stopped by operator: %s" % str(row["stopped"]))
	if row.has("error"):
		lines.append("error      %s" % str(row["error"]))
	if lines.size() == 1:
		lines.append("no further published detail")
	_detail.text = "\n".join(lines)

## Graphical human demonstration entry point.
## Run through the Python CLI or directly with Godot:
## godot --path . --script res://scripts/recording/record_demo.gd -- \
##   --output training/datasets/demo.jsonl
extends SceneTree

var main_scene: Node
var recorder: DemonstrationRecorder
var output_path: String = "training/datasets/human_demo.jsonl"
var duration_seconds: float = 0.0
var elapsed: float = 0.0
var configured: bool = false
var finished: bool = false


func _initialize() -> void:
	var options := _parse_args(OS.get_cmdline_user_args())
	output_path = str(options.get("output", output_path))
	duration_seconds = float(options.get("duration", 0.0))
	main_scene = load("res://scenes/main.tscn").instantiate()
	main_scene.set("environment_count", 1)
	main_scene.set("enemy_count_per_environment", int(options.get("enemy-count", 1)))
	main_scene.set("human_controls_environment_zero", true)
	root.add_child(main_scene)
	call_deferred("_attach_recorder")


func _attach_recorder() -> void:
	var manager: SimulationManager = main_scene.get_node("SimulationManager")
	recorder = DemonstrationRecorder.new()
	(
		recorder
		. start_recording(
			{
				"source": "HumanController",
				"environment_count": 1,
				"curriculum_level": manager.curriculum_level,
			}
		)
	)
	manager.attach_recorder(0, recorder)
	configured = true
	print(
		(
			"Recording demonstrations to %s. Press Escape to release mouse; "
			+ "close the window to save." % output_path
		)
	)


func _process(delta: float) -> void:
	if not configured:
		return
	elapsed += delta
	if duration_seconds > 0.0 and elapsed >= duration_seconds:
		_finish()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_finish()


func _finish() -> void:
	if finished:
		return
	finished = true
	if recorder == null:
		quit(0)
		return
	recorder.stop_recording()
	recorder.save_dataset(output_path)
	print("Saved %d demonstration transitions." % recorder.get_transition_count())
	quit(0)


func _parse_args(args: PackedStringArray) -> Dictionary:
	var parsed: Dictionary = {}
	var index: int = 0
	while index < args.size():
		var arg: String = args[index]
		if (
			arg.begins_with("--")
			and index + 1 < args.size()
			and not args[index + 1].begins_with("--")
		):
			parsed[arg.trim_prefix("--")] = args[index + 1]
			index += 2
		else:
			index += 1
	return parsed

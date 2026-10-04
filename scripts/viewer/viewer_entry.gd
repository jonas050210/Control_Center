## Standalone entry point of the 3D checkpoint viewer.
## Normally launched by `python start.py view` (python/sandboxai/viewer.py),
## which serves a checkpoint on a local port and passes --policy-port.
## Without --policy-port the built-in heuristic plays, so a map can also be
## inspected directly with Godot:
##   godot --path . --script res://scripts/viewer/viewer_entry.gd -- --map compound
extends SceneTree


func _initialize() -> void:
	var packed: PackedScene = load("res://scenes/viewer.tscn") as PackedScene
	if packed == null:
		printerr("[viewer] could not load res://scenes/viewer.tscn")
		quit(2)
		return
	root.add_child(packed.instantiate())

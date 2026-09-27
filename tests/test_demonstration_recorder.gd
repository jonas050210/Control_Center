## Tests the compact JSONL demonstration recorder without requiring a visual
## scene or human input device.
class_name TestDemonstrationRecorder
extends RefCounted


func test_recorder_saves_and_loads_transition_contract() -> SandboxTest:
	var t := SandboxTest.new("demonstration_recorder_round_trip")
	var recorder := DemonstrationRecorder.new()
	recorder.start_recording({"source": "test"})
	var env := EnvironmentCore.new(0, 1)
	env.reset(1)
	var before: PackedFloat32Array = env.get_observations().to_array()
	var result: Dictionary = env.step(Action.idle())
	recorder.record_transition(0, before, Action.idle(), result)
	recorder.stop_recording()
	var path := "user://sandboxai_test_demo.jsonl"
	t.assert_true(recorder.save_dataset(path))
	var loaded := DemonstrationRecorder.new()
	var transitions: Array = loaded.load_dataset(path)
	t.assert_eq(transitions.size(), 1)
	t.assert_eq(transitions[0].observation.size(), Observation.FIELD_COUNT)
	t.assert_eq(transitions[0].next_observation.size(), Observation.FIELD_COUNT)
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	return t

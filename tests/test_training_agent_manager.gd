## Tests for the multi-agent registry over managed trainer processes.
## Controllers are driven through their public status/event surfaces —
## exactly what the backend status files feed at runtime — so no real
## Python process is required.
class_name TestTrainingAgentManager
extends RefCounted

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const TrainingAgentManager = preload("res://scripts/control_center/training_agent_manager.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")


func test_primary_registration_is_idempotent_and_first() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_primary_registration")
	var manager := TrainingAgentManager.new()
	var primary := TrainingRunController.new()
	var first_id: int = manager.register_primary(primary)
	t.assert_eq(first_id, 1, "the primary controller is Agent 1")
	t.assert_eq(manager.register_primary(primary), 1, "re-registering must not duplicate")
	t.assert_eq(manager.agent_count(), 1)
	var snapshot: Dictionary = manager.agent_snapshot(1)
	t.assert_eq(snapshot["label"], "Agent 1")
	t.assert_true(bool(snapshot["is_primary"]))
	t.assert_eq(snapshot["state"], "Idle")
	primary.free()
	manager.free()
	return t


func test_multiple_agents_have_distinct_identities_and_snapshots() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_multiple_agents")
	var manager := TrainingAgentManager.new()
	var primary := TrainingRunController.new()
	manager.register_primary(primary)
	var second := TrainingRunController.new()
	var third := TrainingRunController.new()
	var second_id: int = manager.adopt_agent(second)
	var third_id: int = manager.adopt_agent(third)
	t.assert_eq(second_id, 2)
	t.assert_eq(third_id, 3)
	t.assert_eq(manager.agent_count(), 3)
	t.assert_eq(manager.agent_ids(), [1, 2, 3])

	# Distinct backend states per agent, exactly as published.
	primary.apply_status({"state": "Running", "timesteps": 500, "training_type": "PPO"})
	second.apply_status({"state": "Paused", "epoch": 3, "training_type": "Behavior Cloning"})
	third.apply_status({"state": "Error", "error": "backend exploded"})
	t.assert_eq(manager.active_count(), 2, "Running + Paused are active; Error is not")
	var one: Dictionary = manager.agent_snapshot(1)
	var two: Dictionary = manager.agent_snapshot(2)
	var three: Dictionary = manager.agent_snapshot(3)
	t.assert_eq(one["state"], "Running")
	t.assert_eq(int(one["timesteps"]), 500)
	t.assert_eq(one["algorithm"], "PPO")
	t.assert_eq(two["state"], "Paused")
	t.assert_eq(two["algorithm"], "Behavior Cloning")
	t.assert_eq(three["state"], "Error")
	t.assert_eq(three["last_error"], "backend exploded")

	# Nothing was invented: no agent claims metrics its backend never sent.
	t.assert_false(two.has("timesteps"))
	t.assert_false(three.has("mean_episode_reward"))
	manager.free()  # frees adopted children
	primary.free()
	return t


func test_lifecycle_states_render_through_snapshots() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_lifecycle_states")
	var manager := TrainingAgentManager.new()
	var controller := TrainingRunController.new()
	manager.adopt_agent(controller)
	var agent_id: int = manager.agent_ids()[0]
	for state_name in ["Starting", "Running", "Paused", "Stopping", "Finished"]:
		controller.apply_status({"state": state_name})
		t.assert_eq(
			manager.agent_snapshot(agent_id)["state"],
			state_name,
			"snapshot must mirror the backend state %s" % state_name
		)
	manager.free()
	return t


func test_terminal_agents_stay_visible_until_removed() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_terminal_agents_persist")
	var manager := TrainingAgentManager.new()
	var finished := TrainingRunController.new()
	var running := TrainingRunController.new()
	manager.adopt_agent(finished)
	manager.adopt_agent(running)
	finished.apply_status({"state": "Finished", "timesteps": 1000})
	running.apply_status({"state": "Running"})
	t.assert_eq(manager.agent_count(), 2, "a finished agent remains listed")
	var finished_id: int = manager.agent_ids()[0]
	var running_id: int = manager.agent_ids()[1]
	t.assert_false(manager.remove(running_id), "an active agent cannot be removed")
	t.assert_true(manager.remove(finished_id), "a finished agent can be cleared")
	t.assert_eq(manager.agent_count(), 1)
	t.assert_false(manager.has_agent(finished_id))
	manager.free()
	return t


func test_primary_remove_resets_instead_of_freeing() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_primary_remove_resets")
	var manager := TrainingAgentManager.new()
	var primary := TrainingRunController.new()
	manager.register_primary(primary)
	primary.apply_status({"state": "Error", "error": "boom", "timesteps": 9})
	t.assert_true(manager.remove(1))
	t.assert_true(manager.has_agent(1), "the primary card survives as Idle")
	t.assert_eq(manager.agent_snapshot(1)["state"], "Idle")
	t.assert_false(manager.agent_snapshot(1).has("timesteps"), "reset removed stale run metrics")
	manager.free()
	primary.free()
	return t


func test_pause_resume_stop_propagate_to_the_command_file() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_command_propagation")
	var manager := TrainingAgentManager.new()
	var controller := TrainingRunController.new()
	manager.adopt_agent(controller)
	var agent_id: int = manager.agent_ids()[0]
	var command_path: String = "user://sandboxai_test_agent_command.json"
	controller.command_file = command_path
	controller.apply_status({"state": "Running"})

	t.assert_true(manager.pause(agent_id))
	t.assert_eq(_read_command(command_path), "pause")
	controller.apply_status({"state": "Paused"})
	t.assert_true(manager.resume(agent_id))
	t.assert_eq(_read_command(command_path), "resume")
	controller.apply_status({"state": "Running"})
	t.assert_true(manager.stop(agent_id))
	t.assert_eq(_read_command(command_path), "stop")
	t.assert_eq(
		controller.state,
		TrainingRunController.State.STOPPING,
		"stop is pending until the backend confirms its safe boundary"
	)

	# Commands are rejected in states where the backend cannot honour them.
	controller.apply_status({"state": "Finished"})
	t.assert_false(manager.pause(agent_id))
	t.assert_false(manager.resume(agent_id))
	t.assert_false(manager.stop(agent_id))
	t.assert_false(manager.pause(999), "unknown agents reject commands")
	DirAccess.remove_absolute(ProjectSettings.globalize_path(command_path))
	manager.free()
	return t


func test_event_relay_and_bounded_event_history() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_event_relay_bounded")
	var manager := TrainingAgentManager.new()
	var controller := TrainingRunController.new()
	manager.adopt_agent(controller)
	var agent_id: int = manager.agent_ids()[0]

	var relayed: Array = []
	manager.agent_event.connect(func(id: int, entry: Dictionary): relayed.append([id, entry]))

	# Write more events than the ring keeps, through the real JSONL path.
	var event_path: String = "user://sandboxai_test_agent_events.jsonl"
	var file := FileAccess.open(event_path, FileAccess.WRITE)
	var total: int = TrainingRunController.MAX_RECENT_EVENTS + 50
	for index in range(total):
		(
			file
			. store_line(
				(
					JSON
					. stringify(
						{
							"wall_time": 1700000000.0 + index,
							"category": "system" if index % 2 == 0 else "metric",
							"message": "event %d" % index,
						}
					)
				)
			)
		)
	file.close()
	controller.event_file = event_path
	controller._poll_events()

	t.assert_eq(controller.total_events_ingested, total, "every line was ingested")
	t.assert_eq(
		controller.recent_events.size(),
		TrainingRunController.MAX_RECENT_EVENTS,
		"the in-memory ring stays bounded"
	)
	t.assert_eq(relayed.size(), total, "each event was relayed with its agent id")
	t.assert_eq(relayed[0][0], agent_id)
	var last_entry: Dictionary = relayed[relayed.size() - 1][1]
	t.assert_eq(last_entry["message"], "event %d" % (total - 1))
	var last_kept: Dictionary = controller.recent_events[controller.recent_events.size() - 1]
	t.assert_eq(
		last_kept["message"],
		"event %d" % (total - 1),
		"the newest (terminal) event is preserved by the bounded buffer"
	)
	DirAccess.remove_absolute(ProjectSettings.globalize_path(event_path))
	manager.free()
	return t


func test_runtime_seconds_freezes_on_terminal_state() -> SandboxTest:
	var t := SandboxTest.new("agent_manager_runtime_freeze")
	var manager := TrainingAgentManager.new()
	var primary := TrainingRunController.new()
	manager.register_primary(primary)
	t.assert_null(manager.runtime_seconds(1), "no launch yet -> no runtime claim")
	var config := ControlCenterConfig.new()
	manager.note_primary_started(config)
	primary.apply_status({"state": "Running"})
	var live = manager.runtime_seconds(1)
	t.assert_not_null(live, "a launched agent has measured wall-clock runtime")
	t.assert_gte(float(live), 0.0)
	primary.apply_status({"state": "Finished"})
	var frozen_a = manager.runtime_seconds(1)
	var frozen_b = manager.runtime_seconds(1)
	t.assert_almost_eq(float(frozen_a), float(frozen_b), 0.0001, "terminal runtime is frozen")
	manager.free()
	primary.free()
	return t


static func _read_command(path: String) -> String:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return ""
	var parsed = JSON.parse_string(file.get_as_text())
	if parsed is Dictionary:
		return str((parsed as Dictionary).get("command", ""))
	return ""

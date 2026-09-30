## Focused regression coverage for the managed-training Control Center layer.
class_name TestControlCenterTraining
extends RefCounted

const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterSession = preload("res://scripts/control_center/control_center_session.gd")
const TrainingRunController = preload("res://scripts/control_center/training_run_controller.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_training_configuration_defaults_match_supported_backends() -> SandboxTest:
	var t := SandboxTest.new("control_center_training_config_defaults")
	var config := ControlCenterConfig.new()
	t.assert_eq(config.training_type, ControlCenterConfig.TrainingType.PPO)
	t.assert_eq(config.training_mode, ControlCenterConfig.TrainingMode.VISUAL)
	t.assert_eq(config.total_training_steps, 1_000_000)
	t.assert_eq(config.environment_count, 4)
	t.assert_almost_eq(config.learning_rate, 0.0003, 0.000001)
	t.assert_eq(config.rollout_length, 0)
	t.assert_eq(config.resolved_rollout_length(), 2048)
	t.assert_eq(config.batch_size, 256)
	t.assert_eq(ControlCenterConfig.training_device_argument(config.training_device), "auto")
	t.assert_true(ControlCenterConfig.training_type_available(ControlCenterConfig.TrainingType.PPO))
	t.assert_true(
		ControlCenterConfig.training_type_available(
			ControlCenterConfig.TrainingType.BEHAVIOR_CLONING
		)
	)
	t.assert_false(
		ControlCenterConfig.training_type_available(ControlCenterConfig.TrainingType.SELF_PLAY),
		"the evaluation-only self-play bridge must not be presented as an optimizer"
	)
	return t


func test_ppo_command_propagates_configuration() -> SandboxTest:
	var t := SandboxTest.new("control_center_ppo_command_propagation")
	var config := ControlCenterConfig.new()
	config.training_type = ControlCenterConfig.TrainingType.PPO
	config.training_mode = ControlCenterConfig.TrainingMode.HEADLESS
	config.training_device = ControlCenterConfig.TrainingDevice.GPU
	config.total_training_steps = 345_000
	config.environment_count = 6
	config.enemy_count = 3
	config.curriculum_level = 4
	config.seed = 77
	config.learning_rate = 0.0007
	config.rollout_length = 1024
	config.batch_size = 128
	var controller := TrainingRunController.new()
	var command: PackedStringArray = controller.build_command(config)
	t.assert_eq(command[0], "python")
	t.assert_eq(command[1], "-m")
	t.assert_eq(command[2], "sandboxai")
	t.assert_eq(command[3], "train")
	_assert_option(t, command, "--steps", "345000")
	_assert_option(t, command, "--env-count", "6")
	_assert_option(t, command, "--enemy-count", "3")
	_assert_option(t, command, "--curriculum-level", "4")
	_assert_option(t, command, "--seed", "77")
	_assert_option(t, command, "--device", "cuda")
	_assert_option(t, command, "--rollout-length", "1024")
	_assert_option(t, command, "--batch-size", "128")
	t.assert_true(command.has("--control-file"), "managed runs use the cooperative backend")
	controller.free()
	return t


func test_checkpoint_resume_and_bc_commands_use_real_cli_surfaces() -> SandboxTest:
	var t := SandboxTest.new("control_center_resume_and_bc_commands")
	var config := ControlCenterConfig.new()
	config.resume_from_checkpoint = true
	config.checkpoint_path = "training/runs/example/checkpoints/latest.zip"
	var controller := TrainingRunController.new()
	var ppo: PackedStringArray = controller.build_command(config)
	t.assert_eq(ppo[3], "resume")
	_assert_option(
		t, ppo, "--checkpoint", ProjectSettings.globalize_path("res://%s" % config.checkpoint_path)
	)

	config.training_type = ControlCenterConfig.TrainingType.BEHAVIOR_CLONING
	config.bc_dataset_path = "training/datasets/demo.jsonl"
	config.checkpoint_path = "training/bc_runs/example/latest.pt"
	config.bc_epochs = 40
	var bc: PackedStringArray = controller.build_command(config)
	t.assert_eq(bc[3], "bc-train")
	_assert_option(
		t, bc, "--dataset", ProjectSettings.globalize_path("res://%s" % config.bc_dataset_path)
	)
	_assert_option(t, bc, "--epochs", "40")
	_assert_option(
		t,
		bc,
		"--resume-checkpoint",
		ProjectSettings.globalize_path("res://%s" % config.checkpoint_path)
	)
	controller.free()
	return t


func test_backend_status_drives_runtime_states_and_reset_clears_stale_data() -> SandboxTest:
	var t := SandboxTest.new("control_center_training_runtime_states")
	var controller := TrainingRunController.new()
	for state_name in ["Starting", "Running", "Paused", "Stopping", "Finished"]:
		controller.apply_status({"state": state_name, "timesteps": 123})
		t.assert_eq(TrainingRunController.state_name(controller.state), state_name)
	controller.apply_status({"state": "Error", "error": "backend failed", "timesteps": 456})
	t.assert_eq(controller.state, TrainingRunController.State.ERROR)
	t.assert_eq(controller.last_error, "backend failed")
	t.assert_true(controller.snapshot().has("timesteps"))
	t.assert_true(controller.reset())
	t.assert_eq(controller.state, TrainingRunController.State.IDLE)
	t.assert_false(controller.snapshot().has("timesteps"), "reset removes stale run metrics")
	controller.free()
	return t


func test_visual_headless_and_tile_state_round_trip() -> SandboxTest:
	var t := SandboxTest.new("control_center_training_mode_and_tiles")
	var config := ControlCenterConfig.new()
	config.training_mode = ControlCenterConfig.TrainingMode.HEADLESS
	t.assert_true(config.set_tile_visible("agent", false))
	t.assert_false(config.is_tile_visible("agent"))
	var old_index: int = config.tile_order.find("logs")
	t.assert_true(config.move_tile("logs", -1))
	t.assert_eq(config.tile_order.find("logs"), old_index - 1)

	var restored := ControlCenterConfig.new()
	restored.apply_dict(config.to_dict())
	t.assert_eq(restored.training_mode, ControlCenterConfig.TrainingMode.HEADLESS)
	t.assert_false(restored.is_tile_visible("agent"))
	t.assert_eq(restored.tile_order, config.tile_order)
	t.assert_false(config.set_tile_visible("not-a-tile", true))
	return t


func test_tile_preferences_persist_without_simulation_state() -> SandboxTest:
	var t := SandboxTest.new("control_center_tile_preferences_persist")
	var path: String = "user://sandboxai_control_center_test.cfg"
	var config := ControlCenterConfig.new()
	config.set_tile_visible("logs", false)
	config.left_dock_width = 444
	config.move_tile("logs", -1)
	t.assert_true(config.save_preferences(path))
	var restored := ControlCenterConfig.new()
	t.assert_true(restored.load_preferences(path))
	t.assert_false(restored.is_tile_visible("logs"))
	t.assert_eq(restored.left_dock_width, 444)
	t.assert_eq(restored.tile_order, config.tile_order)
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	return t


func test_agent_dashboard_snapshot_contains_real_display_state() -> SandboxTest:
	var t := SandboxTest.new("control_center_agent_dashboard_binding")
	var config := ControlCenterConfig.new(ControlCenterConfig.Mode.WATCH)
	config.environment_count = 1
	var session := ControlCenterSession.new()
	session.presentation_enabled = false
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop != null:
		loop.root.add_child(session)
	session.setup(config)
	session.presentation_enabled = true
	session.advance(2)
	var snapshot: Dictionary = session.build_snapshot()
	var agent: Dictionary = snapshot["agent"]
	for key in [
		"position",
		"forward",
		"velocity",
		"yaw_deg",
		"pitch_deg",
		"health",
		"alive",
		"on_ground",
		"in_combat",
	]:
		t.assert_true(agent.has(key), "agent display is missing %s" % key)
	var target: Dictionary = snapshot["target"]
	if bool(target.get("has_target", false)):
		t.assert_true(target.has("distance_m"))
		t.assert_true(target.has("direction"))
	t.assert_true(snapshot.has("action"))
	t.assert_true(snapshot.has("episode"))
	if loop != null and session.get_parent() != null:
		loop.root.remove_child(session)
	session.free()
	return t


static func _assert_option(
	test: SandboxTest, command: PackedStringArray, option: String, expected: String
) -> void:
	var index: int = command.find(option)
	test.assert_true(index >= 0, "missing command option %s" % option)
	if index >= 0 and index + 1 < command.size():
		test.assert_eq(command[index + 1], expected, "%s value" % option)

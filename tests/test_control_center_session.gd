## Tests for ControlCenterSession: the only object that connects the
## Control Center to the simulation.
##
## These cover the acceptance criteria that matter most: mode switching is
## safe, the human pipeline is the SAME action pipeline, pausing/stepping/
## resetting work, selection and settings propagate, telemetry is a pure
## read of simulation state, and none of it changes simulation behavior or
## determinism.
class_name TestControlCenterSession
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterSession = preload("res://scripts/control_center/control_center_session.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const HumanController = preload("res://scripts/input/human_controller.gd")
const Observation = preload("res://scripts/core/observation.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


## Builds a headless session (no views, no window) attached to the live
## SceneTree so Node lifecycle callbacks behave exactly as at runtime.
static func _make_session(mode: int, environments: int = 3, enemies: int = 1):
	var config := ControlCenterConfig.new(mode)
	config.environment_count = environments
	config.enemy_count = enemies
	config.seed = 4321
	var session := ControlCenterSession.new()
	session.presentation_enabled = false
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop != null:
		loop.root.add_child(session)
	session.setup(config)
	return session


static func _destroy(session) -> void:
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	if loop != null and session.get_parent() != null:
		loop.root.remove_child(session)
	session.free()


func test_headless_session_creates_no_views_and_no_telemetry() -> SandboxTest:
	var t := SandboxTest.new("session_headless_has_no_presentation")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	t.assert_false(session.simulation_manager.create_visuals, "no visuals without a display")
	for view in session.simulation_manager.views:
		t.assert_null(view, "a headless Control Center must not build EnvironmentViews")
	t.assert_false(session.telemetry_enabled(), "telemetry requires a presentation layer")
	t.assert_true(
		session.build_snapshot().is_empty(), "no snapshot work is done without presentation"
	)
	t.assert_false(
		session.simulation_manager.auto_tick,
		"the session drives stepping itself so pause/step/speed are exact"
	)
	_destroy(session)
	return t


func test_training_mode_disables_logging_and_snapshots() -> SandboxTest:
	var t := SandboxTest.new("session_training_mode_disables_presentation_work")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	session.presentation_enabled = true  # pretend a window exists
	t.assert_true(session.telemetry_enabled())
	t.assert_true(session.event_log.enabled)

	session.set_mode(ControlCenterConfig.Mode.TRAINING)
	t.assert_false(session.telemetry_enabled(), "TRAINING never pays for telemetry")
	t.assert_false(session.event_log.enabled, "TRAINING never buffers log events")
	t.assert_true(session.build_snapshot().is_empty())
	t.assert_false(session.human_input_enabled, "human input is force-released in TRAINING")

	session.advance(5)
	t.assert_eq(session.event_log.size(), 0, "stepping in TRAINING produces no log entries")
	_destroy(session)
	return t


## Regression: `presentation_enabled` was a plain variable, so turning
## presentation on AFTER setup() (what the Control Center scene does once it
## knows a display exists) left every derived flag stale — telemetry
## reported enabled while the event log stayed off and nothing was logged.
func test_toggling_presentation_reapplies_the_derived_runtime_state() -> SandboxTest:
	var t := SandboxTest.new("session_presentation_toggle_reapplies_runtime")
	var session = _make_session(ControlCenterConfig.Mode.WATCH, 1, 1)
	t.assert_false(session.event_log.enabled, "headless setup leaves logging off")

	session.presentation_enabled = true
	t.assert_true(session.telemetry_enabled())
	t.assert_true(session.event_log.enabled, "enabling presentation must enable the log")
	t.assert_true(
		session.get_selected_environment().debug_perception,
		"the selected environment is instrumented for the overlay"
	)
	session.log_system("presentation probe")
	t.assert_gte(float(session.event_log.size()), 1.0, "log entries are buffered again")
	var logged: int = session.event_log.size()

	session.presentation_enabled = false
	t.assert_false(session.telemetry_enabled())
	t.assert_false(session.event_log.enabled, "disabling presentation must disable the log")
	t.assert_false(session.get_selected_environment().debug_perception)
	session.log_system("ignored while headless")
	t.assert_eq(session.event_log.size(), logged, "a disabled log buffers nothing")
	_destroy(session)
	return t


func test_mode_switching_is_safe_and_rebinds_only_the_selected_environment() -> SandboxTest:
	var t := SandboxTest.new("session_mode_switch_rebinds_selected_only")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	var manager = session.simulation_manager
	for index in range(manager.environments.size()):
		t.assert_ne(manager.controllers[index], session.human_controller)

	session.set_mode(ControlCenterConfig.Mode.HUMAN)
	t.assert_eq(
		manager.controllers[session.config.selected_environment],
		session.human_controller,
		"HUMAN mode routes the selected environment through the human controller"
	)
	for index in range(manager.environments.size()):
		if index == session.config.selected_environment:
			continue
		t.assert_ne(
			manager.controllers[index],
			session.human_controller,
			"other environments keep their AI controller so comparisons stay valid"
		)

	# Switching back and forth must not corrupt state or duplicate controllers.
	session.set_mode(ControlCenterConfig.Mode.WATCH)
	session.set_mode(ControlCenterConfig.Mode.HUMAN)
	session.set_mode(ControlCenterConfig.Mode.TRAINING)
	session.set_mode(ControlCenterConfig.Mode.WATCH)
	t.assert_eq(manager.environments.size(), 3, "mode switching never rebuilds environments")
	t.assert_ne(manager.controllers[0], session.human_controller)
	var human_children: int = 0
	for child in session.get_children():
		if child is HumanController:
			human_children += 1
	t.assert_eq(human_children, 1, "exactly one HumanController is ever created")
	_destroy(session)
	return t


## Leaving HUMAN mode must disarm the input pipeline. Otherwise the human
## controller keeps processing input (and, with a real window, keeps the
## mouse captured) while the AI drives.
func test_leaving_human_mode_disarms_the_input_pipeline() -> SandboxTest:
	var t := SandboxTest.new("session_leaving_human_disarms_input")
	var session = _make_session(ControlCenterConfig.Mode.HUMAN)
	session.set_human_input_enabled(true)
	t.assert_true(session.human_input_enabled, "HUMAN mode can arm the input pipeline")

	session.set_mode(ControlCenterConfig.Mode.WATCH)
	t.assert_false(
		session.human_input_enabled, "WATCH must not keep listening to keyboard/mouse"
	)
	t.assert_false(
		session.human_controller.is_processing_unhandled_input(),
		"the human controller stops processing input when it is not driving"
	)

	session.set_mode(ControlCenterConfig.Mode.HUMAN)
	session.set_human_input_enabled(true)
	session.set_mode(ControlCenterConfig.Mode.TRAINING)
	t.assert_false(session.human_input_enabled, "TRAINING also disarms human input")
	_destroy(session)
	return t


func test_human_mode_uses_the_same_action_pipeline() -> SandboxTest:
	var t := SandboxTest.new("session_human_uses_same_action_pipeline")
	var session = _make_session(ControlCenterConfig.Mode.HUMAN)
	var manager = session.simulation_manager
	var selected: int = session.config.selected_environment
	var controller = manager.controllers[selected]
	t.assert_eq(controller, session.human_controller)
	t.assert_true(
		controller.has_method("get_action"),
		"the human controller implements the shared ControllerBase contract"
	)
	var produced = controller.get_action(manager.environments[selected])
	t.assert_true(produced is Action, "human input produces the same Action type as the AI")

	# With input released the agent must receive a genuine idle action
	# rather than stale or fabricated input.
	session.set_human_input_enabled(false)
	session.advance(3)
	var last_action = manager.get_last_actions()[selected]
	t.assert_eq(last_action.move_axis, 0)
	t.assert_eq(last_action.strafe_axis, 0)
	t.assert_false(last_action.shoot)
	t.assert_eq(
		session.get_selected_environment().episode.step_count, 3, "the environment still steps"
	)
	_destroy(session)
	return t


func test_pause_resume_and_single_step() -> SandboxTest:
	var t := SandboxTest.new("session_pause_resume_single_step")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	t.assert_true(session.running)
	session.toggle_running()
	t.assert_false(session.running)

	var before: int = session.get_selected_environment().episode.step_count
	session.request_single_step(4)
	t.assert_false(session.running, "requesting a step keeps the simulation paused")
	session._physics_process(1.0 / 60.0)
	t.assert_eq(
		session.get_selected_environment().episode.step_count,
		before + 4,
		"a queued single-step batch advances exactly that many ticks"
	)

	# A paused session must not advance on its own.
	session._physics_process(1.0 / 60.0)
	t.assert_eq(session.get_selected_environment().episode.step_count, before + 4)

	session.set_running(true)
	session.set_speed(1.0)
	session._physics_process(1.0 / 60.0)
	t.assert_eq(session.get_selected_environment().episode.step_count, before + 5)
	t.assert_almost_eq(
		float(Engine.time_scale), 1.0, 0.0001, "speed control must never touch Engine.time_scale"
	)
	_destroy(session)
	return t


## Bug: advance(0) stepped the simulation once anyway (the loop clamped the
## count to 1), so a zero-step request silently advanced every environment.
func test_advance_zero_advances_nothing() -> SandboxTest:
	var t := SandboxTest.new("session_advance_zero_is_a_noop")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	var before: int = session.get_selected_environment().episode.step_count
	session.advance(0)
	session.advance(-3)
	t.assert_eq(
		session.get_selected_environment().episode.step_count,
		before,
		"a non-positive step count must not advance the simulation"
	)
	_destroy(session)
	return t


func test_speed_multiplier_is_bounded_per_frame() -> SandboxTest:
	var t := SandboxTest.new("session_speed_is_bounded")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	session.set_speed(1000.0)
	t.assert_almost_eq(
		session.config.simulation_speed, ControlCenterConfig.MAX_SPEED, 0.001, "speed is clamped"
	)
	var before: int = session.get_selected_environment().episode.step_count
	session._physics_process(1.0 / 60.0)
	var executed: int = session.get_selected_environment().episode.step_count - before
	t.assert_lte(
		float(executed),
		float(ControlCenterConfig.MAX_STEPS_PER_FRAME_INTERACTIVE),
		"a high speed multiplier can never freeze the frame"
	)
	t.assert_gt(float(executed), 1.0)
	_destroy(session)
	return t


func test_reset_restarts_the_selected_environment_only() -> SandboxTest:
	var t := SandboxTest.new("session_reset_selected_only")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	session.advance(10)
	var other_steps: int = session.simulation_manager.environments[1].episode.step_count
	t.assert_eq(session.get_selected_environment().episode.step_count, 10)

	session.reset_selected_environment(true)
	t.assert_eq(session.get_selected_environment().episode.step_count, 0)
	t.assert_eq(
		session.simulation_manager.environments[1].episode.step_count,
		other_steps,
		"resetting the selected environment must not disturb the others"
	)

	session.reset_all_environments(true)
	for env in session.simulation_manager.environments:
		t.assert_eq(env.episode.step_count, 0)
	_destroy(session)
	return t


func test_deterministic_reset_replays_the_same_episode() -> SandboxTest:
	var t := SandboxTest.new("session_deterministic_reset")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	session.reset_selected_environment(true)
	var first: PackedFloat32Array = session.get_selected_environment().get_observations().to_array()
	session.advance(15)
	session.reset_selected_environment(true)
	var second: PackedFloat32Array = (
		session.get_selected_environment().get_observations().to_array()
	)
	t.assert_eq(first.size(), second.size())
	for index in range(first.size()):
		t.assert_almost_eq(
			float(second[index]),
			float(first[index]),
			0.00001,
			"a deterministic reset must reproduce the identical start state at index %d" % index
		)
	_destroy(session)
	return t


func test_selection_changes_environment_and_agent_slot_honestly() -> SandboxTest:
	var t := SandboxTest.new("session_selection")
	var session = _make_session(ControlCenterConfig.Mode.HUMAN)
	session.select_environment(2)
	t.assert_eq(session.config.selected_environment, 2)
	t.assert_eq(
		session.simulation_manager.controllers[2],
		session.human_controller,
		"the human follows the selection"
	)
	t.assert_ne(session.simulation_manager.controllers[0], session.human_controller)

	session.select_environment(99)
	t.assert_eq(session.config.selected_environment, 2, "out-of-range selection is clamped")

	t.assert_true(session.is_agent_slot_available(0))
	t.assert_false(
		session.is_agent_slot_available(1),
		"slot 1 only exists in the self-play foundation and must be reported unavailable"
	)
	t.assert_false(session.select_agent_slot(1), "an unavailable slot is rejected, not faked")
	t.assert_eq(session.config.selected_agent_slot, 0)
	var slots: Array = session.available_agent_slots()
	t.assert_gte(float(slots.size()), 1.0)
	t.assert_true(bool((slots[0] as Dictionary)["available"]))
	_destroy(session)
	return t


func test_policy_source_rejects_unavailable_backends() -> SandboxTest:
	var t := SandboxTest.new("session_policy_source")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	t.assert_true(session.set_policy_source(ControlCenterConfig.PolicySource.IDLE))
	session.advance(3)
	var action = session.simulation_manager.get_last_actions()[0]
	t.assert_eq(action.move_axis, 0, "the IDLE policy really produces idle actions")

	t.assert_false(
		session.set_policy_source(ControlCenterConfig.PolicySource.EXTERNAL_POLICY),
		"in-engine inference does not exist and must be refused"
	)
	t.assert_eq(session.config.policy_source, ControlCenterConfig.PolicySource.IDLE)

	t.assert_true(session.set_policy_source(ControlCenterConfig.PolicySource.HEURISTIC))
	session.advance(30)
	t.assert_ne(
		session.simulation_manager.controllers[0], null, "the heuristic controller is rebound"
	)
	_destroy(session)
	return t


func test_settings_propagate_with_explicit_rebuild_semantics() -> SandboxTest:
	var t := SandboxTest.new("session_settings_propagation")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)

	# Curriculum level applies live.
	session.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	t.assert_eq(
		session.simulation_manager.curriculum_level, CurriculumConfig.Level.MULTIPLE_ENEMIES
	)
	t.assert_gte(
		float(session.get_selected_environment().enemies.size()),
		float(CurriculumConfig.MULTIPLE_ENEMIES_MIN_COUNT),
		"level 4 enforces its documented minimum enemy count"
	)
	t.assert_false(session.has_pending_settings(), "a live setting never becomes pending")

	# Rebuild settings are pending until explicitly applied.
	session.request_setting("enemy_count", 5)
	session.request_setting("environment_count", 2)
	t.assert_true(session.has_pending_settings())
	t.assert_true(session.pending_setting_keys().has("enemy_count"))
	t.assert_eq(
		session.simulation_manager.environments.size(), 3, "nothing changes before Apply & reset"
	)

	session.apply_pending_settings()
	t.assert_false(session.has_pending_settings())
	t.assert_eq(session.simulation_manager.environments.size(), 2)
	t.assert_eq(session.get_selected_environment().enemies.size(), 5)
	t.assert_eq(
		session.simulation_manager.controllers.size(),
		2,
		"controllers are rebuilt alongside the environments"
	)
	t.assert_false(session.request_setting("not_a_setting", 1), "unknown keys are refused")
	_destroy(session)
	return t


func test_snapshot_is_a_pure_read_of_simulation_state() -> SandboxTest:
	var t := SandboxTest.new("session_snapshot_is_pure_read")
	var session = _make_session(ControlCenterConfig.Mode.WATCH)
	session.presentation_enabled = true  # enable telemetry without creating views
	session.advance(12)

	var before: PackedFloat32Array = (
		session.get_selected_environment().get_observations().to_array()
	)
	var step_before: int = session.get_selected_environment().episode.step_count
	var snapshot: Dictionary = session.build_snapshot()
	t.assert_false(snapshot.is_empty())
	t.assert_eq(
		(snapshot["observation"] as Dictionary)["field_count"],
		Observation.FIELD_COUNT,
		"the inspector reads the real observation contract"
	)
	t.assert_eq(
		((snapshot["observation"] as Dictionary)["rows"] as Array).size(), Observation.FIELD_COUNT
	)
	t.assert_true(snapshot.has("perception"))
	t.assert_true(snapshot.has("agent"))
	t.assert_true(snapshot.has("episode"))

	var after: PackedFloat32Array = (
		session.get_selected_environment().get_observations().to_array()
	)
	t.assert_eq(session.get_selected_environment().episode.step_count, step_before)
	for index in range(before.size()):
		t.assert_almost_eq(
			float(after[index]),
			float(before[index]),
			0.000001,
			"building a snapshot must not mutate the simulation"
		)
	_destroy(session)
	return t


func test_session_stepping_matches_a_bare_environment_exactly() -> SandboxTest:
	var t := SandboxTest.new("session_does_not_change_simulation_behavior")
	var session = _make_session(ControlCenterConfig.Mode.WATCH, 1, 1)
	session.set_policy_source(ControlCenterConfig.PolicySource.IDLE)
	session.reset_selected_environment(true)
	session.advance(40)
	var through_session: PackedFloat32Array = (
		session.get_selected_environment().get_observations().to_array()
	)
	var session_reward: float = session.get_selected_environment().episode.cumulative_reward

	# The same seed driven directly through EnvironmentCore with the same
	# idle actions and the same fixed dt must produce an identical result.
	var reference := EnvironmentCore.new(0, 1)
	reference.set_curriculum_level(session.config.curriculum_level)
	reference.reset(session.config.seed + 0)
	for _step in range(40):
		reference.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	var direct: PackedFloat32Array = reference.get_observations().to_array()

	t.assert_eq(through_session.size(), direct.size())
	for index in range(direct.size()):
		t.assert_almost_eq(
			float(through_session[index]),
			float(direct[index]),
			0.00001,
			"the Control Center must not perturb the simulation (index %d)" % index
		)
	t.assert_almost_eq(
		session_reward,
		reference.episode.cumulative_reward,
		0.00001,
		"rewards are identical with and without the Control Center"
	)
	_destroy(session)
	return t


func test_episode_results_are_recorded_with_their_action_source() -> SandboxTest:
	var t := SandboxTest.new("session_records_episode_source")
	var session = _make_session(ControlCenterConfig.Mode.WATCH, 1, 1)
	session.presentation_enabled = true
	session.get_selected_environment().max_steps = 8
	session.advance(30)
	t.assert_gte(float(session.results.size()), 1.0, "finished episodes are recorded")
	# Oldest first: history(..., newest_first = false)[0] is the FIRST
	# episode that finished, which is the one this test is about.
	var history: Array = session.results.history("", 0, false)
	var record: Dictionary = history[0]
	t.assert_eq(str(record["source"]), "ai", "WATCH episodes are attributed to the AI")
	t.assert_eq(int(record["env_index"]), 0)
	t.assert_eq(
		int(record["episode"]), 1,
		"the FIRST finished episode is episode 1, not 2 (auto-reset had already bumped the counter)"
	)
	for position in range(history.size()):
		var entry: Dictionary = history[position]
		t.assert_eq(
			int(entry["episode"]),
			position + 1,
			"records keep the finished episode numbers in order (record %d)" % position
		)
	t.assert_false(str(record["done_reason"]).is_empty())
	t.assert_gte(float(session.event_log.size()), 1.0, "discrete events are logged")
	_destroy(session)
	return t


## Regression: the record for a finished episode must keep THAT episode's
## number and action source even though SimulationManager.step_all() has
## already auto-reset the environment (and therefore already incremented
## `episode.episode_count`) by the time the result reaches the session.
func test_episode_record_keeps_its_number_after_auto_reset() -> SandboxTest:
	var t := SandboxTest.new("session_record_survives_auto_reset")
	var session = _make_session(ControlCenterConfig.Mode.WATCH, 1, 1)
	session.presentation_enabled = true
	var env = session.get_selected_environment()
	env.max_steps = 4
	t.assert_true(
		session.simulation_manager.auto_reset_on_done, "the Control Center runs with auto-reset on"
	)

	var expected_episode: int = 1
	while expected_episode <= 2:
		var guard: int = 0
		while session.results.size() < expected_episode and guard < 50:
			session.advance(1)
			guard += 1
		t.assert_eq(
			session.results.size(),
			expected_episode,
			"exactly one record per finished episode (episode %d)" % expected_episode
		)
		var record: Dictionary = session.results.history("", 0, false)[expected_episode - 1]
		t.assert_eq(
			int(record["episode"]),
			expected_episode,
			"the record keeps the finished episode's number, not the auto-reset one"
		)
		t.assert_eq(str(record["source"]), "ai", "the finished episode's action source is kept")
		t.assert_eq(
			int(env.episode.episode_count),
			expected_episode + 1,
			"the environment has already been auto-reset into the next episode"
		)
		expected_episode += 1

	_destroy(session)
	return t


func test_status_and_training_command_describe_reality() -> SandboxTest:
	var t := SandboxTest.new("session_status_describes_reality")
	var session = _make_session(ControlCenterConfig.Mode.WATCH, 2, 2)
	var status: Dictionary = session.get_status()
	t.assert_eq(str(status["mode_name"]), "WATCH")
	t.assert_eq(int(status["environment_count"]), 2)
	t.assert_true(bool(status["running"]))
	t.assert_eq(int(status["seed"]), 4321)

	var command: String = session.training_command_line()
	t.assert_true(command.begins_with("sandboxai train"))
	t.assert_true(command.contains("--env-count 2"))
	t.assert_true(command.contains("--seed 4321"))
	_destroy(session)
	return t

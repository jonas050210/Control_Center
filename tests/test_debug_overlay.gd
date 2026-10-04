## Tests for DebugOverlay's pure telemetry-collection/formatting functions and
## for its contact-box geometry (the "what is in view right now" pass that the
## TTK silhouette boxes mirror).
## These deliberately never construct Control/Label nodes: only the data
## pipeline (SimulationManager/EnvironmentCore -> Dictionary -> lines) and the
## screen-space box maths are exercised, keeping the test fast and independent
## of any live scene tree.
class_name TestDebugOverlay
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const DebugOverlay = preload("res://scripts/debug/debug_overlay.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func _make_sim(env_count: int = 2, enemy_count: int = 2) -> SimulationManager:
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(env_count, enemy_count)
	return sim


func test_telemetry_dict_reports_core_fields() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_reports_core_fields")
	var sim := _make_sim(2, 2)
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	t.assert_true(telemetry.has("agent_health"))
	t.assert_true(telemetry.has("enemy_count"))
	t.assert_true(telemetry.has("enemies"))
	t.assert_true(telemetry.has("episode_reward"))
	t.assert_true(telemetry.has("reward_breakdown"))
	t.assert_true(telemetry.has("observation_summary"))
	t.assert_eq(telemetry.enemy_count, 2)
	t.assert_eq(telemetry.active_environments, 2)
	sim.free()
	return t


func test_telemetry_dict_tracks_shots_and_current_action() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_tracks_shots_and_current_action")
	var sim := _make_sim(1, 1)
	sim.step_all([Action.from_discrete(Action.Discrete.SHOOT)])
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	t.assert_eq(telemetry.shots_fired, 1)
	t.assert_true(telemetry.has("current_action"))
	t.assert_eq(int(telemetry.current_action[4]), 1, "recorded action should show shoot=1")
	sim.free()
	return t


func test_telemetry_dict_clamps_out_of_range_focus_index() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_clamps_out_of_range_focus_index")
	var sim := _make_sim(2, 1)
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 99)
	t.assert_eq(telemetry.focused_env_index, 1, "focus index should clamp to the last environment")
	sim.free()
	return t


func test_telemetry_dict_is_empty_with_no_environments() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_is_empty_with_no_environments")
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(null, 0)
	t.assert_true(telemetry.is_empty())
	return t


func test_format_lines_produces_non_empty_readable_output() -> SandboxTest:
	var t := SandboxTest.new("format_lines_produces_non_empty_readable_output")
	var sim := _make_sim(1, 1)
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	var lines: PackedStringArray = DebugOverlay.format_lines(telemetry)
	t.assert_gt(float(lines.size()), 5.0, "overlay should print multiple telemetry lines")
	var joined: String = "\n".join(lines)
	t.assert_true(joined.find("agent hp") >= 0)
	t.assert_true(joined.find("enemies alive") >= 0)
	sim.free()
	return t


func test_format_lines_handles_empty_telemetry_gracefully() -> SandboxTest:
	var t := SandboxTest.new("format_lines_handles_empty_telemetry_gracefully")
	var lines: PackedStringArray = DebugOverlay.format_lines({})
	t.assert_eq(lines.size(), 1)
	return t


## Puts the agent at the origin looking down -Z with one enemy dead ahead and
## one directly behind, then runs a single tick so the enemy list and the
## geometry are settled. Both enemies are alive and inside the arena - the
## only difference between them is where the agent is looking.
##
## Level 0 has no perception layer, so this exercises the geometric path
## (`_geometric_contact_boxes`), which is the path the legacy curriculum
## levels 1-4 take.
func _stage_line_of_sight(env: EnvironmentCore) -> void:
	env.agent.position = Vector3(0.0, 0.0, 0.0)
	env.agent.yaw_deg = 0.0
	env.agent.pitch_deg = 0.0
	env.agent.alive = true
	# Unequal distances on purpose: with both enemies 8 m away the "nearest"
	# slot would be decided by a tie-break, and every assertion below that
	# names the primary contact would be asserting an accident.
	var front = env.enemies[0]
	front.alive = true
	front.health = front.max_health
	front.position = Vector3(0.0, 0.0, -6.0)
	var behind = env.enemies[1]
	behind.alive = true
	behind.health = behind.max_health
	behind.position = Vector3(0.0, 0.0, 9.0)
	env.step(Action.idle(), SandboxConfig.SIMULATION_DT)


func test_contact_boxes_are_empty_without_an_agent() -> SandboxTest:
	var t := SandboxTest.new("contact_boxes_are_empty_without_an_agent")
	t.assert_eq(DebugOverlay.contact_boxes(null).size(), 0)
	var env := EnvironmentCore.new(0, 2)
	env.reset(7)
	env.agent.alive = false
	t.assert_eq(
		DebugOverlay.contact_boxes(env).size(),
		0,
		"a dead agent sees nothing, even while its last beliefs are still stored"
	)
	return t


func test_contact_boxes_mark_only_the_enemies_the_agent_sees() -> SandboxTest:
	var t := SandboxTest.new("contact_boxes_mark_only_the_enemies_the_agent_sees")
	var env := EnvironmentCore.new(0, 2)
	env.reset(7)
	_stage_line_of_sight(env)
	var boxes: Array = DebugOverlay.contact_boxes(env)
	t.assert_gt(float(boxes.size()), 0.0, "the enemy dead ahead of the agent must get a box")
	var seen_indices: Array = []
	for box_value in boxes:
		var box: Dictionary = box_value
		seen_indices.append(int(box["index"]))
		t.assert_true(
			absf(float(box["center_x"])) <= 1.0 and absf(float(box["center_y"])) <= 1.0,
			"box centres are normalised device coordinates, so they stay inside -1..1"
		)
		t.assert_gt(float(box["half_width"]), 0.0, "a contact has to be wider than nothing")
		t.assert_gt(float(box["half_height"]), 0.0, "a contact has to be taller than nothing")
		var exposure: float = float(box["exposure"])
		t.assert_true(
			exposure >= 0.0 and exposure <= 1.0,
			"exposure is the fraction of the body that is actually clear"
		)
	t.assert_true(seen_indices.has(0), "the enemy in front of the agent is in view")
	t.assert_false(
		seen_indices.has(1), "the enemy behind the agent is not in view, so it gets no box"
	)
	return t


func test_contact_boxes_follow_the_agent_gaze() -> SandboxTest:
	var t := SandboxTest.new("contact_boxes_follow_the_agent_gaze")
	var env := EnvironmentCore.new(0, 2)
	env.reset(11)
	_stage_line_of_sight(env)
	# Only the staged contact in front is left, so "no box" can only mean the
	# agent is no longer looking at it. With the second enemy still alive a
	# turn brings *it* into view and the count says nothing.
	env.enemies[1].alive = false
	env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	t.assert_eq(DebugOverlay.contact_boxes(env).size(), 1)
	# Turn right around and the same enemy leaves the screen.
	env.agent.yaw_deg = 180.0
	env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	t.assert_eq(
		DebugOverlay.contact_boxes(env).size(),
		0,
		"looking away removes the box - the overlay shows sight, not proximity"
	)
	return t


func test_contact_boxes_ignore_the_dead() -> SandboxTest:
	var t := SandboxTest.new("contact_boxes_ignore_the_dead")
	var env := EnvironmentCore.new(0, 2)
	env.reset(13)
	_stage_line_of_sight(env)
	t.assert_eq(DebugOverlay.contact_boxes(env).size(), 1)
	var front = env.enemies[0]
	front.alive = false
	front.health = 0.0
	env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	t.assert_eq(
		DebugOverlay.contact_boxes(env).size(),
		0,
		"a corpse is not a contact - only the enemy behind is left, and it is out of view"
	)
	return t


## The text panel and the rectangles must come from one source, or the
## overlay starts arguing with itself.
func test_telemetry_reports_the_same_contacts_the_boxes_draw() -> SandboxTest:
	var t := SandboxTest.new("telemetry_reports_the_same_contacts_the_boxes_draw")
	var sim := _make_sim(1, 2)
	_stage_line_of_sight(sim.environments[0])
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	t.assert_true(telemetry.has("contact_boxes"))
	t.assert_eq(
		telemetry.contact_boxes.size(),
		DebugOverlay.contact_boxes(sim.environments[0]).size(),
		"the label counts the same contacts the boxes frame"
	)
	var joined: String = "\n".join(DebugOverlay.format_lines(telemetry))
	t.assert_true(joined.find("in view:") >= 0)
	sim.free()
	return t


## The perception layer builds a belief per enemy and the vision block of the
## observation is filled from those beliefs, so the overlay has to read the
## very same `screen_box` - not a second, slightly different geometry. This
## pins the two together, and then pins the drawn box to the one that reaches
## the vector, so all three can only ever be one measurement.
func test_contact_boxes_are_the_boxes_the_observation_carries() -> SandboxTest:
	var t := SandboxTest.new("contact_boxes_are_the_boxes_the_observation_carries")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	# Beliefs on an obstacle-free level, without waiting out the acquisition
	# delay: `debug_perception` runs the perception layer on a level that does
	# not gate vision, so one tick is enough and the geometry is the staged
	# one rather than whatever the seeded arena happened to produce.
	env.debug_perception = true
	env.reset(21)
	_stage_line_of_sight(env)
	t.assert_gt(
		float(env.get_beliefs().size()),
		0.0,
		"the perception layer must produce beliefs for this test to mean anything"
	)
	var expected: Array = []
	for belief_value in env.get_beliefs():
		var belief: Dictionary = belief_value
		var box: Dictionary = belief.get("screen_box", {})
		if bool(belief.get("visible", false)) and bool(box.get("in_front", false)):
			expected.append(belief)
	t.assert_gt(float(expected.size()), 0.0, "the staged contact is a sighting with a box")
	var boxes: Array = DebugOverlay.contact_boxes(env)
	t.assert_eq(
		boxes.size(), expected.size(), "one box per visible, in-front belief - no more, no fewer"
	)
	for index in range(mini(boxes.size(), expected.size())):
		var expected_box: Dictionary = expected[index]["screen_box"]
		var drawn: Dictionary = boxes[index]
		t.assert_almost_eq(float(drawn["center_x"]), float(expected_box["center_x"]), 0.0001)
		t.assert_almost_eq(float(drawn["center_y"]), float(expected_box["center_y"]), 0.0001)
		t.assert_almost_eq(float(drawn["half_width"]), float(expected_box["half_width"]), 0.0001)
		t.assert_almost_eq(float(drawn["half_height"]), float(expected_box["half_height"]), 0.0001)
	# And the same box reaches the vector: the overlay reads it from the
	# belief, the observation measures it from ground truth on this level, and
	# both go through one writer.
	var obs = env.get_observations()
	t.assert_almost_eq(obs.primary_enemy_screen_x, float(boxes[0]["center_x"]), 0.0001)
	t.assert_almost_eq(obs.primary_enemy_screen_y, float(boxes[0]["center_y"]), 0.0001)
	t.assert_almost_eq(obs.primary_enemy_screen_half_width, float(boxes[0]["half_width"]), 0.0001)
	t.assert_almost_eq(obs.primary_enemy_screen_half_height, float(boxes[0]["half_height"]), 0.0001)
	return t

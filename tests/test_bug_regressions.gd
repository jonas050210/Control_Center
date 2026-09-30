## Regression tests for the bugs fixed in the audit pass. Every test here
## fails against the previous implementation and documents the exact
## defect it guards.
class_name TestBugRegressions
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentState = preload("res://scripts/agent/agent_state.gd")
const AIStubController = preload("res://scripts/input/ai_stub_controller.gd")
const ControlCenterConfig = preload("res://scripts/control_center/control_center_config.gd")
const ControlCenterResults = preload("res://scripts/control_center/control_center_results.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyBrain = preload("res://scripts/enemy/enemy_brain.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const EpisodeState = preload("res://scripts/core/episode_state.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const ReplayPlayer = preload("res://scripts/replay/replay_player.gd")
const ReplayRecorder = preload("res://scripts/replay/replay_recorder.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


## Deterministic stand-in for RandomNumberGenerator whose randf() always
## fails a hit roll, so a test can force the ranged miss branch without
## depending on any particular engine RNG sequence.
class MissRng:
	extends RefCounted

	func randf() -> float:
		return 0.999999

	func randf_range(from: float, _to: float) -> float:
		return from

	func randi_range(from: int, _to: int) -> int:
		return from


## Bug: SimulationManager.build() called _clear(), which emptied
## `controllers`, and nothing re-attached them. Changing the enemy count
## from the debug overlay therefore detached the HumanController installed
## by main.gd and the player silently lost all input.
func test_rebuilding_environments_preserves_controllers() -> SandboxTest:
	var t := SandboxTest.new("rebuilding_environments_preserves_controllers")
	var manager := SimulationManager.new()
	manager.create_visuals = false
	manager.build(3, 1)
	var controller := AIStubController.new()
	manager.set_controller(1, controller)
	t.assert_eq(manager.controllers[1], controller)

	manager.build(3, 2)
	t.assert_eq(
		manager.controllers[1], controller, "a rebuild must not detach an installed controller"
	)
	t.assert_eq(manager.controllers.size(), 3)

	# Shrinking the environment count drops the controllers that no longer
	# have an environment, but keeps the surviving ones.
	manager.build(2, 2)
	t.assert_eq(manager.controllers.size(), 2)
	t.assert_eq(manager.controllers[1], controller)
	manager.free()
	# AIStubController is a Node, not a RefCounted: it was created outside
	# the SceneTree and never parented, so nothing else can free it. Without
	# this the controller (and the script resource it holds) survived the
	# whole run and was reported as a leaked ObjectDB instance at exit.
	controller.free()
	return t


## Bug: a timed-out episode was reported as `loss = done and not won`, so
## surviving the step limit was indistinguishable from being killed and
## `win_rate + loss_rate` was always exactly 1.0.
func test_timeout_is_reported_as_truncation_not_defeat() -> SandboxTest:
	var t := SandboxTest.new("timeout_is_reported_as_truncation_not_defeat")
	var timeout := EpisodeState.new()
	timeout.start_new_episode()
	timeout.mark_done("timeout")
	var timeout_metrics: Dictionary = timeout.to_metrics(SandboxConfig.SIMULATION_DT, 1, false)
	t.assert_false(bool(timeout_metrics["loss"]))
	t.assert_true(bool(timeout_metrics["truncated"]))

	var death := EpisodeState.new()
	death.start_new_episode()
	death.mark_done("agent_died")
	var death_metrics: Dictionary = death.to_metrics(SandboxConfig.SIMULATION_DT, 1, false)
	t.assert_true(bool(death_metrics["loss"]), "being killed is still a loss")
	t.assert_false(bool(death_metrics["truncated"]))

	var victory := EpisodeState.new()
	victory.start_new_episode()
	victory.mark_done("all_enemies_eliminated")
	var victory_metrics: Dictionary = victory.to_metrics(SandboxConfig.SIMULATION_DT, 1, true)
	t.assert_true(bool(victory_metrics["win"]))
	t.assert_false(bool(victory_metrics["loss"]))
	t.assert_false(bool(victory_metrics["truncated"]))
	return t


## Bug: EnvironmentCore.step() resolved the shoot branch without checking
## `agent.alive`, so an agent killed earlier in the same tick could still
## fire and score kills.
func test_dead_agent_cannot_fire() -> SandboxTest:
	var t := SandboxTest.new("dead_agent_cannot_fire")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	env.reset(5)
	env.agent.alive = false
	env.agent.health = 0.0
	var enemy_health_before: float = env.enemies[0].health
	env.step(Action.new(0, 0, 0, 0, true))
	t.assert_almost_eq(env.enemies[0].health, enemy_health_before)
	t.assert_eq(env.episode.shots_fired, 0)
	return t


## Bug: the seeded reset path did not exist for the Control Center's
## "Reset (random)" button. This asserts the underlying guarantee the fix
## relies on: resetting with an explicit seed always reproduces the same
## episode, and a different seed produces a different one.
func test_reset_with_an_explicit_seed_is_reproducible() -> SandboxTest:
	var t := SandboxTest.new("reset_with_an_explicit_seed_is_reproducible")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)

	env.reset(4242)
	var first: Array = []
	for enemy in env.enemies:
		first.append(enemy.position)

	env.reset(4242)
	for index in range(env.enemies.size()):
		t.assert_vec_almost_eq(env.enemies[index].position, first[index], 0.0001)

	env.reset(9999)
	var identical: bool = true
	for index in range(env.enemies.size()):
		if env.enemies[index].position.distance_to(first[index]) > 0.01:
			identical = false
			break
	t.assert_false(identical, "a different seed must produce a different layout")
	t.assert_eq(env.episode_seed, 9999)
	return t


## Bug: mid-episode curriculum changes only updated the enemy radius,
## leaving speed/cooldown at the previous level's values. Retained as a
## regression guard now that the reaction archetype is applied there too.
func test_curriculum_change_applies_every_enemy_parameter() -> SandboxTest:
	var t := SandboxTest.new("curriculum_change_applies_every_enemy_parameter")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	env.reset(3)
	env.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	var config := CurriculumConfig.new(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	for enemy_value in env.enemies:
		var enemy = enemy_value
		t.assert_almost_eq(
			enemy.move_speed, SandboxConfig.ENEMY_MOVE_SPEED * config.enemy_speed_scale(), 0.001
		)
		t.assert_almost_eq(
			enemy.attack_cooldown_time,
			SandboxConfig.ENEMY_ATTACK_COOLDOWN * config.enemy_cooldown_scale(),
			0.001
		)
		t.assert_eq(enemy.reaction.archetype, config.enemy_archetype())
	return t


## Bug: the action contract grew a `jump` component. Old five-value
## MultiDiscrete actions and old seven-value recorded logs must keep
## working, otherwise every existing checkpoint and demonstration breaks.
func test_legacy_action_encodings_still_decode() -> SandboxTest:
	var t := SandboxTest.new("legacy_action_encodings_still_decode")
	var legacy: Action = Action.from_multidiscrete([2, 0, 1, 1, 1])
	t.assert_eq(legacy.move_axis, 1)
	t.assert_eq(legacy.strafe_axis, -1)
	t.assert_true(legacy.shoot)
	t.assert_false(legacy.jump, "a v1 action must never jump")

	var v2: Action = Action.from_multidiscrete([1, 1, 1, 1, 0, 1])
	t.assert_true(v2.jump)
	t.assert_false(v2.shoot)

	var jump_discrete: Action = Action.from_discrete(Action.Discrete.JUMP)
	t.assert_true(jump_discrete.jump)
	t.assert_eq(Action.MULTI_DISCRETE_SIZE, 6)
	t.assert_eq(Action.idle().to_array().size(), Action.LOG_ARRAY_SIZE)
	return t


## Bug: a MISSED ranged shot at point-blank range fell through to the melee
## check, so the same tick dealt ENEMY_ATTACK_DAMAGE on top of the missed
## shot — a miss dealt MORE damage than a hit (10 vs 9), a hit dealt no
## melee, and a weapon-cooldown tick dealt nothing. A fired shot must end
## the attack for that tick; melee belongs to enemies without ranged
## capability (`allow_ranged` false), per the curriculum contract
## "enemies shoot instead of meleeing".
func test_missed_point_blank_shot_does_not_also_melee() -> SandboxTest:
	var t := SandboxTest.new("missed_point_blank_shot_does_not_also_melee")
	var enemy: EnemyState = EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -1.5))  # inside the 2 m melee range
	enemy.state_time = 10.0  # past the REGULAR archetype reaction delays
	var context: Dictionary = {
		"allow_attack": true,
		"agent_alive": true,
		"agent_position": Vector3.ZERO,
		"allow_ranged": true,
		"world": null,  # no geometry: line of sight is clear
		"rng": MissRng.new(),
	}
	var events: Dictionary = {}
	EnemyBrain._try_attack(enemy, context, events)
	t.assert_true(bool(events.get("shot", false)), "the ranged attempt must fire")
	t.assert_false(bool(events.get("hit", false)), "the rigged roll must miss")
	t.assert_almost_eq(
		float(events.get("damage", 0.0)),
		0.0,
		0.0001,
		"a missed point-blank shot must not convert into melee damage"
	)
	t.assert_almost_eq(
		enemy.attack_cooldown_remaining,
		0.0,
		0.0001,
		"the melee cooldown must stay untouched when a shot was fired"
	)

	# The same enemy without ranged capability must still melee in range.
	var melee_events: Dictionary = {}
	context["allow_ranged"] = false
	EnemyBrain._try_attack(enemy, context, melee_events)
	t.assert_true(bool(melee_events.get("hit", false)), "a non-ranged enemy in range attacks")
	t.assert_almost_eq(
		float(melee_events.get("damage", 0.0)),
		SandboxConfig.ENEMY_ATTACK_DAMAGE,
		0.0001,
		"melee damage must stay available when ranged is disabled"
	)
	return t


## Bug (Godot 4.7 compile cascade): `const X: PackedStringArray =
## PackedStringArray([...])` is not a constant expression, so
## control_center_config.gd, control_center_results.gd and
## perception_model.gd all failed to compile and every test that preloaded
## them failed with them. Reading the constants proves they still exist,
## still hold the same entries and are still usable as lookup tables.
func test_string_list_constants_are_constant_expressions() -> SandboxTest:
	var t := SandboxTest.new("string_list_constants_are_constant_expressions")
	t.assert_true(ControlCenterConfig.REBUILD_SETTINGS.has("environment_count"))
	t.assert_true(ControlCenterConfig.REBUILD_SETTINGS.has("enemy_count"))
	t.assert_true(ControlCenterConfig.REBUILD_SETTINGS.has("seed"))
	t.assert_false(ControlCenterConfig.REBUILD_SETTINGS.has("curriculum_level"))
	t.assert_true(ControlCenterResults.AVERAGED_KEYS.has("reward"))
	t.assert_eq(PerceptionModel.SLOT_LABELS.size(), 3)
	t.assert_eq(str(PerceptionModel.SLOT_LABELS[0]), "primary")
	return t


## Bug: `AgentState` stores aim as yaw/pitch and has no writable `forward`
## property; tests that assigned one raised "Invalid assignment of property
## 'forward'". `set_forward_horizontal()` is the supported inverse of
## `get_forward_horizontal()` and must round-trip.
func test_set_forward_horizontal_round_trips() -> SandboxTest:
	var t := SandboxTest.new("set_forward_horizontal_round_trips")
	var agent := AgentState.new()
	for direction in [
		Vector3(0.0, 0.0, 1.0),
		Vector3(0.0, 0.0, -1.0),
		Vector3(1.0, 0.0, 0.0),
		Vector3(1.0, 0.0, 1.0)
	]:
		agent.set_forward_horizontal(direction)
		t.assert_vec_almost_eq((direction as Vector3).normalized(), agent.get_forward_horizontal())
	# A degenerate direction must not move the aim.
	var yaw_before: float = agent.yaw_deg
	agent.set_forward_horizontal(Vector3(0.0, 1.0, 0.0))
	t.assert_almost_eq(agent.yaw_deg, yaw_before, 0.0001)
	return t


## Bug: `seek_time()` floored `seconds / (1.0 / 60.0)`, and 0.5 / (1/60)
## evaluates to 29.999999999999996 in binary floating point, so seeking to
## half a second landed on tick 29 instead of 30.
func test_replay_seek_time_is_not_off_by_one() -> SandboxTest:
	var t := SandboxTest.new("replay_seek_time_is_not_off_by_one")
	var recorder := ReplayRecorder.new({"map_id": "compound", "simulation_dt": 1.0 / 60.0})
	recorder.start(5)
	for index in range(120):
		recorder.record_step(Action.idle(), 0.0, null, index == 119)
	var player := ReplayPlayer.new(recorder.finish({"done_reason": "done"}))
	t.assert_eq(player.seek_time(0.0), 0)
	t.assert_eq(player.seek_time(0.5), 30)
	t.assert_eq(player.seek_time(1.0), 60)
	t.assert_eq(player.seek_time(1.5), 90)
	t.assert_eq(player.seek_time(-1.0), 0)
	# Mid-tick times must still floor to the tick that contains them.
	t.assert_eq(player.seek_time(0.5 + (1.0 / 120.0)), 30)
	return t


## Bug: `_try_attack()` read `events["damage"]` directly, so a caller that
## passed a fresh dictionary (the documented "events out" contract) crashed
## with "Invalid access to key 'damage'", and it typed the injected RNG as
## `RandomNumberGenerator`, which rejected deterministic test stubs.
func test_try_attack_accepts_empty_events_and_stub_rng() -> SandboxTest:
	var t := SandboxTest.new("try_attack_accepts_empty_events_and_stub_rng")
	var enemy: EnemyState = EnemyState.new()
	enemy.reset(Vector3(0.0, 0.0, -1.5))
	enemy.state_time = 10.0
	var context: Dictionary = {
		"allow_attack": true,
		"agent_alive": true,
		"agent_position": Vector3.ZERO,
		"allow_ranged": false,
		"world": null,
		"rng": MissRng.new(),
	}
	var events: Dictionary = {}
	EnemyBrain._try_attack(enemy, context, events)
	t.assert_true(bool(events.get("hit", false)))
	t.assert_almost_eq(float(events.get("damage", 0.0)), SandboxConfig.ENEMY_ATTACK_DAMAGE, 0.0001)
	return t


## Bug: EnvironmentCore claimed navigation was lazy, but tactical setup
## called `_ensure_navigation()` unconditionally on the first step. Every
## level-5+ environment therefore paid an A* grid bake even when direct
## steering worked for the whole episode.
func test_tactical_navigation_bakes_only_after_an_enemy_is_stuck() -> SandboxTest:
	var t := SandboxTest.new("tactical_navigation_is_actually_lazy")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	env.reset(123)
	t.assert_eq(env.navigation, null)

	env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	t.assert_eq(env.navigation, null, "an ordinary direct-steering tick must not bake a graph")

	var enemy: EnemyState = env.enemies[0]
	enemy.navigation.stuck_time = SandboxConfig.NAV_STUCK_TIME
	env.step(Action.idle(), SandboxConfig.SIMULATION_DT)
	t.assert_true(env.navigation != null, "stuck evidence must trigger the one shared graph bake")
	return t


## Bug: `agent_panel.gd` declared a private helper called `_set`, which is
## the engine's `Object::_set(StringName, Variant) -> bool` virtual. Godot
## 4.7 validates virtual signatures at compile time and rejected the whole
## script. No script may re-declare an engine virtual with a different
## signature.
func test_no_script_redeclares_engine_virtuals() -> SandboxTest:
	var t := SandboxTest.new("no_script_redeclares_engine_virtuals")
	var offenders: Array = []
	for path in _all_script_paths("res://scripts"):
		var source: String = FileAccess.get_file_as_string(path)
		for line in source.split("\n"):
			var text: String = str(line)
			if text.begins_with("func _set(") or text.begins_with("func _get("):
				offenders.append(path)
	t.assert_eq(offenders.size(), 0, "engine virtual re-declared in: %s" % str(offenders))
	return t


func _all_script_paths(root: String) -> Array:
	var found: Array = []
	var directory := DirAccess.open(root)
	if directory == null:
		return found
	directory.list_dir_begin()
	var entry: String = directory.get_next()
	while entry != "":
		var path: String = "%s/%s" % [root, entry]
		if directory.current_is_dir():
			found.append_array(_all_script_paths(path))
		elif entry.ends_with(".gd"):
			found.append(path)
		entry = directory.get_next()
	directory.list_dir_end()
	found.sort()
	return found

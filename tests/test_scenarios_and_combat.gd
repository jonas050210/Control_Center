## Tests for the scenario library, the curriculum extension, the automatic
## curriculum controller, corpse exclusion, multi-enemy target selection and
## the end-to-end environment behaviour on the world-backed levels.
class_name TestScenariosAndCombat
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentPerception = preload("res://scripts/perception/agent_perception.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const CurriculumController = preload("res://scripts/core/curriculum_controller.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const ScenarioLibrary = preload("res://scripts/scenario/scenario_library.gd")

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


func test_all_sixteen_scenarios_resolve() -> SandboxTest:
	var t := SandboxTest.new("all_sixteen_scenarios_resolve")
	var ids: PackedStringArray = ScenarioLibrary.ids()
	t.assert_eq(ids.size(), 16, "the library must ship all sixteen scenarios")
	t.assert_eq(
		ScenarioLibrary.training_ids().size(), 12, "twelve scenarios belong to the regular training pool"
	)
	for scenario_id in ids:
		var resolved: Dictionary = ScenarioLibrary.resolve(scenario_id, 17)
		t.assert_eq(str(resolved["id"]), scenario_id)
		t.assert_not_null(resolved["world"])
		t.assert_gt(
			float((resolved["enemy_spawns"] as Array).size()),
			0.0,
			"%s must place at least one enemy" % scenario_id
		)
		for spawn_value in resolved["enemy_spawns"]:
			var spawn: Vector3 = spawn_value
			t.assert_false(is_nan(spawn.x) or is_nan(spawn.z), "%s produced NaN" % scenario_id)
	return t


func test_same_seed_produces_the_same_world_and_spawns() -> SandboxTest:
	var t := SandboxTest.new("same_seed_produces_the_same_world_and_spawns")
	for scenario_id in ScenarioLibrary.ids():
		var a: Dictionary = ScenarioLibrary.resolve(scenario_id, 5150)
		var b: Dictionary = ScenarioLibrary.resolve(scenario_id, 5150)
		t.assert_vec_almost_eq(a["agent_spawn"], b["agent_spawn"], 0.0001, scenario_id)
		var spawns_a: Array = a["enemy_spawns"]
		var spawns_b: Array = b["enemy_spawns"]
		t.assert_eq(spawns_a.size(), spawns_b.size(), scenario_id)
		for index in range(spawns_a.size()):
			t.assert_vec_almost_eq(spawns_a[index], spawns_b[index], 0.0001, scenario_id)
		t.assert_eq(
			(a["world"] as Object).call("obstacle_count"),
			(b["world"] as Object).call("obstacle_count"),
			scenario_id
		)
	return t


func test_occluded_scenarios_actually_start_without_visual_contact() -> SandboxTest:
	var t := SandboxTest.new("occluded_scenarios_actually_start_without_visual_contact")
	var hidden_starts: int = 0
	for seed_value in range(12):
		var resolved: Dictionary = ScenarioLibrary.resolve("corner_fight", 900 + seed_value)
		var agent_eye: Vector3 = (
			resolved["agent_spawn"] + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0)
		)
		var spawns: Array = resolved["enemy_spawns"]
		if not PerceptionSystem.has_line_of_sight(
			resolved["world"], agent_eye, spawns[0], SandboxConfig.AGENT_HEIGHT
		):
			hidden_starts += 1
	t.assert_gte(
		float(hidden_starts),
		9.0,
		"the corner scenario must nearly always start out of sight (got %d/12)" % hidden_starts
	)
	return t


func test_unknown_scenario_falls_back_instead_of_crashing() -> SandboxTest:
	var t := SandboxTest.new("unknown_scenario_falls_back_instead_of_crashing")
	var resolved: Dictionary = ScenarioLibrary.resolve("not_a_scenario", 3)
	t.assert_eq(str(resolved["id"]), "open_arena")
	t.assert_false(ScenarioLibrary.has_scenario("not_a_scenario"))
	t.assert_true(ScenarioLibrary.has_scenario("ambush"))
	return t


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


func test_curriculum_capability_flags_turn_on_in_order() -> SandboxTest:
	var t := SandboxTest.new("curriculum_capability_flags_turn_on_in_order")
	var legacy := CurriculumConfig.new(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	t.assert_false(legacy.obstacles_enabled(), "levels 1-4 must stay obstacle-free")
	t.assert_false(legacy.perception_enabled())
	t.assert_false(legacy.sound_enabled())
	t.assert_false(legacy.memory_enabled())
	t.assert_false(legacy.vertical_enabled())

	var cover := CurriculumConfig.new(CurriculumConfig.Level.OBSTACLES_COVER)
	t.assert_true(cover.obstacles_enabled())
	t.assert_true(cover.ranged_enemies_enabled())
	t.assert_false(cover.perception_enabled(), "perception gating starts one level later")

	t.assert_true(CurriculumConfig.new(CurriculumConfig.Level.FOV_LOS).perception_enabled())
	t.assert_false(CurriculumConfig.new(CurriculumConfig.Level.FOV_LOS).sound_enabled())
	t.assert_true(CurriculumConfig.new(CurriculumConfig.Level.SOUND).sound_enabled())
	t.assert_false(CurriculumConfig.new(CurriculumConfig.Level.SOUND).memory_enabled())
	t.assert_true(CurriculumConfig.new(CurriculumConfig.Level.MEMORY_LOST_TARGETS).memory_enabled())
	t.assert_true(CurriculumConfig.new(CurriculumConfig.Level.VERTICAL_COMBAT).vertical_enabled())
	t.assert_true(
		CurriculumConfig.new(CurriculumConfig.Level.MIXED_RANDOMIZED).randomized_scenarios_enabled()
	)

	var self_play := CurriculumConfig.new(CurriculumConfig.Level.AGENT_VS_AGENT)
	t.assert_false(self_play.obstacles_enabled(), "self-play uses its own environment")
	t.assert_false(self_play.perception_enabled())
	return t


func test_every_level_has_a_name_and_a_layout() -> SandboxTest:
	var t := SandboxTest.new("every_level_has_a_name_and_a_layout")
	for level in range(
		CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.Level.AGENT_VS_AGENT + 1
	):
		var config := CurriculumConfig.new(level)
		t.assert_ne(CurriculumConfig.level_name(level), "unknown", "level %d" % level)
		t.assert_false(config.layout_id().is_empty(), "level %d" % level)
	return t


func test_automatic_curriculum_controller_promotes_and_demotes() -> SandboxTest:
	var t := SandboxTest.new("automatic_curriculum_controller_promotes_and_demotes")
	var controller: CurriculumController = CurriculumController.create(1, true)
	controller.configure(10, 0.7, 0.2, 10)

	# Below the cooldown nothing can change.
	for _i in range(9):
		controller.record_outcome(true)
	t.assert_eq(controller.level, 1, "a level must not change before the window is full")
	controller.record_outcome(true)
	t.assert_eq(controller.level, 2, "10/10 wins must promote")
	t.assert_almost_eq(float(controller.promotions), 1.0)

	for _i in range(10):
		controller.record_outcome(false)
	t.assert_eq(controller.level, 1, "0/10 wins must demote")
	t.assert_almost_eq(float(controller.demotions), 1.0)

	# Mediocre performance holds the level steady.
	for _i in range(10):
		controller.record_outcome(controller.episodes_observed % 2 == 0)
	t.assert_eq(controller.level, 1)
	return t


func test_disabled_controller_never_changes_the_level() -> SandboxTest:
	var t := SandboxTest.new("disabled_controller_never_changes_the_level")
	var controller: CurriculumController = CurriculumController.create(3, false)
	controller.configure(4, 0.5, 0.1, 0)
	for _i in range(40):
		controller.record_outcome(true)
	t.assert_eq(controller.level, 3)
	t.assert_gt(controller.success_rate(), 0.9, "statistics are still collected while disabled")
	return t


func test_controller_respects_min_and_max_bounds() -> SandboxTest:
	var t := SandboxTest.new("controller_respects_min_and_max_bounds")
	var controller: CurriculumController = CurriculumController.create(
		CurriculumConfig.MAX_COMBAT_LEVEL, true
	)
	controller.configure(4, 0.5, 0.1, 4)
	for _i in range(40):
		controller.record_outcome(true)
	t.assert_eq(controller.level, CurriculumConfig.MAX_COMBAT_LEVEL, "must not exceed max_level")
	return t


# ---------------------------------------------------------------------------
# Corpses and target selection
# ---------------------------------------------------------------------------


func test_a_dead_enemy_becomes_a_permanent_non_targetable_corpse() -> SandboxTest:
	var t := SandboxTest.new("a_dead_enemy_becomes_a_permanent_non_targetable_corpse")
	var enemy := EnemyState.new()
	enemy.enemy_id = 2
	enemy.reset(Vector3(1.0, 0.0, 1.0))
	t.assert_true(enemy.is_targetable())
	enemy.take_damage(enemy.max_health * 2.0)
	t.assert_false(enemy.alive)
	t.assert_true(enemy.corpse)
	t.assert_false(enemy.is_targetable(), "a corpse must never be targetable")
	t.assert_eq(enemy.ai_state, EnemyState.AIState.DEAD)
	t.assert_eq(int(enemy.memory.size()), 0, "a corpse remembers nothing")

	var body: Dictionary = enemy.to_corpse_dict()
	t.assert_eq(int(body["id"]), 2)
	t.assert_false(bool(body["targetable"]))
	t.assert_vec_almost_eq(body["position"], Vector3(1.0, 0.0, 1.0))

	# Healing a corpse must not resurrect it.
	enemy.take_damage(1.0)
	t.assert_true(enemy.corpse)
	return t


func test_target_selection_prefers_visible_then_recent_damage_then_distance() -> SandboxTest:
	var t := SandboxTest.new("target_selection_prefers_visible_then_recent_damage_then_distance")
	var far_visible: Dictionary = {
		"id": 0, "visible": true, "distance": 18.0, "confidence": 1.0, "age": 0.0
	}
	var near_remembered: Dictionary = {
		"id": 1, "visible": false, "distance": 2.0, "confidence": 0.5, "age": 1.0
	}
	var ranked: Array = AgentPerception.rank_beliefs([near_remembered, far_visible])
	t.assert_eq(
		int((ranked[0] as Dictionary)["id"]), 0, "a visible enemy outranks a remembered one"
	)

	var visible_a: Dictionary = {
		"id": 0, "visible": true, "distance": 4.0, "confidence": 1.0, "age": 0.0
	}
	var visible_b: Dictionary = {
		"id": 1, "visible": true, "distance": 9.0, "confidence": 1.0, "age": 0.0
	}
	var by_distance: Array = AgentPerception.rank_beliefs([visible_b, visible_a])
	t.assert_eq(int((by_distance[0] as Dictionary)["id"]), 0, "the nearer visible enemy wins")

	var by_damage: Array = AgentPerception.rank_beliefs([visible_a, visible_b], 1)
	t.assert_eq(
		int((by_damage[0] as Dictionary)["id"]),
		1,
		"the enemy that just damaged the agent is prioritised"
	)
	t.assert_true(
		AgentPerception.selection_reason(by_damage[0], 1).contains("damaged"),
		"the Control Center must get a human-readable reason"
	)
	t.assert_eq(AgentPerception.selection_reason({}, -1), "no target")
	return t


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------


func test_legacy_levels_remain_obstacle_free_and_ungated() -> SandboxTest:
	var t := SandboxTest.new("legacy_levels_remain_obstacle_free_and_ungated")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.ENEMY_ATTACKS)
	env.reset(1234)
	t.assert_null(env.world, "levels 1-4 must not build a world")
	t.assert_eq(env.get_obstacles().size(), 0)
	var observation: Observation = env.get_observations()
	t.assert_true(
		observation.primary_enemy_visible,
		"without perception gating the enemy is reported as visible"
	)
	for _i in range(60):
		env.step(Action.idle())
	t.assert_eq(env.get_sound_events().size(), 0, "sound is off below the sound level")
	return t


func test_world_levels_build_geometry_and_expose_debug_hooks() -> SandboxTest:
	var t := SandboxTest.new("world_levels_build_geometry_and_expose_debug_hooks")
	var env := EnvironmentCore.new(0, 2)
	env.set_curriculum_level(CurriculumConfig.Level.MEMORY_LOST_TARGETS)
	env.reset(77)
	t.assert_not_null(env.world)
	t.assert_gt(float(env.get_obstacles().size()), 0.0)

	for hook in [
		"get_agent_field_of_view",
		"has_line_of_sight",
		"get_sound_events",
		"get_target_memory",
		"get_obstacles",
		"get_navigation_state",
		"get_dead_bodies"
	]:
		t.assert_true(env.has_method(hook), "Control Center hook %s must exist" % hook)

	var fov: Dictionary = env.get_agent_field_of_view()
	t.assert_true(bool(fov["enabled"]))
	t.assert_almost_eq(float(fov["fov_deg"]), SandboxConfig.AGENT_FOV_DEG)

	var navigation: Dictionary = env.get_navigation_state()
	t.assert_ne(str(navigation["layout_id"]), "none")
	t.assert_eq((navigation["enemies"] as Array).size(), env.enemies.size())
	t.assert_eq(env.get_dead_bodies().size(), 0)
	return t


func test_environment_is_deterministic_for_a_seed_on_a_world_level() -> SandboxTest:
	var t := SandboxTest.new("environment_is_deterministic_for_a_seed_on_a_world_level")
	var traces: Array = []
	for _run in range(2):
		var env := EnvironmentCore.new(0, 2)
		env.set_curriculum_level(CurriculumConfig.Level.VERTICAL_COMBAT)
		env.reset(31337)
		var trace: Array = []
		for step_index in range(120):
			var action: Action = Action.new(
				1, 0, 0, 0, step_index % 7 == 0, Vector2.ZERO, step_index % 31 == 0
			)
			var result: Dictionary = env.step(action)
			trace.append([env.agent.position, float(result["reward"]), env.enemies[0].position])
		traces.append(trace)
	var left: Array = traces[0]
	var right: Array = traces[1]
	for index in range(left.size()):
		t.assert_vec_almost_eq(left[index][0], right[index][0], 0.000001, "agent path diverged")
		t.assert_almost_eq(left[index][1], right[index][1], 0.000001, "reward diverged")
		t.assert_vec_almost_eq(left[index][2], right[index][2], 0.000001, "enemy path diverged")
	return t


func test_observation_stays_inside_declared_bounds_on_every_level() -> SandboxTest:
	var t := SandboxTest.new("observation_stays_inside_declared_bounds_on_every_level")
	for level in range(
		CurriculumConfig.Level.STATIONARY_TARGET, CurriculumConfig.MAX_COMBAT_LEVEL + 1
	):
		var env := EnvironmentCore.new(0, 3)
		env.set_curriculum_level(level)
		env.reset(4000 + level)
		for step_index in range(90):
			var action: Action = Action.new(
				1,
				(step_index % 3) - 1,
				1,
				0,
				step_index % 5 == 0,
				Vector2.ZERO,
				step_index % 23 == 0
			)
			env.step(action)
			var values: PackedFloat32Array = env.get_observations().to_array()
			t.assert_eq(values.size(), Observation.FIELD_COUNT, "level %d" % level)
			for index in range(values.size()):
				var value: float = values[index]
				t.assert_false(is_nan(value), "level %d field %d is NaN" % [level, index])
				t.assert_lte(value, 1.0001, "level %d field %d above 1.0" % [level, index])
				t.assert_gte(value, -1.0001, "level %d field %d below -1.0" % [level, index])
	return t


func test_dead_agent_cannot_shoot() -> SandboxTest:
	var t := SandboxTest.new("dead_agent_cannot_shoot")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.ENEMY_ATTACKS)
	env.reset(11)
	env.agent.health = 0.0
	env.agent.alive = false
	var before: int = env.episode.shots_fired
	var result: Dictionary = env.step(Action.new(0, 0, 0, 0, true))
	var events: Dictionary = result["info"]["events"]
	t.assert_eq(env.episode.shots_fired, before, "a dead agent must not register a shot")
	t.assert_false(bool(events["hit"]))
	t.assert_false(bool(events["shot_fired"]))
	return t


func test_corpses_are_excluded_from_observation_and_targeting() -> SandboxTest:
	var t := SandboxTest.new("corpses_are_excluded_from_observation_and_targeting")
	var env := EnvironmentCore.new(0, 3)
	env.set_curriculum_level(CurriculumConfig.Level.MULTIPLE_ENEMIES)
	env.reset(2024)
	var victim: EnemyState = env.get_primary_enemy()
	t.assert_not_null(victim)
	victim.take_damage(victim.max_health * 2.0)
	env.perception.forget(victim.enemy_id)
	env.step(Action.idle())
	t.assert_ne(env.get_primary_enemy(), victim, "a corpse must not be selected as primary")
	t.assert_eq(env.get_corpse_count(), 1)
	for enemy_value in env.enemies:
		var enemy: EnemyState = enemy_value
		if enemy.corpse:
			t.assert_false(enemy.is_targetable())
	return t


func test_timeout_is_not_counted_as_a_loss() -> SandboxTest:
	var t := SandboxTest.new("timeout_is_not_counted_as_a_loss")
	var env := EnvironmentCore.new(0, 1)
	env.set_curriculum_level(CurriculumConfig.Level.STATIONARY_TARGET)
	env.max_steps = 5
	env.reset(1)
	for _i in range(6):
		env.step(Action.idle())
	var metrics: Dictionary = env.get_metrics()
	t.assert_eq(str(metrics["done_reason"]), "timeout")
	t.assert_false(bool(metrics["win"]))
	t.assert_false(bool(metrics["loss"]), "a timeout is a truncation, not a defeat")
	t.assert_true(bool(metrics["truncated"]))
	return t

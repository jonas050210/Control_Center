## Regression tests for the bugs fixed in the audit pass. Every test here
## fails against the previous implementation and documents the exact
## defect it guards.
class_name TestBugRegressions
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AIStubController = preload("res://scripts/input/ai_stub_controller.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const EpisodeState = preload("res://scripts/core/episode_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


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

## Tests for the scripted tactical baselines (EnemyBrain).
##
## These enemies are BASELINES, not the research subject: they exist so a
## learning policy has a non-trivial, deterministic opponent. What is
## tested here is therefore not "is this good play" but "does each
## behaviour follow from information the enemy actually has":
## investigating a noise it never saw, breaking away when critically hurt,
## repositioning after standing in the open too long, and forgetting a
## contact that never materialises.
class_name TestTacticalBehavior
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const EnemyBrain = preload("res://scripts/enemy/enemy_brain.gd")
const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")

const DT: float = SandboxConfig.SIMULATION_DT


func _enemy(at: Vector3) -> EnemyState:
	var enemy := EnemyState.new()
	enemy.reset(at)
	enemy.enemy_id = 0
	return enemy


func _context(agent_position: Vector3, world = null) -> Dictionary:
	return {
		"world": world,
		"navigation": null,
		"sound_bus": null,
		"rng": null,
		"arena_half_extent": SandboxConfig.ARENA_HALF_EXTENT,
		"agent_position": agent_position,
		"agent_eye": agent_position + Vector3(0.0, SandboxConfig.AGENT_EYE_HEIGHT, 0.0),
		"agent_height": SandboxConfig.AGENT_HEIGHT,
		"agent_alive": true,
		"dt": DT,
		"allow_movement": true,
		"allow_ranged": true,
		"allow_attack": true,
	}


func _run(enemy: EnemyState, context: Dictionary, steps: int) -> void:
	for _i in range(steps):
		EnemyBrain.update(enemy, context)


func test_every_state_has_a_name() -> SandboxTest:
	var t := SandboxTest.new("every_state_has_a_name")
	for state in [
		EnemyState.AIState.IDLE,
		EnemyState.AIState.ALERT,
		EnemyState.AIState.ENGAGE,
		EnemyState.AIState.TAKE_COVER,
		EnemyState.AIState.PEEK,
		EnemyState.AIState.SEARCH,
		EnemyState.AIState.INVESTIGATE,
		EnemyState.AIState.RETREAT,
		EnemyState.AIState.DEAD,
	]:
		t.assert_ne(EnemyState.ai_state_name(state), "unknown")
	return t


func test_no_information_means_idle() -> SandboxTest:
	var t := SandboxTest.new("no_information_means_idle")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	# Agent far outside vision range and silent: the enemy must do nothing.
	_run(enemy, _context(Vector3(0.0, 0.0, -200.0)), 30)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.IDLE)
	t.assert_false(enemy.target_confirmed)
	t.assert_eq(int(enemy.memory.size()), 0, "no contact means no memory")
	return t


func test_visible_agent_is_engaged_after_the_reaction_delay() -> SandboxTest:
	var t := SandboxTest.new("visible_agent_is_engaged_after_the_reaction_delay")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	var context: Dictionary = _context(Vector3(0.0, 0.0, -6.0))
	EnemyBrain.update(enemy, context)
	t.assert_false(enemy.target_confirmed, "reaction time is not instant")
	_run(enemy, context, 60)
	t.assert_true(enemy.target_confirmed)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.ENGAGE)
	t.assert_gt(enemy.exposure_time, 0.0, "mutual line of sight is exposure")
	return t


func test_a_noise_it_never_saw_leads_to_investigation() -> SandboxTest:
	var t := SandboxTest.new("a_noise_it_never_saw_leads_to_investigation")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	# The agent is behind the enemy (outside its FOV) but makes noise.
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, 5.0), 99)
	bus.tick(0.3)
	var context: Dictionary = _context(Vector3(0.0, 0.0, 5.0))
	context["sound_bus"] = bus
	EnemyBrain.update(enemy, context)
	t.assert_eq(int(enemy.memory.size()), 1, "hearing creates a contact")
	var track: Dictionary = enemy.memory.get_track(EnemyBrain.AGENT_TRACK_ID)
	t.assert_eq(int(track["source"]), EnemyMemory.Source.SOUND)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.INVESTIGATE)
	t.assert_true(enemy.has_tactical_destination)
	t.assert_false(enemy.target_confirmed, "hearing never confirms a target")
	return t


func test_investigation_gives_up_and_forgets() -> SandboxTest:
	var t := SandboxTest.new("investigation_gives_up_and_forgets")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	var bus: SoundBus = SoundBus.create()
	bus.emit_sound(SoundBus.Category.SHOT, Vector3(0.0, 0.0, 6.0), 99)
	bus.tick(0.3)
	var context: Dictionary = _context(Vector3(0.0, 0.0, 400.0), null)
	context["sound_bus"] = bus
	EnemyBrain.update(enemy, context)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.INVESTIGATE)
	context["sound_bus"] = null
	_run(enemy, context, int(SandboxConfig.ENEMY_SEARCH_DURATION / DT) + 120)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.IDLE, "a noise that leads nowhere is dropped")
	t.assert_eq(int(enemy.memory.size()), 0)
	return t


func test_critical_health_triggers_a_retreat_away_from_the_threat() -> SandboxTest:
	var t := SandboxTest.new("critical_health_triggers_a_retreat_away_from_the_threat")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	var context: Dictionary = _context(Vector3(0.0, 0.0, -6.0))
	_run(enemy, context, 60)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.ENGAGE)

	enemy.health = enemy.max_health * (SandboxConfig.ENEMY_CRITICAL_HEALTH_FRACTION * 0.5)
	EnemyBrain.update(enemy, context)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.RETREAT)
	t.assert_true(enemy.has_tactical_destination)
	t.assert_gt(
		enemy.tactical_destination.z,
		enemy.position.z,
		"the retreat destination must be on the far side from the threat"
	)
	var start_distance: float = enemy.position.distance_to(Vector3(0.0, 0.0, -6.0))
	_run(enemy, context, 60)
	t.assert_gt(
		enemy.position.distance_to(Vector3(0.0, 0.0, -6.0)),
		start_distance,
		"a retreating enemy must actually open the distance"
	)
	return t


func test_standing_in_the_open_too_long_forces_a_reposition() -> SandboxTest:
	var t := SandboxTest.new("standing_in_the_open_too_long_forces_a_reposition")
	var world: ArenaWorld = ArenaWorld.create(SandboxConfig.ARENA_HALF_EXTENT, 3.0)
	world.add_obstacle(
		Obstacle.make(Vector3(3.0, 1.0, 0.0), Vector3(1.0, 1.0, 1.0), Obstacle.Kind.CRATE)
	)
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	var context: Dictionary = _context(Vector3(0.0, 0.0, -6.0), world)
	# Healthy, so neither the low-health nor the critical-health rule can
	# fire: the only thing that can move it is the exposure clock.
	var steps: int = int(SandboxConfig.ENEMY_MAX_EXPOSURE_TIME / DT) + 30
	for _i in range(steps):
		enemy.health = enemy.max_health
		EnemyBrain.update(enemy, context)
	t.assert_ne(enemy.ai_state, EnemyState.AIState.ENGAGE, "it must not stand still forever")
	t.assert_true(
		enemy.tactical_reason.contains("exposed") or enemy.ai_state == EnemyState.AIState.PEEK,
		"the reason must name the exposure, got: %s" % enemy.tactical_reason
	)
	return t


func test_losing_sight_stops_the_exposure_clock() -> SandboxTest:
	var t := SandboxTest.new("losing_sight_stops_the_exposure_clock")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	var context: Dictionary = _context(Vector3(0.0, 0.0, -6.0))
	_run(enemy, context, 60)
	t.assert_gt(enemy.exposure_time, 0.0)
	context["agent_position"] = Vector3(0.0, 0.0, 400.0)
	EnemyBrain.update(enemy, context)
	t.assert_almost_eq(enemy.exposure_time, 0.0, 0.0001)
	return t


func test_a_dead_enemy_does_nothing() -> SandboxTest:
	var t := SandboxTest.new("a_dead_enemy_does_nothing")
	var enemy: EnemyState = _enemy(Vector3(0.0, 0.0, 0.0))
	var context: Dictionary = _context(Vector3(0.0, 0.0, -4.0))
	_run(enemy, context, 30)
	enemy.take_damage(enemy.max_health * 2.0)
	var resting: Vector3 = enemy.position
	var events: Dictionary = EnemyBrain.update(enemy, context)
	t.assert_eq(enemy.ai_state, EnemyState.AIState.DEAD)
	t.assert_almost_eq(float(events["damage"]), 0.0, 0.0001)
	t.assert_vec_almost_eq(enemy.position, resting, 0.0001)
	return t


func test_behaviour_is_deterministic_for_identical_input() -> SandboxTest:
	var t := SandboxTest.new("behaviour_is_deterministic_for_identical_input")
	var positions: Array = []
	var states: Array = []
	for _run_index in range(2):
		var enemy: EnemyState = _enemy(Vector3(1.0, 0.0, 2.0))
		var context: Dictionary = _context(Vector3(0.0, 0.0, -5.0))
		for step in range(180):
			context["agent_position"] = Vector3(
				sin(float(step) * 0.05) * 4.0, 0.0, -5.0 + cos(float(step) * 0.05)
			)
			EnemyBrain.update(enemy, context)
		positions.append(enemy.position)
		states.append(enemy.ai_state)
	t.assert_vec_almost_eq(positions[0], positions[1], 0.000001)
	t.assert_eq(states[0], states[1])
	return t

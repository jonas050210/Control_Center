## Tests for EnemyMemory: last-known positions, information source,
## confidence decay, forgetting and capacity limits.
class_name TestMemory
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_visual_contact_stores_full_confidence_and_position() -> SandboxTest:
	var t := SandboxTest.new("visual_contact_stores_full_confidence_and_position")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_visual(0, Vector3(3.0, 0.0, -4.0), Vector3(0.0, 0.0, -1.0), 0.6)
	var track: Dictionary = memory.get_track(0)
	t.assert_vec_almost_eq(track["position"], Vector3(3.0, 0.0, -4.0))
	t.assert_almost_eq(float(track["confidence"]), 1.0)
	t.assert_almost_eq(float(track["age"]), 0.0)
	t.assert_eq(int(track["source"]), EnemyMemory.Source.VISUAL)
	t.assert_almost_eq(float(track["health_norm"]), 0.6)
	t.assert_true(bool(track["seen_at_least_once"]))
	return t


func test_confidence_decays_with_the_configured_half_life() -> SandboxTest:
	var t := SandboxTest.new("confidence_decays_with_the_configured_half_life")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_visual(0, Vector3(1.0, 0.0, 0.0), Vector3.ZERO)
	var dt: float = 1.0 / 60.0
	var elapsed: float = 0.0
	while elapsed < SandboxConfig.MEMORY_HALF_LIFE:
		memory.tick(dt)
		elapsed += dt
	var track: Dictionary = memory.get_track(0)
	t.assert_almost_eq(
		float(track["confidence"]), 0.5, 0.02, "confidence must halve after one half-life"
	)
	t.assert_almost_eq(float(track["age"]), SandboxConfig.MEMORY_HALF_LIFE, 0.05)
	return t


func test_a_track_is_forgotten_once_confidence_collapses() -> SandboxTest:
	var t := SandboxTest.new("a_track_is_forgotten_once_confidence_collapses")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_visual(0, Vector3(1.0, 0.0, 0.0), Vector3.ZERO)
	for _i in range(60 * 60):
		memory.tick(1.0 / 60.0)
		if not memory.has(0):
			break
	t.assert_false(memory.has(0), "a decayed track must eventually be dropped entirely")
	t.assert_eq(memory.size(), 0)
	return t


func test_memory_position_does_not_follow_a_moving_target() -> SandboxTest:
	var t := SandboxTest.new("memory_position_does_not_follow_a_moving_target")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_visual(0, Vector3(0.0, 0.0, -5.0), Vector3(0.0, 0.0, -1.0))
	# Time passes; the real target moved far away but was never seen again.
	for _i in range(60):
		memory.tick(1.0 / 60.0)
	t.assert_vec_almost_eq(
		memory.get_track(0)["position"],
		Vector3(0.0, 0.0, -5.0),
		0.0001,
		"a remembered position must stay where it was last observed"
	)
	return t


func test_sound_contact_is_weaker_than_a_fresh_visual_fix() -> SandboxTest:
	var t := SandboxTest.new("sound_contact_is_weaker_than_a_fresh_visual_fix")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_sound(0, Vector3(2.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), 1.0)
	var heard: Dictionary = memory.get_track(0)
	t.assert_eq(int(heard["source"]), EnemyMemory.Source.SOUND)
	t.assert_lt(float(heard["confidence"]), 1.0, "hearing must never reach full confidence")

	# A fresh visual fix must not be downgraded by a subsequent noise.
	memory.observe_visual(1, Vector3(5.0, 0.0, 0.0), Vector3.ZERO)
	memory.observe_sound(1, Vector3(-9.0, 0.0, 0.0), Vector3(-1.0, 0.0, 0.0), 1.0)
	var seen: Dictionary = memory.get_track(1)
	t.assert_eq(int(seen["source"]), EnemyMemory.Source.VISUAL)
	t.assert_vec_almost_eq(seen["position"], Vector3(5.0, 0.0, 0.0))
	return t


func test_ranked_orders_by_confidence() -> SandboxTest:
	var t := SandboxTest.new("ranked_orders_by_confidence")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_sound(0, Vector3(1.0, 0.0, 0.0), Vector3.ZERO, 0.3)
	memory.observe_visual(1, Vector3(2.0, 0.0, 0.0), Vector3.ZERO)
	memory.observe_sound(2, Vector3(3.0, 0.0, 0.0), Vector3.ZERO, 0.9)
	var ranked: Array = memory.ranked()
	t.assert_eq(ranked.size(), 3)
	t.assert_eq(int((ranked[0] as Dictionary)["id"]), 1, "the visual fix must rank first")
	t.assert_eq(int((ranked[2] as Dictionary)["id"]), 0, "the faintest noise must rank last")
	return t


func test_capacity_is_enforced_by_dropping_the_least_confident() -> SandboxTest:
	var t := SandboxTest.new("capacity_is_enforced_by_dropping_the_least_confident")
	var memory: EnemyMemory = EnemyMemory.create()
	for index in range(SandboxConfig.MEMORY_MAX_TRACKS + 4):
		# Decreasing strength, so the later ones are the weakest.
		memory.observe_sound(
			index,
			Vector3(float(index), 0.0, 0.0),
			Vector3.ZERO,
			1.0 - float(index) * 0.05
		)
	t.assert_eq(memory.size(), SandboxConfig.MEMORY_MAX_TRACKS)
	t.assert_true(memory.has(0), "the strongest contact must survive")
	t.assert_false(
		memory.has(SandboxConfig.MEMORY_MAX_TRACKS + 3), "the weakest contact must be dropped"
	)
	return t


func test_forget_and_investigate_flags() -> SandboxTest:
	var t := SandboxTest.new("forget_and_investigate_flags")
	var memory: EnemyMemory = EnemyMemory.create()
	memory.observe_visual(4, Vector3(1.0, 0.0, 1.0), Vector3.ZERO)
	t.assert_false(bool(memory.get_track(4)["investigated"]))
	memory.mark_investigated(4)
	t.assert_true(bool(memory.get_track(4)["investigated"]))
	memory.forget(4)
	t.assert_false(memory.has(4), "a dead target's track must be removable immediately")
	t.assert_eq(EnemyMemory.source_name(EnemyMemory.Source.SOUND), "sound")
	return t

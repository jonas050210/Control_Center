## Tests for the weapon handling layer: fire modes, magazines and reloads,
## recoil, bloom, hit zones, and the reward scaling that depends on them.
##
## The single most important property here is **legacy equivalence**: with
## `handling_enabled == false` every new code path must collapse to the
## original cooldown-gated hitscan weapon, byte for byte. Curriculum levels
## 1-4 run in that mode and checkpoints trained before this layer existed
## must keep behaving identically, so that case is tested first and tested
## hardest.
class_name TestWeaponHandling
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of
## the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const AgentState = preload("res://scripts/agent/agent_state.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const RewardSystem = preload("res://scripts/reward/reward_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const WeaponState = preload("res://scripts/weapon/weapon_state.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func _armed(profile_id: String = "rifle") -> WeaponState:
	var weapon := WeaponState.new()
	weapon.configure_profile(profile_id)
	weapon.handling_enabled = true
	weapon.reset()
	return weapon


# -- legacy equivalence ----------------------------------------------------


func test_handling_disabled_is_the_legacy_weapon() -> SandboxTest:
	var t := SandboxTest.new("handling_disabled_is_the_legacy_weapon")
	var weapon := WeaponState.new()
	weapon.configure_profile("rifle")
	weapon.reset()
	t.assert_false(weapon.handling_enabled, "handling must be opt-in")

	# No spread, no movement tax, no reload, no head zone.
	t.assert_almost_eq(weapon.current_spread_deg(1.0, true), 0.0, 0.0001, "legacy has no bloom")
	t.assert_almost_eq(weapon.movement_speed_scale(), 1.0, 0.0001, "legacy never slows you")
	t.assert_false(weapon.is_reloading(), "legacy never reloads")
	t.assert_false(weapon.is_magazine_empty(), "legacy has no magazine")
	t.assert_false(weapon.begin_reload(), "legacy reload is a no-op")

	# tick_handling must be inert apart from the plain cooldown tick.
	weapon.cooldown_remaining = 0.25
	var recovered: Dictionary = weapon.tick_handling(0.1)
	t.assert_almost_eq(float(recovered["pitch_deg"]), 0.0, 0.0001, "legacy has no recoil recovery")
	t.assert_almost_eq(float(recovered["yaw_deg"]), 0.0, 0.0001)
	t.assert_false(bool(recovered["reload_finished"]))
	t.assert_almost_eq(weapon.cooldown_remaining, 0.15, 0.0001, "cooldown must still tick")

	# pull_trigger reduces to try_fire, including the held-trigger case
	# that the handling layer would otherwise block.
	weapon.cooldown_remaining = 0.0
	var first: Dictionary = weapon.pull_trigger(true)
	t.assert_true(bool(first["fired"]), "legacy fires on a held trigger")
	t.assert_almost_eq(float(first["spread_deg"]), 0.0, 0.0001, "legacy shots are pinpoint")
	t.assert_almost_eq(float(first["recoil_pitch_deg"]), 0.0, 0.0001, "legacy has no kick")
	weapon.cooldown_remaining = 0.0
	var second: Dictionary = weapon.pull_trigger(true)
	t.assert_true(bool(second["fired"]), "legacy needs no trigger release")
	return t


func test_handling_disabled_ignores_spread_and_hit_zones() -> SandboxTest:
	var t := SandboxTest.new("handling_disabled_ignores_spread_and_hit_zones")
	var weapon := WeaponState.new()
	weapon.configure_profile("rifle")
	weapon.reset()

	var aim := Vector3.FORWARD
	t.assert_vec_almost_eq(
		weapon.apply_spread(aim, weapon.current_spread_deg(), 0),
		aim,
		0.0001,
		"a zero cone must return the aim direction untouched"
	)

	# With head zones off the resolver reports a plain body hit at x1.
	var body := Vector3(0.0, 1.0, -5.0)
	var head := Vector3(0.0, 1.6, -5.0)
	var zone: Dictionary = weapon.resolve_hit_zone(Vector3(0.0, 1.6, 0.0), aim, body, head, false)
	t.assert_eq(zone["zone"], WeaponState.ZONE_BODY, "head zone must be off")
	t.assert_almost_eq(float(zone["multiplier"]), 1.0, 0.0001, "no damage multiplier")
	return t


func test_curriculum_gates_handling_at_level_five() -> SandboxTest:
	var t := SandboxTest.new("curriculum_gates_handling_at_level_five")
	var curriculum := CurriculumConfig.new()
	for level in range(1, 12):
		curriculum.level = level
		var expected: bool = level >= CurriculumConfig.Level.OBSTACLES_COVER
		t.assert_eq(
			curriculum.weapon_handling_enabled(),
			expected,
			"handling gate wrong at level %d" % level
		)
		t.assert_eq(
			curriculum.hit_zones_enabled(),
			expected,
			"hit-zone gate must track the handling gate at level %d" % level
		)
	return t


func test_level_one_environment_keeps_the_legacy_shot_contract() -> SandboxTest:
	var t := SandboxTest.new("level_one_environment_keeps_the_legacy_shot_contract")
	var env := EnvironmentCore.new(0, 1)
	env.curriculum.level = CurriculumConfig.Level.STATIC_TARGETS
	env.reset(7)
	t.assert_false(env.agent.weapon.handling_enabled, "level 1 must not arm handling")

	# Shooting at nothing is still the old "useless shot", not a
	# trigger-discipline event: the old reward contract is preserved.
	env.agent.position = Vector3(0.0, 0.0, 8.0)
	env.agent.yaw_deg = 180.0
	env.enemies[0].position = Vector3(0.0, 0.0, -8.0)
	env.agent.weapon.cooldown_remaining = 0.0
	var result: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT))
	var events: Dictionary = result.info.events
	t.assert_false(bool(events.get("trigger_discipline", false)), "no discipline events at level 1")
	t.assert_false(bool(events.get("headshot", false)), "no headshots at level 1")
	t.assert_false(bool(events.get("reloading", false)), "no reloading at level 1")
	return t


# -- fire modes ------------------------------------------------------------


func test_semi_auto_requires_a_trigger_release() -> SandboxTest:
	var t := SandboxTest.new("semi_auto_requires_a_trigger_release")
	var weapon := _armed("pistol")
	t.assert_eq(weapon.fire_mode, WeaponState.FIRE_MODE_SEMI, "pistol should be semi-auto")

	var first: Dictionary = weapon.pull_trigger(true)
	t.assert_true(bool(first["fired"]), "first pull fires")

	# Still held, cooldown elapsed: must refuse because it was never let go.
	weapon.cooldown_remaining = 0.0
	var held: Dictionary = weapon.pull_trigger(true)
	t.assert_false(bool(held["fired"]), "a held semi-auto must not fire")
	t.assert_eq(held["blocked_reason"], "needs_release")

	# Release, then pull again.
	weapon.pull_trigger(false)
	weapon.cooldown_remaining = 0.0
	var again: Dictionary = weapon.pull_trigger(true)
	t.assert_true(bool(again["fired"]), "firing resumes after a release")
	return t


func test_auto_fire_keeps_firing_on_a_held_trigger() -> SandboxTest:
	var t := SandboxTest.new("auto_fire_keeps_firing_on_a_held_trigger")
	var weapon := _armed("rifle")
	t.assert_eq(weapon.fire_mode, WeaponState.FIRE_MODE_AUTO, "rifle should be automatic")
	t.assert_true(bool(weapon.pull_trigger(true)["fired"]), "first shot")
	weapon.cooldown_remaining = 0.0
	t.assert_true(bool(weapon.pull_trigger(true)["fired"]), "automatic weapons keep firing")
	return t


func test_cycling_blocks_the_trigger_without_wasting_a_shot() -> SandboxTest:
	var t := SandboxTest.new("cycling_blocks_the_trigger_without_wasting_a_shot")
	var weapon := _armed("rifle")
	var ammo_before: int = weapon.ammo_in_magazine
	t.assert_true(bool(weapon.pull_trigger(true)["fired"]))
	# Cooldown is still running from the shot above.
	var blocked: Dictionary = weapon.pull_trigger(true)
	t.assert_false(bool(blocked["fired"]), "must not fire mid-cycle")
	t.assert_eq(blocked["blocked_reason"], "cycling")
	t.assert_eq(weapon.ammo_in_magazine, ammo_before - 1, "a blocked pull must not eat ammo")
	return t


# -- magazine and reload ---------------------------------------------------


func test_magazine_drains_and_auto_reloads_when_empty() -> SandboxTest:
	var t := SandboxTest.new("magazine_drains_and_auto_reloads_when_empty")
	var weapon := _armed("rifle")
	var capacity: int = weapon.magazine_size
	t.assert_gt(float(capacity), 0.0, "handling weapons need a magazine")

	for i in range(capacity):
		weapon.cooldown_remaining = 0.0
		var shot: Dictionary = weapon.pull_trigger(true)
		t.assert_true(bool(shot["fired"]), "shot %d of the magazine should fire" % i)
	t.assert_eq(weapon.ammo_in_magazine, 0, "magazine should be empty")
	t.assert_true(weapon.is_magazine_empty())

	# The next pull cannot fire, and starts the reload by itself: there is
	# no reload bit in the action space, so a dry weapon must recover
	# without one or the episode becomes unwinnable.
	weapon.cooldown_remaining = 0.0
	var dry: Dictionary = weapon.pull_trigger(true)
	t.assert_false(bool(dry["fired"]), "an empty magazine cannot fire")
	t.assert_eq(dry["blocked_reason"], "empty")
	t.assert_true(weapon.is_reloading(), "an empty weapon must start reloading itself")
	t.assert_false(weapon.is_ready(), "a reloading weapon is not ready")

	# Reloading blocks the trigger for exactly reload_time seconds.
	var elapsed: float = 0.0
	var finished: bool = false
	while elapsed < weapon.reload_time + 1.0 and not finished:
		finished = bool(weapon.tick_handling(0.1)["reload_finished"])
		elapsed += 0.1
	t.assert_true(finished, "the reload must complete")
	t.assert_almost_eq(elapsed, weapon.reload_time, 0.15, "reload should take reload_time")
	t.assert_eq(weapon.ammo_in_magazine, capacity, "the magazine should be full again")
	t.assert_true(weapon.is_ready(), "the weapon is usable after reloading")
	t.assert_eq(weapon.reload_count, 1, "the reload should be counted once")
	return t


func test_weapon_ready_reports_false_while_reloading() -> SandboxTest:
	var t := SandboxTest.new("weapon_ready_reports_false_while_reloading")
	# `weapon_ready` is observation index 15. It is the only channel the
	# policy has for "your trigger currently does nothing", so it must be
	# honest about reloads even though the reload itself is not observed.
	var weapon := _armed("rifle")
	weapon.ammo_in_magazine = 0
	t.assert_false(weapon.is_ready(), "an empty weapon is not ready")
	weapon.begin_reload()
	t.assert_false(weapon.is_ready(), "a reloading weapon is not ready")
	weapon.reload_remaining = 0.0
	weapon.ammo_in_magazine = weapon.magazine_size
	t.assert_true(weapon.is_ready(), "a loaded, cooled weapon is ready")
	return t


func test_reset_restores_a_full_magazine_and_clears_recoil() -> SandboxTest:
	var t := SandboxTest.new("reset_restores_a_full_magazine_and_clears_recoil")
	var weapon := _armed("smg")
	for _i in range(5):
		weapon.cooldown_remaining = 0.0
		weapon.pull_trigger(true)
	t.assert_gt(weapon.recoil_pitch_deg, 0.0, "firing should build recoil")
	t.assert_gt(weapon.bloom_deg, 0.0, "firing should build bloom")

	weapon.reset()
	t.assert_eq(weapon.ammo_in_magazine, weapon.magazine_size, "reset refills the magazine")
	t.assert_almost_eq(weapon.recoil_pitch_deg, 0.0, 0.0001, "reset clears recoil")
	t.assert_almost_eq(weapon.bloom_deg, 0.0, 0.0001, "reset clears bloom")
	t.assert_almost_eq(weapon.reload_remaining, 0.0, 0.0001, "reset cancels a reload")
	t.assert_eq(weapon.shots_fired, 0, "reset restarts the deterministic shot sequence")
	return t


# -- recoil ----------------------------------------------------------------


func test_recoil_accumulates_then_recovers() -> SandboxTest:
	var t := SandboxTest.new("recoil_accumulates_then_recovers")
	var weapon := _armed("rifle")
	var first: Dictionary = weapon.pull_trigger(true)
	t.assert_gt(float(first["recoil_pitch_deg"]), 0.0, "the first shot must kick upward")
	var after_first: float = weapon.recoil_pitch_deg

	weapon.cooldown_remaining = 0.0
	weapon.pull_trigger(true)
	t.assert_gt(weapon.recoil_pitch_deg, after_first, "a second shot adds more kick")

	# Recovery only starts after the delay, then walks the offset back to
	# zero without overshooting into negative pitch.
	var before_recovery: float = weapon.recoil_pitch_deg
	weapon.tick_handling(SandboxConfig.RECOIL_RECOVERY_DELAY * 0.5)
	t.assert_almost_eq(
		weapon.recoil_pitch_deg, before_recovery, 0.0001, "no recovery inside the delay"
	)
	for _i in range(200):
		weapon.tick_handling(0.05)
	t.assert_almost_eq(weapon.recoil_pitch_deg, 0.0, 0.0001, "recoil must settle back to zero")
	t.assert_almost_eq(weapon.recoil_yaw_deg, 0.0, 0.0001, "yaw must settle back to zero")
	return t


func test_accumulated_recoil_is_clamped() -> SandboxTest:
	var t := SandboxTest.new("accumulated_recoil_is_clamped")
	var weapon := _armed("shotgun")
	for _i in range(60):
		weapon.cooldown_remaining = 0.0
		weapon.pull_trigger(true)
		weapon.ammo_in_magazine = weapon.magazine_size  # keep it firing
	t.assert_lte(
		absf(weapon.recoil_pitch_deg),
		SandboxConfig.RECOIL_MAX_DEG + 0.0001,
		"a long spray must saturate, not walk the camera off the map"
	)
	t.assert_lte(absf(weapon.recoil_yaw_deg), SandboxConfig.RECOIL_MAX_DEG + 0.0001)
	return t


func test_recoil_recovery_returns_the_aim_to_where_it_started() -> SandboxTest:
	var t := SandboxTest.new("recoil_recovery_returns_the_aim_to_where_it_started")
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	agent.weapon.configure_profile("rifle")
	agent.weapon.handling_enabled = true
	agent.weapon.reset()

	var start_pitch: float = agent.pitch_deg
	var start_yaw: float = agent.yaw_deg
	for _i in range(6):
		agent.weapon.cooldown_remaining = 0.0
		var shot: Dictionary = agent.weapon.pull_trigger(true)
		agent.apply_recoil(float(shot["recoil_pitch_deg"]), float(shot["recoil_yaw_deg"]))
	t.assert_ne(agent.pitch_deg, start_pitch, "firing must move the aim")

	# Let everything settle: the view must come back to exactly where the
	# agent left it, because a systematic drift would be an unlearnable
	# bias rather than a recoil pattern.
	var idle := Action.new()
	for _i in range(400):
		agent.apply_action(idle, 0.05, 10.0, null)
	t.assert_almost_eq(agent.pitch_deg, start_pitch, 0.001, "pitch must return to its origin")
	t.assert_almost_eq(agent.yaw_deg, start_yaw, 0.001, "yaw must return to its origin")
	return t


func test_recoil_recovery_is_conservative_at_the_pitch_limit() -> SandboxTest:
	var t := SandboxTest.new("recoil_recovery_is_conservative_at_the_pitch_limit")
	# Aiming near vertical, the pitch clamp swallows part of each kick. The
	# weapon still accumulates the full figure, so recovery must be capped
	# at what the view actually absorbed or the aim walks downward.
	var agent := AgentState.new()
	agent.reset(Vector3.ZERO, 0.0)
	agent.weapon.configure_profile("shotgun")
	agent.weapon.handling_enabled = true
	agent.weapon.reset()
	agent.pitch_deg = agent.pitch_limit_deg

	for _i in range(4):
		agent.weapon.cooldown_remaining = 0.0
		agent.weapon.ammo_in_magazine = agent.weapon.magazine_size
		var shot: Dictionary = agent.weapon.pull_trigger(true)
		agent.apply_recoil(float(shot["recoil_pitch_deg"]), float(shot["recoil_yaw_deg"]))
	t.assert_almost_eq(
		agent.pitch_deg, agent.pitch_limit_deg, 0.001, "pitch must stay clamped at the limit"
	)

	var idle := Action.new()
	for _i in range(400):
		agent.apply_action(idle, 0.05, 10.0, null)
	t.assert_almost_eq(
		agent.pitch_deg,
		agent.pitch_limit_deg,
		0.001,
		"recovery must not drag the aim below where the kick ever took it"
	)
	return t


func test_recoil_pattern_is_deterministic_and_two_sided() -> SandboxTest:
	var t := SandboxTest.new("recoil_pattern_is_deterministic_and_two_sided")
	var weapon := _armed("rifle")
	var left: bool = false
	var right: bool = false
	for i in range(16):
		var a: Dictionary = weapon.recoil_kick(i)
		var b: Dictionary = weapon.recoil_kick(i)
		t.assert_almost_eq(
			float(a["pitch_deg"]), float(b["pitch_deg"]), 0.0, "recoil must be reproducible"
		)
		t.assert_almost_eq(float(a["yaw_deg"]), float(b["yaw_deg"]), 0.0)
		if float(a["yaw_deg"]) > 0.0:
			right = true
		if float(a["yaw_deg"]) < 0.0:
			left = true
	t.assert_true(left and right, "horizontal recoil must wander both ways")

	# Sustained fire kicks less than the opening shots, which is what makes
	# a controlled burst better than holding the trigger down.
	var opening: float = float(weapon.recoil_kick(0)["pitch_deg"])
	var sustained: float = float(weapon.recoil_kick(32)["pitch_deg"])
	t.assert_almost_eq(opening, weapon.recoil_vertical_deg, 0.0001, "first shot is the full kick")
	t.assert_lt(sustained, opening, "muzzle climb must saturate")
	return t

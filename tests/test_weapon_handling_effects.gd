## Tests for the observable consequences of the weapon handling layer:
## bloom and movement penalties, head/body hit zones, the damage-scaled hit
## reward, and how all of that surfaces through EnvironmentCore's events,
## metrics and introspection.
##
## Split from tests/test_weapon_handling.gd, which covers the weapon state
## machine itself (legacy equivalence, fire modes, magazines and recoil).
class_name TestWeaponHandlingEffects
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of
## the editor class cache.
const Action = preload("res://scripts/core/action.gd")
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


# -- bloom and movement ----------------------------------------------------
func test_first_settled_shot_is_pinpoint_then_bloom_grows() -> SandboxTest:
	var t := SandboxTest.new("first_settled_shot_is_pinpoint_then_bloom_grows")
	var weapon := _armed("smg")
	t.assert_true(weapon.is_settled(), "a rested weapon is settled")
	var aim := Vector3.FORWARD
	t.assert_vec_almost_eq(
		weapon.apply_spread(aim, weapon.current_spread_deg(), 0),
		aim,
		0.0001,
		"the first settled shot must go exactly where it is aimed"
	)

	var previous: float = weapon.current_spread_deg()
	for i in range(10):
		weapon.cooldown_remaining = 0.0
		weapon.pull_trigger(true)
		var now: float = weapon.current_spread_deg()
		t.assert_gte(now, previous, "bloom must not shrink while firing (shot %d)" % i)
		previous = now
	t.assert_gt(previous, 0.0, "sustained fire must open the cone")
	t.assert_lte(previous, weapon.spread_max_deg + 0.0001, "bloom must respect its ceiling")
	t.assert_false(weapon.is_settled(), "a blooming weapon is not settled")

	# Stop firing and the cone collapses again.
	for _i in range(200):
		weapon.tick_handling(0.05)
	t.assert_almost_eq(weapon.current_spread_deg(), 0.0, 0.0001, "bloom must recover to zero")
	return t


func test_moving_and_airborne_fire_open_the_cone() -> SandboxTest:
	var t := SandboxTest.new("moving_and_airborne_fire_open_the_cone")
	var weapon := _armed("rifle")
	var still: float = weapon.current_spread_deg(0.0, false)
	var running: float = weapon.current_spread_deg(1.0, false)
	var jumping: float = weapon.current_spread_deg(0.0, true)
	t.assert_gt(running, still, "running must cost accuracy")
	t.assert_gt(jumping, still, "shooting mid-air must cost accuracy")
	t.assert_gt(jumping, running, "a jump should be worse than a sprint")
	return t


func test_firing_slows_the_shooter_for_a_bounded_time() -> SandboxTest:
	var t := SandboxTest.new("firing_slows_the_shooter_for_a_bounded_time")
	var weapon := _armed("rifle")
	t.assert_almost_eq(weapon.movement_speed_scale(), 1.0, 0.0001, "no slow before firing")
	weapon.pull_trigger(true)
	t.assert_almost_eq(
		weapon.movement_speed_scale(),
		weapon.move_speed_scale_firing,
		0.0001,
		"firing must plant the shooter"
	)
	t.assert_lt(weapon.move_speed_scale_firing, 1.0, "the slow must be a real cost")

	var waited: float = 0.0
	while waited < SandboxConfig.FIRE_MOVEMENT_SLOW_DURATION + 0.2:
		weapon.tick_handling(0.05)
		waited += 0.05
	t.assert_almost_eq(weapon.movement_speed_scale(), 1.0, 0.0001, "the slow must expire")
	return t


func test_spread_places_shots_inside_the_cone() -> SandboxTest:
	var t := SandboxTest.new("spread_places_shots_inside_the_cone")
	var weapon := _armed("smg")
	var aim := Vector3.FORWARD
	var cone: float = 5.0
	var saw_offset: bool = false
	for i in range(24):
		var shot: Vector3 = weapon.apply_spread(aim, cone, i)
		t.assert_almost_eq(shot.length(), 1.0, 0.0001, "spread must return a unit vector")
		var angle: float = rad_to_deg(acos(clampf(shot.dot(aim), -1.0, 1.0)))
		t.assert_lte(angle, cone + 0.0001, "shot %d escaped the cone" % i)
		if angle > 0.01:
			saw_offset = true
		# Determinism: the same index always yields the same direction.
		t.assert_vec_almost_eq(
			weapon.apply_spread(aim, cone, i), shot, 0.0, "spread must be stable"
		)
	t.assert_true(saw_offset, "a non-zero cone must actually displace shots")
	return t


# -- hit zones -------------------------------------------------------------


func test_head_zone_is_a_subset_of_the_body_and_multiplies_damage() -> SandboxTest:
	var t := SandboxTest.new("head_zone_is_a_subset_of_the_body_and_multiplies_damage")
	var weapon := _armed("rifle")
	var origin := Vector3(0.0, 1.5, 0.0)
	var body := Vector3(0.0, 1.0, -6.0)
	var head := Vector3(0.0, 1.7, -6.0)

	var head_shot: Dictionary = weapon.resolve_hit_zone(
		origin, (head - origin).normalized(), body, head, true
	)
	t.assert_eq(head_shot["zone"], WeaponState.ZONE_HEAD, "aiming at the head should hit it")
	t.assert_almost_eq(
		float(head_shot["multiplier"]),
		weapon.headshot_multiplier,
		0.0001,
		"a headshot must apply the profile multiplier"
	)
	t.assert_gt(weapon.headshot_multiplier, 1.0, "headshots must be worth aiming for")

	var body_shot: Dictionary = weapon.resolve_hit_zone(
		origin, (body - origin).normalized(), body, head, true
	)
	t.assert_eq(body_shot["zone"], WeaponState.ZONE_BODY, "aiming at the chest hits the body")
	t.assert_almost_eq(float(body_shot["multiplier"]), 1.0, 0.0001, "body shots are unscaled")

	# A ray that misses the body entirely cannot produce a headshot.
	var miss: Dictionary = weapon.resolve_hit_zone(origin, Vector3(1.0, 0.0, 0.0), body, head, true)
	t.assert_eq(miss["zone"], WeaponState.ZONE_NONE, "a clean miss is not a hit")
	t.assert_almost_eq(float(miss["multiplier"]), 0.0, 0.0001)

	# Disabling the zone downgrades the same ray to a body hit rather than
	# turning it into a miss.
	var disabled: Dictionary = weapon.resolve_hit_zone(
		origin, (head - origin).normalized(), body, head, false
	)
	t.assert_eq(disabled["zone"], WeaponState.ZONE_BODY, "disabled head zones still hit the body")
	return t


# -- reward scaling --------------------------------------------------------


func test_hit_reward_scale_is_exactly_one_for_the_rifle() -> SandboxTest:
	var t := SandboxTest.new("hit_reward_scale_is_exactly_one_for_the_rifle")
	# The reference damage is the rifle's, so the default weapon must score
	# exactly the historical +1 hit bonus. Anything else silently rescales
	# every reward curve recorded before this layer existed.
	var scale: float = RewardSystem.hit_reward_scale(
		{"weapon_damage": 25.0, "projectiles_fired": 1, "projectiles_hit": 1}
	)
	t.assert_almost_eq(scale, 1.0, 0.0001, "rifle hits must score 1.0")

	# No damage information at all also means "unchanged", so legacy
	# callers and levels 1-4 keep the original contract.
	t.assert_almost_eq(RewardSystem.hit_reward_scale({}), 1.0, 0.0001, "absent damage means 1.0")
	return t


func test_hit_reward_scale_tracks_damage_and_is_clamped() -> SandboxTest:
	var t := SandboxTest.new("hit_reward_scale_tracks_damage_and_is_clamped")
	var weak: float = RewardSystem.hit_reward_scale(
		{"weapon_damage": 12.5, "projectiles_fired": 1, "projectiles_hit": 1}
	)
	t.assert_almost_eq(weak, 0.5, 0.0001, "half damage should score half the bonus")

	var huge: float = RewardSystem.hit_reward_scale(
		{"weapon_damage": 100000.0, "projectiles_fired": 1, "projectiles_hit": 1}
	)
	t.assert_almost_eq(
		huge,
		SandboxConfig.REWARD_HIT_SCALE_MAX,
		0.0001,
		"the hit bonus must be clamped so damage cannot be farmed"
	)
	return t


func test_hit_reward_scale_prorates_shotgun_pellets() -> SandboxTest:
	var t := SandboxTest.new("hit_reward_scale_prorates_shotgun_pellets")
	# A shotgun that lands 2 of 8 pellets grazed the target; it must not
	# collect the same bonus as one that landed the whole volley, or
	# spraying pellets past a target becomes a cheap source of reward.
	var full: float = RewardSystem.hit_reward_scale(
		{"weapon_damage": 112.0, "projectiles_fired": 8, "projectiles_hit": 8}
	)
	var graze: float = RewardSystem.hit_reward_scale(
		{"weapon_damage": 28.0, "projectiles_fired": 8, "projectiles_hit": 2}
	)
	t.assert_gt(full, graze, "a full volley must beat a graze")
	t.assert_almost_eq(graze, (28.0 / 25.0) * 0.25, 0.0001, "graze should be prorated by pellets")
	return t


# -- environment integration ----------------------------------------------


func test_blocked_trigger_is_a_discipline_event_not_a_wasted_shot() -> SandboxTest:
	var t := SandboxTest.new("blocked_trigger_is_a_discipline_event_not_a_wasted_shot")
	var env := EnvironmentCore.new(0, 1)
	env.curriculum.level = CurriculumConfig.Level.OBSTACLES_COVER
	env.reset(11)
	t.assert_true(env.agent.weapon.handling_enabled, "level 5 must arm handling")

	# Face the enemy so the shot is a legitimate attempt, then hold the
	# trigger through the cooldown.
	env.agent.position = Vector3(0.0, 0.0, 6.0)
	env.agent.yaw_deg = 0.0
	env.enemies[0].position = Vector3(0.0, 0.0, 0.0)
	env.agent.weapon.cooldown_remaining = 0.0
	env.step(Action.from_discrete(Action.Discrete.SHOOT))
	var blocked: Dictionary = env.step(Action.from_discrete(Action.Discrete.SHOOT)).info.events
	t.assert_true(
		bool(blocked.get("trigger_discipline", false)),
		"a trigger pull during the cycle is a discipline event"
	)
	t.assert_false(
		bool(blocked.get("useless_shot", false)),
		"holding the trigger must not be punished as a wasted shot"
	)
	return t


func test_episode_metrics_expose_the_handling_counters() -> SandboxTest:
	var t := SandboxTest.new("episode_metrics_expose_the_handling_counters")
	var env := EnvironmentCore.new(0, 1)
	env.curriculum.level = CurriculumConfig.Level.OBSTACLES_COVER
	env.reset(3)
	var metrics: Dictionary = env.episode.to_metrics(SandboxConfig.SIMULATION_DT, 1)
	for key in [
		"headshots", "headshot_rate", "trigger_discipline_events", "reload_starts", "reloading_time"
	]:
		t.assert_true(metrics.has(key), "episode metrics must report %s" % key)
	var env_metrics: Dictionary = env.get_metrics()
	for key in ["weapon_fire_mode", "weapon_handling", "weapon_reloads"]:
		t.assert_true(env_metrics.has(key), "environment metrics must report %s" % key)
	return t


func test_weapon_state_introspection_is_complete() -> SandboxTest:
	var t := SandboxTest.new("weapon_state_introspection_is_complete")
	var env := EnvironmentCore.new(0, 1)
	env.curriculum.level = CurriculumConfig.Level.OBSTACLES_COVER
	env.reset(5)
	var state: Dictionary = env.get_weapon_state()
	for key in [
		"handling_capability",
		"hit_zones_capability",
		"speed_fraction",
		"effective_spread_deg",
		"movement_speed_scale",
		"ammo_in_magazine",
		"fire_mode",
	]:
		t.assert_true(state.has(key), "weapon introspection must report %s" % key)
	t.assert_true(bool(state["handling_capability"]), "level 5 reports the handling capability")
	return t


func test_every_profile_stays_usable_with_handling_on() -> SandboxTest:
	var t := SandboxTest.new("every_profile_stays_usable_with_handling_on")
	# A profile that cannot land a kill on one magazine at its own optimal
	# range would make the reload a tax rather than a decision.
	for profile_id in WeaponState.profile_ids():
		var weapon := _armed(profile_id)
		var optimal: float = minf(weapon.falloff_start_m, weapon.range_m) * 0.5
		var volley: float = weapon.projectile_damage_at_distance(optimal) * weapon.projectile_count
		t.assert_gt(volley, 0.0, "%s must deal damage at %0.1fm" % [profile_id, optimal])
		var shots: int = ceili(100.0 / volley)
		t.assert_lte(
			float(shots),
			float(weapon.magazine_size),
			"%s cannot secure a kill on one magazine" % profile_id
		)
		t.assert_gt(weapon.headshot_multiplier, 0.999, "%s needs a head multiplier" % profile_id)
		t.assert_gt(float(weapon.magazine_size), 0.0, "%s needs a magazine" % profile_id)
		t.assert_gt(weapon.reload_time, 0.0, "%s needs a reload time" % profile_id)
	return t

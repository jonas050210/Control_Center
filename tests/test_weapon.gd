## Tests for WeaponState: fire cooldown and the raycast/hit-test.
class_name TestWeapon
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const WeaponState = preload("res://scripts/weapon/weapon_state.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_first_shot_fires_and_starts_cooldown() -> SandboxTest:
	var t := SandboxTest.new("weapon_first_shot_fires_and_starts_cooldown")
	var weapon := WeaponState.new()
	t.assert_true(weapon.is_ready(), "weapon should start ready")
	var fired := weapon.try_fire()
	t.assert_true(fired, "first shot should fire")
	t.assert_false(weapon.is_ready(), "weapon should be on cooldown right after firing")
	return t


func test_second_shot_within_cooldown_window_fails() -> SandboxTest:
	var t := SandboxTest.new("weapon_second_shot_within_cooldown_fails")
	var weapon := WeaponState.new()
	weapon.try_fire()
	var second_shot := weapon.try_fire()
	t.assert_false(second_shot, "second shot immediately after the first should be rejected")
	return t


func test_cooldown_elapses_and_allows_refire() -> SandboxTest:
	var t := SandboxTest.new("weapon_cooldown_elapses_allows_refire")
	var weapon := WeaponState.new()
	weapon.try_fire()
	weapon.tick(weapon.cooldown_time + 0.01)
	t.assert_true(weapon.is_ready(), "weapon should be ready again once cooldown elapses")
	t.assert_true(weapon.try_fire(), "refire should succeed after cooldown")
	return t


func test_ray_hits_sphere_directly_ahead() -> SandboxTest:
	var t := SandboxTest.new("weapon_ray_hits_sphere_directly_ahead")
	var weapon := WeaponState.new()
	var origin := Vector3.ZERO
	var direction := Vector3(0.0, 0.0, -1.0)
	var target_center := Vector3(0.0, 0.0, -5.0)
	t.assert_true(
		weapon.ray_hits_sphere(origin, direction, target_center),
		"should hit a target directly ahead"
	)
	return t


func test_ray_misses_target_behind_the_shooter() -> SandboxTest:
	var t := SandboxTest.new("weapon_ray_misses_target_behind")
	var weapon := WeaponState.new()
	var origin := Vector3.ZERO
	var direction := Vector3(0.0, 0.0, -1.0)
	var target_center := Vector3(0.0, 0.0, 5.0)
	t.assert_false(
		weapon.ray_hits_sphere(origin, direction, target_center),
		"should not hit a target behind the ray origin"
	)
	return t


func test_ray_misses_target_outside_range() -> SandboxTest:
	var t := SandboxTest.new("weapon_ray_misses_target_outside_range")
	var weapon := WeaponState.new()
	weapon.range_m = 10.0
	var hit := weapon.ray_hits_sphere(
		Vector3.ZERO, Vector3(0.0, 0.0, -1.0), Vector3(0.0, 0.0, -50.0)
	)
	t.assert_false(hit, "should not hit a target far beyond weapon range")
	return t


func test_ray_misses_target_off_to_the_side() -> SandboxTest:
	var t := SandboxTest.new("weapon_ray_misses_off_center_target")
	var weapon := WeaponState.new()
	weapon.hit_radius = 0.5
	var hit := weapon.ray_hits_sphere(
		Vector3.ZERO, Vector3(0.0, 0.0, -1.0), Vector3(5.0, 0.0, -5.0)
	)
	t.assert_false(hit, "target far off the ray's line should not be hit")
	return t


func test_profile_damage_falloff_and_ttk_are_monotonic() -> SandboxTest:
	var t := SandboxTest.new("weapon_profile_falloff_and_ttk")
	for profile_id in WeaponState.profile_ids():
		var weapon := WeaponState.new()
		t.assert_true(weapon.configure_profile(profile_id))
		var near_damage: float = weapon.projectile_damage_at_distance(0.0)
		var far_damage: float = weapon.projectile_damage_at_distance(weapon.range_m)
		t.assert_gt(near_damage, 0.0)
		t.assert_true(far_damage <= near_damage, "falloff cannot increase damage")
		t.assert_true(far_damage >= 0.0, "damage cannot become negative")
		t.assert_true(
			weapon.ideal_ttk_seconds(100.0, weapon.range_m) >= weapon.ideal_ttk_seconds(100.0, 0.0),
			"ideal TTK cannot improve with distance"
		)
	return t


func test_shotgun_pattern_is_deterministic_and_has_eight_pellets() -> SandboxTest:
	var t := SandboxTest.new("shotgun_pattern_deterministic")
	var weapon := WeaponState.new()
	weapon.configure_profile(WeaponState.PROFILE_SHOTGUN)
	var first: Array = weapon.projectile_directions(Vector3.FORWARD)
	var second: Array = weapon.projectile_directions(Vector3.FORWARD)
	t.assert_eq(first.size(), 8)
	t.assert_eq(first, second, "fixed pellet pattern must replay identically")
	return t

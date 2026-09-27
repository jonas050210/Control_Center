## WeaponState
##
## Minimal deterministic hitscan weapon: fixed/configurable damage, range,
## and fire cooldown. No recoil, no ammo/inventory, no random spread.
class_name WeaponState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


var damage: float = SandboxConfig.WEAPON_DAMAGE
var range_m: float = SandboxConfig.WEAPON_RANGE
var cooldown_time: float = SandboxConfig.WEAPON_FIRE_COOLDOWN
var hit_radius: float = SandboxConfig.WEAPON_HIT_RADIUS
var cooldown_remaining: float = 0.0


func _init(
	p_damage: float = SandboxConfig.WEAPON_DAMAGE,
	p_range_m: float = SandboxConfig.WEAPON_RANGE,
	p_cooldown_time: float = SandboxConfig.WEAPON_FIRE_COOLDOWN,
	p_hit_radius: float = SandboxConfig.WEAPON_HIT_RADIUS
) -> void:
	damage = maxf(0.0, p_damage)
	range_m = maxf(0.1, p_range_m)
	cooldown_time = maxf(0.0, p_cooldown_time)
	hit_radius = maxf(0.01, p_hit_radius)
	cooldown_remaining = 0.0


func reset() -> void:
	cooldown_remaining = 0.0


func tick(dt: float) -> void:
	if cooldown_remaining > 0.0:
		cooldown_remaining = maxf(0.0, cooldown_remaining - dt)


func is_ready() -> bool:
	return cooldown_remaining <= 0.0


## Attempts to fire. Returns true and starts the cooldown if the weapon was
## ready; returns false (no effect) if it was still on cooldown.
func try_fire() -> bool:
	if not is_ready():
		return false
	cooldown_remaining = cooldown_time
	return true


## Computes the distance along the ray where it intersects/passes closest
## to the target sphere. Returns the distance `t` (>= 0.0) if a hit occurs
## within `range_m`, or -1.0 if it misses or lies behind the origin.
func ray_hit_distance(origin: Vector3, direction: Vector3, target_center: Vector3) -> float:
	if direction.is_zero_approx():
		return -1.0
	var dir: Vector3 = direction.normalized()
	var to_target: Vector3 = target_center - origin
	var t: float = to_target.dot(dir)
	if t < 0.0 or t > range_m:
		return -1.0
	var closest_point: Vector3 = origin + dir * t
	var dist_sq: float = closest_point.distance_squared_to(target_center)
	if dist_sq <= hit_radius * hit_radius:
		return t
	return -1.0


## Deterministic ray-vs-sphere hit test. Returns true if a ray cast from
## `origin` along unit vector `direction` (within `range_m`) intersects the
## sphere centered at `target_center` with radius `hit_radius`.
func ray_hits_sphere(origin: Vector3, direction: Vector3, target_center: Vector3) -> bool:
	return ray_hit_distance(origin, direction, target_center) >= 0.0


func to_dict() -> Dictionary:
	return {
		"damage": damage,
		"range_m": range_m,
		"cooldown_time": cooldown_time,
		"hit_radius": hit_radius,
		"cooldown_remaining": cooldown_remaining,
		"ready": is_ready(),
	}

## WeaponState
##
## Deterministic hitscan weapon with named handling profiles. The default
## profile preserves the original single-ray rifle contract, while optional
## profiles model the TTK-style roles we care about during controlled tests:
## a mid-range rifle, a close-range pellet shotgun, a precision sidearm and
## an SMG-like fast secondary. Profiles are simulation metadata, not new
## policy actions: the PPO action space still has exactly one shoot button.
class_name WeaponState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const PROFILE_RIFLE: String = "rifle"
const PROFILE_SHOTGUN: String = "shotgun"
const PROFILE_PISTOL: String = "pistol"
const PROFILE_SMG: String = "smg"
const PROFILE_IDS: Array = [PROFILE_RIFLE, PROFILE_SHOTGUN, PROFILE_PISTOL, PROFILE_SMG]

const PROFILE_DEFINITIONS: Dictionary = {
	"rifle": {
		"label": "Rifle",
		"category": "primary",
		"damage": SandboxConfig.WEAPON_DAMAGE,
		"range_m": SandboxConfig.WEAPON_RANGE,
		"cooldown_time": SandboxConfig.WEAPON_FIRE_COOLDOWN,
		"hit_radius": SandboxConfig.WEAPON_HIT_RADIUS,
		"projectile_count": 1,
		"spread_deg": 0.0,
		"falloff_start_m": 12.0,
		"minimum_damage_scale": 0.72,
		"description": "Baseline single-ray rifle: controllable mid-range damage with mild falloff.",
	},
	"shotgun": {
		"label": "Breach shotgun",
		"category": "shotgun",
		"damage": 112.0,
		"range_m": 9.5,
		"cooldown_time": 0.72,
		"hit_radius": 0.55,
		"projectile_count": 8,
		"spread_deg": 5.0,
		"falloff_start_m": 4.0,
		"minimum_damage_scale": 0.22,
		"description": "Close-range 8-pellet pattern; one-shot potential requires a centered close hit.",
	},
	"pistol": {
		"label": "Sidearm",
		"category": "sidearm",
		"damage": 20.0,
		"range_m": 12.0,
		"cooldown_time": 0.24,
		"hit_radius": 0.65,
		"projectile_count": 1,
		"spread_deg": 0.0,
		"falloff_start_m": 7.0,
		"minimum_damage_scale": 0.55,
		"description": "Lower damage precision backup for deliberate finishing drills.",
	},
	"smg": {
		"label": "SMG",
		"category": "secondary_auto",
		"damage": 14.0,
		"range_m": 11.0,
		"cooldown_time": 0.09,
		"hit_radius": 0.62,
		"projectile_count": 1,
		"spread_deg": 0.0,
		"falloff_start_m": 5.5,
		"minimum_damage_scale": 0.45,
		"description": "Fast close-range profile with sharp falloff that rewards sustained tracking.",
	},
}

var profile_id: String = PROFILE_RIFLE
var profile_label: String = "Rifle"
var category: String = "primary"
var damage: float = SandboxConfig.WEAPON_DAMAGE
var range_m: float = SandboxConfig.WEAPON_RANGE
var cooldown_time: float = SandboxConfig.WEAPON_FIRE_COOLDOWN
var hit_radius: float = SandboxConfig.WEAPON_HIT_RADIUS
var projectile_count: int = 1
var spread_deg: float = 0.0
var falloff_start_m: float = SandboxConfig.WEAPON_RANGE
var minimum_damage_scale: float = 1.0
var cooldown_remaining: float = 0.0


func _init(
	p_damage: float = SandboxConfig.WEAPON_DAMAGE,
	p_range_m: float = SandboxConfig.WEAPON_RANGE,
	p_cooldown_time: float = SandboxConfig.WEAPON_FIRE_COOLDOWN,
	p_hit_radius: float = SandboxConfig.WEAPON_HIT_RADIUS
) -> void:
	configure_profile(PROFILE_RIFLE)
	damage = maxf(0.0, p_damage)
	range_m = maxf(0.1, p_range_m)
	cooldown_time = maxf(0.0, p_cooldown_time)
	hit_radius = maxf(0.01, p_hit_radius)
	cooldown_remaining = 0.0


static func has_profile(p_profile_id: String) -> bool:
	return PROFILE_DEFINITIONS.has(p_profile_id)


static func profile_definition(p_profile_id: String) -> Dictionary:
	var resolved: String = p_profile_id if has_profile(p_profile_id) else PROFILE_RIFLE
	return (PROFILE_DEFINITIONS[resolved] as Dictionary).duplicate(true)


static func profile_ids() -> PackedStringArray:
	var out := PackedStringArray()
	for item in PROFILE_IDS:
		out.append(str(item))
	return out


func configure_profile(p_profile_id: String) -> bool:
	if not has_profile(p_profile_id):
		return false
	var spec: Dictionary = profile_definition(p_profile_id)
	profile_id = p_profile_id
	profile_label = str(spec.get("label", p_profile_id))
	category = str(spec.get("category", "primary"))
	damage = maxf(0.0, float(spec.get("damage", SandboxConfig.WEAPON_DAMAGE)))
	range_m = maxf(0.1, float(spec.get("range_m", SandboxConfig.WEAPON_RANGE)))
	cooldown_time = maxf(0.0, float(spec.get("cooldown_time", SandboxConfig.WEAPON_FIRE_COOLDOWN)))
	hit_radius = maxf(0.01, float(spec.get("hit_radius", SandboxConfig.WEAPON_HIT_RADIUS)))
	projectile_count = maxi(1, int(spec.get("projectile_count", 1)))
	spread_deg = maxf(0.0, float(spec.get("spread_deg", 0.0)))
	falloff_start_m = clampf(float(spec.get("falloff_start_m", range_m)), 0.0, range_m)
	minimum_damage_scale = clampf(float(spec.get("minimum_damage_scale", 1.0)), 0.0, 1.0)
	return true


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


func projectile_damage() -> float:
	return damage / float(maxi(1, projectile_count))


## Piecewise-linear deterministic damage falloff. Full damage is retained
## through the role's intended range, then approaches a non-zero floor at
## maximum range. Keeping this separate from hit detection makes boundary
## behavior directly unit-testable and avoids range-dependent RNG.
func damage_scale_at_distance(distance_m: float) -> float:
	if distance_m <= falloff_start_m or range_m <= falloff_start_m:
		return 1.0
	if distance_m >= range_m:
		return minimum_damage_scale
	var alpha: float = (distance_m - falloff_start_m) / (range_m - falloff_start_m)
	return lerpf(1.0, minimum_damage_scale, clampf(alpha, 0.0, 1.0))


func projectile_damage_at_distance(distance_m: float) -> float:
	return projectile_damage() * damage_scale_at_distance(distance_m)


## Ideal center-mass TTK for diagnostics. The first shot occurs at t=0,
## therefore N lethal shots take (N-1) fire intervals.
func ideal_ttk_seconds(
	target_health: float, distance_m: float, pellets_landed: int = -1
) -> float:
	var landed: int = (
		projectile_count
		if pellets_landed < 0
		else clampi(pellets_landed, 0, projectile_count)
	)
	var volley_damage: float = projectile_damage_at_distance(distance_m) * float(landed)
	if volley_damage <= 0.0:
		return INF
	var volleys: int = ceili(maxf(0.0, target_health) / volley_damage)
	return float(maxi(0, volleys - 1)) * cooldown_time


## Deterministic ray directions for the current trigger pull. Single-projectile
## profiles return the exact forward vector. Shotgun-style profiles return a
## center pellet plus a fixed circular pattern, so seeded replays remain stable.
func projectile_directions(direction: Vector3, up_hint: Vector3 = Vector3.UP) -> Array:
	var dir: Vector3 = direction.normalized() if not direction.is_zero_approx() else Vector3.FORWARD
	if projectile_count <= 1 or spread_deg <= 0.0:
		return [dir]

	var right: Vector3 = dir.cross(up_hint).normalized()
	if right.is_zero_approx():
		right = Vector3.RIGHT
	var up: Vector3 = right.cross(dir).normalized()
	if up.is_zero_approx():
		up = Vector3.UP

	var directions: Array = [dir]
	var spread_rad: float = deg_to_rad(spread_deg)
	var ring_count: int = projectile_count - 1
	for pellet_index in range(ring_count):
		var angle: float = TAU * float(pellet_index) / float(maxi(1, ring_count))
		# Slightly stagger radius so the pattern covers the cone interior, not
		# only its rim. Pure math, no RNG, so every replay is byte-stable.
		var radius_scale: float = 0.45 + 0.55 * (float(pellet_index % 3) / 2.0)
		var pellet_angle: float = spread_rad * radius_scale
		var lateral: Vector3 = (
			right * cos(angle) * sin(pellet_angle)
			+ up * sin(angle) * sin(pellet_angle)
		)
		var pellet_dir: Vector3 = (dir * cos(pellet_angle) + lateral).normalized()
		directions.append(pellet_dir)
	return directions


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
		"profile_id": profile_id,
		"profile_label": profile_label,
		"category": category,
		"damage": damage,
		"range_m": range_m,
		"cooldown_time": cooldown_time,
		"hit_radius": hit_radius,
		"projectile_count": projectile_count,
		"projectile_damage": projectile_damage(),
		"spread_deg": spread_deg,
		"falloff_start_m": falloff_start_m,
		"minimum_damage_scale": minimum_damage_scale,
		"ideal_ttk_5m": ideal_ttk_seconds(100.0, 5.0),
		"ideal_ttk_max_range": ideal_ttk_seconds(100.0, range_m),
		"cooldown_remaining": cooldown_remaining,
		"ready": is_ready(),
	}

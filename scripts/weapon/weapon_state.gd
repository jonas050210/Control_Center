# gdlint:ignore=max-public-methods
# The public surface is wide on purpose: this is the single authority for
# everything a weapon does (ballistics, damage falloff, TTK, recoil, bloom,
# fire modes, magazines and hit zones). Splitting handling into a second
# object would mean two places that must agree on the same shot, which is
# exactly the kind of drift that produces a simulator whose documented TTK
# does not match its behavior.
## WeaponState
##
## Deterministic hitscan weapon with named *local calibration* profiles. The
## bundled values preserve historical SandboxAI experiments; they are not a
## claim about TTK Testing's current weapon roster, handling, damage or TTK.
## Treat any profile as a replaceable calibration input only after player-
## visible evidence has been collected (see docs/TTK_TESTING_REFERENCE.md).
## Profiles are simulation metadata, not new policy actions: the PPO action
## space still has exactly one shoot button.
##
## ## Weapon handling (optional layer)
##
## On top of damage/range/cooldown the class models the "weighty handling"
## half of a tactical shooter, because that is what actually separates the
## roles once both sides can shoot:
##
##   * **fire mode** — `auto` fires while the trigger is held; `semi` and
##     `pump` need the trigger released between shots (a rising edge). The
##     action space is unchanged: the edge is derived from the shoot bit.
##   * **recoil** — every shot kicks the shooter's own view up and sideways
##     along a FIXED pattern, and the view recovers after a short pause.
##     The policy perceives this through `agent_forward` / bearing /
##     elevation and counters it with the look axes; nothing was added to
##     the 126-float observation.
##   * **bloom** — sustained fire, movement and being airborne widen the
##     cone of fire. A settled weapon's first shot is *exactly* pinpoint,
##     which keeps the early aiming curriculum learnable.
##   * **magazine / reload** — a shot costs a round; an empty magazine
##     forces a reload during which `is_ready()` is false, so the existing
##     `weapon_ready` observation bit already reports it honestly.
##   * **hit zones** — a smaller head sphere in front of the body sphere
##     with a per-profile damage multiplier, which is what makes the pitch
##     axis worth controlling.
##
## Every one of those is deterministic: recoil follows a closed-form
## pattern and bloom offsets come from a low-discrepancy sequence keyed on
## the shot index, so no RNG is consumed and seeded replays stay stable.
##
## The whole layer is inert unless `handling_enabled` is set (EnvironmentCore
## sets it from `CurriculumConfig.weapon_handling_enabled()`), so curriculum
## levels 1-4, the scripted enemies and every existing unit test keep the
## original cooldown-only behavior bit-for-bit.
class_name WeaponState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const PROFILE_RIFLE: String = "rifle"
const PROFILE_SHOTGUN: String = "shotgun"
const PROFILE_PISTOL: String = "pistol"
const PROFILE_SMG: String = "smg"
const PROFILE_IDS: Array = [PROFILE_RIFLE, PROFILE_SHOTGUN, PROFILE_PISTOL, PROFILE_SMG]

## Trigger semantics. `auto` keeps firing while the bit is held; `semi` and
## `pump` require the policy to release the trigger between shots.
const FIRE_MODE_AUTO: String = "auto"
const FIRE_MODE_SEMI: String = "semi"
const FIRE_MODE_PUMP: String = "pump"
const FIRE_MODE_IDS: Array = [FIRE_MODE_AUTO, FIRE_MODE_SEMI, FIRE_MODE_PUMP]

## Hit zones resolved by `resolve_hit_zone()`.
const ZONE_NONE: String = "none"
const ZONE_HEAD: String = "head"
const ZONE_BODY: String = "body"

const PROFILE_DEFINITIONS: Dictionary = {
	"rifle":
	{
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
		"fire_mode": FIRE_MODE_AUTO,
		"magazine_size": 30,
		"reload_time": 2.2,
		"recoil_vertical_deg": 0.9,
		"recoil_horizontal_deg": 0.35,
		"recoil_recovery_deg_per_s": 18.0,
		"spread_per_shot_deg": 0.55,
		"spread_move_deg": 1.6,
		"spread_air_deg": 4.0,
		"spread_max_deg": 4.5,
		"spread_recovery_deg_per_s": 4.0,
		"headshot_multiplier": 2.0,
		"move_speed_scale_firing": 0.75,
		"description":
		"Baseline single-ray rifle: controllable mid-range damage with mild falloff.",
	},
	"shotgun":
	{
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
		"fire_mode": FIRE_MODE_PUMP,
		"magazine_size": 6,
		"reload_time": 3.0,
		"recoil_vertical_deg": 3.2,
		"recoil_horizontal_deg": 0.8,
		"recoil_recovery_deg_per_s": 14.0,
		"spread_per_shot_deg": 0.0,
		"spread_move_deg": 1.2,
		"spread_air_deg": 3.0,
		"spread_max_deg": 4.2,
		"spread_recovery_deg_per_s": 6.0,
		"headshot_multiplier": 1.5,
		"move_speed_scale_firing": 0.7,
		"description":
		"Close-range 8-pellet pattern; one-shot potential requires a centered close hit.",
	},
	"pistol":
	{
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
		"fire_mode": FIRE_MODE_SEMI,
		"magazine_size": 15,
		"reload_time": 1.8,
		"recoil_vertical_deg": 1.1,
		"recoil_horizontal_deg": 0.45,
		"recoil_recovery_deg_per_s": 22.0,
		"spread_per_shot_deg": 0.3,
		"spread_move_deg": 1.4,
		"spread_air_deg": 3.5,
		"spread_max_deg": 3.0,
		"spread_recovery_deg_per_s": 5.0,
		"headshot_multiplier": 2.0,
		"move_speed_scale_firing": 0.85,
		"description": "Lower damage precision backup for deliberate finishing drills.",
	},
	"smg":
	{
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
		"fire_mode": FIRE_MODE_AUTO,
		"magazine_size": 30,
		"reload_time": 2.0,
		"recoil_vertical_deg": 0.55,
		"recoil_horizontal_deg": 0.45,
		"recoil_recovery_deg_per_s": 16.0,
		"spread_per_shot_deg": 0.42,
		"spread_move_deg": 2.2,
		"spread_air_deg": 4.5,
		"spread_max_deg": 6.0,
		"spread_recovery_deg_per_s": 5.5,
		"headshot_multiplier": 1.8,
		"move_speed_scale_firing": 0.8,
		"description":
		"Fast close-range profile with sharp falloff that rewards sustained tracking.",
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

# --- Handling profile (data) -----------------------------------------------
var fire_mode: String = FIRE_MODE_AUTO
var magazine_size: int = 30
var reload_time: float = 2.2
var recoil_vertical_deg: float = 0.0
var recoil_horizontal_deg: float = 0.0
var recoil_recovery_deg_per_s: float = 18.0
var spread_per_shot_deg: float = 0.0
var spread_move_deg: float = 0.0
var spread_air_deg: float = 0.0
var spread_max_deg: float = 0.0
var spread_recovery_deg_per_s: float = 4.0
var headshot_multiplier: float = 1.0
var move_speed_scale_firing: float = 1.0

# --- Handling state --------------------------------------------------------
## Master switch. False reproduces the pre-handling weapon exactly.
var handling_enabled: bool = false
## Rounds left in the magazine. Only consumed while handling is enabled.
var ammo_in_magazine: int = 30
## Seconds left of the current reload (0 = not reloading).
var reload_remaining: float = 0.0
## Accumulated, not-yet-recovered recoil offset applied to the shooter's aim.
var recoil_pitch_deg: float = 0.0
var recoil_yaw_deg: float = 0.0
## Current bloom contributed by sustained fire (degrees). Movement and
## airborne penalties are added on top at query time.
var bloom_deg: float = 0.0
## Seconds since the last shot. Drives recoil/bloom recovery and the
## post-shot movement slow.
var time_since_fire: float = 999.0
## Trigger state of the previous tick, used for semi/pump edge detection.
var trigger_held: bool = false
## Shots fired since this weapon was reset. Indexes the recoil pattern and
## the deterministic bloom sequence.
var shots_fired: int = 0
## Shots fired in the current uninterrupted burst. Reset by a recovery pause.
var burst_index: int = 0
## Completed reloads this episode (diagnostics only).
var reload_count: int = 0


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
	_configure_handling(spec)
	return true


## Handling half of a profile. Split out so the (long) profile application
## stays readable and so a caller can see at a glance that every handling
## field has a safe, inert default.
func _configure_handling(spec: Dictionary) -> void:
	var mode: String = str(spec.get("fire_mode", FIRE_MODE_AUTO))
	fire_mode = mode if FIRE_MODE_IDS.has(mode) else FIRE_MODE_AUTO
	magazine_size = maxi(1, int(spec.get("magazine_size", 30)))
	reload_time = maxf(0.0, float(spec.get("reload_time", 2.0)))
	recoil_vertical_deg = maxf(0.0, float(spec.get("recoil_vertical_deg", 0.0)))
	recoil_horizontal_deg = maxf(0.0, float(spec.get("recoil_horizontal_deg", 0.0)))
	recoil_recovery_deg_per_s = maxf(0.0, float(spec.get("recoil_recovery_deg_per_s", 18.0)))
	spread_per_shot_deg = maxf(0.0, float(spec.get("spread_per_shot_deg", 0.0)))
	spread_move_deg = maxf(0.0, float(spec.get("spread_move_deg", 0.0)))
	spread_air_deg = maxf(0.0, float(spec.get("spread_air_deg", 0.0)))
	spread_max_deg = maxf(0.0, float(spec.get("spread_max_deg", 0.0)))
	spread_recovery_deg_per_s = maxf(0.0, float(spec.get("spread_recovery_deg_per_s", 4.0)))
	headshot_multiplier = maxf(1.0, float(spec.get("headshot_multiplier", 1.0)))
	move_speed_scale_firing = clampf(float(spec.get("move_speed_scale_firing", 1.0)), 0.1, 1.0)
	ammo_in_magazine = magazine_size


func reset() -> void:
	cooldown_remaining = 0.0
	ammo_in_magazine = magazine_size
	reload_remaining = 0.0
	recoil_pitch_deg = 0.0
	recoil_yaw_deg = 0.0
	bloom_deg = 0.0
	time_since_fire = 999.0
	trigger_held = false
	shots_fired = 0
	burst_index = 0
	reload_count = 0


## Advances cooldown only. Kept byte-identical to the pre-handling version
## so the scripted enemies and every direct caller are unaffected.
func tick(dt: float) -> void:
	if cooldown_remaining > 0.0:
		cooldown_remaining = maxf(0.0, cooldown_remaining - dt)


## Advances cooldown, reload, recoil recovery and bloom recovery, and
## returns the aim correction the shooter should apply this tick:
## `{"pitch_deg": float, "yaw_deg": float, "reload_finished": bool}`.
##
## The returned angles are the amount of previously-applied recoil that has
## now recovered; the caller subtracts them from its own yaw/pitch, which is
## what makes the view settle back toward where the player was aiming.
func tick_handling(dt: float) -> Dictionary:
	tick(dt)
	var recovered := {"pitch_deg": 0.0, "yaw_deg": 0.0, "reload_finished": false}
	if not handling_enabled:
		return recovered
	time_since_fire += dt
	if reload_remaining > 0.0:
		reload_remaining = maxf(0.0, reload_remaining - dt)
		if reload_remaining <= 0.0:
			ammo_in_magazine = magazine_size
			reload_count += 1
			recovered["reload_finished"] = true
	if time_since_fire < SandboxConfig.RECOIL_RECOVERY_DELAY:
		return recovered
	burst_index = 0
	if bloom_deg > 0.0:
		bloom_deg = maxf(0.0, bloom_deg - spread_recovery_deg_per_s * dt)
	var budget: float = recoil_recovery_deg_per_s * dt
	if budget <= 0.0:
		return recovered
	var pitch_step: float = clampf(recoil_pitch_deg, -budget, budget)
	var yaw_step: float = clampf(recoil_yaw_deg, -budget, budget)
	recoil_pitch_deg -= pitch_step
	recoil_yaw_deg -= yaw_step
	recovered["pitch_deg"] = pitch_step
	recovered["yaw_deg"] = yaw_step
	return recovered


func is_reloading() -> bool:
	return handling_enabled and reload_remaining > 0.0


func is_magazine_empty() -> bool:
	return handling_enabled and ammo_in_magazine <= 0


## Whether the weapon could fire *right now* if the trigger were pressed
## correctly. Feeds observation index 15 (`weapon_ready`), which is why a
## reloading weapon must report false: the policy is entitled to know that
## its trigger currently does nothing.
func is_ready() -> bool:
	if cooldown_remaining > 0.0:
		return false
	if handling_enabled and (reload_remaining > 0.0 or ammo_in_magazine <= 0):
		return false
	return true


## Attempts to fire. Returns true and starts the cooldown if the weapon was
## ready; returns false (no effect) if it was still on cooldown.
##
## This is the legacy entry point and stays semantically unchanged for
## `handling_enabled == false`. With handling on it additionally consumes a
## round and starts a reload when the magazine runs dry, but it does NOT
## apply recoil or bloom — callers that want the full model use
## `pull_trigger()`.
func try_fire() -> bool:
	if not is_ready():
		return false
	cooldown_remaining = cooldown_time
	if handling_enabled:
		_consume_round()
	return true


func _consume_round() -> void:
	ammo_in_magazine = maxi(0, ammo_in_magazine - 1)
	shots_fired += 1
	burst_index += 1
	time_since_fire = 0.0
	if ammo_in_magazine <= 0 and reload_time > 0.0:
		reload_remaining = reload_time


## Starts a reload if one is possible and not already running. Returns true
## when a reload was actually started.
func begin_reload() -> bool:
	if not handling_enabled or reload_time <= 0.0:
		return false
	if reload_remaining > 0.0 or ammo_in_magazine >= magazine_size:
		return false
	reload_remaining = reload_time
	return true


## Full trigger resolution for one tick.
##
## `trigger_down` is the raw shoot bit; `speed_fraction` is the shooter's
## horizontal speed as a fraction of its move speed; `airborne` is true
## while it is off the ground. Returns:
##
##   fired            the weapon discharged this tick
##   blocked_reason   "" when it fired, else one of
##                    "released" / "cycling" / "reloading" / "empty" /
##                    "needs_release"
##   recoil_pitch_deg / recoil_yaw_deg   view kick to apply to the shooter
##   spread_deg       cone half-angle this shot was fired with
##   shot_index       index of this shot in the deterministic sequence
##
## The trigger edge is tracked here, so callers must invoke this exactly
## once per tick per weapon.
func pull_trigger(
	trigger_down: bool, speed_fraction: float = 0.0, airborne: bool = false
) -> Dictionary:
	var result: Dictionary = {
		"fired": false,
		"blocked_reason": "released",
		"recoil_pitch_deg": 0.0,
		"recoil_yaw_deg": 0.0,
		"spread_deg": 0.0,
		"shot_index": shots_fired,
	}
	if not handling_enabled:
		# Legacy path: no edge detection, no ammo, no recoil.
		trigger_held = trigger_down
		if not trigger_down:
			return result
		if try_fire():
			result["fired"] = true
			result["blocked_reason"] = ""
		else:
			result["blocked_reason"] = "cycling"
		return result

	var was_held: bool = trigger_held
	trigger_held = trigger_down
	var blocked: String = _trigger_block_reason(trigger_down, was_held)
	if not blocked.is_empty():
		result["blocked_reason"] = blocked
		if blocked == "empty":
			# An empty magazine auto-reloads on the trigger pull: there is
			# no reload button in the action space, and leaving the agent
			# with a permanently dead weapon would make the episode
			# unwinnable.
			begin_reload()
		return result

	var applied_spread: float = current_spread_deg(speed_fraction, airborne)
	var index: int = shots_fired
	cooldown_remaining = cooldown_time
	_consume_round()
	bloom_deg = minf(spread_max_deg, bloom_deg + spread_per_shot_deg)
	var kick: Dictionary = recoil_kick(burst_index - 1)
	var pitch_kick: float = float(kick["pitch_deg"])
	var yaw_kick: float = float(kick["yaw_deg"])
	# Clamp the ACCUMULATED offset, not the per-shot kick, so a long spray
	# saturates instead of walking the camera off the target forever.
	var next_pitch: float = clampf(
		recoil_pitch_deg + pitch_kick, -SandboxConfig.RECOIL_MAX_DEG, SandboxConfig.RECOIL_MAX_DEG
	)
	var next_yaw: float = clampf(
		recoil_yaw_deg + yaw_kick, -SandboxConfig.RECOIL_MAX_DEG, SandboxConfig.RECOIL_MAX_DEG
	)
	result["recoil_pitch_deg"] = next_pitch - recoil_pitch_deg
	result["recoil_yaw_deg"] = next_yaw - recoil_yaw_deg
	recoil_pitch_deg = next_pitch
	recoil_yaw_deg = next_yaw
	result["fired"] = true
	result["blocked_reason"] = ""
	result["spread_deg"] = applied_spread
	result["shot_index"] = index
	return result


## Why this trigger pull cannot fire, or "" when it can. Split out of
## `pull_trigger()` so the decision table is readable in one place.
func _trigger_block_reason(trigger_down: bool, was_held: bool) -> String:
	if not trigger_down:
		return "released"
	if reload_remaining > 0.0:
		return "reloading"
	if ammo_in_magazine <= 0:
		return "empty"
	if cooldown_remaining > 0.0:
		return "cycling"
	if fire_mode != FIRE_MODE_AUTO and was_held:
		return "needs_release"
	return ""


## Deterministic recoil pattern for the n-th shot of a burst.
##
## Vertical climb is strongest on the first shots and saturates toward
## `RECOIL_SUSTAIN_SCALE` of it, which is what a real muzzle-rise curve
## looks like and what makes "tap, tap, pause" beat "hold forever".
## Horizontal drift walks a fixed golden-angle sine so the pattern is
## repeatable (learnable) without being a straight line.
func recoil_kick(index: int) -> Dictionary:
	var n: int = maxi(0, index)
	var pattern_length: float = float(maxi(1, SandboxConfig.RECOIL_PATTERN_LENGTH))
	var progress: float = clampf(float(n) / pattern_length, 0.0, 1.0)
	var vertical: float = (
		recoil_vertical_deg * lerpf(1.0, SandboxConfig.RECOIL_SUSTAIN_SCALE, progress)
	)
	var horizontal: float = (
		recoil_horizontal_deg * sin(float(n) * SandboxConfig.HANDLING_GOLDEN_ANGLE)
	)
	return {"pitch_deg": vertical, "yaw_deg": horizontal}


## Cone half-angle (degrees) a shot fired right now would use.
func current_spread_deg(speed_fraction: float = 0.0, airborne: bool = false) -> float:
	if not handling_enabled:
		return 0.0
	var total: float = bloom_deg
	total += spread_move_deg * clampf(speed_fraction, 0.0, 1.0)
	if airborne:
		total += spread_air_deg
	var ceiling: float = maxf(spread_max_deg, spread_air_deg + spread_move_deg)
	return clampf(total, 0.0, ceiling)


## Whether the weapon is settled enough that the next shot is pinpoint.
func is_settled(speed_fraction: float = 0.0, airborne: bool = false) -> bool:
	return current_spread_deg(speed_fraction, airborne) <= SandboxConfig.SPREAD_SETTLED_EPSILON


## Move-speed multiplier the shooter should use this tick. Firing plants
## you: the slow lasts FIRE_MOVEMENT_SLOW_DURATION after the shot.
func movement_speed_scale() -> float:
	if not handling_enabled:
		return 1.0
	if time_since_fire >= SandboxConfig.FIRE_MOVEMENT_SLOW_DURATION:
		return 1.0
	return move_speed_scale_firing


## Deterministic aim perturbation for shot `index` at `spread` degrees.
##
## Uses an R1 low-discrepancy radius and a golden-angle azimuth, so
## successive shots fill the cone evenly and the exact same sequence
## replays every time. Returns the unit direction to fire along.
func apply_spread(
	direction: Vector3, spread: float, index: int, up_hint: Vector3 = Vector3.UP
) -> Vector3:
	var dir: Vector3 = direction.normalized() if not direction.is_zero_approx() else Vector3.FORWARD
	if spread <= SandboxConfig.SPREAD_SETTLED_EPSILON:
		return dir
	var right: Vector3 = dir.cross(up_hint).normalized()
	if right.is_zero_approx():
		right = Vector3.RIGHT
	var up: Vector3 = right.cross(dir).normalized()
	if up.is_zero_approx():
		up = Vector3.UP
	var n: float = float(maxi(0, index) + 1)
	var radius_fraction: float = fposmod(n * SandboxConfig.HANDLING_R1_ALPHA, 1.0)
	var azimuth: float = n * SandboxConfig.HANDLING_GOLDEN_ANGLE
	var angle: float = deg_to_rad(spread) * sqrt(clampf(radius_fraction, 0.0, 1.0))
	var lateral: Vector3 = right * cos(azimuth) * sin(angle) + up * sin(azimuth) * sin(angle)
	var result: Vector3 = dir * cos(angle) + lateral
	return result.normalized() if not result.is_zero_approx() else dir


## Resolves which body zone a ray hit, given the target's feet position.
##
## The head sphere is tested first and is strictly smaller than the body
## sphere, so a head hit is a sub-region of a body hit rather than an extra
## chance to connect. Returns {"zone", "distance", "multiplier"}; `zone` is
## ZONE_NONE when the ray misses entirely.
func resolve_hit_zone(
	origin: Vector3,
	direction: Vector3,
	body_center: Vector3,
	head_center: Vector3,
	head_zone_enabled: bool = true
) -> Dictionary:
	var body_distance: float = ray_hit_distance(origin, direction, body_center)
	if body_distance < 0.0:
		return {"zone": ZONE_NONE, "distance": -1.0, "multiplier": 0.0}
	if head_zone_enabled and headshot_multiplier > 1.0:
		var head_radius: float = hit_radius * SandboxConfig.HEAD_HIT_RADIUS_SCALE
		var head_distance: float = _ray_sphere_distance(origin, direction, head_center, head_radius)
		if head_distance >= 0.0:
			return {
				"zone": ZONE_HEAD,
				"distance": head_distance,
				"multiplier": headshot_multiplier,
			}
	return {"zone": ZONE_BODY, "distance": body_distance, "multiplier": 1.0}


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
func ideal_ttk_seconds(target_health: float, distance_m: float, pellets_landed: int = -1) -> float:
	var landed: int = (
		projectile_count if pellets_landed < 0 else clampi(pellets_landed, 0, projectile_count)
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
			right * cos(angle) * sin(pellet_angle) + up * sin(angle) * sin(pellet_angle)
		)
		var pellet_dir: Vector3 = (dir * cos(pellet_angle) + lateral).normalized()
		directions.append(pellet_dir)
	return directions


## Computes the distance along the ray where it intersects/passes closest
## to the target sphere. Returns the distance `t` (>= 0.0) if a hit occurs
## within `range_m`, or -1.0 if it misses or lies behind the origin.
func ray_hit_distance(origin: Vector3, direction: Vector3, target_center: Vector3) -> float:
	return _ray_sphere_distance(origin, direction, target_center, hit_radius)


## Shared ray/sphere closest-approach test with an explicit radius, so the
## head zone can reuse the exact same geometry as the body zone instead of
## a second, subtly different implementation.
func _ray_sphere_distance(
	origin: Vector3, direction: Vector3, target_center: Vector3, radius: float
) -> float:
	if direction.is_zero_approx():
		return -1.0
	var dir: Vector3 = direction.normalized()
	var to_target: Vector3 = target_center - origin
	var t: float = to_target.dot(dir)
	if t < 0.0 or t > range_m:
		return -1.0
	var closest_point: Vector3 = origin + dir * t
	var dist_sq: float = closest_point.distance_squared_to(target_center)
	if dist_sq <= radius * radius:
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
		"handling_enabled": handling_enabled,
		"fire_mode": fire_mode,
		"magazine_size": magazine_size,
		"ammo_in_magazine": ammo_in_magazine,
		"reload_time": reload_time,
		"reload_remaining": reload_remaining,
		"reloading": is_reloading(),
		"reload_count": reload_count,
		"recoil_pitch_deg": recoil_pitch_deg,
		"recoil_yaw_deg": recoil_yaw_deg,
		"bloom_deg": bloom_deg,
		"current_spread_deg": current_spread_deg(),
		"headshot_multiplier": headshot_multiplier,
		"shots_fired": shots_fired,
		"burst_index": burst_index,
	}


## Ideal TTK including the magazine/reload cost, which is what actually
## decides a long fight. `pellets_landed` mirrors ideal_ttk_seconds().
func sustained_ttk_seconds(
	target_health: float, distance_m: float, pellets_landed: int = -1
) -> float:
	var landed: int = (
		projectile_count if pellets_landed < 0 else clampi(pellets_landed, 0, projectile_count)
	)
	var volley_damage: float = projectile_damage_at_distance(distance_m) * float(landed)
	if volley_damage <= 0.0:
		return INF
	var volleys: int = ceili(maxf(0.0, target_health) / volley_damage)
	var base: float = float(maxi(0, volleys - 1)) * cooldown_time
	if not handling_enabled or magazine_size <= 0:
		return base
	var reloads: int = int(floor(float(maxi(0, volleys - 1)) / float(magazine_size)))
	return base + float(reloads) * reload_time

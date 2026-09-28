## LightingProfile
##
## Environmental visibility conditions, modelled as something that CHANGES
## PERCEPTION rather than as a flag appended to the observation vector.
##
## The design rule for this layer: a policy must never read "night = true"
## and switch to a memorized night strategy. What it can read is the
## consequences — it acquires targets later, loses them sooner, sees less
## far, and therefore has to lean on sound and memory. The only lighting
## information that reaches the observation is `local_illumination`, i.e.
## how bright it is where the agent is standing, which is exactly what a
## human standing there perceives.
##
## Everything is a deterministic pure function of (mode, seed, position).
## No RandomNumberGenerator is consulted at query time, so two runs of the
## same episode produce identical visibility everywhere, and the Map
## Analyzer can replay a lighting map from its seed alone.
##
## Two mechanisms are combined:
##
##  * **Illumination** — how lit a point is, in [0, 1]. Uniform for the
##    NORMAL/LOW_LIGHT/NIGHT modes; spatially varying (deterministic
##    value-noise patches) for MIXED and HIGH_CONTRAST, which is what gives
##    a map genuinely dark corners next to genuinely bright lanes.
##  * **Transmittance** — how much of the line of sight survives the medium
##    over a distance. Beer-Lambert style exponential falloff driven by
##    `fog_density`, which is what makes FOG a distance limit rather than a
##    brightness limit.
class_name LightingProfile
extends RefCounted

enum Mode {
	NORMAL = 0,
	LOW_LIGHT = 1,
	NIGHT = 2,
	FOG = 3,
	HIGH_CONTRAST = 4,
	MIXED = 5,
}

const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SELF_PATH: String = "res://scripts/perception/lighting_profile.gd"

## Stable string ids, used by map definitions, the CLI and telemetry. Kept
## separate from the enum ordinals because the ordinals are an
## implementation detail while these are persisted.
const MODE_IDS: Array = ["normal", "low_light", "night", "fog", "high_contrast", "mixed"]

## Size (meters) of one illumination patch for the spatially varying modes.
## Roughly a room's worth of light, not a per-pixel texture.
const PATCH_SIZE: float = 4.0

var mode: int = Mode.NORMAL
## Seed for the spatial patch pattern. Two profiles with the same mode and
## seed are identical everywhere.
var seed_value: int = 0
## Illumination in fully lit conditions for this mode, in [0, 1].
var base_illumination: float = 1.0
## How far illumination swings around `base_illumination` between patches.
## Zero for the uniform modes.
var illumination_variance: float = 0.0
## Beer-Lambert extinction coefficient, per meter. Zero means clear air.
var fog_density: float = 0.0


static func create(p_mode: int = Mode.NORMAL, p_seed: int = 0) -> LightingProfile:
	var profile: LightingProfile = (load(SELF_PATH) as GDScript).new() as LightingProfile
	profile.configure(p_mode, p_seed)
	return profile


## Builds a profile from a stable string id. Unknown ids degrade to NORMAL
## rather than failing, so a stale map definition dims nothing instead of
## crashing a training run.
static func from_id(mode_id: String, p_seed: int = 0) -> LightingProfile:
	return create(mode_from_id(mode_id), p_seed)


static func mode_from_id(mode_id: String) -> int:
	var index: int = MODE_IDS.find(mode_id)
	return index if index >= 0 else Mode.NORMAL


static func mode_id(value: int) -> String:
	if value < 0 or value >= MODE_IDS.size():
		return "normal"
	return str(MODE_IDS[value])


func configure(p_mode: int, p_seed: int = 0) -> void:
	mode = p_mode if p_mode >= 0 and p_mode < MODE_IDS.size() else Mode.NORMAL
	seed_value = p_seed
	match mode:
		Mode.LOW_LIGHT:
			base_illumination = 0.55
			illumination_variance = 0.0
			fog_density = 0.0
		Mode.NIGHT:
			base_illumination = 0.25
			illumination_variance = 0.0
			fog_density = 0.0
		Mode.FOG:
			base_illumination = 0.8
			illumination_variance = 0.0
			fog_density = 0.075
		Mode.HIGH_CONTRAST:
			# Bright lanes, near-black shadows: the widest swing of all.
			base_illumination = 0.6
			illumination_variance = 0.45
			fog_density = 0.0
		Mode.MIXED:
			base_illumination = 0.65
			illumination_variance = 0.25
			fog_density = 0.01
		_:
			base_illumination = 1.0
			illumination_variance = 0.0
			fog_density = 0.0


## Perceived brightness at a world position, in [0, 1].
##
## For the uniform modes this ignores the position entirely. For the
## spatially varying modes it is bilinear value noise over PATCH_SIZE
## cells, which produces smooth light/dark regions instead of a checker
## pattern an agent could trivially memorize.
func illumination_at(position: Vector3) -> float:
	if illumination_variance <= 0.0:
		return clampf(base_illumination, 0.0, 1.0)
	var u: float = position.x / PATCH_SIZE
	var v: float = position.z / PATCH_SIZE
	var cell_x: int = int(floor(u))
	var cell_z: int = int(floor(v))
	var fx: float = u - float(cell_x)
	var fz: float = v - float(cell_z)
	# Smoothstep the interpolation weights so patch borders are not creases.
	var wx: float = fx * fx * (3.0 - 2.0 * fx)
	var wz: float = fz * fz * (3.0 - 2.0 * fz)
	var c00: float = _cell_value(cell_x, cell_z)
	var c10: float = _cell_value(cell_x + 1, cell_z)
	var c01: float = _cell_value(cell_x, cell_z + 1)
	var c11: float = _cell_value(cell_x + 1, cell_z + 1)
	var top: float = lerpf(c00, c10, wx)
	var bottom: float = lerpf(c01, c11, wx)
	var noise: float = lerpf(top, bottom, wz) * 2.0 - 1.0
	return clampf(base_illumination + noise * illumination_variance, 0.05, 1.0)


## Deterministic hash in [0, 1) for one patch corner. Integer mixing only:
## no RandomNumberGenerator, no floating-point ordering assumptions.
func _cell_value(cell_x: int, cell_z: int) -> float:
	var h: int = cell_x * 374761393 + cell_z * 668265263 + seed_value * 2147483647
	h = (h ^ (h >> 13)) * 1274126177
	h = h ^ (h >> 16)
	return float(absi(h) % 100000) / 100000.0


## Fraction of a sight line that survives the medium over `distance`.
## 1.0 in clear air; exponential falloff in fog.
func transmittance(distance: float) -> float:
	if fog_density <= 0.0:
		return 1.0
	return exp(-fog_density * maxf(0.0, distance))


## Effective visual acquisition range toward a point, in meters.
##
## Darkness scales the range down linearly (a dark target has to be closer
## before it resolves), and fog imposes a hard horizon where transmittance
## drops below `SandboxConfig.LIGHTING_MIN_TRANSMITTANCE`.
func detection_range(base_range: float, target_position: Vector3) -> float:
	var illumination: float = illumination_at(target_position)
	var scale: float = (
		SandboxConfig.LIGHTING_MIN_RANGE_SCALE
		+ (1.0 - SandboxConfig.LIGHTING_MIN_RANGE_SCALE) * illumination
	)
	var ranged: float = base_range * scale
	if fog_density > 0.0:
		var horizon: float = -log(SandboxConfig.LIGHTING_MIN_TRANSMITTANCE) / fog_density
		ranged = minf(ranged, horizon)
	return maxf(0.5, ranged)


## Multiplier on the observer's visual detection delay. A dim target takes
## measurably longer to register, which is what turns poor visibility into
## a reaction-time disadvantage rather than a binary on/off.
func detection_delay_scale(target_position: Vector3) -> float:
	var illumination: float = illumination_at(target_position)
	return 1.0 + SandboxConfig.LIGHTING_DELAY_GAIN * (1.0 - illumination)


## Multiplier on how long a lost contact keeps being reported. Darkness
## makes losing a target faster, not slower: you cannot keep tracking a
## silhouette you can no longer resolve.
func loss_grace_scale(target_position: Vector3) -> float:
	var illumination: float = illumination_at(target_position)
	return clampf(0.4 + 0.6 * illumination, 0.2, 1.0)


func to_dict() -> Dictionary:
	return {
		"mode": mode,
		"mode_id": mode_id(mode),
		"seed": seed_value,
		"base_illumination": base_illumination,
		"illumination_variance": illumination_variance,
		"fog_density": fog_density,
	}

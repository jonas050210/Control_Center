## CurriculumConfig
##
## Difficulty is data, not a separate environment. Levels 1-4 progressively
## enable the existing enemy behaviors; level 5 reserves the same interface
## for a two-policy self-play match. Each level is a small bundle of flags
## (movement, attacks, spawn variety, strafing, enemy count, aggressiveness)
## layered on top of the same EnemyState/EnvironmentCore code paths — no
## parallel per-level implementation exists.
##
## Progression (see docs/CURRICULUM_AND_COMBAT.md for the full rationale):
##   1 stationary_target   - one enemy, fixed spawn directly ahead, no movement/attacks.
##   2 moving_target       - one moving enemy, spawn position/distance/angle vary per seed.
##   3 enemy_attacks       - moving + attacking enemy, spawn variety, enemies strafe.
##   4 multiple_enemies    - 3+ enemies, spawn variety, strafing, faster/more aggressive.
##   5 agent_vs_agent      - two-policy self-play hook (SelfPlayEnvironmentCore).
class_name CurriculumConfig
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")


enum Level {
	STATIONARY_TARGET = 1,
	MOVING_TARGET = 2,
	ENEMY_ATTACKS = 3,
	MULTIPLE_ENEMIES = 4,
	AGENT_VS_AGENT = 5,
}

## Minimum enemy count enforced once curriculum reaches MULTIPLE_ENEMIES.
## Raised from 2 to 3 so level 4 meaningfully differs from level 3's
## (optionally two-enemy) configuration and forces multi-target tracking.
const MULTIPLE_ENEMIES_MIN_COUNT: int = 3

var level: int = Level.ENEMY_ATTACKS
var configured_enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT


func _init(
	p_level: int = Level.ENEMY_ATTACKS, p_enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT
) -> void:
	level = clampi(p_level, Level.STATIONARY_TARGET, Level.AGENT_VS_AGENT)
	configured_enemy_count = maxi(1, p_enemy_count)


func enemy_movement_enabled() -> bool:
	return level >= Level.MOVING_TARGET


func enemy_attacks_enabled() -> bool:
	return level >= Level.ENEMY_ATTACKS and level < Level.AGENT_VS_AGENT


## Whether enemy spawn positions vary in distance and horizontal angle around
## the agent each episode (deterministic given the environment's seed), or
## stay at the fixed legacy layout directly in front of the agent.
func spawn_variety_enabled() -> bool:
	return level >= Level.MOVING_TARGET


## Whether enemies blend a sinusoidal lateral (strafing) motion into their
## chase/engage movement instead of walking straight at the agent.
func strafing_enabled() -> bool:
	return level >= Level.ENEMY_ATTACKS and level < Level.AGENT_VS_AGENT


## Half-width, in degrees, of the horizontal arc (centered on the agent's
## spawn-facing direction) enemies may spawn within once spawn variety is
## enabled. 0 keeps enemies directly ahead (legacy layout at level 1).
func spawn_angle_spread_deg() -> float:
	match level:
		Level.STATIONARY_TARGET:
			return 0.0
		Level.MOVING_TARGET:
			return 70.0
		Level.ENEMY_ATTACKS:
			return 110.0
		Level.MULTIPLE_ENEMIES:
			return 150.0
		_:
			return 180.0


## [min, max] spawn distance from the agent once spawn variety is enabled.
func spawn_distance_range() -> Vector2:
	return Vector2(SandboxConfig.ENEMY_SPAWN_MIN_DISTANCE, SandboxConfig.ENEMY_SPAWN_MAX_DISTANCE)


func effective_enemy_count() -> int:
	if level >= Level.MULTIPLE_ENEMIES and level < Level.AGENT_VS_AGENT:
		return maxi(MULTIPLE_ENEMIES_MIN_COUNT, configured_enemy_count)
	return configured_enemy_count


func target_radius_scale() -> float:
	return 1.5 if level == Level.STATIONARY_TARGET else 1.0


## Multiplier applied to SandboxConfig.ENEMY_MOVE_SPEED. Higher levels ask
## for slightly more aggressive (faster) enemies.
func enemy_speed_scale() -> float:
	match level:
		Level.MULTIPLE_ENEMIES:
			return 1.15
		Level.AGENT_VS_AGENT:
			return 1.0
		_:
			return 1.0


## Multiplier applied to SandboxConfig.ENEMY_ATTACK_COOLDOWN. Lower is more
## aggressive (enemies attack more often).
func enemy_cooldown_scale() -> float:
	match level:
		Level.MULTIPLE_ENEMIES:
			return 0.85
		_:
			return 1.0


func to_dict() -> Dictionary:
	return {
		"level": level,
		"name": level_name(level),
		"enemy_movement": enemy_movement_enabled(),
		"enemy_attacks": enemy_attacks_enabled(),
		"enemy_count": effective_enemy_count(),
		"target_radius_scale": target_radius_scale(),
		"spawn_variety": spawn_variety_enabled(),
		"strafing": strafing_enabled(),
		"spawn_angle_spread_deg": spawn_angle_spread_deg(),
		"spawn_distance_range": spawn_distance_range(),
		"enemy_speed_scale": enemy_speed_scale(),
		"enemy_cooldown_scale": enemy_cooldown_scale(),
	}


static func level_name(value: int) -> String:
	match value:
		Level.STATIONARY_TARGET:
			return "stationary_target"
		Level.MOVING_TARGET:
			return "moving_target"
		Level.ENEMY_ATTACKS:
			return "enemy_attacks"
		Level.MULTIPLE_ENEMIES:
			return "multiple_enemies"
		Level.AGENT_VS_AGENT:
			return "agent_vs_agent"
		_:
			return "unknown"

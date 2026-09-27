## CurriculumConfig
##
## Difficulty is data, not a separate environment. Levels 1-4 progressively
## enable the existing enemy behaviors; level 5 reserves the same interface
## for a two-policy self-play match.
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


func effective_enemy_count() -> int:
	if level >= Level.MULTIPLE_ENEMIES and level < Level.AGENT_VS_AGENT:
		return maxi(2, configured_enemy_count)
	return configured_enemy_count


func target_radius_scale() -> float:
	return 1.5 if level == Level.STATIONARY_TARGET else 1.0


func to_dict() -> Dictionary:
	return {
		"level": level,
		"name": level_name(level),
		"enemy_movement": enemy_movement_enabled(),
		"enemy_attacks": enemy_attacks_enabled(),
		"enemy_count": effective_enemy_count(),
		"target_radius_scale": target_radius_scale(),
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

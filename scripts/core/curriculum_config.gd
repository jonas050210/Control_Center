# gdlint:ignore=max-public-methods
# The level flags are deliberately one small predicate each: a single
# `capabilities()` Dictionary would be cheaper to lint but far easier to
# typo at the call sites, which are spread across the environment, the
# enemy brain and the Control Center.
## CurriculumConfig
##
## Difficulty is data, not a separate environment. Every level is a small
## bundle of flags layered on top of the same EnemyState / EnemyBrain /
## EnvironmentCore code paths — no parallel per-level implementation
## exists.
##
## The progression teaches one new CAPABILITY at a time rather than simply
## making the enemies stronger (see docs/CURRICULUM_AND_COMBAT.md):
##    1 stationary_target  - basic aiming: one fixed enemy directly ahead.
##    2 moving_target      - moving enemy, seeded spawn distance/angle.
##    3 enemy_attacks      - the enemy fights back and strafes.
##    4 multiple_enemies   - 3+ simultaneous threats, target selection.
##    5 obstacles_cover    - interior geometry, collision, ranged enemies
##                           that use cover. Enemy AI switches to EnemyBrain.
##    6 fov_los            - perception gating: FOV cone + occlusion. The
##                           observation stops being ground truth.
##    7 sound              - footsteps/shots/landings become perceivable.
##    8 memory             - decaying last-known positions; enemies search.
##    9 vertical           - jumping, platforms, elevation-dependent sight.
##   10 mixed_randomized   - randomized layout/scenario every episode.
##   11 agent_vs_agent     - two-policy self-play hook (SelfPlayEnvironmentCore).
##
## Levels 1-4 are bit-for-bit the pre-world behavior (no obstacles, no
## perception gating, analytic melee enemies) so previously trained
## policies and their reward curves remain reproducible.
class_name CurriculumConfig
extends RefCounted

enum Level {
	STATIONARY_TARGET = 1,
	MOVING_TARGET = 2,
	ENEMY_ATTACKS = 3,
	MULTIPLE_ENEMIES = 4,
	OBSTACLES_COVER = 5,
	FOV_LOS = 6,
	SOUND = 7,
	MEMORY_LOST_TARGETS = 8,
	VERTICAL_COMBAT = 9,
	MIXED_RANDOMIZED = 10,
	AGENT_VS_AGENT = 11,
}

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ReactionProfile = preload("res://scripts/perception/reaction_profile.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

## Highest combat level (self-play excluded). Used for clamping and by the
## automatic curriculum controller.
const MAX_COMBAT_LEVEL: int = Level.MIXED_RANDOMIZED

## Minimum enemy count enforced once curriculum reaches MULTIPLE_ENEMIES.
## Raised from 2 to 3 so level 4 meaningfully differs from level 3's
## (optionally two-enemy) configuration and forces multi-target tracking.
const MULTIPLE_ENEMIES_MIN_COUNT: int = 3

## Arena layout generated per level (see WorldGenerator.LAYOUT_IDS). Levels
## not listed use the empty "open_arena".
const LEVEL_LAYOUTS: Dictionary = {
	Level.OBSTACLES_COVER: "scattered_cover",
	Level.FOV_LOS: "corner",
	Level.SOUND: "rooms",
	Level.MEMORY_LOST_TARGETS: "corridor",
	Level.VERTICAL_COMBAT: "vertical",
	Level.MIXED_RANDOMIZED: "randomized",
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


# ---------------------------------------------------------------------------
# World / perception capability flags (levels 5-10)
#
# Each returns false for the legacy levels so levels 1-4 keep the exact
# pre-world dynamics.
# ---------------------------------------------------------------------------


## Interior geometry (walls, cover, crates) exists and characters collide
## with it. Also switches enemies from the analytic `update_ai()` chase to
## the tactical `EnemyBrain`.
func obstacles_enabled() -> bool:
	return level >= Level.OBSTACLES_COVER and level < Level.AGENT_VS_AGENT


## Same condition as obstacles: the tactical brain is what knows how to use
## cover, so the two are enabled together.
func tactical_enemies_enabled() -> bool:
	return obstacles_enabled()


## Enemies shoot instead of meleeing.
func ranged_enemies_enabled() -> bool:
	return obstacles_enabled()


## The agent's observation is gated by field of view and line of sight.
func perception_enabled() -> bool:
	return level >= Level.FOV_LOS and level < Level.AGENT_VS_AGENT


## Sound events are emitted and perceivable.
func sound_enabled() -> bool:
	return level >= Level.SOUND and level < Level.AGENT_VS_AGENT


## Lost contacts decay through EnemyMemory instead of vanishing instantly.
func memory_enabled() -> bool:
	return level >= Level.MEMORY_LOST_TARGETS and level < Level.AGENT_VS_AGENT


## Jumping, gravity-relevant geometry and elevation-dependent sight lines.
func vertical_enabled() -> bool:
	return level >= Level.VERTICAL_COMBAT and level < Level.AGENT_VS_AGENT


## A new randomly chosen scenario/layout every episode.
func randomized_scenarios_enabled() -> bool:
	return level == Level.MIXED_RANDOMIZED


## Arena layout generated for this level (see WorldGenerator.LAYOUT_IDS).
func layout_id() -> String:
	return str(LEVEL_LAYOUTS.get(level, "open_arena"))


## Reaction-latency archetype the enemies of this level use. Higher levels
## get sharper opponents, but never instant ones.
func enemy_archetype() -> int:
	match level:
		Level.OBSTACLES_COVER:
			return ReactionProfile.Archetype.ROOKIE
		Level.FOV_LOS, Level.SOUND:
			return ReactionProfile.Archetype.REGULAR
		Level.MEMORY_LOST_TARGETS, Level.VERTICAL_COMBAT:
			return ReactionProfile.Archetype.VETERAN
		Level.MIXED_RANDOMIZED:
			return ReactionProfile.Archetype.VETERAN
		_:
			return ReactionProfile.Archetype.REGULAR


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
		"obstacles": obstacles_enabled(),
		"tactical_enemies": tactical_enemies_enabled(),
		"ranged_enemies": ranged_enemies_enabled(),
		"perception": perception_enabled(),
		"sound": sound_enabled(),
		"memory": memory_enabled(),
		"vertical": vertical_enabled(),
		"randomized_scenarios": randomized_scenarios_enabled(),
		"layout_id": layout_id(),
		"enemy_archetype": enemy_archetype(),
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
		Level.OBSTACLES_COVER:
			return "obstacles_cover"
		Level.FOV_LOS:
			return "fov_los"
		Level.SOUND:
			return "sound"
		Level.MEMORY_LOST_TARGETS:
			return "memory_lost_targets"
		Level.VERTICAL_COMBAT:
			return "vertical_combat"
		Level.MIXED_RANDOMIZED:
			return "mixed_randomized"
		Level.AGENT_VS_AGENT:
			return "agent_vs_agent"
		_:
			return "unknown"

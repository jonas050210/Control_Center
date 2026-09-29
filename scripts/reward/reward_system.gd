## RewardSystem
##
## Centralized, configurable reward calculation. All magic numbers live in
## SandboxConfig; this class only combines the events that happened during a
## single tick into a scalar reward. Kept as a stateless static utility so
## it is trivial to unit test in isolation.
class_name RewardSystem
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")



## `events` expected keys (all optional, default to "nothing happened"):
##   hit: bool                 -- weapon shot connected with an enemy this tick
##   kill: bool                -- that hit (or a prior one) killed an enemy this tick
##   damage_taken: float       -- HP lost by the agent this tick (>= 0)
##   damage_dealt: float       -- HP removed from enemies this tick (>= 0)
##   died: bool                -- the agent died this tick
##   useless_shot: bool        -- trigger pull that could not plausibly
##                                 connect: weapon still on cooldown (nothing
##                                 fired), no alive target, target blocked /
##                                 out of range, or aim nowhere near a target.
##                                 Penalized with PENALTY_USELESS_SHOT to teach
##                                 trigger discipline.
##   missed_shot: bool         -- weapon actually fired near a clear live
##                                 target but the exact ray did not connect.
##                                 Penalized with the much cheaper
##                                 PENALTY_MISSED_SHOT so that fine-aim
##                                 exploration keeps a positive expected value.
##   trigger_discipline: bool  -- weapon handling ON only: the trigger was
##                                 held while the weapon physically could
##                                 not fire (cycling, reloading, or a
##                                 semi-auto awaiting a release). Holding
##                                 the trigger on an automatic weapon is
##                                 correct play, so this costs far less
##                                 than a useless shot.
##   weapon_damage: float      -- nominal per-trigger-pull damage of the
##                                 weapon that fired, used to normalize the
##                                 flat hit bonus across weapon roles.
##   projectiles_fired/hit: int -- pellet accounting, used to prorate the
##                                 flat hit bonus for shotgun-style volleys.
##   positioning_delta: float  -- meters the agent closed toward the enemy
##                                 this tick while not already at an
##                                 effective engagement range (can be
##                                 negative if it moved away)
##   aiming_delta: float       -- change in forward-vector alignment toward
##                                 the live target this tick
##   target_hittable: bool     -- the current target is clear/in range for the
##                                 true weapon ray (gates aim shaping)
##   valid_target: bool        -- a live target exists
##   meaningful_action: bool   -- movement/valid-aiming progress or a fired shot
##   alive: bool               -- whether the agent is alive at the end of
##                                 the tick
##   survival_reward_allowed: bool -- explicit non-combat opt-in for the legacy
##                                 reward_survive component. Combat never sets
##                                 it, so timeout/passive farming is impossible.
##   exploration_gain: float   -- Map Analyzer mode only: value of the map
##                                 cells newly observed this tick, already
##                                 normalized by grid size by MapAnalyzer.
##                                 Absent (0) in every combat mode.
##   exploration_complete: bool -- the map reached the target coverage this
##                                 tick (paid once per episode)


## Weapon-normalized flat hit bonus.
##
## The flat bonus is paid per CONNECTING TRIGGER PULL. Left unnormalized it
## silently makes the episode return a function of fire rate: a 14 HP SMG
## needs 8 connecting pulls to kill and would collect 8x the flat bonus of
## a 112 HP shotgun shell, so "which weapon did the scenario hand me"
## would dominate cross-weapon evaluation, curriculum promotion thresholds
## and league scores. Scaling by nominal damage relative to the reference
## rifle makes the total flat bonus per kill roughly weapon-independent.
##
## `projectiles_hit / projectiles_fired` additionally prorates pellet
## weapons, so clipping a shotgun target with one pellet is not worth a
## full centered shell. With the reference rifle (damage 25, one
## projectile, fully landed) the factor is exactly 1.0, which is why the
## legacy levels are bit-for-bit unchanged.
static func hit_reward_scale(events: Dictionary) -> float:
	var weapon_damage: float = float(events.get("weapon_damage", 0.0))
	if weapon_damage <= 0.0:
		return 1.0
	var reference: float = maxf(0.0001, SandboxConfig.REWARD_HIT_REFERENCE_DAMAGE)
	var scale: float = clampf(
		weapon_damage / reference, 0.0, SandboxConfig.REWARD_HIT_SCALE_MAX
	)
	var fired: int = int(events.get("projectiles_fired", 0))
	var landed: int = int(events.get("projectiles_hit", 0))
	if fired > 1:
		scale *= clampf(float(landed) / float(fired), 0.0, 1.0)
	return scale


static func compute_components(events: Dictionary) -> Dictionary:
	var components: Dictionary = {}
	components["reward_hit"] = (
		SandboxConfig.REWARD_HIT * hit_reward_scale(events)
		if events.get("hit", false)
		else 0.0
	)
	components["reward_kill"] = SandboxConfig.REWARD_KILL if events.get("kill", false) else 0.0

	var damage_dealt: float = float(events.get("damage_dealt", 0.0))
	components["reward_damage"] = (
		damage_dealt * SandboxConfig.REWARD_DAMAGE_DEALT_PER_HP
		if damage_dealt > 0.0
		else 0.0
	)

	var damage_taken: float = float(events.get("damage_taken", 0.0))
	components["penalty_damage"] = (
		damage_taken * SandboxConfig.PENALTY_DAMAGE_TAKEN_PER_HP
		if damage_taken > 0.0
		else 0.0
	)
	components["penalty_death"] = SandboxConfig.PENALTY_DEATH if events.get("died", false) else 0.0
	components["penalty_useless_shot"] = (
		SandboxConfig.PENALTY_USELESS_SHOT
		if events.get("useless_shot", false)
		else 0.0
	)
	components["penalty_missed_shot"] = (
		SandboxConfig.PENALTY_MISSED_SHOT
		if events.get("missed_shot", false)
		else 0.0
	)
	components["penalty_trigger_discipline"] = (
		SandboxConfig.PENALTY_TRIGGER_DISCIPLINE
		if events.get("trigger_discipline", false)
		else 0.0
	)

	var positioning_delta: float = float(events.get("positioning_delta", 0.0))
	components["reward_positioning"] = clampf(
		positioning_delta * SandboxConfig.REWARD_POSITIONING_SCALE,
		-SandboxConfig.REWARD_POSITIONING_MAX, SandboxConfig.REWARD_POSITIONING_MAX
	) if positioning_delta != 0.0 else 0.0

	var aiming_delta: float = float(events.get("aiming_delta", 0.0))
	var target_hittable: bool = bool(events.get("target_hittable", false))
	components["reward_aiming"] = clampf(
		aiming_delta * SandboxConfig.REWARD_AIMING_SCALE,
		-SandboxConfig.REWARD_AIMING_MAX, SandboxConfig.REWARD_AIMING_MAX
	) if aiming_delta != 0.0 and target_hittable else 0.0
	var valid_target: bool = bool(events.get("valid_target", false))
	components["penalty_passivity"] = (
		SandboxConfig.PENALTY_PASSIVITY
		if valid_target and not events.get("meaningful_action", false)
		else 0.0
	)
	components["penalty_combat_time"] = (
		SandboxConfig.PENALTY_COMBAT_TIME
		if valid_target and events.get("alive", true) and not events.get("died", false)
		else 0.0
	)
	var exploration_gain: float = float(events.get("exploration_gain", 0.0))
	components["reward_exploration"] = exploration_gain
	components["reward_exploration_complete"] = (
		SandboxConfig.REWARD_EXPLORATION_COMPLETE
		if events.get("exploration_complete", false)
		else 0.0
	)

	components["reward_survive"] = (
		SandboxConfig.REWARD_SURVIVE_TICK
		if events.get("alive", true)
		and not events.get("died", false)
		and events.get("survival_reward_allowed", false)
		else 0.0
	)
	return components


static func components_total(components: Dictionary) -> float:
	var total: float = 0.0
	for value in components.values():
		total += float(value)
	return total


static func compute(events: Dictionary) -> float:
	return components_total(compute_components(events))

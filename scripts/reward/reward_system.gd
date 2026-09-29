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
##   died: bool                -- the agent died this tick
##   useless_shot: bool        -- trigger pull that could not possibly
##                                 connect: weapon still on cooldown (nothing
##                                 fired) or no alive target to hit. Penalized
##                                 with PENALTY_USELESS_SHOT to teach trigger
##                                 discipline.
##   missed_shot: bool         -- weapon actually fired at a live target but
##                                 the ray did not connect. Penalized with the
##                                 much cheaper PENALTY_MISSED_SHOT so that
##                                 exploring aim keeps a positive expected
##                                 value while learning.
##   positioning_delta: float  -- meters the agent closed toward the enemy
##                                 this tick while not already at an
##                                 effective engagement range (can be
##                                 negative if it moved away)
##   aiming_delta: float       -- change in forward-vector alignment toward
##                                 the live target this tick
##   valid_target: bool        -- a live target exists
##   meaningful_action: bool   -- movement/aiming progress or a fired shot
##   alive: bool               -- whether the agent is alive at the end of
##                                 the tick (drives the small survive bonus)
##   exploration_gain: float   -- Map Analyzer mode only: value of the map
##                                 cells newly observed this tick, already
##                                 normalized by grid size by MapAnalyzer.
##                                 Absent (0) in every combat mode, so the
##                                 combat reward is byte-identical to before.
##   exploration_complete: bool -- the map reached the target coverage this
##                                 tick (paid once per episode)
static func compute_components(events: Dictionary) -> Dictionary:
	var components: Dictionary = {}
	components["reward_hit"] = SandboxConfig.REWARD_HIT if events.get("hit", false) else 0.0
	components["reward_kill"] = SandboxConfig.REWARD_KILL if events.get("kill", false) else 0.0

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

	var positioning_delta: float = float(events.get("positioning_delta", 0.0))
	components["reward_positioning"] = clampf(
		positioning_delta * SandboxConfig.REWARD_POSITIONING_SCALE,
		-SandboxConfig.REWARD_POSITIONING_MAX, SandboxConfig.REWARD_POSITIONING_MAX
	) if positioning_delta != 0.0 else 0.0

	var aiming_delta: float = float(events.get("aiming_delta", 0.0))
	components["reward_aiming"] = clampf(
		aiming_delta * SandboxConfig.REWARD_AIMING_SCALE,
		-SandboxConfig.REWARD_AIMING_MAX, SandboxConfig.REWARD_AIMING_MAX
	) if aiming_delta != 0.0 else 0.0
	components["penalty_passivity"] = (
		SandboxConfig.PENALTY_PASSIVITY
		if events.get("valid_target", false) and not events.get("meaningful_action", false)
		else 0.0
	)
	var exploration_gain: float = float(events.get("exploration_gain", 0.0))
	components["reward_exploration"] = exploration_gain
	components["reward_exploration_complete"] = (
		SandboxConfig.REWARD_EXPLORATION_COMPLETE
		if events.get("exploration_complete", false)
		else 0.0
	)

	if events.get("alive", true) and not events.get("died", false):
		var valid_target: bool = events.get("valid_target", false)
		var combat_progress: bool = events.get("shot_fired", false) or positioning_delta > 0.0
		components["reward_survive"] = (
			SandboxConfig.REWARD_SURVIVE_TICK
			if not valid_target or combat_progress
			else 0.0
		)
	else:
		components["reward_survive"] = 0.0
	return components


static func components_total(components: Dictionary) -> float:
	var total: float = 0.0
	for value in components.values():
		total += float(value)
	return total


static func compute(events: Dictionary) -> float:
	return components_total(compute_components(events))

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
##   alive: bool               -- whether the agent is alive at the end of
##                                 the tick (drives the small survive bonus)
##   exploration_gain: float   -- Map Analyzer mode only: value of the map
##                                 cells newly observed this tick, already
##                                 normalized by grid size by MapAnalyzer.
##                                 Absent (0) in every combat mode, so the
##                                 combat reward is byte-identical to before.
##   exploration_complete: bool -- the map reached the target coverage this
##                                 tick (paid once per episode)
static func compute(events: Dictionary) -> float:
	var reward: float = 0.0

	if events.get("hit", false):
		reward += SandboxConfig.REWARD_HIT

	if events.get("kill", false):
		reward += SandboxConfig.REWARD_KILL

	var damage_taken: float = float(events.get("damage_taken", 0.0))
	if damage_taken > 0.0:
		reward += damage_taken * SandboxConfig.PENALTY_DAMAGE_TAKEN_PER_HP

	if events.get("died", false):
		reward += SandboxConfig.PENALTY_DEATH

	if events.get("useless_shot", false):
		reward += SandboxConfig.PENALTY_USELESS_SHOT

	if events.get("missed_shot", false):
		reward += SandboxConfig.PENALTY_MISSED_SHOT

	var positioning_delta: float = float(events.get("positioning_delta", 0.0))
	if positioning_delta != 0.0:
		var shaped: float = clampf(
			positioning_delta * SandboxConfig.REWARD_POSITIONING_SCALE,
			-SandboxConfig.REWARD_POSITIONING_MAX,
			SandboxConfig.REWARD_POSITIONING_MAX
		)
		reward += shaped

	var exploration_gain: float = float(events.get("exploration_gain", 0.0))
	if exploration_gain != 0.0:
		reward += exploration_gain

	if events.get("exploration_complete", false):
		reward += SandboxConfig.REWARD_EXPLORATION_COMPLETE

	if events.get("alive", true) and not events.get("died", false):
		reward += SandboxConfig.REWARD_SURVIVE_TICK

	return reward

## RewardSystem
##
## Centralized, configurable reward calculation. All magic numbers live in
## SandboxConfig; this class only combines the events that happened during a
## single tick into a scalar reward. Kept as a stateless static utility so
## it is trivial to unit test in isolation.
class_name RewardSystem
extends RefCounted


## `events` expected keys (all optional, default to "nothing happened"):
##   hit: bool                 -- weapon shot connected with an enemy this tick
##   kill: bool                -- that hit (or a prior one) killed an enemy this tick
##   damage_taken: float       -- HP lost by the agent this tick (>= 0)
##   died: bool                -- the agent died this tick
##   useless_shot: bool        -- trigger pulled but could not possibly land
##                                 (on cooldown or no line-of-sight/target)
##   positioning_delta: float  -- meters the agent closed toward the enemy
##                                 this tick while not already at an
##                                 effective engagement range (can be
##                                 negative if it moved away)
##   alive: bool               -- whether the agent is alive at the end of
##                                 the tick (drives the small survive bonus)
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

	var positioning_delta: float = float(events.get("positioning_delta", 0.0))
	if positioning_delta != 0.0:
		var shaped: float = clampf(
			positioning_delta * SandboxConfig.REWARD_POSITIONING_SCALE,
			-SandboxConfig.REWARD_POSITIONING_MAX,
			SandboxConfig.REWARD_POSITIONING_MAX
		)
		reward += shaped

	if events.get("alive", true) and not events.get("died", false):
		reward += SandboxConfig.REWARD_SURVIVE_TICK

	return reward

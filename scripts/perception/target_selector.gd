## TargetSelector
##
## Explicit, inspectable target selection.
##
## Choosing what to shoot at used to be a private sort inside
## `AgentPerception`. It is promoted to its own abstraction here because it
## is a research object in its own right: every factor is named, weighted by
## a constant in `SandboxConfig`, and reported back so the Control Center
## can show WHY a target was chosen without the policy ever receiving that
## explanation.
##
## Factors, all derived from beliefs the agent actually holds:
##
##   visibility          seeing a target beats remembering one
##   recent_damage       whoever just hurt the agent matters more
##   memory_confidence   a fresh memory beats a stale one
##   proximity           near beats far
##   threat              a target that currently has line of sight to the
##                       agent is dangerous even if it is further away
##   wounded             a nearly dead target is cheap to finish
##   exposure            being exposed to that target raises its priority
##   accessibility       a target that cannot be reached is worth less
##   continuity          a small bonus for the CURRENT target, so the agent
##                       does not oscillate between two equal threats
##
## Nothing here is a personality: the weights describe what information is
## worth, not a play style, and the learned policy is free to ignore the
## resulting ordering entirely — it only decides which enemy fills the
## primary observation slot.
class_name TargetSelector
extends RefCounted

const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

## Names of the scored factors, in report order.
const FACTORS: Array = [
	"visibility",
	"recent_damage",
	"memory_confidence",
	"proximity",
	"threat",
	"wounded",
	"accessibility",
	"continuity",
]


## Largest score any belief can reach. Used to normalize the priority that
## goes into the observation vector, so the policy sees a bounded [0, 1]
## value rather than a raw weight sum.
static func max_score() -> float:
	return (
		SandboxConfig.TARGET_WEIGHT_VISIBLE
		+ SandboxConfig.TARGET_WEIGHT_DAMAGE_SOURCE
		+ SandboxConfig.TARGET_WEIGHT_CONFIDENCE
		+ SandboxConfig.TARGET_WEIGHT_PROXIMITY
		+ SandboxConfig.TARGET_WEIGHT_THREAT
		+ SandboxConfig.TARGET_WEIGHT_WOUNDED
		+ SandboxConfig.TARGET_WEIGHT_CONTINUITY
	)


## Per-factor breakdown for one belief.
##
## `context` keys (all optional):
##   damage_source        int   enemy id that last damaged the agent
##   previous_target_id   int   the target selected on the previous tick
##   unreachable          Array enemy ids the navigation graph cannot reach
static func factors(belief: Dictionary, context: Dictionary) -> Dictionary:
	var out: Dictionary = {}
	out["visibility"] = (
		SandboxConfig.TARGET_WEIGHT_VISIBLE if bool(belief.get("visible", false)) else 0.0
	)
	var damage_source: int = int(context.get("damage_source", -1))
	out["recent_damage"] = (
		SandboxConfig.TARGET_WEIGHT_DAMAGE_SOURCE
		if damage_source >= 0 and int(belief.get("id", -1)) == damage_source
		else 0.0
	)
	out["memory_confidence"] = (
		float(belief.get("confidence", 0.0)) * SandboxConfig.TARGET_WEIGHT_CONFIDENCE
	)
	out["proximity"] = (
		clampf(
			1.0 - float(belief.get("distance", 0.0)) / SandboxConfig.ARENA_MAX_DISTANCE, 0.0, 1.0
		)
		* SandboxConfig.TARGET_WEIGHT_PROXIMITY
	)
	# Threat and exposure are the same measurement seen from two sides: if
	# it can see the agent, the agent is exposed to it.
	out["threat"] = (
		SandboxConfig.TARGET_WEIGHT_THREAT if bool(belief.get("threatening", false)) else 0.0
	)
	out["wounded"] = (
		clampf(1.0 - float(belief.get("health_norm", 1.0)), 0.0, 1.0)
		* SandboxConfig.TARGET_WEIGHT_WOUNDED
	)
	var unreachable: Array = context.get("unreachable", [])
	out["accessibility"] = (
		-SandboxConfig.TARGET_PENALTY_UNREACHABLE
		if unreachable.has(int(belief.get("id", -1)))
		else 0.0
	)
	var previous: int = int(context.get("previous_target_id", -1))
	out["continuity"] = (
		SandboxConfig.TARGET_WEIGHT_CONTINUITY
		if previous >= 0 and int(belief.get("id", -2)) == previous
		else 0.0
	)
	return out


static func score(belief: Dictionary, context: Dictionary) -> float:
	var total: float = 0.0
	var breakdown: Dictionary = factors(belief, context)
	for name_value in FACTORS:
		total += float(breakdown[name_value])
	return total


## Beliefs sorted best-target-first. Ties break on distance, so the order is
## total and never depends on the input order.
static func rank(beliefs: Array, context: Dictionary) -> Array:
	var ranked: Array = beliefs.duplicate()
	ranked.sort_custom(
		func(a, b):
			var score_a: float = score(a, context)
			var score_b: float = score(b, context)
			if absf(score_a - score_b) > 0.000001:
				return score_a > score_b
			return float(a["distance"]) < float(b["distance"])
	)
	return ranked


## Full selection result.
##
## Returns:
##   target        the chosen belief ({} when the agent has no contact)
##   ranked        every belief, best first
##   priority_norm the winner's score normalized to [0, 1]
##   switched      whether this is a different target than last tick
##   reason        human-readable justification (Control Center only)
##   factors       the winner's per-factor breakdown (Control Center only)
static func select(beliefs: Array, context: Dictionary) -> Dictionary:
	var ranked: Array = rank(beliefs, context)
	if ranked.is_empty():
		return {
			"target": {},
			"ranked": ranked,
			"priority_norm": 0.0,
			"switched": false,
			"reason": "no target",
			"factors": {},
		}
	var winner: Dictionary = ranked[0]
	var previous: int = int(context.get("previous_target_id", -1))
	var winner_id: int = int(winner.get("id", -1))
	return {
		"target": winner,
		"ranked": ranked,
		"priority_norm": clampf(score(winner, context) / maxf(max_score(), 0.0001), 0.0, 1.0),
		"switched": previous >= 0 and winner_id != previous,
		"reason": reason(winner, context),
		"factors": factors(winner, context),
	}


## Human-readable justification. Debug/UI only: this string is never part of
## the observation, because "why" is the policy's job to work out.
static func reason(belief: Dictionary, context: Dictionary) -> String:
	if belief.is_empty():
		return "no target"
	var damage_source: int = int(context.get("damage_source", -1))
	if bool(belief.get("visible", false)):
		if int(belief.get("id", -1)) == damage_source:
			return "visible and recently damaged me"
		if bool(belief.get("threatening", false)):
			return "visible and has line of sight to me"
		return "visible, nearest threat"
	if int(belief.get("source", EnemyMemory.Source.NONE)) == EnemyMemory.Source.SOUND:
		return "heard only, last known position"
	if context.get("unreachable", []).has(int(belief.get("id", -1))):
		return "remembered but currently unreachable"
	return "remembered, %.1fs since contact" % float(belief.get("age", 0.0))

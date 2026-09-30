## SkillMetrics
##
## Diagnostic research metrics for the Control Center's METRICS tab,
## grouped into the same eight categories as python/sandboxai/metrics.py so
## a number shown in the window means the same thing as the number in an
## exported report.
##
## Two rules govern this file:
##
## 1. **Diagnostic only.** Nothing here is a reward and nothing here is
##    wired into the reward system. A metric tells a researcher what the
##    policy did; converting every metric into a reward term is how a
##    sandbox ends up training to its own instrumentation.
## 2. **Two clearly separated origins.** `ai_available()` is computed from
##    what the agent could perceive; `ground_truth()` is computed from
##    simulator state the agent never sees. They are returned as separate
##    dictionaries, with `GROUND_TRUTH_KEYS` naming the second set, so a
##    renderer cannot accidentally present a debug value as something the
##    policy knew — and so no caller can feed one back into an observation.
class_name SkillMetrics
extends RefCounted

## Report order, mirroring metrics.CATEGORIES on the Python side.
const CATEGORIES: Array = [
	"aim",
	"reaction",
	"awareness",
	"positioning",
	"movement",
	"combat",
	"survival",
	"exploration",
]

## Keys that come from simulator state rather than perception. Anything
## listed here is debug information and is labelled as such in the UI.
const GROUND_TRUTH_KEYS: Array = [
	"true_enemy_distance_m",
	"true_enemy_health",
	"true_enemy_position",
	"alive_enemies",
	"total_enemies",
]


## Per-category metrics derived from a Control Center telemetry snapshot.
## Missing sections degrade to empty categories rather than to invented
## numbers.
static func from_snapshot(snapshot: Dictionary) -> Dictionary:
	var episode: Dictionary = snapshot.get("episode", {})
	var agent: Dictionary = snapshot.get("agent", {})
	var perception: Dictionary = snapshot.get("perception", {})
	var action: Dictionary = snapshot.get("action", {})

	var shots: int = int(episode.get("shots_fired", 0))
	var hits: int = int(episode.get("shots_hit", 0))
	var steps: int = maxi(1, int(episode.get("step", 0)))
	var seconds: float = maxf(0.0001, float(episode.get("time_seconds", 0.0)))
	var max_health: float = maxf(1.0, float(agent.get("max_health", 100.0)))

	var categories: Dictionary = {}
	var headshots: int = int(episode.get("headshots", 0))
	categories["aim"] = {
		"shots_fired": shots,
		"shots_hit": hits,
		"accuracy": float(hits) / float(maxi(1, shots)) if shots > 0 else 0.0,
		"damage_per_shot": float(episode.get("damage_dealt", 0.0)) / float(maxi(1, shots)),
		"headshots": headshots,
		"headshot_rate": float(headshots) / float(maxi(1, hits)) if hits > 0 else 0.0,
	}
	categories["reaction"] = {
		"steps": steps,
		"shots_per_second": float(shots) / seconds,
	}
	categories["awareness"] = {
		"visible_enemies": int(perception.get("visible_count", 0)),
		"remembered_enemies": int(perception.get("remembered_count", 0)),
		"has_target": bool(snapshot.get("target", {}).get("has_target", false)),
		"hearing_events": int(perception.get("sound_count", 0)),
	}
	categories["positioning"] = {
		"health_fraction": float(agent.get("health", 0.0)) / max_health,
		"damage_taken": float(episode.get("damage_received", 0.0)),
		"damage_ratio":
		(
			float(episode.get("damage_dealt", 0.0))
			/ maxf(1.0, float(episode.get("damage_received", 0.0)))
		),
	}
	categories["movement"] = {
		"speed": (agent.get("velocity", Vector3.ZERO) as Vector3).length(),
		"moving": bool(action.get("moving", false)),
		"turning": bool(action.get("turning", false)),
	}
	# Weapon handling. `trigger_discipline_events` counts trigger pulls the
	# weapon refused (cycling, reloading, empty, semi-auto not released);
	# it is reported separately from wasted shots because the two describe
	# different mistakes. All four degrade to zero when the handling layer
	# is disabled by the curriculum, which is the correct reading: on
	# levels 1-4 there is no magazine and no reload to get wrong.
	var discipline: int = int(episode.get("trigger_discipline_events", 0))
	categories["combat"] = {
		"kills": int(episode.get("kills", 0)),
		"damage_dealt": float(episode.get("damage_dealt", 0.0)),
		"shooting": bool(action.get("shooting", false)),
		"trigger_discipline_events": discipline,
		"trigger_discipline_rate":
		float(discipline) / float(maxi(1, shots + discipline)) if shots + discipline > 0 else 0.0,
		"reload_starts": int(episode.get("reload_starts", 0)),
		"reloading_time": float(episode.get("reloading_time", 0.0)),
		"weapon_ammo": int(agent.get("weapon_ammo", 0)),
		"weapon_reloading": bool(agent.get("weapon_reloading", false)),
		"weapon_bloom_deg": float(agent.get("weapon_bloom_deg", 0.0)),
	}
	categories["survival"] = {
		"alive": bool(agent.get("alive", true)),
		"deaths": int(episode.get("deaths", 0)),
		"episode_seconds": seconds,
		"reward": float(episode.get("reward", 0.0)),
	}
	categories["exploration"] = {
		"coverage": float(perception.get("exploration_coverage", 0.0)),
	}
	return categories


## The subset a policy could in principle have derived from its own
## observation. Safe to show next to the AI's own view.
static func ai_available(snapshot: Dictionary) -> Dictionary:
	var categories: Dictionary = from_snapshot(snapshot)
	for key in GROUND_TRUTH_KEYS:
		for category in categories:
			(categories[category] as Dictionary).erase(key)
	return categories


## Simulator-side facts the agent never perceives. Debug only; shown in the
## UI under an explicit ground-truth heading and never merged into the
## category tables above.
static func ground_truth(snapshot: Dictionary) -> Dictionary:
	var episode: Dictionary = snapshot.get("episode", {})
	var target: Dictionary = snapshot.get("target", {})
	var out: Dictionary = {
		"alive_enemies": int(episode.get("alive_enemies", 0)),
		"total_enemies": int(episode.get("total_enemies", 0)),
	}
	if bool(target.get("has_target", false)):
		out["true_enemy_distance_m"] = float(target.get("distance_m", 0.0))
		out["true_enemy_health"] = float(target.get("health", 0.0))
		out["true_enemy_position"] = target.get("position", Vector3.ZERO)
	return out


## Flat "category.metric" rows, ready for a table or a CSV export.
static func rows(categories: Dictionary) -> Array:
	var out: Array = []
	for category_value in CATEGORIES:
		var category: String = str(category_value)
		if not categories.has(category):
			continue
		var values: Dictionary = categories[category]
		var keys: Array = values.keys()
		keys.sort()
		for key in keys:
			out.append({"category": category, "metric": str(key), "value": values[key]})
	return out


static func format_categories(categories: Dictionary) -> String:
	var lines: Array = []
	for category_value in CATEGORIES:
		var category: String = str(category_value)
		if not categories.has(category):
			continue
		var values: Dictionary = categories[category]
		var parts: Array = []
		var keys: Array = values.keys()
		keys.sort()
		for key in keys:
			parts.append("%s=%s" % [str(key), _format_value(values[key])])
		lines.append("%-12s %s" % [category.to_upper(), " ".join(parts)])
	return "\n".join(lines)


static func _format_value(value: Variant) -> String:
	if typeof(value) == TYPE_FLOAT:
		return "%.3f" % float(value)
	if typeof(value) == TYPE_BOOL:
		return "yes" if bool(value) else "no"
	return str(value)

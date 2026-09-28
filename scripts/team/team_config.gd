## TeamConfig
##
## Foundation for team play. **Disabled by default**, and a disabled config
## is canonicalized to exactly the single-agent behaviour the project has
## today: every slot on team 0, no communication, no team reward weight.
## Nothing in the existing environment changes unless a caller explicitly
## enables teams.
##
## The Python mirror is python/sandboxai/teamplay.py; the two share the
## same symbol vocabulary and the same friendly-fire and isolation rules so
## a team experiment configured on either side means the same thing.
##
## What this file deliberately does NOT do: it does not give a teammate
## privileged knowledge. A teammate report is built from what the *teammate*
## perceived and then decays; it is never a window into ground truth. That
## is enforced by FORBIDDEN_REPORT_FIELDS and by the perception isolation
## tests.
class_name TeamConfig
extends RefCounted

## The team every agent belongs to when teams are off.
const SOLO_TEAM: int = 0

## Fixed communication vocabulary. Symbols, not free text: a learned policy
## can only emit one of these, which keeps the channel a small discrete
## space instead of an unbounded side-channel for ground truth.
const COMMS_SYMBOLS: Array = [
	"none",
	"contact",
	"contact_lost",
	"need_help",
	"holding",
	"advancing",
	"falling_back",
	"enemy_down",
]

## Objectives a team can be scored on. Hooks only: the reward system is not
## rewired here, and metrics remain diagnostic.
const TEAM_OBJECTIVES: Array = ["eliminate", "survive", "control", "explore"]

## Fields a teammate report may carry.
const REPORT_FIELDS: Array = [
	"teammate_slot",
	"team_id",
	"alive",
	"health_norm",
	"bearing_norm",
	"distance_norm",
	"last_symbol",
	"confidence",
	"age_norm",
]

## Fields a teammate report must never carry, because none of them are
## things the reporting agent could have perceived.
const FORBIDDEN_REPORT_FIELDS: Array = [
	"enemy_position",
	"enemy_health",
	"map_geometry",
	"hidden_entities",
	"ground_truth_position",
	"world_position",
]

## Absolute path to this script, used by the static factory methods. See
## the note in scripts/core/action.gd: referencing the global class name in
## a value context requires the editor class cache, which does not exist
## during a standalone headless run.
const SELF_PATH: String = "res://scripts/team/team_config.gd"

var enabled: bool = false
## Slot index -> team id. Empty means "one solo agent in slot 0".
var slot_teams: Dictionary = {}
var friendly_fire: bool = false
var comms_enabled: bool = false
## Maximum messages one team may put on the channel per tick.
var comms_budget: int = 2
var objective: String = "eliminate"
## Weight blending the team score into an agent's shaped reward. Zero while
## teams are disabled, so the reward path is bit-for-bit unchanged.
var team_reward_weight: float = 0.0
var friendly_fire_penalty: float = 0.0


func _init(p_enabled: bool = false, p_slot_teams: Dictionary = {}) -> void:
	enabled = p_enabled
	slot_teams = p_slot_teams.duplicate()
	canonicalize()


## Forces a disabled config back to the solo baseline. Called from _init and
## again after any mutation, so "disabled" can never be half-on.
func canonicalize() -> void:
	if not TEAM_OBJECTIVES.has(objective):
		objective = "eliminate"
	comms_budget = maxi(0, comms_budget)
	if enabled:
		return
	for slot in slot_teams:
		slot_teams[slot] = SOLO_TEAM
	comms_enabled = false
	team_reward_weight = 0.0
	friendly_fire = false
	friendly_fire_penalty = 0.0


static func solo() -> TeamConfig:
	return load(SELF_PATH).new(false, {})


## Two teams of `per_team` slots each, laid out slot 0..n-1 = team 0,
## n..2n-1 = team 1.
static func versus(per_team: int = 1) -> TeamConfig:
	var slots: Dictionary = {}
	var count: int = maxi(1, per_team)
	for index in range(count * 2):
		slots[index] = 0 if index < count else 1
	return load(SELF_PATH).new(true, slots)


func team_of(slot: int) -> int:
	return int(slot_teams.get(slot, SOLO_TEAM))


func slots_of(team_id: int) -> Array:
	var out: Array = []
	for slot in slot_teams:
		if int(slot_teams[slot]) == team_id:
			out.append(int(slot))
	out.sort()
	return out


func teammates_of(slot: int) -> Array:
	var out: Array = []
	for other in slots_of(team_of(slot)):
		if int(other) != slot:
			out.append(int(other))
	return out


func are_teammates(slot_a: int, slot_b: int) -> bool:
	if slot_a == slot_b:
		return true
	if not enabled:
		# With teams off there is only one agent; "teammate" is meaningless
		# and must not silently become "everyone".
		return false
	return team_of(slot_a) == team_of(slot_b)


## Damage permission. With friendly fire off, a teammate simply cannot be
## hurt — the shot is wasted rather than penalized, and the penalty hook is
## separate so an experiment can choose either.
func can_damage(attacker_slot: int, victim_slot: int) -> bool:
	if attacker_slot == victim_slot:
		return false
	if not enabled:
		return true
	if are_teammates(attacker_slot, victim_slot):
		return friendly_fire
	return true


func team_count() -> int:
	var seen: Dictionary = {}
	for slot in slot_teams:
		seen[int(slot_teams[slot])] = true
	return maxi(1, seen.size())


func is_valid_symbol(symbol: String) -> bool:
	return COMMS_SYMBOLS.has(symbol)


## Validates a teammate report: correct fields, nothing forbidden, values
## in range. Returns a list of problems (empty means valid).
func validate_report(report: Dictionary) -> Array:
	var problems: Array = []
	for key in report:
		var name: String = str(key)
		if FORBIDDEN_REPORT_FIELDS.has(name):
			problems.append("teammate report leaks ground truth via '%s'" % name)
		elif not REPORT_FIELDS.has(name):
			problems.append("unknown teammate report field: '%s'" % name)
	for bounded in ["health_norm", "bearing_norm", "distance_norm", "confidence", "age_norm"]:
		if report.has(bounded):
			var value: float = float(report[bounded])
			if value < -1.0 or value > 1.0:
				problems.append("%s out of range: %f" % [bounded, value])
	if report.has("last_symbol") and not is_valid_symbol(str(report["last_symbol"])):
		problems.append("unknown comms symbol: %s" % str(report["last_symbol"]))
	return problems


func to_dictionary() -> Dictionary:
	return {
		"enabled": enabled,
		"slot_teams": slot_teams.duplicate(),
		"teams": team_count(),
		"friendly_fire": friendly_fire,
		"comms_enabled": comms_enabled,
		"comms_budget": comms_budget,
		"objective": objective,
		"team_reward_weight": team_reward_weight,
		"friendly_fire_penalty": friendly_fire_penalty,
	}

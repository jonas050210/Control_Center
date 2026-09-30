## ControlCenterEpisodeProbe
##
## Per-episode diagnostics for the SELECTED environment, moved verbatim out
## of ControlCenterSession: discrete combat/perception event logging plus
## the counters (shots, misses, target switches, reaction clock) that end
## up on the episode record. Pure observer — it never mutates the
## simulation, only reads step events and writes to the event log.
extends RefCounted

const ControlCenterEventLog = preload("res://scripts/control_center/control_center_event_log.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

var event_log: ControlCenterEventLog

var _data: Dictionary = {}
var _last_target_index: int = -1


func _init(p_event_log: ControlCenterEventLog = null) -> void:
	event_log = p_event_log
	reset()


func track_selected_events(events: Dictionary) -> void:
	var probe: Dictionary = _data
	if bool(events.get("shot_fired", false)):
		probe["shots"] = int(probe.get("shots", 0)) + 1
		if int(probe.get("first_shot_step", -1)) < 0:
			probe["first_shot_step"] = int(probe.get("steps", 0))
	if bool(events.get("hit", false)):
		event_log.log_event(
			ControlCenterEventLog.Category.COMBAT,
			"hit enemy for %.0f damage" % float(events.get("damage_dealt", 0.0)),
			{},
			"hit"
		)
	if bool(events.get("kill", false)):
		event_log.log_event(ControlCenterEventLog.Category.COMBAT, "enemy eliminated")
	if float(events.get("damage_taken", 0.0)) > 0.0:
		event_log.log_event(
			ControlCenterEventLog.Category.COMBAT,
			"took %.0f damage" % float(events.get("damage_taken", 0.0)),
			{},
			"damage_taken"
		)
	if bool(events.get("useless_shot", false)):
		probe["useless_shots"] = int(probe.get("useless_shots", 0)) + 1
		event_log.log_event(
			ControlCenterEventLog.Category.REWARD,
			"useless trigger pull (weapon on cooldown or no live target)",
			{},
			"useless_shot"
		)
	if bool(events.get("missed_shot", false)):
		probe["missed_shots"] = int(probe.get("missed_shots", 0)) + 1
		event_log.log_event(
			ControlCenterEventLog.Category.REWARD, "shot missed a live target", {}, "missed_shot"
		)
	if bool(events.get("died", false)):
		event_log.log_event(ControlCenterEventLog.Category.COMBAT, "agent died")
	probe["steps"] = int(probe.get("steps", 0)) + 1


## Target changes are a perception event: the observation's primary slot
## now refers to a different enemy.
func track_target_change(env) -> void:
	var target_index: int = PerceptionModel.current_target_index(env)
	if target_index == _last_target_index:
		return
	if _last_target_index >= 0 and target_index >= 0:
		_data["target_switches"] = int(_data.get("target_switches", 0)) + 1
	event_log.log_event(
		ControlCenterEventLog.Category.PERCEPTION,
		(
			"target -> enemy #%d" % target_index
			if target_index >= 0
			else "target lost (no alive enemy in observation)"
		),
		{"previous": _last_target_index, "current": target_index}
	)
	_last_target_index = target_index

	# First moment an enemy is inside weapon range starts the reaction clock.
	if (
		int(_data.get("engagement_step", -1)) < 0
		and target_index >= 0
		and (
			env.agent.position.distance_to(env.enemies[target_index].position)
			<= env.agent.weapon.range_m
		)
	):
		_data["engagement_step"] = int(_data.get("steps", 0))


## Seconds between the first tick with an enemy inside weapon range and the
## first shot fired after that. Returns -1.0 when either never happened,
## so "not measured" is never reported as a real number.
func measured_reaction_time() -> float:
	var engagement: int = int(_data.get("engagement_step", -1))
	var first_shot: int = int(_data.get("first_shot_step", -1))
	if engagement < 0 or first_shot < engagement:
		return -1.0
	return float(first_shot - engagement) * SandboxConfig.SIMULATION_DT


func count(key: String) -> int:
	return int(_data.get(key, 0))


func reset() -> void:
	_data = {
		"steps": 0,
		"shots": 0,
		"useless_shots": 0,
		"missed_shots": 0,
		"target_switches": 0,
		"engagement_step": -1,
		"first_shot_step": -1,
	}
	_last_target_index = -1

## EpisodeState
## Per-environment episode bookkeeping, combat metrics and reward breakdown.
class_name EpisodeState
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const RewardSystem = preload("res://scripts/reward/reward_system.gd")


var step_count: int = 0
var episode_count: int = 0
var cumulative_reward: float = 0.0
var last_reward: float = 0.0
var done: bool = false
var done_reason: String = ""

## Episode metrics. They reset at the beginning of every episode.
var kills: int = 0
var deaths: int = 0
var damage_dealt: float = 0.0
var damage_taken: float = 0.0
var shots_fired: int = 0
var shots_hit: int = 0
var trigger_pulls: int = 0
var near_miss_shots: int = 0
var useless_shots: int = 0
var cooldown_shots: int = 0
## Weapon-handling diagnostics. All measurement only: none of them is a
## reward term, and none of them reaches the observation vector.
var headshots: int = 0
var trigger_discipline_events: int = 0
var reload_starts: int = 0
var reloading_steps: int = 0
var last_shot_result: String = "none"
var survival_steps: int = 0

## Reward breakdown per episode.
var reward_hits: float = 0.0
var reward_kills: float = 0.0
var reward_damage: float = 0.0
var reward_survive: float = 0.0
var reward_positioning: float = 0.0
var reward_aiming: float = 0.0
var penalty_passivity: float = 0.0
var penalty_combat_time: float = 0.0
var reward_exploration: float = 0.0
var reward_exploration_complete: float = 0.0
var penalty_damage: float = 0.0
var penalty_death: float = 0.0
var penalty_useless_shot: float = 0.0
var penalty_missed_shot: float = 0.0
var penalty_trigger_discipline: float = 0.0

## Lifetime counters, useful for the Godot debug overlay.
var total_kills: int = 0
var total_deaths: int = 0


func start_new_episode() -> void:
	episode_count += 1
	step_count = 0
	cumulative_reward = 0.0
	last_reward = 0.0
	done = false
	done_reason = ""
	kills = 0
	deaths = 0
	damage_dealt = 0.0
	damage_taken = 0.0
	shots_fired = 0
	shots_hit = 0
	trigger_pulls = 0
	near_miss_shots = 0
	useless_shots = 0
	cooldown_shots = 0
	headshots = 0
	trigger_discipline_events = 0
	reload_starts = 0
	reloading_steps = 0
	last_shot_result = "none"
	survival_steps = 0
	reward_hits = 0.0
	reward_kills = 0.0
	reward_damage = 0.0
	reward_survive = 0.0
	reward_positioning = 0.0
	reward_aiming = 0.0
	penalty_passivity = 0.0
	penalty_combat_time = 0.0
	reward_exploration = 0.0
	reward_exploration_complete = 0.0
	penalty_damage = 0.0
	penalty_death = 0.0
	penalty_useless_shot = 0.0
	penalty_missed_shot = 0.0
	penalty_trigger_discipline = 0.0


func record_step(reward: float) -> void:
	step_count += 1
	last_reward = reward
	cumulative_reward += reward
	if not done:
		survival_steps += 1


func record_reward_breakdown(events: Dictionary) -> void:
	var components: Dictionary = RewardSystem.compute_components(events)
	reward_hits += float(components.get("reward_hit", 0.0))
	reward_kills += float(components.get("reward_kill", 0.0))
	reward_damage += float(components.get("reward_damage", 0.0))
	penalty_damage += float(components.get("penalty_damage", 0.0))
	penalty_death += float(components.get("penalty_death", 0.0))
	penalty_useless_shot += float(components.get("penalty_useless_shot", 0.0))
	penalty_missed_shot += float(components.get("penalty_missed_shot", 0.0))
	penalty_trigger_discipline += float(components.get("penalty_trigger_discipline", 0.0))
	reward_positioning += float(components.get("reward_positioning", 0.0))
	reward_aiming += float(components.get("reward_aiming", 0.0))
	penalty_passivity += float(components.get("penalty_passivity", 0.0))
	penalty_combat_time += float(components.get("penalty_combat_time", 0.0))
	reward_exploration += float(components.get("reward_exploration", 0.0))
	reward_exploration_complete += float(components.get("reward_exploration_complete", 0.0))
	reward_survive += float(components.get("reward_survive", 0.0))
	_record_shot_diagnostics(events)


func _record_shot_diagnostics(events: Dictionary) -> void:
	if bool(events.get("reloading", false)):
		reloading_steps += 1
	var result: String = str(events.get("shot_result", "none"))
	if result.is_empty() or result == "none":
		return
	last_shot_result = result
	trigger_pulls += 1
	if result == "near_miss":
		near_miss_shots += 1
	if result == "cooldown":
		cooldown_shots += 1
	if result == "reload_started":
		reload_starts += 1
	if bool(events.get("useless_shot", false)):
		useless_shots += 1
	if bool(events.get("trigger_discipline", false)):
		trigger_discipline_events += 1


func record_shot(hit: bool, headshot: bool = false) -> void:
	shots_fired += 1
	if hit:
		shots_hit += 1
	if headshot:
		headshots += 1


func record_damage_dealt(amount: float) -> void:
	damage_dealt += maxf(amount, 0.0)


func record_damage_taken(amount: float) -> void:
	damage_taken += maxf(amount, 0.0)


func record_kill() -> void:
	kills += 1
	total_kills += 1


func record_death() -> void:
	deaths += 1
	total_deaths += 1


func mark_done(reason: String) -> void:
	done = true
	done_reason = reason


func is_timeout(max_steps: int) -> bool:
	return step_count >= max_steps


func get_reward_breakdown() -> Dictionary:
	return {
		"reward_hits": reward_hits,
		"reward_kills": reward_kills,
		"reward_damage": reward_damage,
		"reward_survive": reward_survive,
		"reward_positioning": reward_positioning,
		"reward_aiming": reward_aiming,
		"penalty_passivity": penalty_passivity,
		"penalty_combat_time": penalty_combat_time,
		"reward_exploration": reward_exploration,
		"reward_exploration_complete": reward_exploration_complete,
		"penalty_damage": penalty_damage,
		"penalty_death": penalty_death,
		"penalty_useless_shot": penalty_useless_shot,
		"penalty_missed_shot": penalty_missed_shot,
		"penalty_trigger_discipline": penalty_trigger_discipline,
		"total": cumulative_reward,
	}


## Episode summary.
##
## `loss` counts only episodes that were actually LOST. It used to be
## `done and not won`, which classified every timeout as a defeat: a run
## that survived the full 1200 steps without clearing the arena was
## reported identically to one where the agent was killed, so `win_rate +
## loss_rate` was always 1.0 and the timeout rate was invisible. A
## truncated episode is neither a win nor a loss, which is also what
## Gymnasium's `TimeLimit.truncated` convention means.
func to_metrics(simulation_dt: float, enemy_count: int, won: bool = false) -> Dictionary:
	var accuracy: float = float(shots_hit) / float(shots_fired) if shots_fired > 0 else 0.0
	var truncated: bool = done_reason == "timeout"
	return {
		# The episode these metrics describe. Captured here — inside the
		# episode — so a consumer that receives the metrics after the
		# environment has already been auto-reset still reports the number
		# of the episode that produced them.
		"episode": episode_count,
		"episode_reward": cumulative_reward,
		"episode_length": step_count,
		"kills": kills,
		"deaths": deaths,
		"damage_dealt": damage_dealt,
		"damage_received": damage_taken,
		"survival_time": float(survival_steps) * simulation_dt,
		"accuracy": accuracy,
		"shots_fired": shots_fired,
		"shots_hit": shots_hit,
		"trigger_pulls": trigger_pulls,
		"near_miss_shots": near_miss_shots,
		"useless_shots": useless_shots,
		"cooldown_shots": cooldown_shots,
		"headshots": headshots,
		"headshot_rate": float(headshots) / float(shots_hit) if shots_hit > 0 else 0.0,
		"trigger_discipline_events": trigger_discipline_events,
		"reload_starts": reload_starts,
		"reloading_time": float(reloading_steps) * simulation_dt,
		"last_shot_result": last_shot_result,
		"win": won,
		"loss": done and not won and not truncated,
		"truncated": truncated,
		"done_reason": done_reason,
		"enemy_count": enemy_count,
		"reward_breakdown": get_reward_breakdown(),
	}

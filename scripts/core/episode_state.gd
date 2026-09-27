## EpisodeState
## Per-environment episode bookkeeping and combat metrics.
class_name EpisodeState
extends RefCounted

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
var survival_steps: int = 0

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
	survival_steps = 0


func record_step(reward: float) -> void:
	step_count += 1
	last_reward = reward
	cumulative_reward += reward
	if not done:
		survival_steps += 1


func record_shot(hit: bool) -> void:
	shots_fired += 1
	if hit:
		shots_hit += 1


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


func to_metrics(simulation_dt: float, enemy_count: int, won: bool = false) -> Dictionary:
	var accuracy: float = float(shots_hit) / float(shots_fired) if shots_fired > 0 else 0.0
	return {
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
		"win": won,
		"loss": done and not won,
		"done_reason": done_reason,
		"enemy_count": enemy_count,
	}

## EpisodeState
##
## Tracks per-environment episode/session bookkeeping: step counter, episode
## counter, cumulative reward and lifetime kill/death counts. Kept separate
## from AgentState/EnemyState so combat state and episode bookkeeping don't
## get tangled together.
class_name EpisodeState
extends RefCounted

var step_count: int = 0
var episode_count: int = 0
var cumulative_reward: float = 0.0
var last_reward: float = 0.0
var done: bool = false
var done_reason: String = ""

## Lifetime counters (persist across resets, useful for telemetry).
var total_kills: int = 0
var total_deaths: int = 0


func start_new_episode() -> void:
	episode_count += 1
	step_count = 0
	cumulative_reward = 0.0
	last_reward = 0.0
	done = false
	done_reason = ""


func record_step(reward: float) -> void:
	step_count += 1
	last_reward = reward
	cumulative_reward += reward


func mark_done(reason: String) -> void:
	done = true
	done_reason = reason


func is_timeout(max_steps: int) -> bool:
	return step_count >= max_steps

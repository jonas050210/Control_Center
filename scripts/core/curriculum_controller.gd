## CurriculumController
##
## Automatic curriculum progression driven by ROLLING PERFORMANCE rather
## than by hard-coded step counts. A step-count schedule promotes a policy
## that is still failing and holds back one that already generalizes; a
## performance gate does neither.
##
## Usage (the environment/trainer owns one instance):
##     controller.record_episode(metrics)   # after every finished episode
##     if controller.pending_level != controller.level: ...apply it...
##
## Promotion rule: over the last `window` episodes, the success rate must
## reach `promote_threshold` AND at least `window` episodes must have been
## observed since the last level change (so a level is never skipped on a
## single lucky episode). Demotion is symmetric and deliberately more
## reluctant (`demote_threshold` is much lower), because oscillating
## between levels destroys the value of a curriculum.
##
## Everything is configurable and the controller is entirely optional:
## `enabled = false` leaves the level exactly where the operator set it.
class_name CurriculumController
extends RefCounted

const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")

const SELF_PATH: String = "res://scripts/core/curriculum_controller.gd"

var enabled: bool = false
var level: int = CurriculumConfig.Level.STATIONARY_TARGET
var min_level: int = CurriculumConfig.Level.STATIONARY_TARGET
var max_level: int = CurriculumConfig.MAX_COMBAT_LEVEL

## Number of recent episodes the success rate is measured over.
var window: int = 20
## Success rate at or above which the level is raised.
var promote_threshold: float = 0.7
## Success rate below which the level is lowered.
var demote_threshold: float = 0.15
## Episodes that must elapse after a change before another one is allowed.
var cooldown_episodes: int = 20

var episodes_observed: int = 0
var episodes_since_change: int = 0
var last_change_reason: String = ""
var promotions: int = 0
var demotions: int = 0

## Rolling window of 1.0 (win) / 0.0 (loss) outcomes.
var _outcomes: Array = []


static func create(
	p_level: int = CurriculumConfig.Level.STATIONARY_TARGET, p_enabled: bool = false
) -> CurriculumController:
	var controller: CurriculumController = (
		(load(SELF_PATH) as GDScript).new() as CurriculumController
	)
	controller.level = p_level
	controller.enabled = p_enabled
	return controller


func configure(p_window: int, p_promote: float, p_demote: float, p_cooldown: int = -1) -> void:
	window = maxi(1, p_window)
	promote_threshold = clampf(p_promote, 0.0, 1.0)
	demote_threshold = clampf(p_demote, 0.0, promote_threshold)
	cooldown_episodes = window if p_cooldown < 0 else maxi(0, p_cooldown)


func reset() -> void:
	_outcomes.clear()
	episodes_observed = 0
	episodes_since_change = 0
	last_change_reason = ""
	promotions = 0
	demotions = 0


## Feeds one finished episode's metrics (the Dictionary returned by
## `EpisodeState.to_metrics()`). Returns the level that should now be
## active — unchanged unless a threshold was crossed.
func record_episode(metrics: Dictionary) -> int:
	var success: bool = bool(metrics.get("win", false))
	return record_outcome(success)


## Lower-level entry point used by tests and by callers that compute their
## own success criterion.
func record_outcome(success: bool) -> int:
	episodes_observed += 1
	episodes_since_change += 1
	_outcomes.append(1.0 if success else 0.0)
	while _outcomes.size() > window:
		_outcomes.remove_at(0)
	if not enabled:
		return level
	if _outcomes.size() < window or episodes_since_change < cooldown_episodes:
		return level

	var rate: float = success_rate()
	if rate >= promote_threshold and level < max_level:
		level += 1
		promotions += 1
		episodes_since_change = 0
		_outcomes.clear()
		last_change_reason = (
			"promoted to %d: %.0f%% success over %d episodes" % [level, rate * 100.0, window]
		)
	elif rate <= demote_threshold and level > min_level:
		level -= 1
		demotions += 1
		episodes_since_change = 0
		_outcomes.clear()
		last_change_reason = (
			"demoted to %d: %.0f%% success over %d episodes" % [level, rate * 100.0, window]
		)
	return level


func success_rate() -> float:
	if _outcomes.is_empty():
		return 0.0
	var total: float = 0.0
	for value in _outcomes:
		total += float(value)
	return total / float(_outcomes.size())


func to_dict() -> Dictionary:
	return {
		"enabled": enabled,
		"level": level,
		"level_name": CurriculumConfig.level_name(level),
		"min_level": min_level,
		"max_level": max_level,
		"window": window,
		"promote_threshold": promote_threshold,
		"demote_threshold": demote_threshold,
		"cooldown_episodes": cooldown_episodes,
		"episodes_observed": episodes_observed,
		"episodes_since_change": episodes_since_change,
		"success_rate": success_rate(),
		"promotions": promotions,
		"demotions": demotions,
		"last_change_reason": last_change_reason,
	}

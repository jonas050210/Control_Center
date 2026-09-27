## RLAdapter
##
## Thin, framework-agnostic façade over SimulationManager. This is the class
## an eventual training loop (e.g. a Python PPO trainer talking over a
## future socket/GDExtension bridge, or an in-engine imitation-learning
## script) is meant to call. It intentionally does NOT depend on any
## external RL library — adding a heavy dependency before it is needed would
## contradict the "keep it minimal" milestone goal. When a real trainer is
## connected, only this file (and a transport layer) should need to change.
##
## Interface mirrors the classic Gym-style contract requested by the
## milestone spec:
##   reset()             -> Array[Observation]
##   step(actions)        -> Dictionary{observations, rewards, dones, infos}
##   get_observations()   -> Array[Observation]
##   get_rewards()        -> Array[float]
##   is_done()            -> Array[bool]
class_name RLAdapter
extends RefCounted

var simulation_manager: SimulationManager


func _init(p_simulation_manager: SimulationManager) -> void:
	simulation_manager = p_simulation_manager


## Static description of the action space (for a future Python-side gym
## wrapper to introspect without hardcoding magic numbers).
static func action_space_info() -> Dictionary:
	return {
		"type": "multi_discrete_plus_continuous",
		"discrete_choices": Action.DISCRETE_COUNT,
		"fields": ["move_axis", "strafe_axis", "look_yaw_axis", "look_pitch_axis", "shoot"],
		"continuous_reserved": ["look_delta.x", "look_delta.y"],
	}


## Static description of the observation space.
static func observation_space_info() -> Dictionary:
	return {
		"type": "structured_float_vector",
		"size": Observation.FIELD_COUNT,
		"mode": SandboxConfig.ACTIVE_OBSERVATION_MODE,
	}


func reset(seed_base: int = -1) -> Array:
	return simulation_manager.reset_all(seed_base)


## `actions` is an Array of either `Action` instances or raw discrete ints
## (0-9); ints are converted via `Action.from_discrete()` for convenience.
func step(actions: Array) -> Dictionary:
	var resolved: Array = []
	for a in actions:
		if a is Action:
			resolved.append(a)
		elif typeof(a) == TYPE_INT:
			resolved.append(Action.from_discrete(a))
		else:
			resolved.append(Action.idle())

	var results: Array = simulation_manager.step_all(resolved)
	var observations: Array = []
	var rewards: Array = []
	var dones: Array = []
	var infos: Array = []
	for r in results:
		observations.append(r.observation)
		rewards.append(r.reward)
		dones.append(r.done)
		infos.append(r.info)

	return {
		"observations": observations,
		"rewards": rewards,
		"dones": dones,
		"infos": infos,
	}


func get_observations() -> Array:
	return simulation_manager.get_observations()


func get_rewards() -> Array:
	return simulation_manager.get_rewards()


func is_done() -> Array:
	return simulation_manager.is_done_all()

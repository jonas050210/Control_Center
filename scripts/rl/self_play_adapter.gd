## SelfPlayAdapter
##
## Batch façade over SelfPlayEnvironmentCore instances for the headless
## JSON-lines bridge (rl_server `--self-play` mode). Each environment is a
## deterministic two-agent match: the caller provides one action per policy
## slot and receives both slots' observations, rewards and infos back.
##
## Wire shapes mirror RLAdapter with an added slot axis of size 2:
##   step actions      [[action_a, action_b], ... per environment]
##   observations      [[obs_a, obs_b], ... per environment]
##   rewards/infos     [[r_a, r_b] / [info_a, info_b], ... per environment]
##
## Determinism model: every reset is fully seeded (seed_base + env index;
## slot B derives +1000003 inside SelfPlayEnvironmentCore), so a match is
## reproducible from its seed alone. Episodes do NOT auto-reset on done;
## the caller drives match boundaries explicitly, which is what a league
## needs to attribute results.
class_name SelfPlayAdapter
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SelfPlayEnvironmentCore = preload("res://scripts/self_play/self_play_environment.gd")

## Stride between the two policy slots' seeds, matching
## SelfPlayEnvironmentCore's own derivation.
const SLOT_SEED_STRIDE: int = 1000003

var environments: Array = []  # Array[SelfPlayEnvironmentCore]


func _init(environment_count: int = 1, base_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED) -> void:
	var count: int = maxi(1, environment_count)
	for index in range(count):
		var env := SelfPlayEnvironmentCore.new()
		if env == null:
			# A script that failed to COMPILE still loads as a resource, so
			# this preload chain works, but .new() returns null. Without this
			# guard the null reset() call below aborts _init BEFORE the
			# append, leaving `environments` empty — every later reset then
			# silently serializes as [] over the bridge. Fail loudly instead
			# (the compile error itself is on stderr) and keep going with the
			# environments that did construct.
			push_error(
				"SelfPlayEnvironmentCore.new() returned null: the script failed to compile"
			)
			continue
		env.reset(base_seed + index)
		environments.append(env)


func environment_count() -> int:
	return environments.size()


## Resets every environment with an explicit deterministic seed. A negative
## seed_base falls back to the project default seed — self-play resets are
## ALWAYS fully seeded so match results stay reproducible.
func reset(seed_base: int = -1) -> Array:
	var base: int = seed_base if seed_base >= 0 else SandboxConfig.DEFAULT_RANDOM_SEED
	var observations: Array = []
	for i in range(environments.size()):
		var env: SelfPlayEnvironmentCore = environments[i]
		observations.append(_serialize_pair(env.reset(base + i)))
	return observations


## Steps every environment with one action pair each. No auto-reset: a
## finished environment returns its already_done result until reset.
func step(actions: Array) -> Dictionary:
	var observations: Array = []
	var rewards: Array = []
	var dones: Array = []
	var infos: Array = []
	for i in range(environments.size()):
		var env: SelfPlayEnvironmentCore = environments[i]
		var pair: Array = actions[i] if i < actions.size() and actions[i] is Array else []
		var action_a: Action = RLAdapter._resolve_action(pair[0] if pair.size() > 0 else null)
		var action_b: Action = RLAdapter._resolve_action(pair[1] if pair.size() > 1 else null)
		var result: Dictionary = env.step([action_a, action_b])
		observations.append(_serialize_pair(result["observations"]))
		rewards.append([float(result["rewards"][0]), float(result["rewards"][1])])
		dones.append(bool(result["done"]))
		infos.append(result["infos"])
	return {
		"observations": observations,
		"rewards": rewards,
		"dones": dones,
		"infos": infos,
	}


func health_check() -> Array:
	var reports: Array = []
	for env in environments:
		reports.append((env as SelfPlayEnvironmentCore).health_check())
	return reports


func set_map(map_id: String) -> bool:
	var all_ok: bool = true
	for env in environments:
		if not (env as SelfPlayEnvironmentCore).set_map(map_id):
			all_ok = false
	return all_ok


func set_layout(layout_id: String) -> bool:
	var all_ok: bool = true
	for env in environments:
		if not (env as SelfPlayEnvironmentCore).set_layout(layout_id):
			all_ok = false
	return all_ok


func set_lighting_mode(mode_id: String) -> bool:
	var all_ok: bool = true
	for env in environments:
		if not (env as SelfPlayEnvironmentCore).set_lighting_mode(mode_id):
			all_ok = false
	return all_ok


func set_curriculum_level(level: int) -> void:
	for env in environments:
		(env as SelfPlayEnvironmentCore).set_curriculum_level(level)


func get_episode_conditions() -> Array:
	var conditions: Array = []
	for env in environments:
		conditions.append((env as SelfPlayEnvironmentCore).get_episode_condition())
	return conditions


static func _serialize_pair(observations: Array) -> Array:
	return [
		RLAdapter._observation_to_array(observations[0]),
		RLAdapter._observation_to_array(observations[1]),
	]

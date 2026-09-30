## Tests for PerceptionModel: the "what does the AI see?" data source.
##
## The critical invariants are (a) REAL WORLD and AI PERCEPTION are kept in
## separate branches, (b) AI PERCEPTION is decoded exclusively from the
## observation vector, so debug visualization can never leak hidden world
## state into the policy's view, and (c) features the simulation does not
## implement are reported as unavailable instead of faked.
class_name TestPerceptionModel
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const PerceptionModel = preload("res://scripts/control_center/perception_model.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_build_separates_real_world_from_ai_perception() -> SandboxTest:
	var t := SandboxTest.new("perception_separates_real_world_and_ai")
	var env := EnvironmentCore.new(0, 2)
	env.reset(123)
	var perception: Dictionary = PerceptionModel.build(env)

	t.assert_true(perception.has("real_world"), "ground truth lives in its own branch")
	t.assert_true(perception.has("ai_perception"), "the AI's view lives in its own branch")
	var ai: Dictionary = perception["ai_perception"]
	t.assert_eq(
		str(ai["source"]),
		"observation_vector",
		"the AI branch must declare the observation vector as its only source"
	)
	t.assert_eq(
		(ai["slots"] as Array).size(),
		Observation.MAX_TRACKED_ENEMIES,
		"the AI branch has exactly as many enemy slots as the contract"
	)
	var real_world: Dictionary = perception["real_world"]
	t.assert_eq((real_world["enemies"] as Array).size(), 2)
	t.assert_true(perception.is_empty() == false)
	t.assert_true(PerceptionModel.build(null).is_empty(), "no environment means no perception")
	return t


func test_ai_perception_matches_the_observation_not_the_world() -> SandboxTest:
	var t := SandboxTest.new("perception_ai_branch_decodes_observation")
	var env := EnvironmentCore.new(0, 1)
	env.reset(55)
	for _step in range(10):
		env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))
	var observation: Observation = env.get_observations()
	var perception: Dictionary = PerceptionModel.build(env, observation)
	var primary: Dictionary = (perception["ai_perception"]["slots"] as Array)[0]

	t.assert_almost_eq(
		float(primary["distance_m"]),
		observation.enemy_distance_norm * SandboxConfig.ARENA_MAX_DISTANCE,
		0.001,
		"distance is the de-normalized observation value, not a world measurement"
	)
	t.assert_almost_eq(float(primary["bearing_deg"]), observation.enemy_bearing_norm * 180.0, 0.001)
	t.assert_almost_eq(float(primary["health_norm"]), observation.enemy_health_norm, 0.001)
	t.assert_true(
		bool(primary["alive"]) == (observation.alive_enemy_count_norm > 0.0),
		"slot occupancy is decoded from the alive-enemy count field, not from world state"
	)
	return t


## Regression guard for the strict "AI PERCEPTION = observation vector"
## rule: `Observation.enemy_alive` is internal bookkeeping that is NOT part
## of to_array(), so it must not influence the AI branch at all.
func test_ai_branch_ignores_members_outside_the_observation_vector() -> SandboxTest:
	var t := SandboxTest.new("perception_ai_branch_ignores_non_vector_members")
	var env := EnvironmentCore.new(0, 1)
	env.reset(77)

	var observation := Observation.new()
	# Contradictory on purpose: the non-vector member claims a live enemy,
	# the vector (alive_enemy_count_norm, index 18) says there is none.
	observation.enemy_alive = true
	observation.alive_enemy_count_norm = 0.0
	var perception: Dictionary = PerceptionModel.build(env, observation)
	var primary: Dictionary = (perception["ai_perception"]["slots"] as Array)[0]
	t.assert_false(
		bool(primary["alive"]),
		"a member that never reaches the policy must not show up as AI knowledge"
	)

	# And the opposite direction: the vector alone is enough to occupy the slot.
	observation.enemy_alive = false
	observation.alive_enemy_count_norm = 1.0
	observation.enemy_distance_norm = 0.5
	var second: Dictionary = (
		PerceptionModel.build(env, observation)["ai_perception"]["slots"] as Array
	)[0]
	t.assert_true(bool(second["alive"]), "the vector is the only source of AI knowledge")
	t.assert_almost_eq(float(second["distance_m"]), 0.5 * SandboxConfig.ARENA_MAX_DISTANCE, 0.001)
	return t


func test_enemies_beyond_the_contract_budget_are_flagged_hidden() -> SandboxTest:
	var t := SandboxTest.new("perception_flags_hidden_enemies")
	var env := EnvironmentCore.new(0, 6)
	env.set_curriculum_level(4)
	env.reset(808)
	var perception: Dictionary = PerceptionModel.build(env)

	var tracked: Array = perception["tracked_enemy_indices"]
	t.assert_eq(
		tracked.size(),
		Observation.MAX_TRACKED_ENEMIES,
		"only the contract's tracked enemies are represented"
	)
	var hidden: Array = perception["hidden_from_ai"]
	t.assert_eq(
		hidden.size(),
		env.get_alive_enemy_count() - Observation.MAX_TRACKED_ENEMIES,
		"every remaining alive enemy is reported as hidden from the AI"
	)
	for entry_value in hidden:
		var entry: Dictionary = entry_value
		t.assert_eq(str(entry["reason"]), "beyond_tracked_enemy_budget")
		t.assert_false(tracked.has(int(entry["index"])), "hidden enemies are not also tracked")
	return t


func test_tracked_indices_follow_the_observation_ranking_rule() -> SandboxTest:
	var t := SandboxTest.new("perception_tracked_matches_observation_ranking")
	var env := EnvironmentCore.new(0, 4)
	env.set_curriculum_level(4)
	env.reset(1717)
	for _step in range(25):
		env.step(Action.from_discrete(Action.Discrete.MOVE_FORWARD))

	var perception: Dictionary = PerceptionModel.build(env)
	var tracked: Array = perception["tracked_enemy_indices"]
	var ranked: Array = Observation.rank_alive_enemies(env.enemies, env.agent.position)
	for rank in range(mini(ranked.size(), Observation.MAX_TRACKED_ENEMIES)):
		t.assert_eq(
			env.enemies[int(tracked[rank])],
			ranked[rank],
			"tracked order must equal Observation.rank_alive_enemies order"
		)
	t.assert_eq(
		int(perception["target_index"]),
		PerceptionModel.current_target_index(env),
		"the overlay target index comes from the same ranking rule"
	)
	return t


## Originally asserted that FOV/LOS/sound/memory were all unavailable,
## because none of them existed. They exist now, so the invariant this
## guards has been inverted rather than deleted: a feature is reported
## available if and only if the environment really implements the hook, and
## an unavailable one must still carry an explanation instead of being
## silently faked.
func test_perception_features_are_reported_honestly() -> SandboxTest:
	var t := SandboxTest.new("perception_features_reported_honestly")
	var env := EnvironmentCore.new(0, 1)
	env.reset(9)
	var perception: Dictionary = PerceptionModel.build(env)
	var capabilities: Dictionary = perception["capabilities"]

	for feature_id in [
		"field_of_view",
		"line_of_sight",
		"sound_events",
		"target_memory",
		"obstacles_cover",
		"navigation",
		"dead_bodies"
	]:
		t.assert_true(capabilities.has(feature_id), "%s must be described" % feature_id)
		var capability: Dictionary = capabilities[feature_id]
		var method_name: String = str(capability["method"])
		t.assert_eq(
			bool(capability["available"]),
			env.has_method(method_name),
			"%s availability must match has_method(%s)" % [feature_id, method_name]
		)
		if not bool(capability["available"]):
			t.assert_false(
				str(capability["note"]).is_empty(),
				"%s must explain why it is unavailable" % feature_id
			)

	t.assert_eq(
		(perception["unavailable_features"] as Array).size(),
		0,
		"EnvironmentCore implements every perception hook"
	)
	return t


## An environment WITHOUT the hooks must still be described honestly: this
## is what keeps the Control Center from inventing perception data for the
## self-play environment or a future adapter.
func test_environment_without_hooks_reports_every_feature_unavailable() -> SandboxTest:
	var t := SandboxTest.new("environment_without_hooks_reports_unavailable")
	var capabilities: Dictionary = PerceptionModel.capabilities(RefCounted.new())
	t.assert_eq(capabilities.size(), PerceptionModel.OPTIONAL_FEATURES.size())
	for feature_id in capabilities.keys():
		var capability: Dictionary = capabilities[feature_id]
		t.assert_false(bool(capability["available"]), "%s must not be faked" % feature_id)
		t.assert_false(str(capability["note"]).is_empty())
	return t


func test_format_lines_labels_both_worlds() -> SandboxTest:
	var t := SandboxTest.new("perception_format_lines")
	var env := EnvironmentCore.new(0, 2)
	env.reset(31)
	var lines: PackedStringArray = PerceptionModel.format_lines(PerceptionModel.build(env))
	var joined: String = "\n".join(lines)
	t.assert_true(joined.contains("REAL WORLD"), "ground truth section is labelled")
	t.assert_true(joined.contains("AI PERCEPTION"), "AI section is labelled")
	t.assert_eq(PerceptionModel.format_lines({})[0], "no environment selected")
	return t

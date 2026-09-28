## Tests the team-play foundation: assignment, isolation, friendly fire and
## — most importantly — that teams stay off and change nothing by default.
class_name TestTeamConfig
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const TeamConfig = preload("res://scripts/team/team_config.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_teams_are_disabled_by_default() -> SandboxTest:
	var t := SandboxTest.new("team_disabled_by_default")
	var config := TeamConfig.new()
	t.assert_false(config.enabled)
	t.assert_false(config.comms_enabled)
	t.assert_false(config.friendly_fire)
	t.assert_almost_eq(config.team_reward_weight, 0.0)
	t.assert_eq(config.team_of(0), TeamConfig.SOLO_TEAM)
	t.assert_eq(config.team_count(), 1)
	t.assert_eq(config.teammates_of(0).size(), 0)
	return t


func test_disabled_config_is_canonicalized() -> SandboxTest:
	var t := SandboxTest.new("team_disabled_is_canonical")
	var config := TeamConfig.new(false, {0: 0, 1: 1, 2: 1})
	# A disabled config must not keep a half-configured team layout around.
	t.assert_eq(config.team_of(1), TeamConfig.SOLO_TEAM)
	t.assert_eq(config.team_of(2), TeamConfig.SOLO_TEAM)
	config.comms_enabled = true
	config.team_reward_weight = 0.5
	config.friendly_fire = true
	config.canonicalize()
	t.assert_false(config.comms_enabled)
	t.assert_almost_eq(config.team_reward_weight, 0.0)
	t.assert_false(config.friendly_fire)
	return t


func test_versus_layout_assigns_two_teams() -> SandboxTest:
	var t := SandboxTest.new("team_versus_layout")
	var config := TeamConfig.versus(2)
	t.assert_true(config.enabled)
	t.assert_eq(config.team_count(), 2)
	t.assert_eq(config.team_of(0), 0)
	t.assert_eq(config.team_of(1), 0)
	t.assert_eq(config.team_of(2), 1)
	t.assert_eq(config.team_of(3), 1)
	t.assert_eq(config.slots_of(0), [0, 1])
	t.assert_eq(config.slots_of(1), [2, 3])
	t.assert_eq(config.teammates_of(2), [3])
	return t


func test_team_isolation() -> SandboxTest:
	var t := SandboxTest.new("team_isolation")
	var config := TeamConfig.versus(2)
	t.assert_true(config.are_teammates(0, 1))
	t.assert_false(config.are_teammates(0, 2))
	t.assert_false(config.are_teammates(1, 3))
	# An unassigned slot falls back to the solo team rather than joining
	# someone else's team by accident.
	t.assert_eq(config.team_of(99), TeamConfig.SOLO_TEAM)
	t.assert_false(config.are_teammates(2, 99))
	return t


func test_solo_config_has_no_teammates() -> SandboxTest:
	var t := SandboxTest.new("team_solo_has_no_teammates")
	var config := TeamConfig.solo()
	t.assert_false(config.are_teammates(0, 1))
	t.assert_eq(config.teammates_of(0).size(), 0)
	return t


func test_friendly_fire_rules() -> SandboxTest:
	var t := SandboxTest.new("team_friendly_fire")
	var config := TeamConfig.versus(2)
	t.assert_false(config.can_damage(0, 0), "an agent cannot shoot itself")
	t.assert_false(config.can_damage(0, 1), "friendly fire is off by default")
	t.assert_true(config.can_damage(0, 2))
	config.friendly_fire = true
	t.assert_true(config.can_damage(0, 1))

	# With teams off, everyone that is not you is a valid target — exactly
	# the pre-existing single-agent behaviour.
	var solo := TeamConfig.solo()
	t.assert_true(solo.can_damage(0, 1))
	t.assert_false(solo.can_damage(0, 0))
	return t


func test_comms_symbols_are_a_closed_vocabulary() -> SandboxTest:
	var t := SandboxTest.new("team_comms_vocabulary")
	var config := TeamConfig.versus()
	t.assert_true(config.is_valid_symbol("contact"))
	t.assert_true(config.is_valid_symbol("none"))
	t.assert_false(config.is_valid_symbol("enemy_is_at_12_7_3"))
	t.assert_eq(TeamConfig.COMMS_SYMBOLS.size(), 8)
	return t


func test_teammate_reports_cannot_leak_ground_truth() -> SandboxTest:
	var t := SandboxTest.new("team_report_validation")
	var config := TeamConfig.versus()
	var good: Dictionary = {
		"teammate_slot": 1,
		"team_id": 0,
		"alive": true,
		"health_norm": 0.6,
		"bearing_norm": -0.2,
		"distance_norm": 0.4,
		"last_symbol": "contact",
		"confidence": 0.8,
		"age_norm": 0.1,
	}
	t.assert_eq(config.validate_report(good).size(), 0, str(config.validate_report(good)))

	var leaky: Dictionary = good.duplicate()
	leaky["enemy_position"] = [1.0, 2.0, 3.0]
	t.assert_true(config.validate_report(leaky).size() > 0)

	var unknown: Dictionary = good.duplicate()
	unknown["secret_channel"] = 1
	t.assert_true(config.validate_report(unknown).size() > 0)

	var out_of_range: Dictionary = good.duplicate()
	out_of_range["confidence"] = 7.0
	t.assert_true(config.validate_report(out_of_range).size() > 0)

	var bad_symbol: Dictionary = good.duplicate()
	bad_symbol["last_symbol"] = "coordinates_incoming"
	t.assert_true(config.validate_report(bad_symbol).size() > 0)
	return t


func test_objective_falls_back_to_a_known_value() -> SandboxTest:
	var t := SandboxTest.new("team_objective_fallback")
	var config := TeamConfig.versus()
	config.objective = "win_at_all_costs"
	config.canonicalize()
	t.assert_eq(config.objective, "eliminate")
	config.objective = "explore"
	config.canonicalize()
	t.assert_eq(config.objective, "explore")
	return t


func test_serialization_round_trip() -> SandboxTest:
	var t := SandboxTest.new("team_serialization")
	var config := TeamConfig.versus(3)
	var payload: Dictionary = config.to_dictionary()
	t.assert_true(bool(payload["enabled"]))
	t.assert_eq(int(payload["teams"]), 2)
	t.assert_eq((payload["slot_teams"] as Dictionary).size(), 6)
	t.assert_eq(str(payload["objective"]), "eliminate")
	# The returned dictionary is a copy: mutating it must not reconfigure
	# a live environment.
	(payload["slot_teams"] as Dictionary)[0] = 9
	t.assert_eq(config.team_of(0), 0)
	return t

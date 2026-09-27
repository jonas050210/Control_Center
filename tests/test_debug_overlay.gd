## Tests for DebugOverlay's pure telemetry-collection/formatting functions.
## These deliberately never construct Control/Label nodes: only the data
## pipeline (SimulationManager/EnvironmentCore -> Dictionary -> lines) is
## exercised, keeping the test fast and independent of any live scene tree.
class_name TestDebugOverlay
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const DebugOverlay = preload("res://scripts/debug/debug_overlay.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func _make_sim(env_count: int = 2, enemy_count: int = 2) -> SimulationManager:
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(env_count, enemy_count)
	return sim


func test_telemetry_dict_reports_core_fields() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_reports_core_fields")
	var sim := _make_sim(2, 2)
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	t.assert_true(telemetry.has("agent_health"))
	t.assert_true(telemetry.has("enemy_count"))
	t.assert_true(telemetry.has("enemies"))
	t.assert_true(telemetry.has("episode_reward"))
	t.assert_true(telemetry.has("reward_breakdown"))
	t.assert_true(telemetry.has("observation_summary"))
	t.assert_eq(telemetry.enemy_count, 2)
	t.assert_eq(telemetry.active_environments, 2)
	sim.free()
	return t


func test_telemetry_dict_tracks_shots_and_current_action() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_tracks_shots_and_current_action")
	var sim := _make_sim(1, 1)
	sim.step_all([Action.from_discrete(Action.Discrete.SHOOT)])
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	t.assert_eq(telemetry.shots_fired, 1)
	t.assert_true(telemetry.has("current_action"))
	t.assert_eq(int(telemetry.current_action[4]), 1, "recorded action should show shoot=1")
	sim.free()
	return t


func test_telemetry_dict_clamps_out_of_range_focus_index() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_clamps_out_of_range_focus_index")
	var sim := _make_sim(2, 1)
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 99)
	t.assert_eq(telemetry.focused_env_index, 1, "focus index should clamp to the last environment")
	sim.free()
	return t


func test_telemetry_dict_is_empty_with_no_environments() -> SandboxTest:
	var t := SandboxTest.new("telemetry_dict_is_empty_with_no_environments")
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(null, 0)
	t.assert_true(telemetry.is_empty())
	return t


func test_format_lines_produces_non_empty_readable_output() -> SandboxTest:
	var t := SandboxTest.new("format_lines_produces_non_empty_readable_output")
	var sim := _make_sim(1, 1)
	var telemetry: Dictionary = DebugOverlay.build_telemetry_dict(sim, 0)
	var lines: PackedStringArray = DebugOverlay.format_lines(telemetry)
	t.assert_gt(float(lines.size()), 5.0, "overlay should print multiple telemetry lines")
	var joined: String = "\n".join(lines)
	t.assert_true(joined.find("agent hp") >= 0)
	t.assert_true(joined.find("enemies alive") >= 0)
	sim.free()
	return t


func test_format_lines_handles_empty_telemetry_gracefully() -> SandboxTest:
	var t := SandboxTest.new("format_lines_handles_empty_telemetry_gracefully")
	var lines: PackedStringArray = DebugOverlay.format_lines({})
	t.assert_eq(lines.size(), 1)
	return t

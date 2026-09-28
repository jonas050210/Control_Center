## Tests the deterministic replay system: recording, serialization,
## loading, rejection of corrupt or incompatible files, playback control
## and — the point of the whole subsystem — that replaying a recorded
## action stream through a freshly seeded environment reproduces the same
## rewards.
class_name TestReplay
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const ReplayFormat = preload("res://scripts/replay/replay_format.gd")
const ReplayPlayer = preload("res://scripts/replay/replay_player.gd")
const ReplayRecorder = preload("res://scripts/replay/replay_recorder.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")

const TEST_PATH: String = "user://sandboxai_test_replay.jsonl"


func test_header_defaults_state_the_current_contract() -> SandboxTest:
	var t := SandboxTest.new("replay_header_defaults")
	var header: Dictionary = ReplayFormat.default_header()
	t.assert_eq(str(header["magic"]), ReplayFormat.MAGIC)
	t.assert_eq(int(header["version"]), ReplayFormat.FORMAT_VERSION)
	t.assert_eq(int(header["observation_dim"]), Observation.FIELD_COUNT)
	t.assert_eq((header["action_nvec"] as Array).size(), Action.MULTI_DISCRETE_SIZE)
	t.assert_eq(ReplayFormat.validate_header(header).size(), 0)
	return t


func test_recorder_records_ticks_and_events() -> SandboxTest:
	var t := SandboxTest.new("replay_records_ticks_and_events")
	var recorder := ReplayRecorder.new({"map_id": "open_field", "seed": 7})
	recorder.start(7)
	t.assert_true(recorder.is_recording())
	for index in range(5):
		recorder.record_step(Action.idle(), float(index), null, index == 4)
	recorder.record_step_events({"shot_fired": true, "hit": true, "damage_dealt": 25.0})
	recorder.record_step_events({"kill": true})
	var episode: Dictionary = recorder.finish({"done_reason": "target_eliminated"})
	t.assert_false(recorder.is_recording())
	t.assert_eq((episode["ticks"] as Array).size(), 5)
	t.assert_eq(int(episode["header"]["seed"]), 7)
	t.assert_eq(str(episode["result"]["done_reason"]), "target_eliminated")
	# episode_start + combat + death + episode_end
	t.assert_eq((episode["events"] as Array).size(), 4)
	t.assert_almost_eq(ReplayRecorder.total_reward(episode), 10.0)
	return t


func test_light_recording_stores_no_observations() -> SandboxTest:
	var t := SandboxTest.new("replay_light_omits_observations")
	var env := EnvironmentCore.new(0, 1)
	env.reset(3)
	var recorder := ReplayRecorder.new({}, ReplayFormat.DETAIL_LIGHT)
	recorder.start(3)
	recorder.record_step(Action.idle(), 0.0, env.get_observations(), false)
	var light: Dictionary = recorder.to_dictionary()
	t.assert_false((light["ticks"][0] as Dictionary).has("observation"))

	var detailed_recorder := ReplayRecorder.new({}, ReplayFormat.DETAIL_DETAILED)
	detailed_recorder.start(3)
	detailed_recorder.record_step(Action.idle(), 0.0, env.get_observations(), false)
	var detailed: Dictionary = detailed_recorder.to_dictionary()
	t.assert_true((detailed["ticks"][0] as Dictionary).has("observation"))
	t.assert_eq(
		((detailed["ticks"][0] as Dictionary)["observation"] as Array).size(),
		Observation.FIELD_COUNT
	)
	# The whole point of the default: light must be smaller.
	t.assert_true(
		recorder.to_lines().size() <= detailed_recorder.to_lines().size(),
		"light recording should not be larger than detailed"
	)
	return t


func test_save_and_load_round_trip() -> SandboxTest:
	var t := SandboxTest.new("replay_save_load_round_trip")
	var recorder := ReplayRecorder.new(
		{"map_id": "two_rooms", "scenario": "cover_fight", "lighting": "fog", "policy_id": "brain_a"}
	)
	recorder.start(11)
	for index in range(8):
		recorder.record_step(Action.from_multidiscrete([2, 1, 0, 1, 1, 0]), 0.5, null, index == 7)
	recorder.add_event("target_change", "enemy_1", {"reason": "nearest_visible"})
	recorder.finish({"done_reason": "timeout", "total_reward": 4.0})
	t.assert_true(recorder.save(TEST_PATH))

	var loaded: Dictionary = ReplayRecorder.load_replay(TEST_PATH)
	t.assert_eq((loaded["problems"] as Array).size(), 0, str(loaded["problems"]))
	t.assert_eq((loaded["ticks"] as Array).size(), 8)
	t.assert_eq(str(loaded["header"]["map_id"]), "two_rooms")
	t.assert_eq(str(loaded["header"]["policy_id"]), "brain_a")
	t.assert_eq(str(loaded["result"]["done_reason"]), "timeout")
	t.assert_almost_eq(ReplayRecorder.total_reward(loaded), 4.0)
	var actions: Array = ReplayRecorder.actions_of(loaded)
	t.assert_eq(actions.size(), 8)
	t.assert_eq((actions[0] as Action).move_axis, 1)
	t.assert_true((actions[0] as Action).shoot)
	DirAccess.remove_absolute(ProjectSettings.globalize_path(TEST_PATH))
	return t


func test_replaying_actions_reproduces_the_episode() -> SandboxTest:
	var t := SandboxTest.new("replay_is_deterministic")
	var recorded_rewards: Array = []
	var env := EnvironmentCore.new(0, 2)
	env.reset(1234)
	var recorder := ReplayRecorder.new({"seed": 1234, "enemy_count": 2})
	recorder.start(1234)
	for index in range(30):
		var action := Action.from_multidiscrete([1, 1, 2 if index % 3 == 0 else 1, 1, 1, 0])
		var result: Dictionary = env.step(action)
		recorded_rewards.append(float(result.get("reward", 0.0)))
		recorder.record_step(action, float(result.get("reward", 0.0)), null, false)
	recorder.finish({"done_reason": "recorded"})

	var replay: Dictionary = recorder.to_dictionary()
	var replay_env := EnvironmentCore.new(0, 2)
	replay_env.reset(int(replay["header"]["seed"]))
	var mismatches: int = 0
	var actions: Array = ReplayRecorder.actions_of(replay)
	for index in range(actions.size()):
		var result: Dictionary = replay_env.step(actions[index])
		if absf(float(result.get("reward", 0.0)) - float(recorded_rewards[index])) > 0.0001:
			mismatches += 1
	t.assert_eq(mismatches, 0, "replayed rewards diverged from the recording")
	return t


func test_corrupt_and_incompatible_replays_are_rejected() -> SandboxTest:
	var t := SandboxTest.new("replay_rejects_bad_files")

	var no_header: Dictionary = ReplayRecorder.parse(PackedStringArray(['{"tick":{"t":0,"a":[]}}']))
	t.assert_true((no_header["problems"] as Array).size() > 0, "missing header must be reported")

	var garbage: Dictionary = ReplayRecorder.parse(PackedStringArray(["not json at all"]))
	t.assert_true((garbage["problems"] as Array).size() > 0, "malformed line must be reported")

	var wrong_magic: Dictionary = ReplayFormat.default_header()
	wrong_magic["magic"] = "some.other.tool"
	t.assert_true(ReplayFormat.validate_header(wrong_magic).size() > 0)

	var future: Dictionary = ReplayFormat.default_header()
	future["version"] = ReplayFormat.FORMAT_VERSION + 99
	t.assert_true(ReplayFormat.validate_header(future).size() > 0)

	var old_contract: Dictionary = ReplayFormat.default_header()
	old_contract["observation_dim"] = Observation.LEGACY_FIELD_COUNT
	t.assert_true(ReplayFormat.validate_header(old_contract).size() > 0)
	# ... but it stays inspectable when the caller opts out of the strict check.
	t.assert_eq(ReplayFormat.validate_header(old_contract, false).size(), 0)

	var short_action: Dictionary = ReplayFormat.default_header()
	short_action["action_nvec"] = [3, 3, 3, 3, 2]
	t.assert_true(ReplayFormat.validate_header(short_action).size() > 0)

	var missing: Dictionary = ReplayRecorder.load_replay("user://does_not_exist_replay.jsonl")
	t.assert_true((missing["problems"] as Array).size() > 0)
	return t


func test_out_of_order_ticks_and_unknown_events_are_reported() -> SandboxTest:
	var t := SandboxTest.new("replay_structural_validation")
	var episode: Dictionary = {
		"header": ReplayFormat.default_header(),
		"ticks": [
			{"tick": 0, "action": [1, 1, 1, 1, 0, 0], "reward": 0.0},
			{"tick": 5, "action": [1, 1, 1, 1, 0, 0], "reward": 0.0},
		],
		"events": [{"tick": 0, "kind": "teleport_hack", "label": "", "data": {}}],
		"result": {},
	}
	var problems: Array = ReplayFormat.validate_episode(episode)
	t.assert_true(problems.size() >= 2, str(problems))
	return t


func test_unknown_event_kinds_are_not_recorded() -> SandboxTest:
	var t := SandboxTest.new("replay_rejects_unknown_event_kind")
	var recorder := ReplayRecorder.new()
	recorder.start(1)
	t.assert_false(recorder.add_event("wallhack", "nope"))
	# Only the episode_start event exists.
	t.assert_eq((recorder.to_dictionary()["events"] as Array).size(), 1)
	return t


func test_start_resets_previous_recording() -> SandboxTest:
	var t := SandboxTest.new("replay_start_resets")
	var recorder := ReplayRecorder.new()
	recorder.start(1)
	recorder.record_step(Action.idle(), 1.0)
	recorder.finish({"done_reason": "first"})
	recorder.start(2)
	t.assert_eq(recorder.tick_count(), 0)
	t.assert_eq(recorder.current_tick(), 0)
	t.assert_eq(int(recorder.header["seed"]), 2)
	t.assert_true((recorder.to_dictionary()["result"] as Dictionary).is_empty())
	return t


func test_recording_while_stopped_is_a_no_op() -> SandboxTest:
	var t := SandboxTest.new("replay_stopped_recorder_ignores_steps")
	var recorder := ReplayRecorder.new()
	t.assert_eq(recorder.record_step(Action.idle(), 1.0), -1)
	t.assert_eq(recorder.tick_count(), 0)
	return t


func _sample_episode(tick_count: int = 60) -> Dictionary:
	var recorder := ReplayRecorder.new({"map_id": "compound", "simulation_dt": 1.0 / 60.0})
	recorder.start(5)
	for index in range(tick_count):
		recorder.record_step(Action.idle(), 1.0, null, index == tick_count - 1)
		if index == 20:
			recorder.add_event("combat", "shot", {})
		if index == 40:
			recorder.add_event("death", "enemy_killed", {})
	return recorder.finish({"done_reason": "done"})


func test_player_pause_step_and_speed() -> SandboxTest:
	var t := SandboxTest.new("replay_player_transport")
	var player := ReplayPlayer.new(_sample_episode())
	t.assert_eq(player.tick_count(), 60)
	t.assert_almost_eq(player.duration(), 1.0, 0.01)

	# Paused by default: time passing must not move the cursor.
	t.assert_eq(player.advance(1.0).size(), 0)
	t.assert_eq(player.cursor, 0)

	# Frame stepping works while paused.
	player.step(3)
	t.assert_eq(player.cursor, 3)

	player.play()
	player.set_speed(2.0)
	var crossed: Array = player.advance(0.1)
	t.assert_eq(crossed.size(), 12, "2x speed for 0.1s at 60Hz should cross 12 ticks")
	t.assert_eq(player.cursor, 15)

	player.pause()
	t.assert_eq(player.advance(1.0).size(), 0)

	player.reset()
	t.assert_eq(player.cursor, 0)
	t.assert_false(player.playing)
	return t


func test_player_seeking_and_event_jumps() -> SandboxTest:
	var t := SandboxTest.new("replay_player_seek")
	var player := ReplayPlayer.new(_sample_episode())
	t.assert_eq(player.seek(30), 30)
	t.assert_eq(player.seek(-5), 0)
	t.assert_eq(player.seek(9999), 60)
	t.assert_eq(player.seek_time(0.5), 30)

	player.reset()
	var first: Dictionary = player.next_event()
	t.assert_eq(str(first.get("kind", "")), "episode_start")
	player.step(1)
	var second: Dictionary = player.next_event()
	t.assert_eq(int(second.get("tick", -1)), 20)
	t.assert_eq(player.cursor, 20)
	var third: Dictionary = player.next_event()
	t.assert_eq(int(third.get("tick", -1)), 20)
	player.step(1)
	t.assert_eq(int(player.next_event().get("tick", -1)), 40)

	var previous: Dictionary = player.previous_event()
	t.assert_eq(int(previous.get("tick", -1)), 20)
	t.assert_eq(player.events_at(20).size(), 1)
	return t


## Regression: an event was anchored to the NEXT tick, because record_step()
## had already advanced the cursor by the time the caller translated that
## step's events. "Jump to next event" then landed one tick after the shot
## and `events_at(n)` found nothing.
func test_events_are_anchored_to_the_tick_they_describe() -> SandboxTest:
	var t := SandboxTest.new("replay_events_anchored_to_their_tick")
	var recorder := ReplayRecorder.new({"map_id": "compound"})
	recorder.start(11)
	for index in range(5):
		recorder.record_step(Action.idle(), 0.0, null, index == 4)
		if index == 2:
			recorder.record_step_events({"shot_fired": true, "hit": true})
	var episode: Dictionary = recorder.finish({"done_reason": "done"})
	var by_kind: Dictionary = {}
	for event_value in episode["events"]:
		var event: Dictionary = event_value
		by_kind[str(event["kind"])] = int(event["tick"])
	t.assert_eq(int(by_kind["episode_start"]), 0, "episode_start belongs to tick 0")
	t.assert_eq(int(by_kind["combat"]), 2, "the shot fired during tick 2 belongs to tick 2")
	t.assert_eq(int(by_kind["episode_end"]), 4, "the episode ends on its last tick")

	var player := ReplayPlayer.new(episode)
	player.seek(2)
	t.assert_eq(player.events_at(2).size(), 1, "the event must be found at its own tick")
	return t


func test_player_status_is_presentation_only() -> SandboxTest:
	var t := SandboxTest.new("replay_player_status")
	var player := ReplayPlayer.new(_sample_episode(10))
	player.step(4)
	var status: Dictionary = player.status()
	t.assert_eq(int(status["tick"]), 4)
	t.assert_eq(int(status["tick_count"]), 10)
	t.assert_eq(str(status["map_id"]), "compound")
	t.assert_eq(str(status["done_reason"]), "done")
	t.assert_almost_eq(float(status["total_reward"]), 10.0)
	t.assert_false(bool(status["playing"]))
	# The status dictionary is derived data only: mutating it must not
	# change the replay it came from.
	status["tick"] = 999
	t.assert_eq(player.cursor, 4)
	return t


func test_empty_replay_is_handled_without_errors() -> SandboxTest:
	var t := SandboxTest.new("replay_empty_episode")
	var player := ReplayPlayer.new({})
	t.assert_eq(player.tick_count(), 0)
	t.assert_true(player.is_finished())
	t.assert_eq(player.step(5).size(), 0)
	t.assert_eq(player.current_tick_record().size(), 0)
	t.assert_eq(player.next_event().size(), 0)
	return t

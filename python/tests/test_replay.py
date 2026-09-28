"""Tests for the deterministic replay system (Phase 2).

Every test here runs without Godot: the determinism check is driven by a
scripted environment whose dynamics are a pure function of (seed, action
history), which is exactly the property a replay is supposed to verify.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.replay import (
    DetailLevel,
    IMPORTANT_EVENT_KINDS,
    READABLE_VERSIONS,
    REPLAY_FORMAT_VERSION,
    ReplayError,
    ReplayEpisode,
    ReplayHeader,
    ReplayIncompatibleError,
    ReplayPlayer,
    ReplayRecorder,
    load_replay,
    parse_replay,
    save_episode,
    validate_replay,
    verify_determinism,
)


class ScriptedEnv:
    """Deterministic stand-in for the Godot bridge.

    Observations and rewards depend only on the seed and the action
    history, so two runs with the same seed must agree exactly. The
    ``jitter`` flag deliberately breaks that, which is how the negative
    determinism test proves the checker can actually fail.
    """

    def __init__(self, jitter: bool = False) -> None:
        self.jitter = jitter
        self.seed = 0
        self.tick = 0
        self._noise = 0.0

    def reset(self, seed: int | None = None):
        self.seed = int(seed or 0)
        self.tick = 0
        self._noise = 0.0
        return self._observation(), {}

    def _observation(self):
        base = (self.seed % 97) / 100.0
        return [
            round(min(1.0, base + 0.001 * ((self.tick + index) % 13)) + self._noise, 6)
            for index in range(OBSERVATION_FIELD_COUNT)
        ]

    def step(self, action):
        self.tick += 1
        if self.jitter and self.tick == 4:
            self._noise = 0.5
        reward = 0.01 * sum(int(value) for value in action) + self._noise
        terminated = self.tick >= 12
        return self._observation(), reward, terminated, False, {"metrics": {}}

    def close(self) -> None:
        return None


def _record(detail: str = DetailLevel.LIGHT, env: ScriptedEnv | None = None, steps: int = 8):
    """Records `steps` ticks of ScriptedEnv through a recorder."""
    environment = env or ScriptedEnv()
    recorder = ReplayRecorder(
        header=ReplayHeader(
            seed=4242,
            map_id="blind_corner",
            scenario="corner_fight",
            lighting="low_light",
            enemy_count=2,
            curriculum_level=6,
            policy_id="brain_a",
            checkpoint="training/runs/a/latest.zip",
        ),
        detail=detail,
    )
    recorder.start(seed=4242)
    observation, _info = environment.reset(seed=4242)
    for index in range(steps):
        action = [index % 3, 1, 1, 1, index % 2, 0]
        observation, reward, terminated, truncated, _info = environment.step(action)
        recorder.record_step(action, reward, observation, done=bool(terminated or truncated))
        if index == 2:
            recorder.event("target_change", "enemy_1", {"reason": "closest_visible"})
        if index == 3:
            recorder.record_events({"shot_fired": True, "hit": True, "damage_dealt": 25.0})
        if terminated or truncated:
            break
    recorder.finish({"done_reason": "timeout", "win": False})
    return recorder


class ReplayRecordingTests(unittest.TestCase):
    def test_light_recording_stores_actions_rewards_and_events(self):
        recorder = _record()
        episode = recorder.episode()
        self.assertEqual(len(episode.ticks), 8)
        self.assertEqual(episode.header.detail, DetailLevel.LIGHT)
        self.assertTrue(all(tick.observation is None for tick in episode.ticks))
        kinds = {event.kind for event in episode.events}
        self.assertIn("episode_start", kinds)
        self.assertIn("target_change", kinds)
        self.assertIn("combat", kinds)
        self.assertIn("episode_end", kinds)

    def test_events_are_anchored_to_the_tick_they_describe(self):
        """Regression: events were filed one tick after the step they belong to.

        ``record_step`` advances the cursor, so a recorder that anchored
        events to the *current* tick put the shot fired during tick 3 at
        tick 4 — the Control Center's "jump to event" then landed after the
        shot and ``events_at(3)`` found nothing.
        """
        episode = _record().episode()
        by_kind = {event.kind: event for event in episode.events}
        self.assertEqual(by_kind["episode_start"].tick, 0)
        self.assertEqual(by_kind["target_change"].tick, 2)
        self.assertEqual(by_kind["combat"].tick, 3)
        self.assertEqual(by_kind["episode_end"].tick, len(episode.ticks) - 1)
        player = ReplayPlayer(episode)
        player.seek(3)
        self.assertEqual([event.kind for event in player.events_at()], ["combat"])

    def test_detailed_recording_stores_observations(self):
        recorder = _record(DetailLevel.DETAILED)
        episode = recorder.episode()
        self.assertTrue(episode.detailed)
        self.assertEqual(len(episode.observations()), 8)
        self.assertEqual(len(episode.observations()[0]), OBSERVATION_FIELD_COUNT)

    def test_light_replay_refuses_to_invent_observations(self):
        episode = _record().episode()
        with self.assertRaises(ReplayError):
            episode.observations()

    def test_detailed_recording_is_larger_than_light(self):
        self.assertGreater(
            _record(DetailLevel.DETAILED).size_estimate_bytes(),
            _record(DetailLevel.LIGHT).size_estimate_bytes() * 4,
        )

    def test_start_resets_previous_episode_state(self):
        recorder = _record()
        recorder.start(seed=7, map_id="open_field")
        self.assertEqual(recorder.tick, 0)
        self.assertEqual(recorder.ticks, [])
        self.assertEqual(recorder.header.seed, 7)
        self.assertEqual(recorder.header.map_id, "open_field")
        # The fresh episode_start event is the only one left.
        self.assertEqual([event.kind for event in recorder.events], ["episode_start"])

    def test_recording_after_finish_is_rejected(self):
        recorder = _record()
        with self.assertRaises(ReplayError):
            recorder.record_step([1, 1, 1, 1, 0, 0], 0.0)

    def test_unknown_event_kind_is_rejected(self):
        recorder = ReplayRecorder()
        recorder.start(seed=1)
        with self.assertRaises(ValueError):
            recorder.event("telepathy")

    def test_detailed_observation_width_is_checked(self):
        recorder = ReplayRecorder(detail=DetailLevel.DETAILED)
        recorder.start(seed=1)
        with self.assertRaises(ReplayError):
            recorder.record_step([1, 1, 1, 1, 0, 0], 0.0, observation=[0.0, 0.0])

    def test_unknown_header_field_is_rejected(self):
        recorder = ReplayRecorder()
        with self.assertRaises(AttributeError):
            recorder.start(seed=1, mystery_field=3)


class ReplayPersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_everything(self):
        recorder = _record(DetailLevel.DETAILED)
        original = recorder.episode()
        with tempfile.TemporaryDirectory() as tmp:
            path = recorder.save(Path(tmp) / "episode.jsonl")
            loaded = load_replay(path)
        self.assertEqual(loaded.header.seed, original.header.seed)
        self.assertEqual(loaded.header.policy_id, "brain_a")
        self.assertEqual(loaded.header.map_id, "blind_corner")
        self.assertEqual(len(loaded.ticks), len(original.ticks))
        self.assertEqual(loaded.actions(), original.actions())
        self.assertAlmostEqual(loaded.total_reward(), original.total_reward(), places=5)
        self.assertEqual(len(loaded.events), len(original.events))
        self.assertEqual(loaded.result["done_reason"], "timeout")

    def test_save_episode_round_trips_a_loaded_episode(self):
        recorder = _record()
        with tempfile.TemporaryDirectory() as tmp:
            first = recorder.save(Path(tmp) / "a.jsonl")
            loaded = load_replay(first)
            second = save_episode(loaded, Path(tmp) / "b.jsonl")
            self.assertEqual(
                Path(first).read_text(encoding="utf-8"),
                Path(second).read_text(encoding="utf-8"),
            )

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_replay("/nonexistent/replay.jsonl")


class ReplayRejectionTests(unittest.TestCase):
    def test_corrupt_json_is_rejected(self):
        lines = list(_record().iter_lines())
        lines[2] = "{not json"
        with self.assertRaises(ReplayError):
            parse_replay(lines)

    def test_missing_header_is_rejected(self):
        lines = list(_record().iter_lines())[1:]
        with self.assertRaises(ReplayError):
            parse_replay(lines)

    def test_foreign_file_is_rejected(self):
        with self.assertRaises(ReplayError):
            parse_replay(['{"header": {"magic": "some.other.format", "version": 1}}'])

    def test_future_version_is_rejected(self):
        lines = list(_record().iter_lines())
        lines[0] = lines[0].replace(
            f'"version":{REPLAY_FORMAT_VERSION}', f'"version":{max(READABLE_VERSIONS) + 99}'
        )
        with self.assertRaises(ReplayIncompatibleError):
            parse_replay(lines)

    def test_contract_drift_is_rejected(self):
        lines = list(_record().iter_lines())
        lines[0] = lines[0].replace(
            f'"observation_dim":{OBSERVATION_FIELD_COUNT}',
            f'"observation_dim":{OBSERVATION_FIELD_COUNT - 1}',
        )
        with self.assertRaises(ReplayIncompatibleError):
            parse_replay(lines)
        # ...but a tool that knowingly wants the old data can opt out.
        episode = parse_replay(lines, strict_contract=False)
        self.assertEqual(episode.header.observation_dim, OBSERVATION_FIELD_COUNT - 1)

    def test_action_width_drift_is_rejected(self):
        lines = list(_record().iter_lines())
        lines[0] = lines[0].replace(
            f'"action_nvec":{list(ACTION_NVEC)}'.replace(" ", ""), '"action_nvec":[3,3,3,3,2]'
        )
        with self.assertRaises(ReplayIncompatibleError):
            parse_replay(lines)

    def test_non_contiguous_ticks_are_rejected(self):
        lines = list(_record().iter_lines())
        del lines[3]
        with self.assertRaises(ReplayError):
            parse_replay(lines)

    def test_unknown_record_type_is_rejected(self):
        lines = list(_record().iter_lines())
        lines.insert(2, '{"telemetry": {"cpu": 1}}')
        with self.assertRaises(ReplayError):
            parse_replay(lines)

    def test_validate_reports_out_of_range_actions(self):
        episode = _record().episode()
        episode.ticks[1].action[0] = 99
        problems = validate_replay(episode)
        self.assertTrue(any("outside" in problem for problem in problems))

    def test_validate_accepts_a_good_replay(self):
        self.assertEqual(validate_replay(_record(DetailLevel.DETAILED).episode()), [])


class ReplayPlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episode: ReplayEpisode = _record(DetailLevel.DETAILED).episode()
        self.player = ReplayPlayer(self.episode)

    def test_paused_player_does_not_advance(self):
        self.assertEqual(self.player.advance(10.0), [])
        self.assertEqual(self.player.cursor, 0)

    def test_step_advances_regardless_of_pause(self):
        self.player.step(3)
        self.assertEqual(self.player.cursor, 3)

    def test_play_advances_by_wall_clock_and_speed(self):
        dt = self.episode.header.simulation_dt
        self.player.play()
        self.player.advance(dt * 2.0)
        self.assertEqual(self.player.cursor, 2)
        self.player.set_speed(2.0)
        self.player.advance(dt * 2.0)
        self.assertEqual(self.player.cursor, 6)

    def test_speed_is_clamped(self):
        self.assertEqual(self.player.set_speed(1000.0), ReplayPlayer.MAX_SPEED)
        self.assertEqual(self.player.set_speed(0.0), ReplayPlayer.MIN_SPEED)

    def test_playback_stops_at_the_end(self):
        self.player.play()
        self.player.advance(60.0)
        self.assertTrue(self.player.finished)
        self.assertFalse(self.player.playing)

    def test_seek_and_reset(self):
        self.player.seek(5)
        self.assertEqual(self.player.cursor, 5)
        self.player.seek(9999)
        self.assertEqual(self.player.cursor, len(self.episode.ticks))
        self.player.seek(-4)
        self.assertEqual(self.player.cursor, 0)
        self.player.step(4)
        self.player.play()
        self.player.reset()
        self.assertEqual(self.player.cursor, 0)
        self.assertFalse(self.player.playing)

    def test_seek_time(self):
        self.player.seek_time(self.episode.header.simulation_dt * 4)
        self.assertEqual(self.player.cursor, 4)

    def test_jump_between_important_events(self):
        event = self.player.jump_to_next_event()
        self.assertIsNotNone(event)
        self.assertEqual(self.player.cursor, event.tick)
        self.assertIn(event.kind, IMPORTANT_EVENT_KINDS)
        self.player.seek(len(self.episode.ticks))
        previous = self.player.jump_to_previous_event()
        self.assertIsNotNone(previous)
        self.assertLess(previous.tick, len(self.episode.ticks) + 1)

    def test_timeline_has_time_for_every_event(self):
        timeline = self.episode.timeline()
        self.assertEqual(len(timeline), len(self.episode.events))
        for row in timeline:
            self.assertAlmostEqual(row["time"], row["tick"] * self.episode.header.simulation_dt)

    def test_state_is_presentation_only_and_complete(self):
        self.player.step(2)
        state = self.player.state()
        for key in ("tick", "time", "progress", "playing", "speed", "policy_id", "seed", "map"):
            self.assertIn(key, state)
        self.assertEqual(state["tick"], 2)
        self.assertEqual(state["policy_id"], "brain_a")
        # Reading state must not move the cursor.
        self.assertEqual(self.player.cursor, 2)

    def test_summary_counts_events_by_kind(self):
        summary = self.episode.summary()
        self.assertEqual(summary["ticks"], 8)
        self.assertEqual(summary["event_kinds"]["episode_start"], 1)
        self.assertEqual(summary["policy_id"], "brain_a")


class ReplayDeterminismTests(unittest.TestCase):
    def test_replay_of_a_deterministic_environment_reproduces_it(self):
        episode = _record(DetailLevel.DETAILED).episode()
        report = verify_determinism(episode, ScriptedEnv)
        self.assertTrue(report.deterministic, report.detail)
        self.assertEqual(report.ticks_compared, len(episode.ticks))
        self.assertEqual(report.first_divergence_tick, -1)

    def test_light_replay_still_verifies_rewards(self):
        episode = _record(DetailLevel.LIGHT).episode()
        report = verify_determinism(episode, ScriptedEnv)
        self.assertTrue(report.deterministic, report.detail)
        self.assertEqual(report.observation_mismatch, 0)

    def test_non_deterministic_environment_is_detected(self):
        episode = _record(DetailLevel.DETAILED).episode()
        report = verify_determinism(episode, lambda: ScriptedEnv(jitter=True))
        self.assertFalse(report.deterministic)
        self.assertGreaterEqual(report.first_divergence_tick, 0)
        self.assertIn("diverged", report.detail)

    def test_two_recordings_with_the_same_seed_are_identical(self):
        first = _record(DetailLevel.DETAILED).episode()
        second = _record(DetailLevel.DETAILED).episode()
        self.assertEqual(first.actions(), second.actions())
        self.assertEqual(first.observations(), second.observations())
        self.assertEqual(first.rewards(), second.rewards())


if __name__ == "__main__":
    unittest.main()

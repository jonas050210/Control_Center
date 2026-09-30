"""Contract tests for the external-game adapter boundary (Phase 9)."""

from __future__ import annotations

import unittest

from sandboxai.contract import (
    ACTION_NVEC,
    OBSERVATION_FIELD_COUNT,
    OBSERVATION_GROUPS,
    OBSERVATION_INDEX,
    GameAdapter,
)
from sandboxai.external_adapter import (
    ALLOWED_DATA_SOURCES,
    EPISODE_STATES,
    EXTERNAL_CONTRACT_VERSION,
    FORBIDDEN_DATA_SOURCES,
    AdapterCapabilities,
    AdapterContractChecker,
    AdapterContractError,
    EpisodeStatus,
    ExternalEnvironment,
    MockExternalEnvironment,
    TimingContract,
    contract_summary,
    validate_action,
    validate_observation,
)


class CapabilityTests(unittest.TestCase):
    def test_default_capabilities_declare_every_channel(self):
        capabilities = AdapterCapabilities()
        capabilities.validate()
        self.assertEqual(set(capabilities.channels), set(OBSERVATION_GROUPS))
        self.assertEqual(capabilities.missing_channels, [])
        self.assertEqual(capabilities.neutral_indices(), [])

    def test_an_undeclared_channel_is_rejected(self):
        capabilities = AdapterCapabilities()
        capabilities.channels.pop("sound")
        with self.assertRaises(AdapterContractError):
            capabilities.validate()

    def test_an_unknown_channel_is_rejected(self):
        capabilities = AdapterCapabilities()
        capabilities.channels["wallhack"] = True
        with self.assertRaises(AdapterContractError):
            capabilities.validate()

    def test_forbidden_data_sources_are_rejected(self):
        for source in FORBIDDEN_DATA_SOURCES:
            capabilities = AdapterCapabilities(data_sources=[source])
            with self.assertRaises(AdapterContractError, msg=source):
                capabilities.validate()

    def test_unknown_data_sources_are_rejected(self):
        with self.assertRaises(AdapterContractError):
            AdapterCapabilities(data_sources=["telepathy"]).validate()

    def test_allowed_and_forbidden_sources_do_not_overlap(self):
        self.assertFalse(set(ALLOWED_DATA_SOURCES) & set(FORBIDDEN_DATA_SOURCES))

    def test_missing_channel_maps_to_the_right_indices(self):
        capabilities = AdapterCapabilities()
        capabilities.channels["sound"] = False
        indices = capabilities.neutral_indices()
        expected = []
        for name in OBSERVATION_GROUPS["sound"]:
            start, width = OBSERVATION_INDEX[name]
            expected.extend(range(start, start + width))
        self.assertEqual(indices, sorted(expected))


class TimingTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        TimingContract().validate()

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(AdapterContractError):
            TimingContract(mode="whenever").validate()

    def test_non_positive_rate_is_rejected(self):
        with self.assertRaises(AdapterContractError):
            TimingContract(tick_hz=0.0).validate()

    def test_fixed_mode_cannot_promise_more_than_one_tick_of_latency(self):
        with self.assertRaises(AdapterContractError):
            TimingContract(mode="fixed", tick_hz=60.0, max_latency=0.5).validate()

    def test_determinism_defaults_to_false_for_an_external_game(self):
        self.assertFalse(TimingContract().deterministic)


class ValidationTests(unittest.TestCase):
    def test_correct_observation_passes(self):
        validate_observation([0.0] * OBSERVATION_FIELD_COUNT)

    def test_wrong_width_is_rejected(self):
        with self.assertRaises(AdapterContractError):
            validate_observation([0.0] * (OBSERVATION_FIELD_COUNT - 1))

    def test_out_of_range_value_is_rejected(self):
        observation = [0.0] * OBSERVATION_FIELD_COUNT
        observation[3] = 5.0
        with self.assertRaises(AdapterContractError):
            validate_observation(observation)

    def test_nan_is_rejected(self):
        observation = [0.0] * OBSERVATION_FIELD_COUNT
        observation[7] = float("nan")
        with self.assertRaises(AdapterContractError):
            validate_observation(observation)

    def test_a_missing_channel_must_be_zero(self):
        capabilities = AdapterCapabilities()
        capabilities.channels["sound"] = False
        observation = [0.0] * OBSERVATION_FIELD_COUNT
        index, _width = OBSERVATION_INDEX["last_sound_bearing_norm"]
        observation[index] = 0.7
        with self.assertRaises(AdapterContractError) as context:
            validate_observation(observation, capabilities)
        self.assertIn("neutral encoding", str(context.exception))

    def test_action_validation(self):
        validate_action([1, 1, 1, 1, 0, 0])
        with self.assertRaises(AdapterContractError):
            validate_action([1, 1, 1, 1, 0])
        with self.assertRaises(AdapterContractError):
            validate_action([1, 1, 1, 1, 0, 9])
        with self.assertRaises(AdapterContractError):
            validate_action([-1, 1, 1, 1, 0, 0])


class EpisodeStatusTests(unittest.TestCase):
    def test_known_states(self):
        for state in EPISODE_STATES:
            EpisodeStatus(state=state)

    def test_unknown_state_is_rejected(self):
        with self.assertRaises(AdapterContractError):
            EpisodeStatus(state="winning")

    def test_finished_covers_unavailability(self):
        self.assertFalse(EpisodeStatus(state="running").finished)
        self.assertTrue(EpisodeStatus(state="terminated").finished)
        self.assertTrue(EpisodeStatus(state="truncated").finished)
        self.assertTrue(EpisodeStatus(state="unavailable").finished)


class MockAdapterTests(unittest.TestCase):
    def test_mock_is_a_game_adapter(self):
        adapter = MockExternalEnvironment()
        self.assertIsInstance(adapter, GameAdapter)
        self.assertIsInstance(adapter, ExternalEnvironment)

    def test_mock_passes_the_full_contract_check(self):
        problems = AdapterContractChecker(MockExternalEnvironment()).run()
        self.assertEqual(problems, [], "\n".join(problems))

    def test_mock_with_missing_channels_still_passes(self):
        capabilities = AdapterCapabilities(name="partial")
        capabilities.channels["sound"] = False
        capabilities.channels["exploration"] = False
        adapter = MockExternalEnvironment(capabilities=capabilities)
        problems = AdapterContractChecker(adapter).run()
        self.assertEqual(problems, [], "\n".join(problems))
        self.assertEqual(adapter.capabilities.missing_channels, ["exploration", "sound"])

    def test_observation_is_in_contract_shape(self):
        adapter = MockExternalEnvironment()
        observation = adapter.reset(seed=3)
        self.assertEqual(len(observation), OBSERVATION_FIELD_COUNT)
        validate_observation(observation, adapter.capabilities)

    def test_stepping_before_reset_is_rejected(self):
        with self.assertRaises(AdapterContractError):
            MockExternalEnvironment().step([1, 1, 1, 1, 0, 0])

    def test_stepping_a_finished_episode_is_rejected(self):
        adapter = MockExternalEnvironment(episode_length=2)
        adapter.reset(seed=1)
        adapter.step([1, 1, 1, 1, 0, 0])
        adapter.step([1, 1, 1, 1, 0, 0])
        self.assertTrue(adapter.status().finished)
        with self.assertRaises(AdapterContractError):
            adapter.step([1, 1, 1, 1, 0, 0])

    def test_invalid_action_is_rejected_by_the_adapter(self):
        adapter = MockExternalEnvironment()
        adapter.reset(seed=1)
        with self.assertRaises(AdapterContractError):
            adapter.step([9, 1, 1, 1, 0, 0])

    def test_closed_adapter_refuses_work(self):
        adapter = MockExternalEnvironment()
        adapter.reset(seed=1)
        adapter.close()
        self.assertTrue(adapter.closed)
        with self.assertRaises(AdapterContractError):
            adapter.reset(seed=1)
        with self.assertRaises(AdapterContractError):
            adapter.step([1, 1, 1, 1, 0, 0])

    def test_truncation_and_termination_are_distinguished(self):
        timeout = MockExternalEnvironment(episode_length=3)
        timeout.reset(seed=1)
        for _ in range(3):
            timeout.step([1, 1, 1, 1, 0, 0])
        self.assertEqual(timeout.status().state, "truncated")

        killer = MockExternalEnvironment(episode_length=500, seed=5)
        killer.reset(seed=5)
        for _ in range(500):
            if killer.status().finished:
                break
            # Turn toward the target, then fire when centred.
            observation = killer._observation()
            bearing_index, _ = OBSERVATION_INDEX["primary_enemy_bearing_norm"]
            bearing = observation[bearing_index]
            yaw = 2 if bearing > 0.02 else (0 if bearing < -0.02 else 1)
            shoot = 1 if abs(bearing) < 0.05 else 0
            killer.step([1, 1, yaw, 1, shoot, 0])
        self.assertEqual(killer.status().state, "terminated")
        self.assertEqual(killer.status().reason, "target_eliminated")

    def test_describe_is_machine_readable(self):
        description = MockExternalEnvironment().describe()
        self.assertEqual(description["observation_dim"], OBSERVATION_FIELD_COUNT)
        self.assertEqual(description["action_nvec"], list(ACTION_NVEC))
        self.assertEqual(description["contract_version"], EXTERNAL_CONTRACT_VERSION)
        self.assertIn("capabilities", description)


class LeakDetectionTests(unittest.TestCase):
    def test_the_checker_catches_an_adapter_that_fabricates_a_channel(self):
        class LeakyAdapter(MockExternalEnvironment):
            """Declares it has no sound, then reports a sound bearing."""

            def _observation(self):
                vector = super()._observation()
                index, _width = OBSERVATION_INDEX["last_sound_bearing_norm"]
                vector[index] = 0.9
                return vector

        capabilities = AdapterCapabilities(name="leaky")
        capabilities.channels["sound"] = False
        problems = AdapterContractChecker(LeakyAdapter(capabilities=capabilities)).run()
        self.assertTrue(problems)
        self.assertTrue(any("neutral encoding" in problem for problem in problems))

    def test_the_checker_catches_an_out_of_range_observation(self):
        class BrokenAdapter(MockExternalEnvironment):
            def _observation(self):
                vector = super()._observation()
                vector[0] = 12.0
                return vector

        problems = AdapterContractChecker(BrokenAdapter()).run()
        self.assertTrue(any("outside" in problem for problem in problems))

    def test_the_checker_catches_a_done_flag_that_disagrees_with_status(self):
        class LyingAdapter(MockExternalEnvironment):
            def step(self, action):
                observation, reward, _done, info = super().step(action)
                return observation, reward, True, info

        problems = AdapterContractChecker(LyingAdapter()).run()
        self.assertTrue(any("disagrees" in problem for problem in problems))


class ContractSummaryTests(unittest.TestCase):
    def test_summary_is_complete_and_states_the_scope(self):
        summary = contract_summary()
        self.assertEqual(summary["observation"]["dimension"], OBSERVATION_FIELD_COUNT)
        self.assertEqual(summary["action"]["nvec"], list(ACTION_NVEC))
        self.assertEqual(set(summary["observation"]["channels"]), set(OBSERVATION_GROUPS))
        self.assertEqual(summary["episode_states"], list(EPISODE_STATES))
        scope = summary["scope"].lower()
        for phrase in ("no client modification", "memory reading", "anti-cheat"):
            self.assertIn(phrase, scope)

    def test_no_module_in_the_package_mentions_an_exploit_technique(self):
        # Guards against the adapter boundary drifting into territory the
        # project explicitly refuses to enter.
        import re
        from pathlib import Path

        package = Path(__file__).resolve().parents[1] / "sandboxai"
        # Whole-word technique names. The forbidden-source *vocabulary*
        # (e.g. "client_modification") is allowed to name what it forbids;
        # an actual implementation verb is not.
        banned = (
            r"\bcode_injection\b",
            r"\bdll_inject\w*",
            r"\bhook_process\b",
            r"\breadprocessmemory\b",
            r"\bwriteprocessmemory\b",
            r"\bbypass_anticheat\b",
            r"\bpacket_spoof\w*",
        )
        for module in package.rglob("*.py"):
            text = module.read_text(encoding="utf-8").lower()
            for pattern in banned:
                self.assertIsNone(re.search(pattern, text), f"{module.name} matches {pattern}")


if __name__ == "__main__":
    unittest.main()

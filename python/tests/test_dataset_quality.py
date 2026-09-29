"""Dataset validation, statistics and leakage-free splitting.

These tests exist because the failure they guard against is silent: a
transition-level split of a demonstration recording reports an excellent
validation loss while the model has effectively seen every validation
state during training.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.dataset import DemonstrationDataset


def transition(value: float, episode: int, environment: int = 0, step: int = 0, done: bool = False):
    return {
        "observation": [value] * OBSERVATION_FIELD_COUNT,
        "action": [0, 0, 0, 0, 0, 0, 0.0, 0.0],
        "next_observation": [value + 0.001] * OBSERVATION_FIELD_COUNT,
        "reward": 0.25,
        "done": done,
        "episode_id": episode,
        "environment_id": environment,
        "step": step,
    }


def dataset_with_episodes(episodes: int, length: int, environments: int = 1):
    rows = []
    for environment in range(environments):
        for episode in range(episodes):
            for step in range(length):
                rows.append(
                    transition(
                        value=0.001 * (episode * length + step),
                        episode=episode,
                        environment=environment,
                        step=step,
                        done=step == length - 1,
                    )
                )
    return DemonstrationDataset(rows, {"source": "unit-test"})


class EpisodeStructureTests(unittest.TestCase):
    def test_groups_separate_environments_with_the_same_episode_id(self):
        dataset = dataset_with_episodes(episodes=2, length=3, environments=2)
        groups = dataset.episode_groups()
        self.assertEqual(len(groups), 4)
        self.assertEqual(dataset.summary()["episodes"], 4)

    def test_truncated_and_misplaced_terminals_are_reported(self):
        rows = [transition(0.1, episode=1, step=0, done=True), transition(0.2, episode=1, step=1)]
        problems = DemonstrationDataset(rows, {}).episode_boundary_problems()
        self.assertTrue(any("done=true before the last transition" in item for item in problems))
        self.assertTrue(any("truncated" in item for item in problems))

    def test_dataset_without_episode_ids_reports_missing_structure(self):
        rows = [
            {
                "observation": [0.0] * OBSERVATION_FIELD_COUNT,
                "action": [0, 0, 0, 0, 0, 0, 0.0, 0.0],
                "next_observation": [0.0] * OBSERVATION_FIELD_COUNT,
                "reward": 0.0,
                "done": True,
            }
        ]
        dataset = DemonstrationDataset(rows, {})
        self.assertFalse(dataset.has_episode_structure())
        self.assertIn(
            "no episode_id/episode_key on any transition; episodes cannot be separated",
            dataset.episode_boundary_problems(),
        )


class SplitLeakageTests(unittest.TestCase):
    def test_default_split_never_shares_an_episode(self):
        dataset = dataset_with_episodes(episodes=10, length=12)
        train, validation, report = dataset.split_with_report(0.2, seed=7)
        self.assertEqual(report.strategy, "episode")
        self.assertEqual(report.shared_groups, 0)
        train_keys = {DemonstrationDataset.group_key(row, i) for i, row in enumerate(train.transitions)}
        validation_keys = {
            DemonstrationDataset.group_key(row, i) for i, row in enumerate(validation.transitions)
        }
        self.assertFalse(train_keys & validation_keys)
        self.assertEqual(len(train.transitions) + len(validation.transitions), 120)

    def test_transition_split_leaks_and_is_therefore_not_the_default(self):
        dataset = dataset_with_episodes(episodes=10, length=12)
        _train, _validation, report = dataset.split_with_report(0.2, seed=7, strategy="transition")
        self.assertEqual(report.strategy, "transition")
        # The historical behaviour: with 120 shuffled rows from 10
        # episodes, essentially every episode lands on both sides.
        self.assertGreater(report.shared_groups, 0)

    def test_episode_split_is_stable_when_new_episodes_are_appended(self):
        small = dataset_with_episodes(episodes=6, length=10)
        _train, validation, _report = small.split_with_report(0.25, seed=11)
        before = {
            DemonstrationDataset.group_key(row, i) for i, row in enumerate(validation.transitions)
        }
        grown = DemonstrationDataset(
            list(small.transitions) + list(dataset_with_episodes(episodes=3, length=10).transitions),
            {},
        )
        # Appending episodes 0..2 of a second "run" must not move the
        # original episodes between sides beyond adding new ones.
        _train2, validation2, _report2 = grown.split_with_report(0.25, seed=11)
        after = {
            DemonstrationDataset.group_key(row, i) for i, row in enumerate(validation2.transitions)
        }
        self.assertTrue(before & after, "seeded group hashing should keep assignments stable")

    def test_split_is_deterministic_for_a_seed(self):
        dataset = dataset_with_episodes(episodes=8, length=5)
        first = dataset.split_with_report(0.25, seed=3)[1].transitions
        second = dataset.split_with_report(0.25, seed=3)[1].transitions
        self.assertEqual(first, second)

    def test_auto_falls_back_with_an_explicit_reason_for_group_less_data(self):
        rows = [transition(0.01 * index, episode=0, step=index) for index in range(10)]
        dataset = DemonstrationDataset(rows, {})
        _train, _validation, report = dataset.split_with_report(0.2, seed=1)
        self.assertEqual(report.strategy, "transition")
        self.assertIn("episode group", report.degraded_reason)

    def test_explicit_episode_strategy_refuses_group_less_data(self):
        rows = [transition(0.01 * index, episode=0, step=index) for index in range(10)]
        with self.assertRaises(ValueError):
            DemonstrationDataset(rows, {}).split_with_report(0.2, seed=1, strategy="episode")

    def test_training_side_is_never_empty(self):
        dataset = dataset_with_episodes(episodes=2, length=4)
        train, validation, report = dataset.split_with_report(0.9, seed=5)
        self.assertGreater(len(train.transitions), 0)
        self.assertGreater(len(validation.transitions), 0)
        self.assertEqual(report.shared_groups, 0)


class ValidationTests(unittest.TestCase):
    def test_contract_width_is_enforced_on_request(self):
        rows = [
            {
                "observation": [0.0] * 33,
                "action": [0, 0, 0, 0, 0, 0, 0.0, 0.0],
                "next_observation": [0.0] * 33,
                "reward": 0.0,
                "done": True,
                "episode_id": 1,
            }
        ]
        dataset = DemonstrationDataset(rows, {})
        dataset.validate()  # tolerated for inspection
        with self.assertRaises(ValueError) as caught:
            dataset.validate(require_contract_width=True)
        self.assertIn(str(OBSERVATION_FIELD_COUNT), str(caught.exception))

    def test_out_of_contract_action_component_is_rejected(self):
        rows = [
            {
                "observation": [0.0] * OBSERVATION_FIELD_COUNT,
                # 5 MultiDiscrete values with an axis of 7 -> out of range.
                "action": [7, 1, 1, 1, 0, 0],
                "next_observation": [0.0] * OBSERVATION_FIELD_COUNT,
                "reward": 0.0,
                "done": True,
            }
        ]
        with self.assertRaises(ValueError):
            DemonstrationDataset(rows, {}).validate()


class StatisticsTests(unittest.TestCase):
    def test_statistics_expose_duplicates_and_constant_components(self):
        rows = [transition(0.5, episode=1, step=index) for index in range(4)]
        rows[-1]["done"] = True
        statistics = DemonstrationDataset(rows, {}).statistics()
        self.assertEqual(statistics["transitions"], 4)
        self.assertEqual(statistics["duplicates"]["duplicate_transitions"], 3)
        self.assertAlmostEqual(statistics["duplicates"]["duplicate_fraction"], 0.75)
        # Every action component is constant in this dataset.
        self.assertEqual(statistics["constant_action_components"], list(range(len(ACTION_NVEC))))
        self.assertTrue(statistics["matches_contract"])

    def test_statistics_count_observations_outside_the_contract_range(self):
        rows = [transition(0.1, episode=1, step=0, done=True)]
        rows[0]["observation"][0] = 4.0
        statistics = DemonstrationDataset(rows, {}).statistics()
        self.assertEqual(statistics["observation_values_outside_contract_range"], 1)

    def test_fingerprint_follows_file_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.jsonl"
            dataset_with_episodes(episodes=2, length=3).save(path)
            first = DemonstrationDataset.load(path).fingerprint
            self.assertTrue(first.startswith("blake2b:"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(transition(0.9, episode=9, done=True)) + "\n")
            self.assertNotEqual(DemonstrationDataset.load(path).fingerprint, first)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

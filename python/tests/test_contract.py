"""Consistency checks for the Observation/Action contract description.

These tests validate the Python-side contract module used as the target for
a future external (e.g. Roblox) adapter. They intentionally do not require a
running Godot process; keeping OBSERVATION_SPEC in sync with
scripts/core/observation.gd is a manual responsibility documented in
contract.py's module docstring.
"""
import unittest

from sandboxai.contract import (
    ACTION_NVEC,
    ACTION_SPEC,
    OBSERVATION_FIELD_COUNT,
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
    OBSERVATION_MAX_TRACKED_ENEMIES,
    OBSERVATION_SPEC,
    GameAdapter,
    validate_observation_spec,
)


class ObservationContractTests(unittest.TestCase):
    def test_field_indices_are_contiguous_and_cover_the_full_vector(self):
        validate_observation_spec()

    def test_observation_field_count_matches_godot_contract(self):
        # Mirrors Observation.FIELD_COUNT in scripts/core/observation.gd.
        self.assertEqual(OBSERVATION_FIELD_COUNT, 33)

    def test_observation_bounds_are_symmetric_and_normalized(self):
        self.assertEqual(OBSERVATION_LOW, -1.0)
        self.assertEqual(OBSERVATION_HIGH, 1.0)

    def test_tracked_enemy_budget_matches_primary_plus_two_extra(self):
        # primary (implicit) + secondary + tertiary == 3 tracked enemies.
        self.assertEqual(OBSERVATION_MAX_TRACKED_ENEMIES, 3)

    def test_field_names_are_unique(self):
        names = [field.name for field in OBSERVATION_SPEC]
        self.assertEqual(len(names), len(set(names)))


class ActionContractTests(unittest.TestCase):
    def test_action_nvec_matches_multidiscrete_shape(self):
        self.assertEqual(ACTION_NVEC, (3, 3, 3, 3, 2))

    def test_action_field_indices_are_sequential(self):
        for expected_index, field in enumerate(ACTION_SPEC):
            self.assertEqual(field.index, expected_index)


class GameAdapterInterfaceTests(unittest.TestCase):
    def test_game_adapter_is_abstract_and_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            GameAdapter()  # type: ignore[abstract]

    def test_a_minimal_concrete_adapter_can_implement_the_interface(self):
        class DummyAdapter(GameAdapter):
            def reset(self, seed=None):
                return [0.0] * OBSERVATION_FIELD_COUNT

            def step(self, action):
                return [0.0] * OBSERVATION_FIELD_COUNT, 0.0, False, {}

            def close(self):
                pass

        adapter = DummyAdapter()
        observation = adapter.reset(seed=1)
        self.assertEqual(len(observation), OBSERVATION_FIELD_COUNT)
        obs, reward, done, info = adapter.step([1, 1, 1, 1, 0])
        self.assertEqual(len(obs), OBSERVATION_FIELD_COUNT)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertIsInstance(info, dict)
        adapter.close()
        self.assertEqual(len(DummyAdapter.observation_spec()), len(OBSERVATION_SPEC))
        self.assertEqual(len(DummyAdapter.action_spec()), len(ACTION_SPEC))


if __name__ == "__main__":
    unittest.main()

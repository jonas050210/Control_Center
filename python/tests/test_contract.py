"""Consistency checks for the Observation/Action contract description.

These tests validate the Python-side contract module used as the target for
a future external (e.g. Roblox) adapter. They intentionally do not require a
running Godot process; keeping OBSERVATION_SPEC in sync with
scripts/core/observation.gd is a manual responsibility documented in
contract.py's module docstring. The GodotSourceDriftTests below narrow that
gap statically: they parse the GDScript sources and fail loudly when the
Godot-side constants/field layout no longer match the Python contract.
"""
import re
import unittest
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


class GodotSourceDriftTests(unittest.TestCase):
    """Static cross-language drift guards for the observation/action contract.

    The contract is implemented twice (GDScript + Python) and no build step
    generates one from the other. These tests parse the Godot sources and
    compare the declared constants and the to_array() field layout against
    the Python contract, so an accidental change on either side fails here
    instead of silently producing incompatible checkpoints.
    """

    def _godot_source(self, relative: str) -> str:
        path = PROJECT_ROOT / relative
        self.assertTrue(path.is_file(), f"missing Godot source: {path}")
        return path.read_text(encoding="utf-8")

    def test_godot_observation_field_count_matches_python_contract(self):
        source = self._godot_source("scripts/core/observation.gd")
        match = re.search(r"const FIELD_COUNT:\s*int\s*=\s*(\d+)", source)
        self.assertIsNotNone(match, "Observation.FIELD_COUNT declaration not found")
        self.assertEqual(
            int(match.group(1)),
            OBSERVATION_FIELD_COUNT,
            "Observation.FIELD_COUNT in scripts/core/observation.gd no longer matches "
            "OBSERVATION_FIELD_COUNT in python/sandboxai/contract.py",
        )

    def test_godot_to_array_assigns_every_contract_index_exactly_once(self):
        source = self._godot_source("scripts/core/observation.gd")
        start = source.find("func to_array()")
        end = source.find("\nfunc ", start + 1)
        self.assertGreater(start, -1, "to_array() not found in observation.gd")
        body = source[start:end if end != -1 else len(source)]
        indices = [int(value) for value in re.findall(r"arr\[(\d+)\]\s*=", body)]
        self.assertEqual(
            sorted(indices),
            list(range(OBSERVATION_FIELD_COUNT)),
            "to_array() must assign exactly indices 0..N-1 with no gaps or duplicates; "
            "update the field table in docs/OBSERVATION_ACTION_CONTRACT.md and "
            "python/sandboxai/contract.py together with scripts/core/observation.gd",
        )
        self.assertIn(
            "arr.resize(FIELD_COUNT)",
            body,
            "to_array() must pre-allocate the packed array to FIELD_COUNT",
        )

    def test_godot_action_nvec_matches_python_contract(self):
        source = self._godot_source("scripts/core/action.gd")
        match = re.search(
            r"const MULTI_DISCRETE_NVECS:\s*Array\s*=\s*\[([0-9,\s]+)\]", source
        )
        self.assertIsNotNone(match, "Action.MULTI_DISCRETE_NVECS declaration not found")
        self.assertEqual(
            tuple(int(value) for value in match.group(1).split(",")),
            ACTION_NVEC,
            "Action.MULTI_DISCRETE_NVECS in scripts/core/action.gd no longer matches "
            "ACTION_NVEC in python/sandboxai/contract.py",
        )

    def test_godot_tracked_enemy_budget_matches_python_contract(self):
        source = self._godot_source("scripts/core/sandbox_config.gd")
        match = re.search(r"const OBSERVATION_MAX_TRACKED_ENEMIES:\s*int\s*=\s*(\d+)", source)
        self.assertIsNotNone(
            match, "SandboxConfig.OBSERVATION_MAX_TRACKED_ENEMIES declaration not found"
        )
        self.assertEqual(int(match.group(1)), OBSERVATION_MAX_TRACKED_ENEMIES)


if __name__ == "__main__":
    unittest.main()

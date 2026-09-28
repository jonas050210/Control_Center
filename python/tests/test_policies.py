"""Tests for the multi-policy architecture (Phase 5)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT, OBSERVATION_INDEX
from sandboxai.policies import (
    PolicyError,
    PolicyHandle,
    PolicyRole,
    PolicyRoster,
    PolicySpec,
    ScriptedBaseline,
    SlotAssignment,
    assert_independent_weights,
    matchup,
    standard_matchups,
    weights_differ,
)


class FakeParameter(list):
    """Stands in for a torch tensor: a list with .flatten()/.tolist()."""

    def flatten(self):
        return self

    def tolist(self):
        return list(self)


class FakeModel:
    """Duck-typed SB3 model with its own parameter objects."""

    def __init__(self, values, action=None):
        self._parameters = [FakeParameter(values)]
        self._action = action or [1, 1, 1, 1, 0, 0]
        self.predict_calls = 0

    def parameters(self):
        return list(self._parameters)

    def predict(self, observation, deterministic: bool = True):
        self.predict_calls += 1
        return list(self._action), None


def obs(**fields: float) -> list[float]:
    vector = [0.0] * OBSERVATION_FIELD_COUNT
    for name, value in fields.items():
        index, _width = OBSERVATION_INDEX[name]
        vector[index] = float(value)
    return vector


class PolicySpecTests(unittest.TestCase):
    def test_checkpoint_policy_requires_a_checkpoint(self):
        with self.assertRaises(ValueError):
            PolicySpec(policy_id="brain_a")

    def test_scripted_policy_needs_no_checkpoint(self):
        spec = PolicySpec(policy_id="base", kind="scripted", role=PolicyRole.BASELINE)
        self.assertFalse(spec.trainable)

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError):
            PolicySpec(policy_id="x", checkpoint="a.zip", role="aggressive")

    def test_only_a_learner_checkpoint_is_trainable(self):
        self.assertTrue(
            PolicySpec(policy_id="a", checkpoint="a.zip", role=PolicyRole.LEARNER).trainable
        )
        self.assertFalse(
            PolicySpec(policy_id="b", checkpoint="b.zip", role=PolicyRole.FROZEN).trainable
        )


class RosterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roster = PolicyRoster()
        self.roster.add(PolicySpec("brain_a", "runs/a/latest.zip", PolicyRole.LEARNER))
        self.roster.add(PolicySpec("brain_b", "runs/b/latest.zip", PolicyRole.FROZEN))
        self.roster.add(PolicySpec("brain_c", "runs/c/latest.zip", PolicyRole.EXTERNAL))
        self.roster.add_baseline()

    def test_roster_holds_three_independent_brains_plus_a_baseline(self):
        self.assertEqual(len(self.roster), 4)
        self.assertEqual(self.roster.ids(PolicyRole.LEARNER), ["brain_a"])
        self.assertEqual(self.roster.ids(PolicyRole.FROZEN), ["brain_b"])
        self.assertEqual(self.roster.ids(PolicyRole.EXTERNAL), ["brain_c"])
        self.assertEqual(self.roster.ids(PolicyRole.BASELINE), ["scripted_baseline"])

    def test_duplicate_policy_id_is_rejected(self):
        with self.assertRaises(PolicyError):
            self.roster.add(PolicySpec("brain_a", "runs/other.zip"))

    def test_two_policies_may_not_share_a_checkpoint(self):
        with self.assertRaises(PolicyError) as context:
            self.roster.add(PolicySpec("brain_d", "runs/a/latest.zip"))
        self.assertIn("independent weights", str(context.exception))

    def test_roster_round_trips_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.roster.save(Path(tmp) / "roster.json")
            reloaded = PolicyRoster.load(path)
        self.assertEqual(reloaded.ids(), self.roster.ids())
        self.assertEqual(reloaded.get("brain_b").spec.role, PolicyRole.FROZEN)

    def test_unknown_policy_lookup_raises(self):
        with self.assertRaises(KeyError):
            self.roster.get("brain_z")


class LoadingTests(unittest.TestCase):
    def test_each_handle_loads_its_own_model_object(self):
        roster = PolicyRoster()
        roster.add(PolicySpec("brain_a", "runs/a.zip", PolicyRole.LEARNER))
        roster.add(PolicySpec("brain_b", "runs/b.zip", PolicyRole.FROZEN))
        created: list[str] = []

        def loader(checkpoint: str, device: str):
            created.append(checkpoint)
            return FakeModel([float(len(created))])

        roster.load_all("cpu", loader)
        self.assertEqual(created, ["runs/a.zip", "runs/b.zip"])
        self.assertIsNot(roster.get("brain_a").model, roster.get("brain_b").model)

    def test_scripted_policy_loads_the_baseline(self):
        handle = PolicyHandle(PolicySpec("base", kind="scripted", role=PolicyRole.BASELINE))
        model = handle.load()
        self.assertIsInstance(model, ScriptedBaseline)

    def test_predicting_before_loading_raises(self):
        handle = PolicyHandle(PolicySpec("brain_a", "a.zip"))
        with self.assertRaises(PolicyError):
            handle.predict([0.0] * OBSERVATION_FIELD_COUNT)

    def test_frozen_policies_always_predict_deterministically(self):
        class RecordingModel(FakeModel):
            def __init__(self):
                super().__init__([1.0])
                self.deterministic_flags: list[bool] = []

            def predict(self, observation, deterministic: bool = True):
                self.deterministic_flags.append(deterministic)
                return [1, 1, 1, 1, 0, 0], None

        frozen = PolicyHandle(PolicySpec("frozen", "f.zip", PolicyRole.FROZEN), RecordingModel())
        learner = PolicyHandle(PolicySpec("learner", "l.zip", PolicyRole.LEARNER), RecordingModel())
        frozen.predict([0.0] * OBSERVATION_FIELD_COUNT, deterministic=False)
        learner.predict([0.0] * OBSERVATION_FIELD_COUNT, deterministic=False)
        self.assertEqual(frozen.model.deterministic_flags, [True])
        self.assertEqual(learner.model.deterministic_flags, [False])


class IndependenceTests(unittest.TestCase):
    def test_distinct_models_are_independent(self):
        a = PolicyHandle(PolicySpec("brain_a", "a.zip"), FakeModel([1.0, 2.0]))
        b = PolicyHandle(PolicySpec("brain_b", "b.zip"), FakeModel([1.0, 2.0]))
        report = assert_independent_weights([a, b])
        self.assertTrue(report["independent"])
        self.assertEqual(report["parameters_checked"], 2)

    def test_shared_parameter_tensor_is_caught(self):
        shared = FakeParameter([1.0])
        model_a = FakeModel([0.0])
        model_b = FakeModel([0.0])
        model_a._parameters = [shared]
        model_b._parameters = [shared]
        a = PolicyHandle(PolicySpec("brain_a", "a.zip"), model_a)
        b = PolicyHandle(PolicySpec("brain_b", "b.zip"), model_b)
        with self.assertRaises(PolicyError) as context:
            assert_independent_weights([a, b])
        self.assertIn("same brain", str(context.exception))

    def test_shared_checkpoint_is_caught_even_before_loading(self):
        a = PolicyHandle(PolicySpec("brain_a", "runs/x.zip"))
        b = PolicyHandle(PolicySpec("brain_b", "runs/x.zip"))
        with self.assertRaises(PolicyError):
            assert_independent_weights([a, b])

    def test_duplicate_ids_are_caught(self):
        a = PolicyHandle(PolicySpec("brain_a", "a.zip"))
        b = PolicyHandle(PolicySpec("brain_a", "b.zip"))
        with self.assertRaises(PolicyError):
            assert_independent_weights([a, b])

    def test_weights_differ_detects_value_differences(self):
        self.assertTrue(weights_differ(FakeModel([1.0, 2.0]), FakeModel([1.0, 2.5])))
        self.assertFalse(weights_differ(FakeModel([1.0, 2.0]), FakeModel([1.0, 2.0])))

    def test_a_snapshot_is_independent_even_with_equal_values(self):
        a = PolicyHandle(PolicySpec("brain_a", "a.zip"), FakeModel([1.0]))
        b = PolicyHandle(PolicySpec("brain_a@1000", "a_1000.zip"), FakeModel([1.0]))
        assert_independent_weights([a, b])  # identity, not value
        self.assertFalse(weights_differ(a.model, b.model))


class SlotAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roster = PolicyRoster()
        self.roster.add(PolicySpec("brain_a", "a.zip", PolicyRole.LEARNER))
        self.roster.add(PolicySpec("brain_b", "b.zip", PolicyRole.FROZEN))
        self.roster.add(PolicySpec("brain_c", "c.zip", PolicyRole.FROZEN))

    def test_a_vs_b(self):
        assignment = matchup("brain_a", "brain_b")
        assignment.validate(self.roster)
        self.assertEqual(assignment.policy_for(0), "brain_a")
        self.assertEqual(assignment.policy_for(1), "brain_b")
        self.assertEqual(assignment.label, "brain_a vs brain_b")

    def test_a_vs_b_vs_a_is_one_brain_in_two_seats(self):
        assignment = matchup("brain_a", "brain_b", "brain_a")
        self.assertEqual(assignment.distinct_policies, ["brain_a", "brain_b"])
        self.assertEqual(assignment.slots_for("brain_a"), [0, 2])

    def test_a_vs_b_vs_c(self):
        assignment = matchup("brain_a", "brain_b", "brain_c")
        assignment.validate(self.roster)
        self.assertEqual(len(assignment.distinct_policies), 3)

    def test_standard_matchups(self):
        matchups = standard_matchups("brain_a", "brain_b", "brain_c")
        self.assertEqual([len(item) for item in matchups], [2, 3, 3])
        self.assertEqual(matchups[1].slots, ["brain_a", "brain_b", "brain_a"])

    def test_unknown_policy_in_an_assignment_is_rejected(self):
        with self.assertRaises(PolicyError):
            matchup("brain_a", "ghost").validate(self.roster)

    def test_slot_index_is_bounds_checked(self):
        with self.assertRaises(IndexError):
            matchup("brain_a", "brain_b").policy_for(5)

    def test_empty_assignment_is_rejected(self):
        with self.assertRaises(ValueError):
            SlotAssignment([])


class ScriptedBaselineTests(unittest.TestCase):
    def test_actions_are_inside_the_action_space(self):
        baseline = ScriptedBaseline()
        for bearing in (-0.9, -0.01, 0.0, 0.02, 0.8):
            action, _ = baseline.predict(
                obs(primary_enemy_visible=1.0, primary_enemy_bearing_norm=bearing, weapon_ready=1.0)
            )
            self.assertEqual(len(action), len(ACTION_NVEC))
            for index, value in enumerate(action):
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, ACTION_NVEC[index])

    def test_it_only_shoots_a_visible_centred_target(self):
        baseline = ScriptedBaseline()
        centred, _ = baseline.predict(
            obs(primary_enemy_visible=1.0, primary_enemy_bearing_norm=0.0, weapon_ready=1.0)
        )
        off_centre, _ = baseline.predict(
            obs(primary_enemy_visible=1.0, primary_enemy_bearing_norm=0.6, weapon_ready=1.0)
        )
        invisible, _ = baseline.predict(obs(primary_enemy_visible=0.0, weapon_ready=1.0))
        self.assertEqual(centred[4], 1)
        self.assertEqual(off_centre[4], 0)
        self.assertEqual(invisible[4], 0)

    def test_it_turns_toward_sound_when_nothing_is_visible(self):
        baseline = ScriptedBaseline()
        action, _ = baseline.predict(obs(last_sound_bearing_norm=0.6, last_sound_loudness=0.8))
        self.assertEqual(action[2], 2)
        action, _ = baseline.predict(obs(last_sound_bearing_norm=-0.6, last_sound_loudness=0.8))
        self.assertEqual(action[2], 0)

    def test_it_is_deterministic_for_the_same_observation(self):
        baseline = ScriptedBaseline(seed=7)
        observation = obs(primary_enemy_visible=1.0, primary_enemy_bearing_norm=0.3)
        first, _ = baseline.predict(observation)
        second, _ = baseline.predict(observation)
        self.assertEqual(first, second)

    def test_batched_observations_produce_one_action_each(self):
        baseline = ScriptedBaseline()
        batch = [obs(primary_enemy_visible=1.0), obs()]
        actions, _ = baseline.predict(batch)
        self.assertEqual(len(actions), 2)


if __name__ == "__main__":
    unittest.main()

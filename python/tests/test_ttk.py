"""Manual TTK trial schema, validation, statistics and holdout splitting."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sandboxai.cli import main
from sandboxai.ttk import (
    TTKDataset,
    TTKTrial,
    bootstrap_interval,
    compare_with_simulator,
    format_summary,
    interquartile_mean,
    summarize_values,
    trials_from_iterable,
    validate_trial,
)


def trial(**overrides):
    values = {
        "trial_id": "t1",
        "source": "manual_observation",
        "weapon_profile": "rifle",
        "distance_m": 8.0,
        "target_health": 100.0,
        "outcome": "kill",
        "acquisition_time": 1.0,
        "first_trigger_time": 1.25,
        "first_damage_time": 1.30,
        "lethal_time": 2.80,
        "shots_fired": 6,
        "shots_hit": 4,
        "movement_state": "strafing",
        "hit_zone": "body",
        "tester": "tester-a",
        "annotator": "annotator-a",
        "consent": True,
        "frame_rate": 60.0,
    }
    values.update(overrides)
    return values


class ValidationTests(unittest.TestCase):
    def test_valid_trial_passes(self):
        self.assertEqual(validate_trial(trial()), [])

    def test_consent_is_mandatory(self):
        problems = validate_trial(trial(consent=False))
        self.assertTrue(any("consent" in item for item in problems))

    def test_non_perceivable_fields_are_rejected(self):
        for field in ("server_authoritative_position", "memory_read_health", "packet_timestamp"):
            with self.subTest(field=field):
                problems = validate_trial(trial(**{field: 1.0}))
                self.assertTrue(any("player-visible" in item for item in problems))

    def test_times_must_be_ordered(self):
        problems = validate_trial(trial(first_damage_time=0.5))
        self.assertTrue(any("non-decreasing" in item for item in problems))

    def test_kill_requires_a_lethal_time_and_censored_trials_must_not_have_one(self):
        self.assertTrue(any("lethal_time" in p for p in validate_trial(trial(lethal_time=None))))
        censored = trial(outcome="target_escaped")
        self.assertTrue(any("lethal_time" in p for p in validate_trial(censored)))

    def test_shots_hit_cannot_exceed_shots_fired(self):
        self.assertTrue(any("shots_hit" in p for p in validate_trial(trial(shots_hit=9))))

    def test_unknown_outcome_and_source_are_rejected(self):
        self.assertTrue(validate_trial(trial(outcome="vibes")))
        self.assertTrue(validate_trial(trial(source="private_api")))

    def test_trials_from_iterable_raises_on_any_invalid_row(self):
        with self.assertRaises(ValueError):
            trials_from_iterable([trial(), trial(consent=False)])


class DerivedValueTests(unittest.TestCase):
    def test_derived_intervals(self):
        item = TTKTrial.from_dict(trial())
        self.assertAlmostEqual(item.reaction_time, 0.25)
        self.assertAlmostEqual(item.trigger_to_kill, 1.55)
        self.assertAlmostEqual(item.damage_to_kill, 1.50)
        self.assertAlmostEqual(item.encounter_time, 1.80)
        self.assertAlmostEqual(item.accuracy, 4 / 6)
        self.assertFalse(item.censored)

    def test_condition_key_buckets_distance(self):
        near = TTKTrial.from_dict(trial(distance_m=2.0))
        far = TTKTrial.from_dict(trial(distance_m=8.0))
        self.assertNotEqual(near.condition_key(3.0), far.condition_key(3.0))
        self.assertEqual(near.condition_key(3.0), TTKTrial.from_dict(trial(distance_m=1.0)).condition_key(3.0))


class StatisticsTests(unittest.TestCase):
    def test_iqm_ignores_extremes_without_dropping_them_from_the_data(self):
        values = [1.0, 1.1, 1.2, 1.3, 50.0]
        self.assertLess(interquartile_mean(values), 2.0)
        self.assertEqual(summarize_values(values)["n"], 5)
        self.assertEqual(summarize_values(values)["max"], 50.0)

    def test_bootstrap_is_deterministic_for_a_seed(self):
        values = [1.0, 1.4, 0.9, 1.2, 1.6, 1.1]
        self.assertEqual(bootstrap_interval(values, seed=5), bootstrap_interval(values, seed=5))
        self.assertNotEqual(bootstrap_interval(values, seed=5), bootstrap_interval(values, seed=6))

    def test_single_value_has_no_interval(self):
        result = summarize_values([1.0])
        self.assertEqual(result["n"], 1)
        self.assertNotEqual(result["bootstrap_ci95_low"], result["bootstrap_ci95_low"])  # NaN


class DatasetTests(unittest.TestCase):
    def make_dataset(self):
        rows = []
        for index in range(8):
            rows.append(
                TTKTrial.from_dict(
                    trial(
                        trial_id=f"t{index}",
                        tester=f"tester-{index % 4}",
                        lethal_time=2.5 + 0.1 * index,
                        distance_m=4.0 + index,
                    )
                )
            )
        rows.append(TTKTrial.from_dict(trial(trial_id="censored", outcome="target_escaped", lethal_time=None, tester="tester-1")))
        return TTKDataset(rows, {"study": "unit-test"})

    def test_censoring_is_counted_not_dropped(self):
        dataset = self.make_dataset()
        censoring = dataset.censoring_report()
        self.assertEqual(censoring["trials"], 9)
        self.assertEqual(censoring["censored"], 1)
        # The censored trial contributes to accuracy/reaction but not TTK.
        summary = dataset.summary()
        self.assertEqual(summary["trigger_to_kill"]["n"], 8)
        self.assertEqual(summary["reaction_time"]["n"], 9)

    def test_round_trip_through_jsonl(self):
        dataset = self.make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            dataset.save(path)
            loaded = TTKDataset.load(path)
            self.assertEqual(len(loaded.trials), len(dataset.trials))
            self.assertEqual(loaded.metadata["study"], "unit-test")

    def test_loading_rejects_an_invalid_file_as_a_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps(trial()) + "\n" + json.dumps(trial(trial_id="t2", consent=False)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                TTKDataset.load(path)
            # Non-strict keeps the good rows and reports the rest.
            lenient = TTKDataset.load(path, strict=False)
            self.assertEqual(len(lenient.trials), 1)
            self.assertTrue(lenient.rejected)

    def test_holdout_split_separates_testers(self):
        dataset = self.make_dataset()
        calibration, holdout = dataset.holdout_split(0.25, seed=3)
        calibration_testers = {t.tester for t in calibration.trials}
        holdout_testers = {t.tester for t in holdout.trials}
        self.assertFalse(calibration_testers & holdout_testers)
        self.assertTrue(holdout_testers)

    def test_holdout_split_requires_two_groups(self):
        single = TTKDataset([TTKTrial.from_dict(trial())], {})
        with self.assertRaises(ValueError):
            single.holdout_split()

    def test_by_condition_reports_thin_cells(self):
        report = self.make_dataset().by_condition(3.0)
        self.assertTrue(report)
        for cell in report.values():
            self.assertIn("trials", cell)
            self.assertIn("trigger_to_kill", cell)

    def test_format_summary_is_printable(self):
        text = format_summary(self.make_dataset().summary())
        self.assertIn("trigger_to_kill", text)


class SimulatorComparisonTests(unittest.TestCase):
    def test_comparison_uses_the_parsed_weapon_profiles(self):
        dataset = TTKDataset([TTKTrial.from_dict(trial(weapon_profile="rifle"))], {})
        report = compare_with_simulator(dataset)
        rifle = report["weapons"]["rifle"]
        self.assertIsNotNone(rifle["simulator"])
        self.assertIn("ideal_ttk", rifle["simulator"])
        self.assertEqual(report["unknown_weapon_profiles"], [])

    def test_unknown_weapon_is_reported_not_skipped(self):
        dataset = TTKDataset([TTKTrial.from_dict(trial(weapon_profile="railgun"))], {})
        report = compare_with_simulator(dataset)
        self.assertEqual(report["unknown_weapon_profiles"], ["railgun"])


class CliTests(unittest.TestCase):
    def test_ttk_report_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            TTKDataset(
                [TTKTrial.from_dict(trial(trial_id=f"t{i}", tester=f"tester-{i%2}")) for i in range(4)],
                {},
            ).save(path)
            self.assertEqual(
                main(["ttk-report", "--trials", str(path), "--json", "--compare-simulator"]), 0
            )
            self.assertEqual(main(["ttk-report", "--trials", str(path), "--by-condition"]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

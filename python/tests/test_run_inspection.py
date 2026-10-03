"""Read-only run inspection: discovery, tolerance and honest reporting."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sandboxai.run_inspection import (
    RUN_INDEX_FORMAT,
    RUN_REPORT_FORMAT,
    discover_run_directories,
    format_run_index,
    format_run_report,
    inspect_run,
    inspect_runs,
    is_run_directory,
    read_json,
    tail_jsonl,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _make_run(
    root: Path,
    name: str,
    *,
    finished: bool = True,
    manifest: bool = True,
    checkpoints: tuple[str, ...] = ("latest.zip", "best_eval.zip"),
    evaluations: int = 2,
    dirty: bool = False,
) -> Path:
    run = root / "runs" / name
    _write(
        run / "config.json",
        {
            "run_id": name,
            "seed": 1234,
            "environment_count": 8,
            "env_workers": 2,
            "total_training_steps": 1000,
            "curriculum_mode": "auto",
            "checkpoint_selection_metric": "mean_episode_reward",
        },
    )
    if manifest:
        _write(
            run / "run_manifest.json",
            {
                "format": "sandboxai.run_manifest/v2",
                "run_id": name,
                "experiment_id": "exp",
                "seed": 1234,
                "contract": {"observation_dim": 126, "action_nvec": [3, 3, 3, 3, 2, 2]},
                "code": {"commit": "abc123def456", "branch": "main", "dirty": dirty},
                "host": {"python": "3.11.2", "system": "Linux", "logical_cpus": 12},
                "godot": {"version": "4.7.2.stable"},
                "checkpoint_selection": {
                    "metric": "mean_episode_reward",
                    "goal": "max",
                    "min_delta": 0.0,
                },
            },
        )
    for checkpoint in checkpoints:
        (run / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run / "checkpoints" / checkpoint).write_bytes(b"zip")
    for index in range(evaluations):
        (run / "evaluations" / f"step_{index:09d}").mkdir(parents=True, exist_ok=True)
    if evaluations:
        _write(
            run / "evaluations" / "latest.json",
            {"timesteps": 500, "mean_episode_reward": 3.5, "win_rate": 0.5},
        )
        _write(
            run / "evaluations" / "best.json",
            {"mean_reward": 3.5, "score": 3.5, "timesteps": 500},
        )
    _write(run / "logs" / "training.jsonl", '{"event":"training_start"}\n{"event":"rollout"}\n')
    if finished:
        _write(run / "run_summary.json", {"timesteps": 1000, "stopped": False, "device": "cpu"})
    return run


class DiscoveryTests(unittest.TestCase):
    def test_discovers_runs_under_output_root_and_runs_dir(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, "20260101-000000")
            _make_run(root, "20260102-000000")
            self.assertEqual(len(discover_run_directories(root)), 2)
            self.assertEqual(len(discover_run_directories(root / "runs")), 2)

    def test_a_single_run_directory_is_accepted_directly(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "solo")
            self.assertTrue(is_run_directory(run))
            self.assertEqual(discover_run_directories(run), [run])

    def test_non_run_directories_are_ignored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, "real")
            (root / "runs" / "not_a_run").mkdir(parents=True)
            (root / "runs" / "not_a_run" / "notes.txt").write_text("hi", encoding="utf-8")
            self.assertEqual([path.name for path in discover_run_directories(root)], ["real"])

    def test_missing_root_is_an_empty_index_not_an_error(self):
        index = inspect_runs("/definitely/not/here")
        self.assertEqual(index["format"], RUN_INDEX_FORMAT)
        self.assertEqual(index["run_count"], 0)
        self.assertIn("(none)", format_run_index(index))


class ReportTests(unittest.TestCase):
    def test_finished_run_reports_state_progress_and_inventory(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "finished")
            report = inspect_run(run)
        self.assertEqual(report["format"], RUN_REPORT_FORMAT)
        self.assertEqual(report["status"]["state"], "finished")
        self.assertEqual(report["status"]["source"], "run_summary.json")
        self.assertEqual(report["progress"]["timesteps"], 1000)
        self.assertEqual(report["progress"]["target_timesteps"], 1000)
        self.assertAlmostEqual(report["progress"]["fraction"], 1.0)
        self.assertEqual(report["checkpoints"]["count"], 2)
        self.assertTrue(report["checkpoints"]["has_best"])
        self.assertEqual(report["evaluation"]["evaluation_count"], 2)
        self.assertEqual(report["manifest"]["godot"]["version"], "4.7.2.stable")
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["problems"], [])

    def test_unfinished_run_is_reported_as_incomplete_not_guessed(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "killed", finished=False)
            report = inspect_run(run)
        self.assertEqual(report["status"]["state"], "incomplete")
        self.assertIn(
            "run has no run_summary.json; it did not reach a clean end", report["warnings"]
        )
        # Progress still readable from the evaluation trail.
        self.assertEqual(report["progress"]["timesteps"], 500)
        self.assertEqual(report["progress"]["source"], "evaluations/latest.json")

    def test_control_status_supplies_the_state_for_a_live_run(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "live", finished=False)
            _write(run / "status.json", {"state": "Running", "pid": 4242, "timesteps": 750})
            report = inspect_run(run)
        self.assertEqual(report["status"]["state"], "running")
        self.assertEqual(report["status"]["source"], "status.json")
        self.assertEqual(report["control"]["pid"], 4242)
        self.assertEqual(report["progress"]["timesteps"], 750)

    def test_dirty_tree_is_surfaced_as_a_warning(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "dirty", dirty=True)
            report = inspect_run(run)
        self.assertTrue(any("dirty" in warning for warning in report["warnings"]))

    def test_contract_mismatch_is_surfaced(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "oldcontract")
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            manifest["contract"]["observation_dim"] = 64
            _write(run / "run_manifest.json", manifest)
            report = inspect_run(run)
        self.assertTrue(any("64-float observation" in w for w in report["warnings"]))

    def test_missing_checkpoints_and_manifest_are_warnings(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "bare", manifest=False, checkpoints=(), evaluations=0)
            report = inspect_run(run)
        self.assertIn("no checkpoints were written", report["warnings"])
        self.assertTrue(any("no run_manifest.json" in w for w in report["warnings"]))
        self.assertFalse(report["artifacts"]["run_manifest.json"])

    def test_corrupt_json_is_a_problem_not_an_exception(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "corrupt")
            _write(run / "run_summary.json", "{not json")
            report = inspect_run(run)
        self.assertTrue(any("run_summary.json" in problem for problem in report["problems"]))
        # Still a usable report.
        self.assertEqual(report["checkpoints"]["count"], 2)

    def test_missing_directory_reports_instead_of_raising(self):
        report = inspect_run("/definitely/not/a/run")
        self.assertFalse(report["exists"])
        self.assertIn("run directory does not exist", report["problems"])

    def test_inspection_never_writes_to_the_run_directory(self):
        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "readonly")
            before = {
                str(path.relative_to(run)): path.stat().st_mtime
                for path in run.rglob("*")
                if path.is_file()
            }
            inspect_run(run, event_limit=5)
            inspect_runs(Path(tmp), event_limit=5)
            after = {
                str(path.relative_to(run)): path.stat().st_mtime
                for path in run.rglob("*")
                if path.is_file()
            }
        self.assertEqual(before, after, "inspection must not touch the run directory")

    def test_report_and_index_render_as_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, "20260101-000000")
            _make_run(root, "20260102-000000", finished=False, dirty=True)
            index = inspect_runs(root)
            text = format_run_index(index)
            detail = format_run_report(index["runs"][0])
        self.assertIn("20260101-000000", text)
        self.assertIn("Warnings:", text)
        self.assertIn("checkpoints", detail)
        self.assertIn("4.7.2.stable", detail)

    def test_limit_keeps_the_newest_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(4):
                _make_run(root, f"2026010{index}-000000")
            index = inspect_runs(root, limit=2)
        self.assertEqual(index["run_count"], 2)
        self.assertEqual(index["runs"][-1]["run_id"], "20260103-000000")

    def test_stamp_cache_avoids_reparsing_unchanged_run_and_invalidates_on_change(self):
        from sandboxai import run_inspection

        with TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "cached-run", finished=False)
            first = inspect_run(run)
            self.assertEqual(first["status"]["state"], "incomplete")
            second = inspect_run(run)
            self.assertEqual(second["run_id"], "cached-run")
            # Mutating the returned dict must not corrupt the cached entry.
            second["status"]["state"] = "mutated"
            self.assertEqual(inspect_run(run)["status"]["state"], "incomplete")
            # Writing run_summary.json changes the directory/file stamp and invalidates cache.
            _write(run / "run_summary.json", {"timesteps": 1000, "stopped": False, "device": "cpu"})
            updated = inspect_run(run)
            self.assertEqual(updated["status"]["state"], "finished")
            ckpt_only = run_inspection.inspect_run_checkpoints(run)
            self.assertEqual(ckpt_only["run_id"], "cached-run")
            self.assertEqual(ckpt_only["checkpoints"]["count"], 2)


class LineCountTests(unittest.TestCase):
    """Log line counts are memoised, and only extended for grown files."""

    def setUp(self):
        from sandboxai import run_inspection

        run_inspection._LINE_COUNTS.clear()

    def test_a_full_count_is_the_reference(self):
        from sandboxai.run_inspection import _count_lines, _count_lines_cached

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            path.write_text("a\nb\nc\n", encoding="utf-8")
            self.assertEqual(_count_lines(path), 3)
            self.assertEqual(_count_lines_cached(path), 3)
            # A second call must agree with the first, cache or not.
            self.assertEqual(_count_lines_cached(path), 3)

    def test_an_unchanged_file_is_not_read_again(self):
        from sandboxai.run_inspection import _count_lines_cached

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            path.write_text("a\nb\n", encoding="utf-8")
            self.assertEqual(_count_lines_cached(path), 2)

            reads = {"count": 0}
            real_open = Path.open

            def counting_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                reads["count"] += 1
                return real_open(self, *args, **kwargs)

            Path.open = counting_open  # type: ignore[method-assign]
            try:
                self.assertEqual(_count_lines_cached(path), 2)
                self.assertGreaterEqual(_count_lines_cached(path), 2)
            finally:
                Path.open = real_open  # type: ignore[method-assign]
        self.assertEqual(reads["count"], 0, "an unchanged log was read again")

    def test_an_appended_file_is_extended_not_recounted(self):
        from sandboxai import run_inspection
        from sandboxai.run_inspection import _count_lines, _count_lines_cached

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            path.write_text("a\nb\n", encoding="utf-8")
            self.assertEqual(_count_lines_cached(path), 2)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("c\nd\n")
            self.assertEqual(_count_lines_cached(path), 4)
            self.assertEqual(_count_lines_cached(path), _count_lines(path))
            # A trailing partial line is counted once, not twice.
            with path.open("a", encoding="utf-8") as handle:
                handle.write("e")
            self.assertEqual(_count_lines_cached(path), 5)
            self.assertEqual(_count_lines_cached(path), _count_lines(path))
            # Completing that partial line does not add another line.
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            self.assertEqual(_count_lines_cached(path), 5)
            self.assertEqual(_count_lines_cached(path), _count_lines(path))
            self.assertIsNotNone(run_inspection._LINE_COUNTS)

    def test_a_shrunk_file_is_recounted_from_zero(self):
        from sandboxai.run_inspection import _count_lines, _count_lines_cached

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            path.write_text("a\nb\nc\nd\n", encoding="utf-8")
            self.assertEqual(_count_lines_cached(path), 4)
            # A new run reusing the path: the old offset is not a safe place
            # to resume from, so the count starts over (the same rule
            # IncrementalJsonlTailer uses).
            path.write_text("x\n", encoding="utf-8")
            self.assertEqual(_count_lines_cached(path), 1)
            self.assertEqual(_count_lines_cached(path), _count_lines(path))

    def test_the_count_is_bounded(self):
        from sandboxai import run_inspection

        original = run_inspection._LINE_COUNTS_LIMIT
        run_inspection._LINE_COUNTS_LIMIT = 3
        try:
            with TemporaryDirectory() as tmp:
                for index in range(5):
                    path = Path(tmp) / f"log-{index}.jsonl"
                    path.write_text("a\n", encoding="utf-8")
                    run_inspection._count_lines_cached(path)
            self.assertLessEqual(len(run_inspection._LINE_COUNTS), 3)
        finally:
            run_inspection._LINE_COUNTS_LIMIT = original


class TailTests(unittest.TestCase):
    def test_tail_returns_the_last_objects(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "".join(json.dumps({"event": index}) + "\n" for index in range(100)),
                encoding="utf-8",
            )
            rows = tail_jsonl(path, 3)
        self.assertEqual([row["event"] for row in rows], [97, 98, 99])

    def test_tail_skips_a_half_written_final_line(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"event": 1}\n{"event": 2}\n{"eve', encoding="utf-8")
            rows = tail_jsonl(path, 5)
        self.assertEqual([row["event"] for row in rows], [1, 2])

    def test_tail_of_a_missing_file_is_empty(self):
        self.assertEqual(tail_jsonl(Path("/nope/events.jsonl"), 5), [])

    def test_tail_bounds_the_bytes_it_reads(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(60_000):
                    handle.write(json.dumps({"event": index, "pad": "x" * 40}) + "\n")
            self.assertGreater(path.stat().st_size, 1 << 20)
            rows = tail_jsonl(path, 2)
        self.assertEqual([row["event"] for row in rows], [59_998, 59_999])

    def test_tail_parses_only_the_rows_it_returns(self):
        """A megabyte window must not be decoded row by row to return five.

        The Dashboard tails the newest run's log on every poll tick; parsing
        every line in the trailing window made that the page's most expensive
        operation, so the window is walked from its end and stops as soon as
        the requested rows are in hand.
        """
        from sandboxai import run_inspection

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(20_000):
                    handle.write(json.dumps({"step": index, "reward": 0.5}) + "\n")

            parses = {"count": 0}
            real_loads = run_inspection.json.loads

            def counting_loads(payload, *args, **kwargs):
                parses["count"] += 1
                return real_loads(payload, *args, **kwargs)

            run_inspection.json.loads = counting_loads
            try:
                rows = tail_jsonl(path, 5)
            finally:
                run_inspection.json.loads = real_loads

        self.assertEqual([row["step"] for row in rows], [19_995, 19_996, 19_997, 19_998, 19_999])
        # Five rows plus the tolerance for blank/partial trailing lines; the
        # point is that it is not the ~20,000 lines of the window.
        self.assertLess(parses["count"], 20)

    def test_read_json_distinguishes_absent_from_broken(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.json"
            value, problem = read_json(missing)
            self.assertIsNone(value)
            self.assertIsNone(problem)
            broken = Path(tmp) / "broken.json"
            broken.write_text("{", encoding="utf-8")
            value, problem = read_json(broken)
            self.assertIsNone(value)
            self.assertIn("invalid JSON", problem)


class CliTests(unittest.TestCase):
    def test_cli_lists_and_details_runs(self):
        from sandboxai.cli import main

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root, "cli-run")
            self.assertEqual(main(["inspect-runs", "--root", str(root)]), 0)
            self.assertEqual(main(["inspect-runs", "--root", str(root), "--json"]), 0)
            self.assertEqual(main(["inspect-runs", "--run", str(run)]), 0)
            self.assertEqual(main(["inspect-runs", "--run", str(root / "nope")]), 1)


if __name__ == "__main__":
    unittest.main()

"""Analytics helpers: CSV reading, kill-only TTK statistics and heatmap payloads."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server import analytics as analytics_module

from server.analytics import (
    _kill_ttks,
    benchmark_payload,
    heatmap_payload,
    read_csv_rows,
    training_stats,
)

EVENT_FIELDS = [
    "timestamp", "episode", "map", "win", "draw", "killed", "ttk", "weapon",
    "opponent_weapon", "distance", "shots_fired", "bullets_fired", "hits",
    "headshots", "accuracy", "headshot_pct", "avg_kill_distance", "death_x",
    "death_y", "kill_x", "kill_y",
]
METRIC_FIELDS = [
    "timestamp", "steps", "fps", "episodes", "win_rate", "kill_rate", "avg_reward",
    "avg_ttk", "headshot_pct", "accuracy", "elapsed", "map",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def kill_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": "2026-01-01T00:00:00+00:00", "episode": 1, "map": "Dust",
        "win": 1, "draw": 0, "killed": 1, "ttk": 5.0, "weapon": "AK-47",
        "opponent_weapon": "Pistol", "distance": 12.0, "accuracy": 0.5,
        "headshot_pct": 0.2, "avg_kill_distance": 11.0,
        "death_x": "", "death_y": "", "kill_x": 3.0, "kill_y": -4.0,
    }
    row.update(overrides)
    return row


class KillTtkTests(unittest.TestCase):
    def test_only_confirmed_kills_count(self) -> None:
        rows = [
            {"ttk": "9.5", "win": "1", "killed": "1"},   # kill
            {"ttk": "60.0", "win": "1", "killed": "0"},  # win on the time limit, no kill
            {"ttk": "12.0", "win": "0", "killed": "0"},  # defeat
            {"ttk": "8.0", "win": "1", "draw": "0"},     # legacy row without "killed"
            {"ttk": "8.0", "win": "1", "draw": "1"},     # legacy draw – ignored
            {"ttk": "oops", "win": "1", "killed": "1"},  # unparsable, ignored
        ]
        self.assertEqual(_kill_ttks(rows), [9.5, 8.0])


class CsvReadingTests(unittest.TestCase):
    def test_partially_written_last_line_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            write_csv(path, ["ttk", "win"], [{"ttk": "1", "win": "1"}, {"ttk": "2", "win": "0"}])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("3")  # no newline, as while a training run is writing
            rows = read_csv_rows(path)
            self.assertEqual([row["ttk"] for row in rows], ["1", "2"])

    def test_tail_window_skips_old_rows_without_losing_the_header(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ttk", "win"])
                for index in range(5000):
                    writer.writerow([f"{index}.5", 1])
            with mock.patch.object(analytics_module, "TAIL_READ_BYTES", 2048):
                rows = read_csv_rows(path, limit=50)
            self.assertEqual(len(rows), 50)
            self.assertEqual(set(rows[0]), {"ttk", "win"})
            self.assertEqual(rows[-1]["ttk"], "4999.5")
            self.assertEqual([int(float(row["ttk"])) for row in rows],
                             list(range(4950, 5000)))

    def test_limit_keeps_the_newest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            write_csv(path, ["ttk"], [{"ttk": str(index)} for index in range(10)])
            self.assertEqual([row["ttk"] for row in read_csv_rows(path, limit=3)], ["7", "8", "9"])


class TrainingStatsTests(unittest.TestCase):
    def test_summary_uses_kill_ttk_and_reports_kill_count(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            write_csv(logs / "heatmap_events.csv", EVENT_FIELDS, [
                kill_row(ttk=4.0),
                kill_row(episode=2, killed=0, ttk=90.0),          # time-limit decision
                kill_row(episode=3, win=0, killed=0, ttk=6.0),     # defeat
            ])
            write_csv(logs / "training_metrics.csv", METRIC_FIELDS, [{
                "timestamp": "2026-01-01T00:00:00+00:00", "steps": 4096, "fps": 220,
                "episodes": 3, "win_rate": 0.66, "kill_rate": 0.33,
                "avg_reward": 1.2, "avg_ttk": 4.0,
                "headshot_pct": 0.3, "accuracy": 0.4, "elapsed": 18, "map": "Dust",
            }])
            payload = training_stats(logs)
            summary = payload["summary"]
            self.assertAlmostEqual(summary["avg_ttk"], 4.0, places=6)
            self.assertEqual(summary["kill_count"], 1)
            self.assertEqual(summary["logged_episodes"], 3)
            self.assertEqual(payload["histogram"]["ttk"], [4.0])
            self.assertEqual(payload["summary"]["episodes"], 3)
            self.assertEqual(payload["weapons"], {"AK-47": 3})

    def test_missing_files_produce_an_empty_but_valid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            payload = training_stats(Path(folder))
            self.assertEqual(payload["summary"]["episodes"], 0)
            self.assertEqual(payload["summary"]["kill_count"], 0)
            self.assertEqual(payload["histogram"]["ttk"], [])
            self.assertEqual(payload["series"]["steps"], [])
            self.assertIn("resources", payload)


class HeatmapPayloadTests(unittest.TestCase):
    def _write_events(self, logs: Path) -> None:
        rows = []
        for index in range(12):
            alive = index % 3 != 0
            rows.append(kill_row(
                episode=index + 1,
                weapon="Pistol" if index % 3 == 0 else "AK-47",
                opponent_weapon="Pistol" if index % 3 == 1 else "AK-47",
                win=1 if alive else 0,
                killed=1 if alive else 0,
                kill_x=-20 + index * 3 if alive else "",
                kill_y=5 + index if alive else "",
                death_x=10 if not alive else "",
                death_y=-10 if not alive else "",
                distance=8.0 + index,
            ))
        write_csv(logs / "heatmap_events.csv", EVENT_FIELDS, rows)

    def test_grid_shape_counts_and_filter_options(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            self._write_events(logs)
            payload = heatmap_payload("Dust", logs_dir=logs, bins=48)
            self.assertEqual(payload["map"], "Dust")
            self.assertEqual(len(payload["grid"]), 48)
            self.assertEqual(len(payload["grid"][0]), 48)
            self.assertEqual(payload["events"], 12)
            self.assertEqual(payload["kills"], 8)
            self.assertEqual(payload["deaths"], 4)
            self.assertIn("All weapons", payload["weapons"])
            self.assertGreaterEqual(len(payload["weapons"]), 3)  # "All weapons" + both guns
            self.assertEqual(payload["max_episode"], 12)
            self.assertIsNotNone(payload["hottest_kill"])

    def test_weapon_and_episode_filters_reduce_the_sample(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            self._write_events(logs)
            everything = heatmap_payload("Dust", logs_dir=logs, bins=24)
            filtered = heatmap_payload("Dust", logs_dir=logs, bins=24, weapon="Pistol")
            narrowed = heatmap_payload("Dust", logs_dir=logs, bins=24, episode_range=(1, 3))
            self.assertEqual(filtered["events"], 8)  # 4 own + 4 opponent Pistols
            self.assertEqual(narrowed["events"], 3)
            self.assertLess(filtered["events"], everything["events"])
            self.assertEqual(heatmap_payload("Warehouse", logs_dir=logs, bins=8)["events"], 0)

    def test_extra_events_are_merged_without_touching_the_log(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            self._write_events(logs)
            payload = heatmap_payload("Dust", logs_dir=logs, bins=16,
                                      extra_events=[{"map": "Dust", "kill_x": 1.0, "kill_y": 2.0}])
            self.assertEqual(payload["events"], 13)
            self.assertEqual(read_csv_rows(logs / "heatmap_events.csv").__len__(), 12)


class BenchmarkPayloadTests(unittest.TestCase):
    def test_rows_are_ranked_and_progress_is_bounded(self) -> None:
        snapshot = {
            "status": "complete",
            "results": [
                {"workers": 2, "envs_per_worker": 2, "total_envs": 4,
                 "steps_per_second": 100.0, "status": "complete"},
                {"workers": 4, "envs_per_worker": 2, "total_envs": 8,
                 "steps_per_second": 150.0, "status": "complete"},
                {"workers": 8, "envs_per_worker": 2, "total_envs": 16,
                 "steps_per_second": 0.0, "status": "running"},
            ],
            "current": [8, 2],
            "error": None,
        }
        payload = benchmark_payload(snapshot, total=3)
        self.assertEqual(payload["best"]["total_envs"], 8)
        self.assertEqual(payload["completed"], 2)
        self.assertEqual(payload["current"], [8, 2])
        self.assertAlmostEqual(payload["progress"], 2 / 3, places=6)
        self.assertEqual(payload["total"], 3)

    def test_empty_benchmark_has_no_best_row(self) -> None:
        payload = benchmark_payload({"status": "stopped", "results": [], "current": None}, total=0)
        self.assertIsNone(payload["best"])
        self.assertEqual(payload["progress"], 0.0)
        self.assertEqual(payload["results"], [])


if __name__ == "__main__":
    unittest.main()

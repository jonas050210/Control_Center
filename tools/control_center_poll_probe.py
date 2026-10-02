#!/usr/bin/env python3
"""What one Control Center poll tick reads from disk.

Every page's ``refresh()`` submits adapter calls on the 600 ms timer. This
probe builds a synthetic ``training/`` tree that looks like a real one - runs
with configs, summaries, live ``status.json``, checkpoints, evaluations and
training logs, including a large log for the newest runs, which is the shape
that actually costs something - and times each of those calls, cold and warm.

It exists because "the GUI reads the disk on a timer" is a claim that has to
be measured before it is fixed or claimed fixed. It reports milliseconds and
nothing else.

What this is NOT
----------------
Not a Godot benchmark, and not a training-throughput number. It measures this
repository's own file handling on a synthetic corpus; use it to compare a
change against itself.

Usage
-----
    python tools/control_center_poll_probe.py --runs 60 --big-log-mb 64
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sandboxai.adapter import SandboxAIAdapter  # noqa: E402

LOG_LINE = json.dumps(
    {"step": 0, "reward": 0.125, "loss": 0.5, "policy_loss": 0.25, "value_loss": 0.25}
)


def build_corpus(root: Path, runs: int, big_log_mb: float, big_logs: int) -> int:
    """Write ``runs`` run directories; the newest ``big_logs`` get a big log."""
    total = 0
    log_rows = max(1, int(big_log_mb * 1e6 / len(LOG_LINE)))
    for index in range(runs):
        run = root / "runs" / f"2026-10-01T00-{index:04d}"
        (run / "logs").mkdir(parents=True, exist_ok=True)
        (run / "checkpoints").mkdir(exist_ok=True)
        (run / "evaluations" / "step_000050000").mkdir(parents=True, exist_ok=True)
        (run / "run_manifest.json").write_text(
            json.dumps({"run_id": f"run-{index:04d}", "experiment_id": "e1"}),
            encoding="utf-8",
        )
        (run / "config.json").write_text(
            json.dumps({"run_id": f"run-{index:04d}", "total_training_steps": 100000}),
            encoding="utf-8",
        )
        (run / "run_summary.json").write_text(
            json.dumps({"total_timesteps": index * 500, "mean_episode_reward": 0.9}),
            encoding="utf-8",
        )
        (run / "status.json").write_text(
            json.dumps({"state": "finished", "timesteps": index * 500}), encoding="utf-8"
        )
        for name in ("latest.zip", "best_eval.zip"):
            (run / "checkpoints" / name).write_bytes(b"x" * 512)
        (run / "evaluations" / "latest.json").write_text(
            json.dumps({"timesteps": index * 500, "mean_episode_reward": 1.0}),
            encoding="utf-8",
        )
        (run / "evaluations" / "step_000050000" / "summary.json").write_text(
            json.dumps({"mean_episode_reward": 1.0}), encoding="utf-8"
        )
        rows = log_rows if index >= runs - big_logs else 200
        with (run / "logs" / "training.jsonl").open("w", encoding="utf-8") as handle:
            for step in range(rows):
                handle.write(LOG_LINE.replace('"step": 0', f'"step": {step}') + "\n")
        total += (run / "logs" / "training.jsonl").stat().st_size
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60, help="run directories to create")
    parser.add_argument("--big-log-mb", type=float, default=64.0, help="size of the big logs")
    parser.add_argument("--big-logs", type=int, default=3, help="how many runs get a big log")
    parser.add_argument("--output", help="write the JSON report to this path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="poll-probe-") as workdir:
        work = Path(workdir)
        log_bytes = build_corpus(work, args.runs, args.big_log_mb, args.big_logs)
        adapter = SandboxAIAdapter(project_root=REPO_ROOT, output_root=work)
        # Warm the directory listing so the first measurement is not the
        # filesystem's first look at a fresh tree.
        adapter.list_runs()

        calls = [
            ("dashboard_snapshot", lambda: adapter.dashboard_snapshot()),
            ("agents.views", lambda: adapter.agents.views()),
            ("discover_checkpoints", lambda: adapter.discover_checkpoints()),
            ("discover_evaluations", lambda: adapter.discover_evaluations()),
            ("list_runs", lambda: adapter.list_runs()),
            ("benchmark_pipeline_history", lambda: adapter.benchmark_pipeline_history()),
            ("list_replays", lambda: adapter.list_replays()),
        ]
        rows = []
        for label, fn in calls:
            start = time.perf_counter()
            fn()
            cold = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            fn()
            warm = (time.perf_counter() - start) * 1000.0
            rows.append({"call": label, "cold_ms": round(cold, 1), "warm_ms": round(warm, 1)})

    report = {
        "format": "sandboxai.control_center_poll_probe/v1",
        "synthetic": True,
        "note": "File I/O of the Control Center's own poll calls; not a Godot measurement.",
        "runs": args.runs,
        "big_log_mb": args.big_log_mb,
        "big_logs": args.big_logs,
        "log_bytes": log_bytes,
        "calls": rows,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

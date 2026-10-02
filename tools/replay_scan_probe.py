#!/usr/bin/env python3
"""Reproducible probe for the Control Center's replay-folder rescan.

What this measures
------------------
``SandboxAIAdapter.list_replays()``, the call behind the Stats page's replay
picker. It runs on the page's poll timer, so its warm-path cost is what an
operator browsing a large ``training/`` folder actually feels.

The probe writes a synthetic corpus of real-format replay files (header line
plus tick lines, observations included - structurally what the recorder
produces, not what the game plays), then times the call cold (empty cache)
and warm (the state a poll tick sees). It reports milliseconds per call,
never a rate it did not observe.

What this is NOT
----------------
This is not a Godot benchmark and not a training-throughput number. It
measures file I/O and caching in this repository's own adapter code, and the
corpus is synthetic: use it to compare a change against itself, not to
predict a machine's absolute performance.

Usage
-----
    python tools/replay_scan_probe.py --files 1200 --ticks 200 --repeats 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sandboxai.adapter import SandboxAIAdapter  # noqa: E402
from sandboxai.contract import OBSERVATION_FIELD_COUNT  # noqa: E402


def build_corpus(root: Path, files: int, ticks: int) -> int:
    """Write ``files`` detailed replays of ``ticks`` steps each; return bytes."""
    total = 0
    for index in range(files):
        folder = root / "runs" / f"run-{index:04d}" / "replays"
        folder.mkdir(parents=True, exist_ok=True)
        rows = [
            json.dumps(
                {
                    "header": {
                        "seed": index,
                        "map_id": "blind_corner",
                        "scenario": "corner_fight",
                        "detail": "detailed",
                        "observation_dim": OBSERVATION_FIELD_COUNT,
                    }
                }
            )
        ]
        for tick in range(ticks):
            rows.append(
                json.dumps(
                    {
                        "tick": {
                            "i": tick,
                            "a": [0, 1, 1, 1, 0, 0],
                            "r": 0.125,
                            "d": 1 if tick == ticks - 1 else 0,
                            "o": [0.0] * OBSERVATION_FIELD_COUNT,
                        }
                    }
                )
            )
        path = folder / "episode_0001.jsonl"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        total += path.stat().st_size
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=1200, help="recordings on disk")
    parser.add_argument("--ticks", type=int, default=200, help="steps per recording")
    parser.add_argument("--limit", type=int, default=200, help="list_replays(limit=...)")
    parser.add_argument("--repeats", type=int, default=4, help="warm scans to average")
    parser.add_argument("--output", help="write the JSON report to this path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="replay-scan-probe-") as workdir:
        work = Path(workdir)
        corpus_bytes = build_corpus(work, args.files, args.ticks)
        adapter = SandboxAIAdapter(project_root=REPO_ROOT, output_root=work)

        start = time.perf_counter()
        entries = adapter.list_replays(limit=args.limit)
        cold_ms = (time.perf_counter() - start) * 1000.0

        warm_ms: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            adapter.list_replays(limit=args.limit)
            warm_ms.append((time.perf_counter() - start) * 1000.0)

    report = {
        "format": "sandboxai.replay_scan_probe/v1",
        "synthetic": True,
        "note": "File I/O and caching only; not a Godot or training measurement.",
        "files": args.files,
        "ticks_per_file": args.ticks,
        "limit": args.limit,
        "corpus_mb": round(corpus_bytes / 1e6, 1),
        "listed": len(entries),
        "cold_ms": round(cold_ms, 1),
        "warm_min_ms": round(min(warm_ms), 1),
        "warm_median_ms": round(statistics.median(warm_ms), 1),
        "warm_max_ms": round(max(warm_ms), 1),
        "repeats": args.repeats,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Architecture-level scaling probe for the SandboxAI bridge.

What this measures
------------------
How Python-side end-to-end step throughput scales with (a) environments per
Godot process and (b) the number of concurrently simulating Godot
processes, using the *simulated* bridge from ``python/tests``. The simulated
bridge burns a configurable amount of CPU per environment step, which
reproduces the property that matters architecturally: one bridge process
steps its environments serially on one core.

What this is NOT
----------------
This is **not** a Godot benchmark and it must never be reported as one.
It cannot tell you how many microseconds ``EnvironmentCore.step`` takes on
your machine. Use ``sandboxai benchmark --worker-counts ...`` with a real
Godot binary for that; that path deliberately refuses to invent numbers.

What it is good for: measuring the transport, serialization, batching and
process-parallelism behaviour of the bridge itself, which is exactly the
part this repository owns.

Usage
-----
    python tools/bridge_scaling_probe.py --step-usec 200 \
        --env-counts 4,8,12,16,20 --worker-counts 1,2,4,8 --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "tests"))

from simulated_bridge import SimulatedBridgeExecutable, supported  # noqa: E402

from sandboxai.godot_env import make_batch_client  # noqa: E402


def measure(
    executable: str,
    environment_count: int,
    worker_count: int,
    steps: int,
    warmup: int = 20,
) -> dict[str, float]:
    client = make_batch_client(
        project_path=REPO_ROOT,
        godot_executable=executable,
        environment_count=environment_count,
        enemy_count=1,
        seed=1234,
        curriculum_level=3,
        worker_count=worker_count,
        request_timeout=120.0,
    )
    try:
        client.reset(1234)
        actions = [[1, 1, 1, 1, 0, 0] for _ in range(environment_count)]
        for _ in range(warmup):
            client.step(actions)
        started = time.perf_counter()
        for _ in range(steps):
            client.step(actions)
        elapsed = max(time.perf_counter() - started, 1e-9)
    finally:
        client.close()
    return {
        "environments": environment_count,
        "workers": worker_count,
        "vector_steps": steps,
        "env_steps": steps * environment_count,
        "elapsed_seconds": elapsed,
        "env_steps_per_second": steps * environment_count / elapsed,
        "vector_steps_per_second": steps / elapsed,
        "ms_per_vector_step": elapsed / steps * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-counts", default="4,8,12,16,20")
    parser.add_argument("--worker-counts", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--step-usec",
        type=float,
        default=200.0,
        help="synthetic per-environment CPU cost of one simulation step",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not supported():
        print("simulated bridge requires a POSIX shell", file=sys.stderr)
        return 2

    os.environ["SANDBOXAI_FAKE_STEP_USEC"] = str(args.step_usec)
    os.environ["SANDBOXAI_FAKE_EPISODE_LENGTH"] = "101"
    environment_counts = [int(v) for v in args.env_counts.split(",") if v.strip()]
    worker_counts = [int(v) for v in args.worker_counts.split(",") if v.strip()]
    rows: list[dict[str, float]] = []
    with SimulatedBridgeExecutable() as bridge:
        for environment_count in environment_counts:
            planned = sorted({min(w, environment_count) for w in worker_counts})
            baseline = None
            for worker_count in planned:
                row = measure(bridge.path, environment_count, worker_count, args.steps)
                if worker_count == 1:
                    baseline = row["env_steps_per_second"]
                row["speedup_vs_single_process"] = (
                    row["env_steps_per_second"] / baseline if baseline else None
                )
                rows.append(row)
                print(
                    f"envs={environment_count:>3} workers={worker_count:>2} "
                    f"{row['env_steps_per_second']:>10.1f} env-steps/s "
                    f"({row['ms_per_vector_step']:.2f} ms/vector step, "
                    f"speedup {row['speedup_vs_single_process']:.2f}x)"
                )
    report = {
        "format": "sandboxai.bridge_scaling_probe/v1",
        "synthetic": True,
        "step_usec": args.step_usec,
        "note": "Simulated bridge, not Godot. Measures transport/process parallelism only.",
        "cpu_count": os.cpu_count(),
        "rows": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

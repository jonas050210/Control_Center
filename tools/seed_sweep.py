"""Short CPU seed sweep: which run produces the first reliable kills?

PPO on a CPU budget is noisy, so a single run proves nothing. This helper trains
N short runs (default: 2 x 100k steps) with the honest perception model, each in
its own models/ folder, and prints the peak and best kill rate per run. The
winner can be copied into ``models/`` afterwards.

Usage (from the repository root, inside .venv):

    python3 tools/seed_sweep.py 3 100000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/Control_Center")
from training.train import TrainingConfig, TrainingController


def run_one(index: int, seed: int, steps: int) -> dict:
    root = Path(f"/tmp/sweep/{index}")
    config = TrainingConfig(
        map_name="Dust",
        total_timesteps=steps,
        duration_seconds=0.0,
        n_workers=2,
        self_play=False,
        curriculum=True,
        vision_mode="coarse_los",
        episode_seconds=30.0,
        curriculum_min_win_rate=0.35,
        gamma=0.99,
        seed=seed,
        models_dir=root / "models",
        logs_dir=root / "logs",
    )
    controller = TrainingController(config)
    started = time.monotonic()
    controller.start()
    peak_kill = 0.0
    while True:
        snapshot = controller.snapshot()
        metrics = snapshot.get("metrics", {})
        peak_kill = max(peak_kill, float(metrics.get("kill_rate", 0.0)))
        if snapshot.get("status") in {"complete", "error", "stopped"}:
            best = root / "models" / "best_model.json"
            payload = json.loads(best.read_text()) if best.exists() else {}
            return {
                "index": index, "seed": seed, "status": snapshot.get("status"),
                "error": snapshot.get("error"), "steps": metrics.get("timesteps", 0),
                "peak_kill_rate": peak_kill,
                "best_kill_rate": payload.get("kill_rate", 0.0),
                "best_win_rate": payload.get("win_rate", 0.0),
                "best_phase": payload.get("phase"),
                "seconds": round(time.monotonic() - started, 1),
                "models_dir": str(root / "models"),
            }
        time.sleep(5)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
    results = []
    for index in range(1, count + 1):
        result = run_one(index, seed=1000 * index + 7, steps=steps)
        results.append(result)
        print(json.dumps(result), flush=True)
    print("SWEEP DONE", json.dumps(results), flush=True)

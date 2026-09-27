"""Terminal dashboard for SandboxAI state and experiment history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry


def display_dashboard(
    state_file: str = "logs/system_state.json",
    datasets_dir: str = "datasets",
    experiments_dir: str = "logs/experiments",
) -> None:
    telemetry = SystemTelemetry(state_file)
    telemetry.update_dataset_stats(datasets_dir)
    state = telemetry.state
    hardware = state.get("hardware", {})
    datasets = state.get("datasets", {})
    bc = state.get("bc", {})
    rl = state.get("rl", {})
    runtime = state.get("runtime", {})
    benchmark = state.get("evaluation", {}).get("latest_benchmark", {})
    runs = ExperimentTracker.list_runs(experiments_dir)

    print("=" * 72)
    print("  SandboxAI — local learning platform")
    print("=" * 72)
    print(f"Stage            : {state.get('pipeline_stage', 'UNKNOWN')}")
    print(f"Last update      : {state.get('last_updated', 'N/A')}")
    print(f"Host             : {hardware.get('os', 'N/A')}")
    print(
        f"CPU / RAM        : {hardware.get('cpu_percent', 0)}% / "
        f"{hardware.get('ram_used_gb', 0)} of {hardware.get('ram_total_gb', 0)} GB"
    )
    print(f"GPU              : {hardware.get('gpu', 'N/A')}")
    print("\nDATA")
    print(
        f"  sessions={datasets.get('total_sessions', 0)} "
        f"steps={datasets.get('total_steps', 0)} size={datasets.get('total_mb', 0)} MB"
    )
    print("MODELS")
    print(
        f"  BC  checkpoint={bc.get('active_checkpoint') or '-'} "
        f"best_val_loss={bc.get('best_val_loss') if bc.get('best_val_loss') is not None else '-'}"
    )
    print(
        f"  PPO checkpoint={rl.get('active_checkpoint') or '-'} "
        f"steps={rl.get('timesteps', 0)} speed={rl.get('steps_per_sec', 0)} steps/s"
    )
    print("RUNTIME")
    print(
        f"  fps={runtime.get('fps', 0)} steps/s={runtime.get('steps_per_sec', 0)} "
        f"reward={runtime.get('episode_reward', 0)} action={runtime.get('last_action', {})}"
    )
    if benchmark:
        print("LATEST BENCHMARK")
        for name, metrics in benchmark.items():
            if not isinstance(metrics, dict) or "mean_reward" not in metrics:
                continue
            print(
                f"  {name.upper():<10} reward={metrics['mean_reward']:>8} "
                f"hits={metrics.get('total_hits', 0):>4} "
                f"kills={metrics.get('total_kills', 0):>3} "
                f"accuracy={metrics.get('hit_rate_pct', 0):>6.1f}%"
            )
    print("RECENT EXPERIMENTS")
    if not runs:
        print("  (none)")
    for run in runs[:5]:
        print(
            f"  {run.get('run_id', '?'):<42} {run.get('status', '?'):<10} {run.get('kind', '?')}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="View SandboxAI status")
    parser.add_argument("--state_file", default="logs/system_state.json")
    parser.add_argument("--datasets_dir", default="datasets")
    parser.add_argument("--experiments_dir", default="logs/experiments")
    args = parser.parse_args()
    display_dashboard(args.state_file, args.datasets_dir, args.experiments_dir)


if __name__ == "__main__":
    main()

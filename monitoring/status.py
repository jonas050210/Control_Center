"""CLI Dashboard for SandboxAI System Telemetry and Pipeline Status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.state import SystemTelemetry


def display_dashboard(state_file: str = "logs/system_state.json") -> None:
    telemetry = SystemTelemetry(state_file=state_file)
    telemetry.update_dataset_stats("datasets")
    state = telemetry.state

    hw = state.get("hardware", {})
    ds = state.get("datasets", {})
    bc = state.get("bc", {})
    rl = state.get("rl", {})
    ev = state.get("evaluation", {}).get("latest_benchmark", {})

    print("============================================================")
    print("  SandboxAI System Control & Telemetry Dashboard")
    print("============================================================")
    print(f"Pipeline Stage   : {state.get('pipeline_stage', 'UNKNOWN')}")
    print(f"Last Updated     : {state.get('last_updated', 'N/A')}")
    print(f"Platform / Host  : {hw.get('os', 'N/A')}")
    print(f"CPU Utilization  : {hw.get('cpu_percent', 0)}% ({hw.get('cpu_count', 0)} cores)")
    print(f"RAM Utilization  : {hw.get('ram_used_gb', 0)} / {hw.get('ram_total_gb', 0)} GB ({hw.get('ram_percent', 0)}%)")
    print(f"GPU Device       : {hw.get('gpu', 'N/A')}")
    print("")
    print("--- Dataset Repository ---")
    print(f"  Total Sessions : {ds.get('total_sessions', 0)}")
    print(f"  Total Steps    : {ds.get('total_steps', 0)}")
    print(f"  Storage Size   : {ds.get('total_mb', 0)} MB")
    print("")
    print("--- Behavioral Cloning (BC) ---")
    print(f"  Active Model   : {bc.get('active_checkpoint', 'None')}")
    print(f"  Best Val Loss  : {bc.get('best_val_loss', 'N/A')}")
    print("")
    print("--- Reinforcement Learning (PPO) ---")
    print(f"  Active Policy  : {rl.get('active_checkpoint', 'None')}")
    print(f"  Total Timesteps: {rl.get('timesteps', 0)}")
    print(f"  Training Speed : {rl.get('steps_per_sec', 0)} steps/s")

    if ev:
        print("")
        print("--- Latest Evaluation Benchmark ---")
        for p_name, metrics in ev.items():
            rew_str = f"{metrics.get('mean_reward', 0):+.2f} ± {metrics.get('std_reward', 0):.2f}"
            print(f"  {p_name.upper():<10}: Reward={rew_str:<18} Hits={metrics.get('total_hits', 0):<4} Fire={metrics.get('fire_rate_pct', 0):.1f}%")

    print("============================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="View SandboxAI System Status Dashboard")
    parser.add_argument("--state_file", type=str, default="logs/system_state.json", help="Path to state file")
    args = parser.parse_args()
    display_dashboard(args.state_file)


if __name__ == "__main__":
    main()

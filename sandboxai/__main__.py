"""Unified SandboxAI command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict


def doctor() -> Dict[str, Any]:
    dependencies = {}
    for name in ("numpy", "PIL", "torch", "gymnasium", "stable_baselines3", "mss", "pynput", "psutil"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = {"available": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            dependencies[name] = {"available": False, "error": str(exc)}
    exports = {
        str(path): path.is_file()
        for path in (Path("build/sandbox_windows.exe"), Path("build/sandbox_linux.x86_64"))
    }
    required = {"numpy", "PIL", "torch", "gymnasium", "stable_baselines3", "mss", "psutil"}
    if platform.system() == "Windows":
        required.add("pynput")
    report = {
        "dependencies": dependencies,
        "godot_project": Path("sandbox/godot_project/project.godot").is_file(),
        "exports": exports,
        "pipeline_ready": all(dependencies[name]["available"] for name in required),
        "note": "pynput is required for live input capture on Windows" if platform.system() != "Windows" else None,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sandboxai", description="Local end-to-end FPS learning platform"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check local dependencies and sandbox exports")

    e2e = subparsers.add_parser("e2e", help="Run record/data/BC/RL/evaluation workflow")
    e2e.add_argument("--data_dir", default="datasets")
    e2e.add_argument("--checkpoint_dir", default="checkpoints")
    e2e.add_argument("--logs_dir", default="logs")
    e2e.add_argument("--generate_demo", action="store_true")
    e2e.add_argument("--demo_duration", type=float, default=1.0)
    e2e.add_argument("--bc_epochs", type=int, default=3)
    e2e.add_argument("--rl_timesteps", type=int, default=4096)
    e2e.add_argument("--eval_episodes", type=int, default=5)
    e2e.add_argument("--eval_steps", type=int, default=250)
    e2e.add_argument("--device", default="auto")
    e2e.add_argument("--backend", choices=["python", "godot"], default="python")
    e2e.add_argument("--env_path", default=None)
    e2e.add_argument("--quick", action="store_true")
    e2e.add_argument("--seed", type=int, default=42)

    status = subparsers.add_parser("status", help="Print pipeline status")
    status.add_argument("--state_file", default="logs/system_state.json")
    status.add_argument("--datasets_dir", default="datasets")
    status.add_argument("--experiments_dir", default="logs/experiments")

    dashboard = subparsers.add_parser("dashboard", help="Serve local monitoring UI")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--state_file", default="logs/system_state.json")
    dashboard.add_argument("--experiments_dir", default="logs/experiments")

    play = subparsers.add_parser("play", help="Launch the exported sandbox for manual play")
    play.add_argument("--env_path", default="build/sandbox_windows.exe")

    args = parser.parse_args()
    if args.command == "doctor":
        report = doctor()
        raise SystemExit(0 if report["pipeline_ready"] else 1)
    if args.command == "e2e":
        from sandboxai.workflow import run_end_to_end

        result = run_end_to_end(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            logs_dir=args.logs_dir,
            generate_demo=args.generate_demo,
            demo_duration=args.demo_duration,
            bc_epochs=args.bc_epochs,
            rl_timesteps=args.rl_timesteps,
            eval_episodes=args.eval_episodes,
            eval_steps=args.eval_steps,
            device=args.device,
            env_path=args.env_path,
            backend=args.backend,
            quick=args.quick,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "status":
        from monitoring.status import display_dashboard

        display_dashboard(args.state_file, args.datasets_dir, args.experiments_dir)
        return
    if args.command == "dashboard":
        from monitoring.server import serve

        serve(args.host, args.port, args.state_file, args.experiments_dir)
        return
    if args.command == "play":
        path = Path(args.env_path)
        if not path.is_file():
            raise SystemExit(
                f"Sandbox export not found: {path}. Run python -m sandbox.export first."
            )
        raise SystemExit(subprocess.run([str(path.resolve())], check=False).returncode)


if __name__ == "__main__":
    main()

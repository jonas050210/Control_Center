"""Command-line workflow for recording, BC, PPO, evaluation and benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .config import BCConfig, TrainingConfig


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON TrainingConfig file")
    parser.add_argument("--env-count", type=int, dest="environment_count")
    parser.add_argument("--enemy-count", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--rollout-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--steps", type=int, dest="total_training_steps")
    parser.add_argument("--checkpoint-frequency", type=int)
    parser.add_argument("--evaluation-frequency", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--curriculum-level", type=int)
    parser.add_argument("--godot-executable", default=None)
    parser.add_argument("--project-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--bc-checkpoint", default=None)


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    values: dict[str, Any] = {}
    if args.config:
        values = json.loads(Path(args.config).read_text(encoding="utf-8"))
    fields = set(TrainingConfig.__dataclass_fields__)
    for key, value in vars(args).items():
        if key in fields and value is not None:
            values[key] = value
    return TrainingConfig.from_dict(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sandboxai", description="SandboxAI local Godot + PyTorch research workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="print the recommended local dependency installation command")
    install.add_argument("--cuda", action="store_true", help="also print the CUDA PyTorch index example")

    train = sub.add_parser("train", help="train PPO against headless Godot")
    _add_training_options(train)

    resume = sub.add_parser("resume", help="resume PPO from a .zip checkpoint")
    resume.add_argument("--checkpoint", required=True)
    _add_training_options(resume)

    evaluate = sub.add_parser("evaluate", help="evaluate a frozen PPO checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=9001)
    evaluate.add_argument("--env-count", type=int, default=1, dest="environment_count")
    evaluate.add_argument("--enemy-count", type=int, default=1)
    evaluate.add_argument("--curriculum-level", type=int, default=3)
    evaluate.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    evaluate.add_argument("--godot-executable", default="godot")
    evaluate.add_argument("--project-path", default="")
    evaluate.add_argument("--output-dir", default="training/evaluations/cli")

    record = sub.add_parser("record", help="open the graphical Godot human demonstration recorder")
    record.add_argument("--output", default="training/datasets/human_demo.jsonl")
    record.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means until the window closes")
    record.add_argument("--enemy-count", type=int, default=1)
    record.add_argument("--godot-executable", default="godot")
    record.add_argument("--project-path", default="")

    bc = sub.add_parser("bc-train", help="train a PyTorch behavior-cloning policy")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--epochs", type=int, default=25)
    bc.add_argument("--batch-size", type=int, default=512)
    bc.add_argument("--learning-rate", type=float, default=3e-4)
    bc.add_argument("--validation-fraction", type=float, default=0.1)
    bc.add_argument("--seed", type=int, default=1234)
    bc.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bc.add_argument("--output-dir", default="training/bc_runs/latest")
    bc.add_argument("--resume-checkpoint")

    inspect = sub.add_parser("inspect-dataset", help="validate and summarise a demonstration JSONL")
    inspect.add_argument("--dataset", required=True)

    benchmark = sub.add_parser("benchmark", help="measure headless Godot throughput")
    benchmark.add_argument("--env-counts", default="1,4,8,16")
    benchmark.add_argument("--steps", type=int, default=10000)
    benchmark.add_argument("--enemy-count", type=int, default=1)
    benchmark.add_argument("--seed", type=int, default=1234)
    benchmark.add_argument("--curriculum-level", type=int, default=3)
    benchmark.add_argument("--godot-executable", default="godot")
    benchmark.add_argument("--project-path", default="")
    benchmark.add_argument("--output-dir", default="training/benchmarks/latest")
    return parser


def _run_record(args: argparse.Namespace) -> int:
    project = Path(args.project_path).expanduser().resolve() if args.project_path else Path(__file__).resolve().parents[2]
    command = [
        args.godot_executable,
        "--path",
        str(project),
        "--script",
        "res://scripts/recording/record_demo.gd",
        "--",
        "--output",
        args.output,
        "--enemy-count",
        str(args.enemy_count),
    ]
    if args.duration > 0:
        command.extend(["--duration", str(args.duration)])
    return subprocess.call(command)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        print("python -m pip install -e '.[training]'")
        if args.cuda:
            print("For CUDA, install the matching PyTorch wheel from https://pytorch.org/ before the command above.")
        print("Godot 4.7.2 must be installed separately and available as 'godot' (or pass --godot-executable).")
        return 0
    if args.command == "inspect-dataset":
        from .dataset import DemonstrationDataset
        print(json.dumps(DemonstrationDataset.load(args.dataset).summary(), indent=2, default=str))
        return 0
    if args.command == "bc-train":
        from .bc import train_behavior_cloning
        config = BCConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            device=args.device,
            output_root=str(Path(args.output_dir).parent),
        )
        print(json.dumps(train_behavior_cloning(args.dataset, config, args.output_dir, args.resume_checkpoint), indent=2))
        return 0
    if args.command in {"train", "resume"}:
        from .ppo import train_ppo
        config = _config_from_args(args)
        result = train_ppo(config, args.checkpoint if args.command == "resume" else None)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "evaluate":
        from stable_baselines3 import PPO  # type: ignore
        from .evaluation import evaluate_model, format_summary
        from .config import TrainingConfig
        config = TrainingConfig(
            environment_count=args.environment_count,
            enemy_count=args.enemy_count,
            curriculum_level=args.curriculum_level,
            seed=args.seed,
            device=args.device,
            godot_executable=args.godot_executable,
            project_path=args.project_path,
        )
        model = PPO.load(args.checkpoint, device=config.resolved_device())
        result = evaluate_model(
            model,
            {
                "project_path": config.project,
                "godot_executable": config.godot_executable,
                "environment_count": 1,
                "enemy_count": config.enemy_count,
                "seed": config.seed,
                "curriculum_level": config.curriculum_level,
            },
            args.episodes,
            args.seed,
            args.output_dir,
        )
        print(format_summary(result))
        print(json.dumps({key: value for key, value in result.items() if key != "episodes_detail"}, indent=2, default=str))
        return 0
    if args.command == "benchmark":
        from .benchmark import benchmark_simulation
        project = Path(args.project_path).expanduser().resolve() if args.project_path else Path(__file__).resolve().parents[2]
        counts = [int(value) for value in args.env_counts.split(",") if value.strip()]
        result = benchmark_simulation(project, args.godot_executable, counts, args.steps, args.enemy_count, args.seed, args.curriculum_level, args.output_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    raise RuntimeError(f"unhandled command {args.command}")


if __name__ == "__main__":
    sys.exit(main())

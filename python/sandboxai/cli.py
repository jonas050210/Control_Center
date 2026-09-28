"""Command-line workflow for recording, BC, PPO, evaluation, benchmarks and smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .config import BCConfig, TrainingConfig, find_godot_executable


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
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--bc-checkpoint", default=None)
    # --- Integrated research pipeline (defaults live in TrainingConfig) ---
    parser.add_argument(
        "--curriculum-mode",
        choices=["auto", "fixed"],
        default=None,
        help="auto = curriculum-driven episode plans (integrated pipeline); fixed = one level, historical behavior",
    )
    parser.add_argument("--curriculum-start-level", type=int, default=None)
    parser.add_argument(
        "--adaptive-curriculum",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="off = curriculum plans every episode deterministically but never promotes/demotes",
    )
    parser.add_argument(
        "--skill-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="per-tick research metrics during training (measurement only, never rewards)",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["off", "interesting", "every_n", "all", "evaluation"],
        default=None,
    )
    parser.add_argument("--replay-every-n", type=int, default=None)
    parser.add_argument("--replay-detail", choices=["light", "detailed"], default=None)
    parser.add_argument("--replay-max-per-run", type=int, default=None)
    parser.add_argument(
        "--checkpoint-condition-eval", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--checkpoint-generalization-eval", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--condition-eval-episodes", type=int, default=None)
    parser.add_argument("--generalization-episodes-per-cell", type=int, default=None)
    parser.add_argument(
        "--checkpoint-league-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="play frozen checkpoint snapshots in the self-play league at each evaluation",
    )
    parser.add_argument("--league-matches-per-checkpoint", type=int, default=None)
    parser.add_argument("--league-max-opponents", type=int, default=None)


def _resolve_project_path(project_path: str | None) -> Path:
    if project_path:
        return Path(project_path).expanduser().resolve()
    # python/sandboxai/cli.py -> repository root
    return Path(__file__).resolve().parents[2]


def build_record_command(
    godot_executable: str,
    project_path: str | None,
    output: str,
    duration: float,
    enemy_count: int,
) -> list[str]:
    """Build the Godot invocation for the graphical demonstration recorder.

    The output path is resolved to an absolute path so the recording is saved
    predictably regardless of the Godot process working directory.
    """
    executable = find_godot_executable(godot_executable)
    project = _resolve_project_path(project_path)
    return [
        executable,
        "--path",
        str(project),
        "--script",
        "res://scripts/recording/record_demo.gd",
        "--",
        "--output",
        str(Path(output).expanduser().resolve()),
        "--duration",
        str(duration),
        "--enemy-count",
        str(enemy_count),
    ]


def build_control_center_command(
    godot_executable: str,
    project_path: str | None,
    mode: str,
    environment_count: int,
    enemy_count: int,
    curriculum_level: int,
    seed: int,
    scenario: str = "",
) -> list[str]:
    """Build the Godot invocation for the graphical Control Center.

    This launches ``scenes/control_center.tscn`` in a normal (non-headless)
    Godot window: the Control Center is an operator/inspection tool and is
    deliberately never part of the headless RL training path. Arguments after
    ``--`` are read by ``ControlCenterMain._apply_command_line``.
    """
    executable = find_godot_executable(godot_executable)
    project = _resolve_project_path(project_path)
    command = [
        executable,
        "--path",
        str(project),
        "res://scenes/control_center.tscn",
        "--",
        f"--mode={mode}",
        f"--env-count={environment_count}",
        f"--enemy-count={enemy_count}",
        f"--curriculum-level={curriculum_level}",
        f"--seed={seed}",
    ]
    if scenario:
        command.append(f"--scenario={scenario}")
    return command


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

    control_center = sub.add_parser(
        "control-center",
        help="open the graphical Control Center (watch the AI, play as a human, inspect the simulation)",
    )
    control_center.add_argument("--mode", default="watch", choices=["training", "watch", "human"])
    control_center.add_argument("--env-count", type=int, default=4, dest="environment_count")
    control_center.add_argument("--enemy-count", type=int, default=1)
    control_center.add_argument("--curriculum-level", type=int, default=3)
    control_center.add_argument("--seed", type=int, default=1234)
    control_center.add_argument(
        "--scenario",
        default="",
        help=(
            "optional scenario preset id (target_practice, duel, three_way, overwhelmed, "
            "cover_fight, corner_fight, sound_only, lost_target, vertical, randomized)"
        ),
    )
    control_center.add_argument("--godot-executable", default="godot")
    control_center.add_argument("--project-path", default="")

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
    benchmark.add_argument("--env-counts", default="1,2,4,8,16,24,32,48,64")
    benchmark.add_argument("--steps", type=int, default=2000)
    benchmark.add_argument("--enemy-count", type=int, default=1)
    benchmark.add_argument("--seed", type=int, default=1234)
    benchmark.add_argument("--curriculum-level", type=int, default=3)
    benchmark.add_argument("--godot-executable", default="godot")
    benchmark.add_argument("--project-path", default="")
    benchmark.add_argument("--output-dir", default="training/benchmarks/latest")
    benchmark.add_argument("--max-seconds-per-config", type=float, default=20.0)

    benchmark_suites = sub.add_parser(
        "benchmark-suites",
        help="run the four comparable benchmark suites at 1/4/8/16/32/64 environments",
    )
    benchmark_suites.add_argument("--godot-executable", default="godot")
    benchmark_suites.add_argument("--project-path", default="")
    benchmark_suites.add_argument("--output-dir", default="training/benchmarks/suites")
    benchmark_suites.add_argument(
        "--plan-only",
        action="store_true",
        help="print the plan without measuring anything (no Godot required)",
    )

    replay = sub.add_parser("replay", help="inspect or validate a recorded episode replay")
    replay.add_argument("--path", required=True)
    replay.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help="load a replay recorded against an older observation/action contract",
    )
    replay.add_argument("--timeline", action="store_true", help="print the event timeline")

    curriculum = sub.add_parser(
        "curriculum", help="print the integrated curriculum ladder (levels 1-11)"
    )
    curriculum.add_argument("--json", action="store_true")

    adapter = sub.add_parser(
        "adapter-contract", help="print the external-game adapter contract (Roblox boundary)"
    )
    adapter.add_argument("--check-mock", action="store_true", help="run the mock adapter contract check")

    validate_rt = sub.add_parser(
        "validate-runtime",
        help="run automated live headless validation against a real Godot executable",
    )
    validate_rt.add_argument("--godot-executable", default="godot")
    validate_rt.add_argument("--project-path", default="")
    validate_rt.add_argument("--timeout", type=float, default=15.0)
    validate_rt.add_argument("--env-count", type=int, default=2)
    validate_rt.add_argument("--json", action="store_true")

    compare_exp = sub.add_parser(
        "compare-experiments",
        help="compare candidate experiment summary against a baseline and detect regressions",
    )
    compare_exp.add_argument("--baseline", required=True, help="path to baseline summary.json")
    compare_exp.add_argument("--candidate", required=True, help="path to candidate summary.json")
    compare_exp.add_argument("--threshold", type=float, default=0.05, help="regression threshold (default 0.05)")
    compare_exp.add_argument("--json", action="store_true")

    summarize_exp = sub.add_parser(
        "summarize-experiment",
        help="summarize multi-seed experiment runs in a directory",
    )
    summarize_exp.add_argument("--path", required=True, help="directory containing seed run summaries")
    summarize_exp.add_argument("--json", action="store_true")

    smoke = sub.add_parser("smoke-test", help="run end-to-end sanity verification of the Python & ML stack")
    smoke.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])

    return parser


def run_smoke_test(device: str = "cpu") -> dict[str, Any]:
    """Runs a fast, self-contained smoke test verifying all Python ML components."""
    print("Running SandboxAI ML stack smoke test...")
    results: dict[str, Any] = {}
    failures: list[str] = []

    # 1. Config test
    config = TrainingConfig(environment_count=2, rollout_length=64, batch_size=32, total_training_steps=128, device=device).validate()
    results["config_valid"] = True

    # 2. Dataset creation and validation. Observations use the real
    # 84-field contract dimension (OBSERVATION_FIELD_COUNT) so the smoke
    # test exercises (and produces checkpoints compatible with) the actual
    # observation space.
    from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
    from .dataset import DemonstrationDataset, DemonstrationRecorder
    recorder = DemonstrationRecorder({"source": "smoke_test"})
    recorder.start()
    for i in range(20):
        obs = [0.1 * ((i + j) % 10) for j in range(OBSERVATION_FIELD_COUNT)]
        next_obs = [0.1 * ((i + j + 1) % 10) for j in range(OBSERVATION_FIELD_COUNT)]
        action = [0, 0, 1, 0, 1 if i % 4 == 0 else 0, 0.0, 0.0]
        done = (i == 19)
        recorder.append(obs, action, next_obs, 1.0 if done else 0.01, done, episode_id=0)
    recorder.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = Path(tmp_dir) / "smoke_demo.jsonl"
        recorder.save(dataset_path)
        dataset = DemonstrationDataset.load(dataset_path)
        summary = dataset.summary()
        results["dataset_transitions"] = summary["transitions"]
        results["observation_dim"] = summary["observation_dim"]
        if summary["observation_dim"] != OBSERVATION_FIELD_COUNT:
            failures.append(f"observation_dim {summary['observation_dim']} != contract {OBSERVATION_FIELD_COUNT}")

        # 3. BC training
        from .bc import train_behavior_cloning, load_bc_checkpoint, load_bc_into_sb3_policy
        bc_config = BCConfig(epochs=2, batch_size=4, device=device, output_root=tmp_dir)
        bc_result = train_behavior_cloning(dataset_path, bc_config, output_dir=Path(tmp_dir) / "bc_out")
        results["bc_best_checkpoint"] = bc_result["best_checkpoint"]

        # 4. BC checkpoint load & predict
        model = load_bc_checkpoint(bc_result["best_checkpoint"], device=device)
        sample_pred = model.predict([0.0] * OBSERVATION_FIELD_COUNT)
        results["bc_prediction_shape"] = list(sample_pred.shape)
        expected_action_dim = [len(ACTION_NVEC)]
        if list(sample_pred.shape) != expected_action_dim:
            failures.append(f"BC prediction shape {list(sample_pred.shape)} != {expected_action_dim}")

        # 5. SB3 warm start verification
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO
            action_space = gym.spaces.MultiDiscrete(list(ACTION_NVEC))
            obs_space = gym.spaces.Box(-1.0, 1.0, shape=(OBSERVATION_FIELD_COUNT,))

            class MockGymEnv(gym.Env):
                def __init__(self):
                    self.observation_space = obs_space
                    self.action_space = action_space

            ppo_model = PPO("MlpPolicy", MockGymEnv(), n_steps=16, batch_size=16, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}}, device=device)
            transfer_res = load_bc_into_sb3_policy(ppo_model.policy, bc_result["best_checkpoint"], device=device)
            results["sb3_weight_transfer"] = transfer_res["transferred"]
            if not transfer_res["transferred"]:
                failures.append("SB3 weight transfer reported transferred=False")
        except Exception as e:
            results["sb3_weight_transfer_error"] = str(e)
            failures.append(f"SB3 weight transfer failed: {e}")

    results["failures"] = failures
    results["all_passed"] = not failures
    if failures:
        print("Smoke test FAILED:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("Smoke test completed successfully!")
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        print("python -m pip install -e '.[training]'")
        if args.cuda:
            print("For CUDA, install the matching PyTorch wheel from https://pytorch.org/ before the command above.")
        print("Godot 4.7.2 must be installed separately and available as 'godot' (or pass --godot-executable).")
        return 0
    if args.command == "smoke-test":
        res = run_smoke_test(args.device)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("all_passed") else 1
    if args.command == "inspect-dataset":
        from .dataset import DemonstrationDataset
        print(json.dumps(DemonstrationDataset.load(args.dataset).summary(), indent=2, default=str))
        return 0
    if args.command == "record":
        command = build_record_command(
            args.godot_executable,
            args.project_path,
            args.output,
            args.duration,
            args.enemy_count,
        )
        print("Launching Godot demonstration recorder:", " ".join(command))
        try:
            return subprocess.call(command)
        except OSError as exc:
            print(
                f"Could not launch Godot executable {command[0]!r}: {exc}. "
                "Install Godot 4.7.2 and put it on PATH or pass --godot-executable.",
                file=sys.stderr,
            )
            return 1
    if args.command == "control-center":
        command = build_control_center_command(
            args.godot_executable,
            args.project_path,
            args.mode,
            args.environment_count,
            args.enemy_count,
            args.curriculum_level,
            args.seed,
            args.scenario,
        )
        print("Launching SandboxAI Control Center:", " ".join(command))
        try:
            return subprocess.call(command)
        except OSError as exc:
            print(
                f"Could not launch Godot executable {command[0]!r}: {exc}. "
                "Install Godot 4.7.2 and put it on PATH or pass --godot-executable.",
                file=sys.stderr,
            )
            return 1
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
        ).validate()
        model = PPO.load(args.checkpoint, device=config.resolved_device())
        result = evaluate_model(
            model,
            {
                "project_path": config.project,
                "godot_executable": config.godot_executable,
                "environment_count": max(1, int(args.environment_count or 1)),
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
        from .benchmark import benchmark_simulation, summarize_scaling
        project = _resolve_project_path(args.project_path)
        counts = [int(value) for value in args.env_counts.split(",") if value.strip()]
        result = benchmark_simulation(
            project,
            args.godot_executable,
            counts,
            args.steps,
            args.enemy_count,
            args.seed,
            args.curriculum_level,
            args.output_dir,
            args.max_seconds_per_config,
        )
        print(json.dumps(result, indent=2, default=str))
        print(json.dumps({"scaling_summary": summarize_scaling(result)}, indent=2, default=str))
        return 0
    if args.command == "benchmark-suites":
        from .benchmark_suites import describe_plan, format_report, run_suites
        if args.plan_only:
            print(json.dumps(describe_plan(), indent=2, default=str))
            return 0
        project = _resolve_project_path(args.project_path)
        report = run_suites(project, args.godot_executable, output_dir=args.output_dir)
        print(format_report(report))
        return 0
    if args.command == "replay":
        from .replay import load_replay, validate_replay
        episode = load_replay(args.path, strict_contract=not args.allow_contract_mismatch)
        problems = validate_replay(episode, strict_contract=not args.allow_contract_mismatch)
        print(json.dumps(episode.summary(), indent=2, default=str))
        if args.timeline:
            print(json.dumps(episode.timeline(), indent=2, default=str))
        if problems:
            print(json.dumps({"problems": problems}, indent=2), file=sys.stderr)
            return 1
        return 0
    if args.command == "curriculum":
        from .curriculum_stages import describe_progression, format_progression
        if args.json:
            print(json.dumps(describe_progression(), indent=2, default=str))
        else:
            print(format_progression(), end="")
        return 0
    if args.command == "adapter-contract":
        from .external_adapter import (
            AdapterContractChecker,
            MockExternalEnvironment,
            contract_summary,
        )
        print(json.dumps(contract_summary(), indent=2, default=str))
        if args.check_mock:
            problems = AdapterContractChecker(MockExternalEnvironment()).run()
            print(json.dumps({"mock_adapter_problems": problems}, indent=2))
            return 1 if problems else 0
        return 0
    if args.command == "validate-runtime":
        from .runtime_validation import RuntimeValidator, format_validation_report
        validator = RuntimeValidator(
            project_path=args.project_path,
            godot_executable=args.godot_executable,
            timeout=args.timeout,
        )
        report = validator.validate(env_count=args.env_count)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(format_validation_report(report))
        return 0 if report.status in {"passed", "unavailable"} else 1
    if args.command == "compare-experiments":
        from .experiment import compare_experiments, format_experiment_report
        base_data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        cand_data = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        res = compare_experiments(base_data, cand_data, threshold=args.threshold)
        if args.json:
            print(json.dumps(res, indent=2, default=str))
        else:
            print(format_experiment_report(cand_data, comparison=res))
        return 0 if res["status"] != "regression_warning" else 1
    if args.command == "summarize-experiment":
        from .experiment import aggregate_seed_runs, format_experiment_report
        target_dir = Path(args.path)
        summary_files = list(target_dir.glob("**/run_summary.json")) or list(target_dir.glob("*.json"))
        run_dicts = [json.loads(p.read_text(encoding="utf-8")) for p in summary_files]
        agg = aggregate_seed_runs(run_dicts)
        if args.json:
            print(json.dumps(agg, indent=2, default=str))
        else:
            print(format_experiment_report(agg))
        return 0
    raise RuntimeError(f"unhandled command {args.command}")


if __name__ == "__main__":
    sys.exit(main())

"""Command-line workflow for recording, BC, PPO, evaluation, benchmarks and smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # noqa: F401  (kept as the documented mock seam for CLI launch tests)
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    BCConfig,
    TrainingConfig,
    find_godot_executable,
    load_godot_executable_setting,
    save_godot_executable_setting,
)
from .contract import GODOT_VERSION
from .wsl import GodotLaunchError, WindowsInterop, is_windows_shell, normalize_host_path


def _env_workers_argument(value: str) -> int:
    """``--env-workers N|auto`` -> the TrainingConfig integer (0 = auto)."""
    text = str(value).strip().lower()
    if text == "auto":
        return 0
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--env-workers expects a positive integer or 'auto', got {value!r}"
        ) from exc
    if number < 1:
        raise argparse.ArgumentTypeError("--env-workers must be >= 1 (or 'auto')")
    return number


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON TrainingConfig file")
    parser.add_argument("--env-count", type=int, dest="environment_count")
    parser.add_argument(
        "--env-workers",
        type=_env_workers_argument,
        dest="env_workers",
        help="Godot processes hosting the environments: 1 = single process (default), "
        "N = shard the environments over N concurrently simulating processes, "
        "'auto' = size the pool from the host CPU. Sharding preserves per-environment "
        "seeds and therefore results; only wall time changes.",
    )
    parser.add_argument("--enemy-count", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--rollout-length",
        type=int,
        help="per-environment PPO horizon; 0 (default) keeps the aggregate rollout near 16k as env-count changes",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--torch-threads", type=int, help="PyTorch CPU threads; 0 = bounded auto")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--steps", type=int, dest="total_training_steps")
    parser.add_argument("--checkpoint-frequency", type=int)
    parser.add_argument("--evaluation-frequency", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument(
        "--eval-env-count",
        type=int,
        dest="evaluation_environment_count",
        help="bridge environments for exact plan-scheduled normal evaluation (default 8)",
    )
    parser.add_argument(
        "--checkpoint-eval-env-count",
        type=int,
        dest="checkpoint_eval_environment_count",
        help="bridge environments for the checkpoint battery (planned episodes are "
        "result-invariant under parallelism, so this is a pure speed knob)",
    )
    parser.add_argument(
        "--inference-device",
        choices=["auto", "cpu", "cuda"],
        help="device for rollout/evaluation inference while PPO updates stay on --device "
        "(cpu removes the per-step host<->device round trip that makes CUDA slower "
        "than CPU for this tiny policy)",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        default=None,
        help="evaluation-summary key that selects best_eval.zip (default "
        "mean_episode_reward); dotted paths reach mirrored report sections, "
        "e.g. condition_evaluation.mean_win_rate",
    )
    parser.add_argument(
        "--checkpoint-selection-goal",
        choices=["max", "min"],
        default=None,
        help="whether the selection metric is maximised (default) or minimised",
    )
    parser.add_argument(
        "--checkpoint-selection-min-delta",
        type=float,
        default=None,
        help="minimum improvement that replaces best_eval.zip (default 0.0 = any "
        "strict improvement)",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--curriculum-level", type=int)
    parser.add_argument("--godot-executable", default=None)
    parser.add_argument("--project-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument(
        "--profile-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write logs/training_profile.json with PPO/bridge/callback wall-time breakdowns",
    )
    parser.add_argument(
        "--compact-training-infos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="omit redundant non-terminal metrics/reward components from bridge step responses",
    )
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
    # Cooperative process boundary used by the Godot Control Center. These
    # stay optional: an ordinary terminal run does not create or poll files.
    parser.add_argument("--control-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--status-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--event-log-file", default=None, help=argparse.SUPPRESS)


def _resolve_project_path(project_path: str | None) -> Path:
    if project_path:
        # WSL users may hand over the Windows form (C:\...) of the path.
        return Path(normalize_host_path(project_path)).expanduser().resolve()
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
    predictably regardless of the Godot process working directory. Paths are
    handed to a Windows Godot binary from WSL in Windows form.
    """
    executable = find_godot_executable(godot_executable)
    project = _resolve_project_path(project_path)
    output_path = Path(normalize_host_path(output)).expanduser().resolve()
    interop = WindowsInterop(executable)
    return [
        executable,
        "--path",
        interop.windows_path(project),
        "--script",
        "res://scripts/recording/record_demo.gd",
        "--",
        "--output",
        interop.windows_path(output_path),
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
        WindowsInterop(executable).windows_path(project),
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


def _call_godot_process(command: list[str]) -> int:
    """Launch a graphical Godot tool and wait for it, WSL/Windows aware.

    On WSL with a Windows Godot binary the paths in `command` are already in
    Windows form (the builders convert them); the wrapper retries a refused
    direct .exe launch through ``cmd.exe /C call``. Any launch failure is a
    clean CLI error (exit code 1), matching the previous OSError behavior.
    """
    try:
        return WindowsInterop(command[0]).call(command)
    except GodotLaunchError as exc:
        # Already carries the executable name and all attempted launches.
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(
            f"Could not launch Godot executable {command[0]!r}: {exc}. "
            "Install Godot 4.7.2 and put it on PATH or pass --godot-executable.",
            file=sys.stderr,
        )
        return 1


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    values: dict[str, Any] = {}
    if args.config:
        values = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    fields = set(TrainingConfig.__dataclass_fields__)
    for key, value in vars(args).items():
        if key in fields and value is not None:
            values[key] = value
    # A Godot executable configured once (e.g. via `validate-runtime
    # --godot-executable ...`) is remembered in .sandboxai/settings.json.
    # When neither the CLI flag nor a --config file names one, seed it into
    # the training config explicitly so GodotVecEnv/GodotProcessTransport
    # receive the configured executable and the run's config.json records
    # the binary that was actually used.
    if "godot_executable" not in values:
        remembered = load_godot_executable_setting()
        if remembered:
            values["godot_executable"] = remembered
    return TrainingConfig.from_dict(values)


def _remember_godot_executable(args: argparse.Namespace) -> None:
    """Remember an explicitly-passed --godot-executable for future runs.

    Every Godot-touching command resolves its binary through
    ``config.find_godot_executable``, which consults the remembered setting,
    so configuring the executable once (for validation, recording, a
    benchmark, ...) makes it available to every later command — training
    included — without repeating the flag. Only verified-resolvable
    executables are remembered (typos never poison later runs), and the
    documented default (``godot`` on PATH) is never persisted.
    """
    executable = getattr(args, "godot_executable", None)
    if not executable or executable == "godot":
        return
    if is_windows_shell(executable):
        # Passing cmd.exe was a workaround for the WSL launch problems the
        # interop layer now handles; remembering it would poison later runs.
        return
    if not (shutil.which(executable) or Path(executable).expanduser().is_file()):
        return
    previous = load_godot_executable_setting()
    try:
        saved = save_godot_executable_setting(executable)
    except OSError:
        return  # persistence is best-effort; this command still runs
    if saved is None:
        return
    remembered = load_godot_executable_setting()
    if remembered and remembered != previous:
        print(
            f"Remembered Godot executable for future runs: {remembered} ({saved})",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sandboxai", description="SandboxAI local Godot + PyTorch research workflow"
    )
    # Both numbers appear in every bug report worth acting on: the package
    # version and the engine the observation/action contract is pinned to.
    parser.add_argument(
        "--version",
        action="version",
        version=f"sandboxai {__version__} (Godot {GODOT_VERSION})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser(
        "install", help="print the recommended local dependency installation command"
    )
    install.add_argument(
        "--cuda", action="store_true", help="also print the CUDA PyTorch index example"
    )

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
    record.add_argument(
        "--duration", type=float, default=0.0, help="seconds; 0 means until the window closes"
    )
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

    desktop = sub.add_parser(
        "control-center-desktop",
        help="open the real Python desktop Control Center backed by SandboxAIAdapter",
    )
    desktop.add_argument(
        "--project-path",
        default=None,
        help="project root the adapter reads/launches training from (default: this "
        "editable install's repository root)",
    )
    desktop.add_argument(
        "--output-root",
        default="training",
        help="run/benchmark/evaluation output directory, relative to --project-path unless absolute "
        "(default: training)",
    )

    bc = sub.add_parser("bc-train", help="train a PyTorch behavior-cloning policy")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--epochs", type=int, default=25)
    bc.add_argument("--batch-size", type=int, default=512)
    bc.add_argument("--learning-rate", type=float, default=3e-4)
    bc.add_argument("--validation-fraction", type=float, default=0.1)
    bc.add_argument(
        "--split-strategy",
        choices=["auto", "episode", "transition"],
        default="auto",
        help="auto = split by episode when the dataset has episode structure (default, "
        "leakage-free); episode = require it; transition = the historical shuffle, "
        "which leaks adjacent states between train and validation",
    )
    bc.add_argument("--seed", type=int, default=1234)
    bc.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bc.add_argument("--output-dir", default="training/bc_runs/latest")
    bc.add_argument("--resume-checkpoint")
    bc.add_argument("--control-file", default=None, help=argparse.SUPPRESS)
    bc.add_argument("--status-file", default=None, help=argparse.SUPPRESS)
    bc.add_argument("--event-log-file", default=None, help=argparse.SUPPRESS)

    inspect = sub.add_parser("inspect-dataset", help="validate and summarise a demonstration JSONL")
    inspect.add_argument("--dataset", required=True)
    inspect.add_argument(
        "--statistics",
        action="store_true",
        help="full dataset report: episode structure, action histograms, duplicates, "
        "observation range violations and boundary problems",
    )

    inspect_runs = sub.add_parser(
        "inspect-runs",
        help="read-only inspection of training runs on disk (state, progress, "
        "checkpoints, evaluations, provenance)",
    )
    inspect_runs.add_argument(
        "--root",
        default="training",
        help="training output root, its runs/ directory, or a single run directory "
        "(default: training)",
    )
    inspect_runs.add_argument(
        "--run",
        default=None,
        help="inspect exactly one run directory and print its full report",
    )
    inspect_runs.add_argument(
        "--limit",
        type=int,
        default=0,
        help="show only the newest N runs (0 = all)",
    )
    inspect_runs.add_argument(
        "--events",
        type=int,
        default=0,
        help="include the last N telemetry/control events per run",
    )
    inspect_runs.add_argument("--json", action="store_true", help="emit the raw JSON report")

    benchmark = sub.add_parser("benchmark", help="measure headless Godot throughput")
    benchmark.add_argument("--env-counts", default="1,2,4,8,16,24,32,48,64")
    benchmark.add_argument(
        "--worker-counts",
        default="1",
        help="comma-separated Godot worker-process counts to sweep per environment count "
        "(e.g. 1,2,4,8); 1 is the single-process baseline. Worker counts above the "
        "environment count are clamped.",
    )
    benchmark.add_argument("--steps", type=int, default=2000)
    benchmark.add_argument("--enemy-count", type=int, default=1)
    benchmark.add_argument("--seed", type=int, default=1234)
    benchmark.add_argument("--curriculum-level", type=int, default=3)
    benchmark.add_argument("--godot-executable", default="godot")
    benchmark.add_argument("--project-path", default="")
    benchmark.add_argument("--output-dir", default="training/benchmarks/latest")
    benchmark.add_argument("--max-seconds-per-config", type=float, default=20.0)
    benchmark.add_argument(
        "--full-infos",
        action="store_true",
        help="benchmark diagnostic-heavy wire responses instead of PPO's compact training path",
    )

    benchmark_suites = sub.add_parser(
        "benchmark-suites",
        help="run the five comparable benchmark suites at 1/4/8/16/32/64 environments",
    )
    benchmark_suites.add_argument("--godot-executable", default="godot")
    benchmark_suites.add_argument("--project-path", default="")
    benchmark_suites.add_argument("--output-dir", default="training/benchmarks/suites")
    benchmark_suites.add_argument(
        "--plan-only",
        action="store_true",
        help="print the plan without measuring anything (no Godot required)",
    )

    hardware = sub.add_parser(
        "hardware-wizard",
        help="measure CPU/Hybrid/CUDA throughput and persist a hardware profile",
    )
    hardware.add_argument("--project-path", default="")
    hardware.add_argument("--godot-executable", default=None)
    hardware.add_argument(
        "--steps",
        type=int,
        default=None,
        help="steps to time per device candidate (default: the wizard's 5,000-step budget)",
    )
    hardware.add_argument(
        "--no-save",
        action="store_true",
        help="print the measured profile without persisting it to .sandboxai/",
    )
    hardware.add_argument(
        "--show",
        action="store_true",
        help="print the persisted profile without measuring anything",
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

    weapons = sub.add_parser(
        "weapon-table",
        help="print the weapon TTK / role matrix parsed from the Godot weapon tables",
    )
    weapons.add_argument("--json", action="store_true")
    weapons.add_argument(
        "--distances",
        default="2,5,8,11,14",
        help="comma-separated engagement distances in metres",
    )
    weapons.add_argument(
        "--health", type=float, default=100.0, help="target health used for the TTK maths"
    )

    ttk = sub.add_parser(
        "ttk-report",
        help="validate manually annotated TTK trials and compare them with the simulator",
    )
    ttk.add_argument("--trials", required=True, help="JSONL file of annotated TTK trials")
    ttk.add_argument("--json", action="store_true")
    ttk.add_argument(
        "--by-condition",
        action="store_true",
        help="per weapon/distance-band/movement/hit-zone aggregates instead of the global summary",
    )
    ttk.add_argument(
        "--compare-simulator",
        action="store_true",
        help="compare measured human TTK against the analytic TTK parsed from WeaponState",
    )
    ttk.add_argument("--distance-bucket", type=float, default=3.0)
    ttk.add_argument(
        "--seed", type=int, default=1234, help="bootstrap seed (reports are reproducible)"
    )

    adapter = sub.add_parser(
        "adapter-contract", help="print the external-game adapter contract (Roblox boundary)"
    )
    adapter.add_argument(
        "--check-mock", action="store_true", help="run the mock adapter contract check"
    )

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
    compare_exp.add_argument(
        "--threshold", type=float, default=0.05, help="regression threshold (default 0.05)"
    )
    compare_exp.add_argument("--json", action="store_true")

    summarize_exp = sub.add_parser(
        "summarize-experiment",
        help="summarize multi-seed experiment runs in a directory",
    )
    summarize_exp.add_argument(
        "--path", required=True, help="directory containing seed run summaries"
    )
    summarize_exp.add_argument("--json", action="store_true")

    smoke = sub.add_parser(
        "smoke-test", help="run end-to-end sanity verification of the Python & ML stack"
    )
    smoke.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])

    return parser


def run_smoke_test(device: str = "cpu") -> dict[str, Any]:
    """Runs a fast, self-contained smoke test verifying all Python ML components."""
    print("Running SandboxAI ML stack smoke test...")
    results: dict[str, Any] = {}
    failures: list[str] = []

    # 1. Config test
    TrainingConfig(
        environment_count=2,
        rollout_length=64,
        batch_size=32,
        total_training_steps=128,
        device=device,
    ).validate()
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
        done = i == 19
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
            failures.append(
                f"observation_dim {summary['observation_dim']} != contract {OBSERVATION_FIELD_COUNT}"
            )

        # 3. BC training
        from .bc import load_bc_checkpoint, load_bc_into_sb3_policy, train_behavior_cloning

        bc_config = BCConfig(epochs=2, batch_size=4, device=device, output_root=tmp_dir)
        bc_result = train_behavior_cloning(
            dataset_path, bc_config, output_dir=Path(tmp_dir) / "bc_out"
        )
        results["bc_best_checkpoint"] = bc_result["best_checkpoint"]

        # 4. BC checkpoint load & predict
        model = load_bc_checkpoint(bc_result["best_checkpoint"], device=device)
        sample_pred = model.predict([0.0] * OBSERVATION_FIELD_COUNT)
        results["bc_prediction_shape"] = list(sample_pred.shape)
        expected_action_dim = [len(ACTION_NVEC)]
        if list(sample_pred.shape) != expected_action_dim:
            failures.append(
                f"BC prediction shape {list(sample_pred.shape)} != {expected_action_dim}"
            )

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

            ppo_model = PPO(
                "MlpPolicy",
                MockGymEnv(),
                n_steps=16,
                batch_size=16,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                device=device,
            )
            transfer_res = load_bc_into_sb3_policy(
                ppo_model.policy, bc_result["best_checkpoint"], device=device
            )
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


# --- subcommand handlers -------------------------------------------------
#
# One function per subcommand, dispatched through _COMMANDS below. `main`
# used to be a 300-line chain of `if args.command == ...` blocks with a
# cyclomatic complexity of 48, which meant no single command could be read,
# tested or changed in isolation. Each handler takes the parsed namespace and
# returns the process exit code. Heavy imports stay inside the handlers so
# that `sandboxai --help` and the numpy-only commands never pay for torch.


def _cmd_install(args: argparse.Namespace) -> int:
    print("python -m pip install -e '.[training]'")
    if args.cuda:
        print(
            "For CUDA, install the matching PyTorch wheel from "
            "https://pytorch.org/ before the command above."
        )
    print(
        f"Godot {GODOT_VERSION} must be installed separately and available as "
        "'godot' (or pass --godot-executable)."
    )
    return 0


def _cmd_control_center_desktop(args: argparse.Namespace) -> int:
    from .control_center_desktop import main as desktop_main

    return desktop_main(project_root=args.project_path or None, output_root=args.output_root)


def _cmd_smoke_test(args: argparse.Namespace) -> int:
    res = run_smoke_test(args.device)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("all_passed") else 1


def _cmd_inspect_dataset(args: argparse.Namespace) -> int:
    from .dataset import DemonstrationDataset

    dataset = DemonstrationDataset.load(args.dataset)
    report = dataset.statistics() if args.statistics else dataset.summary()
    print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_inspect_runs(args: argparse.Namespace) -> int:
    from .run_inspection import format_run_index, format_run_report, inspect_run
    from .run_inspection import inspect_runs as inspect_runs_index

    if args.run:
        report = inspect_run(args.run, event_limit=max(0, args.events))
        print(json.dumps(report, indent=2, default=str) if args.json else format_run_report(report))
        return 0 if report.get("exists") else 1
    index = inspect_runs_index(args.root, limit=max(0, args.limit), event_limit=max(0, args.events))
    print(json.dumps(index, indent=2, default=str) if args.json else format_run_index(index))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    try:
        command = build_record_command(
            args.godot_executable,
            args.project_path,
            args.output,
            args.duration,
            args.enemy_count,
        )
    except ValueError as exc:
        print(f"Invalid Godot executable: {exc}", file=sys.stderr)
        return 1
    print("Launching Godot demonstration recorder:", " ".join(command))
    return _call_godot_process(command)


def _cmd_control_center(args: argparse.Namespace) -> int:
    try:
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
    except ValueError as exc:
        print(f"Invalid Godot executable: {exc}", file=sys.stderr)
        return 1
    print("Launching SandboxAI Control Center:", " ".join(command))
    return _call_godot_process(command)


def _run_under_control(control: Any, work: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Runs `work`, reporting success or failure to an optional RunControl.

    Both training entry points need the same three-way handshake (start
    already done by the caller, finish on success, fail on exception) and
    getting it wrong leaks a run that the Control Center shows as forever
    "running".
    """
    try:
        result = work()
    except Exception as exc:
        if control is not None:
            control.fail(exc)
        raise
    if control is not None:
        control.finish(**result)
    return result


def _cmd_bc_train(args: argparse.Namespace) -> int:
    from .bc import train_behavior_cloning
    from .run_control import from_cli_paths

    control = from_cli_paths(args.control_file, args.status_file, args.event_log_file)
    config = BCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        output_root=str(Path(args.output_dir).parent),
        split_strategy=args.split_strategy,
    )
    if control is not None:
        control.start(training_type="behavior_cloning", total_epochs=config.epochs)
    result = _run_under_control(
        control,
        lambda: train_behavior_cloning(
            args.dataset,
            config,
            args.output_dir,
            args.resume_checkpoint,
            run_control=control,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Handles both `train` and `resume`; they differ only in the checkpoint."""
    from .ppo import train_ppo
    from .run_control import from_cli_paths

    control = from_cli_paths(args.control_file, args.status_file, args.event_log_file)
    config = _config_from_args(args)
    if control is not None:
        control.start(
            training_type="ppo",
            total_training_steps=config.total_training_steps,
            environment_count=config.environment_count,
        )
    result = _run_under_control(
        control,
        lambda: train_ppo(
            config,
            args.checkpoint if args.command == "resume" else None,
            run_control=control,
        ),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from stable_baselines3 import PPO  # type: ignore

    from .config import TrainingConfig
    from .evaluation import evaluate_model, format_summary

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
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "episodes_detail"},
            indent=2,
            default=str,
        )
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import benchmark_simulation, summarize_scaling

    project = _resolve_project_path(args.project_path)
    counts = [int(value) for value in args.env_counts.split(",") if value.strip()]
    workers = [int(value) for value in str(args.worker_counts).split(",") if value.strip()]
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
        worker_counts=workers or (1,),
        compact_infos=not args.full_infos,
    )
    print(json.dumps(result, indent=2, default=str))
    print(json.dumps({"scaling_summary": summarize_scaling(result)}, indent=2, default=str))
    return 0


def _cmd_benchmark_suites(args: argparse.Namespace) -> int:
    from .benchmark_suites import describe_plan, format_report, run_suites

    if args.plan_only:
        print(json.dumps(describe_plan(), indent=2, default=str))
        return 0
    project = _resolve_project_path(args.project_path)
    report = run_suites(project, args.godot_executable, output_dir=args.output_dir)
    print(format_report(report))
    return 0


def _cmd_hardware_wizard(args: argparse.Namespace) -> int:
    from .hardware_profile import (
        DEFAULT_MEASUREMENT_STEPS,
        load_profile,
        run_hardware_wizard,
    )

    if args.show:
        profile = load_profile()
        if profile is None:
            print(json.dumps({"available": False, "note": "no hardware profile persisted yet"}))
            return 1
        print(json.dumps(profile.to_dict(), indent=2, default=str))
        return 0

    project = _resolve_project_path(args.project_path)
    steps = args.steps if args.steps is not None else DEFAULT_MEASUREMENT_STEPS

    def _on_progress(measurement: Any) -> None:
        status = measurement.status
        rate = (
            f"{measurement.steps_per_second:.1f} steps/s"
            if measurement.steps_per_second is not None
            else status
        )
        print(f"  {measurement.label}: {rate}", file=sys.stderr)

    profile = run_hardware_wizard(
        project_path=project,
        godot_executable=getattr(args, "godot_executable", None),
        steps=steps,
        on_progress=_on_progress,
        save=not args.no_save,
    )
    print(json.dumps(profile.to_dict(), indent=2, default=str))
    return 0 if not profile.fallback else 1


def _cmd_replay(args: argparse.Namespace) -> int:
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


def _cmd_curriculum(args: argparse.Namespace) -> int:
    from .curriculum_stages import describe_progression, format_progression

    if args.json:
        print(json.dumps(describe_progression(), indent=2, default=str))
    else:
        print(format_progression(), end="")
    return 0


def _cmd_ttk_report(args: argparse.Namespace) -> int:
    from .ttk import TTKDataset, compare_with_simulator, format_summary

    dataset = TTKDataset.load(args.trials)
    payload: dict[str, Any] = {"summary": dataset.summary(seed=args.seed)}
    if args.by_condition:
        payload["by_condition"] = dataset.by_condition(args.distance_bucket, seed=args.seed)
    if args.compare_simulator:
        payload["simulator_comparison"] = compare_with_simulator(dataset, seed=args.seed)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(format_summary(payload["summary"]))
    for key in ("by_condition", "simulator_comparison"):
        if key in payload:
            print(json.dumps(payload[key], indent=2, default=str))
    return 0


def _cmd_weapon_table(args: argparse.Namespace) -> int:
    from .weapons import format_ttk_table, role_ranking, ttk_table

    distances = [float(v) for v in str(args.distances).split(",") if v.strip()]
    table = ttk_table(distances, target_health=args.health)
    ranking = role_ranking(target_health=args.health)
    if args.json:
        payload = dict(table)
        payload["role_ranking"] = {
            band: [{"profile": name, "ttk": value} for name, value in entries]
            for band, entries in ranking.items()
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(format_ttk_table(table))
    print()
    print("Best profile per engagement band (by ideal TTK):")
    for band, entries in ranking.items():
        ranked = ", ".join(
            f"{name} {value:.2f}s" for name, value in entries if value != float("inf")
        )
        print(f"  {band:<12} {ranked or 'nothing reaches this band'}")
    return 0


def _cmd_adapter_contract(args: argparse.Namespace) -> int:
    from .external_adapter import (
        AdapterContractChecker,
        MockExternalEnvironment,
        contract_summary,
    )

    print(json.dumps(contract_summary(), indent=2, default=str))
    if not args.check_mock:
        return 0
    problems = AdapterContractChecker(MockExternalEnvironment()).run()
    print(json.dumps({"mock_adapter_problems": problems}, indent=2))
    return 1 if problems else 0


def _cmd_validate_runtime(args: argparse.Namespace) -> int:
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


def _cmd_compare_experiments(args: argparse.Namespace) -> int:
    from .experiment import compare_experiments, format_experiment_report

    base_data = json.loads(Path(args.baseline).read_text(encoding="utf-8-sig"))
    cand_data = json.loads(Path(args.candidate).read_text(encoding="utf-8-sig"))
    res = compare_experiments(base_data, cand_data, threshold=args.threshold)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(format_experiment_report(cand_data, comparison=res))
    return 0 if res["status"] != "regression_warning" else 1


def _cmd_summarize_experiment(args: argparse.Namespace) -> int:
    from .experiment import aggregate_seed_runs, format_experiment_report

    target_dir = Path(args.path)
    summary_files = list(target_dir.glob("**/run_summary.json")) or list(target_dir.glob("*.json"))
    run_dicts = [json.loads(p.read_text(encoding="utf-8-sig")) for p in summary_files]
    agg = aggregate_seed_runs(run_dicts)
    if args.json:
        print(json.dumps(agg, indent=2, default=str))
    else:
        print(format_experiment_report(agg))
    return 0


# Subcommand name -> handler. build_parser() is the only other place that
# knows these names; test_cli.py asserts the two stay in sync, so a parser
# entry without a handler fails the suite instead of reaching a user as an
# "unhandled command" crash.
_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "install": _cmd_install,
    "control-center-desktop": _cmd_control_center_desktop,
    "smoke-test": _cmd_smoke_test,
    "inspect-dataset": _cmd_inspect_dataset,
    "inspect-runs": _cmd_inspect_runs,
    "record": _cmd_record,
    "control-center": _cmd_control_center,
    "bc-train": _cmd_bc_train,
    "train": _cmd_train,
    "resume": _cmd_train,
    "evaluate": _cmd_evaluate,
    "benchmark": _cmd_benchmark,
    "benchmark-suites": _cmd_benchmark_suites,
    "hardware-wizard": _cmd_hardware_wizard,
    "replay": _cmd_replay,
    "curriculum": _cmd_curriculum,
    "ttk-report": _cmd_ttk_report,
    "weapon-table": _cmd_weapon_table,
    "adapter-contract": _cmd_adapter_contract,
    "validate-runtime": _cmd_validate_runtime,
    "compare-experiments": _cmd_compare_experiments,
    "summarize-experiment": _cmd_summarize_experiment,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # A verified --godot-executable is remembered so later commands (train
    # in particular) can use the configured binary without repeating the
    # flag. Runs before any dispatch: every subcommand accepts the option.
    _remember_godot_executable(args)
    try:
        handler = _COMMANDS[args.command]
    except KeyError:
        raise RuntimeError(f"unhandled command {args.command}") from None
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

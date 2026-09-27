"""One coherent record -> BC -> sandbox -> RL -> comparison workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from bc.sandbox_runner import run_bc_in_sandbox
from bc.train import train_bc
from data_pipeline.catalog import build_catalog
from data_pipeline.recorder import SessionRecorder
from evaluation.benchmark import run_benchmark
from evaluation.imitation import evaluate_imitation
from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry
from rl.train_ppo import train_ppo_sandbox


def run_end_to_end(
    data_dir: Union[str, Path] = "datasets",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    logs_dir: Union[str, Path] = "logs",
    generate_demo: bool = False,
    demo_duration: float = 1.0,
    bc_epochs: int = 3,
    rl_timesteps: int = 4096,
    eval_episodes: int = 5,
    eval_steps: int = 250,
    device: str = "auto",
    env_path: Optional[str] = None,
    backend: str = "python",
    quick: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute every pipeline stage and persist a machine-readable summary.

    Synthetic generation exists for installation verification only.  Real use
    should point ``data_dir`` at manually recorded human sessions.
    """
    data_root, checkpoints, logs = Path(data_dir), Path(checkpoint_dir), Path(logs_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    experiments = logs / "experiments"
    state_file = logs / "system_state.json"
    tracker = ExperimentTracker(
        "end_to_end",
        {
            "data_dir": str(data_root),
            "checkpoint_dir": str(checkpoints),
            "generate_demo": generate_demo,
            "bc_epochs": bc_epochs,
            "rl_timesteps": rl_timesteps,
            "backend": backend,
            "quick": quick,
            "seed": seed,
        },
        experiments,
    )
    telemetry = SystemTelemetry(state_file)
    summary: Dict[str, Any] = {"run_id": tracker.run_id, "stages": {}}
    try:
        existing = list(data_root.rglob("metadata.json"))
        if generate_demo:
            telemetry.update_stage("RECORDING")
            duration = 0.45 if quick else demo_duration
            for index in range(2):
                recorder = SessionRecorder(
                    output_dir=data_root,
                    session_id=f"synthetic_demo_{tracker.run_id}_{index}",
                    source_name="mock_synthetic",
                    target_fps=15,
                    frame_width=84 if quick else 160,
                    frame_height=84 if quick else 120,
                    is_mock=True,
                    telemetry_state_file=state_file,
                )
                recorder.record(max_duration=duration)
            summary["stages"]["record"] = {
                "mode": "synthetic_installation_check",
                "sessions_created": 2,
            }
        elif existing:
            summary["stages"]["record"] = {
                "mode": "existing_manual_data",
                "sessions_found": len(existing),
            }
        else:
            raise ValueError(
                "No dataset sessions found. Record manual gameplay first, or use "
                "--generate_demo only for an installation smoke test."
            )

        telemetry.update_stage("VALIDATING")
        catalog_path = logs / "dataset_catalog.json"
        catalog = build_catalog(data_root, catalog_path)
        summary["stages"]["dataset"] = catalog["summary"]
        if not catalog["sessions"]:
            raise ValueError("No sessions are available; record manual gameplay first")
        if catalog["summary"]["invalid_sessions"]:
            raise ValueError(
                f"Dataset catalog contains {catalog['summary']['invalid_sessions']} invalid session(s)"
            )
        tracker.add_artifact(catalog_path, "dataset_catalog")

        bc_result = train_bc(
            data_dir=data_root,
            checkpoint_dir=checkpoints,
            epochs=1 if quick else bc_epochs,
            batch_size=8 if quick else 32,
            target_h=84 if quick else 120,
            target_w=84 if quick else 160,
            seq_len=2 if quick else 4,
            use_gru=True,
            device_str="cpu" if quick else (None if device == "auto" else device),
            seed=seed,
            augment=not quick,
            latent_dim=64 if quick else 256,
            channel_scales=(8, 16, 16) if quick else (16, 32, 32),
            experiment_root=experiments,
            state_file=state_file,
        )
        summary["stages"]["bc_train"] = bc_result
        bc_checkpoint = bc_result["checkpoint_best"]

        held_out_range = bc_result["validation_ranges"][0]
        held_out = held_out_range["path"]
        imitation_path = logs / "imitation_evaluation.json"
        imitation = evaluate_imitation(
            checkpoint_path=bc_checkpoint,
            data_path=held_out,
            device="cpu" if quick else ("cpu" if device == "auto" else device),
            start_step=held_out_range["start_step"],
            end_step=held_out_range["end_step"],
            context_start_step=held_out_range["context_start_step"],
            output_report=imitation_path,
            experiment_root=experiments,
        )
        summary["stages"]["imitation_evaluation"] = imitation

        closed_loop_path = logs / "bc_closed_loop.json"
        closed_loop = run_bc_in_sandbox(
            bc_checkpoint,
            num_episodes=2 if quick else eval_episodes,
            max_steps_per_episode=40 if quick else eval_steps,
            env_path=env_path,
            backend=backend,
            output_report=closed_loop_path,
            experiment_root=experiments,
            state_file=state_file,
        )
        summary["stages"]["bc_closed_loop"] = closed_loop

        rl_result = train_ppo_sandbox(
            timesteps=128 if quick else rl_timesteps,
            checkpoint_dir=checkpoints,
            env_path=env_path,
            n_envs=1 if quick else 2,
            learning_rate=3e-4,
            n_steps=64 if quick else 128,
            batch_size=32 if quick else 64,
            seed=seed,
            device="cpu" if quick else device,
            bc_checkpoint=bc_checkpoint,
            backend=backend,
            checkpoint_freq=0 if quick else max(1000, rl_timesteps // 4),
            max_episode_steps=80 if quick else 500,
            experiment_root=experiments,
            state_file=state_file,
        )
        summary["stages"]["rl_train"] = rl_result

        benchmark_path = logs / "benchmark.json"
        benchmark = run_benchmark(
            bc_checkpoint=bc_checkpoint,
            ppo_checkpoint=rl_result["checkpoint_path"],
            num_episodes=2 if quick else eval_episodes,
            max_steps_per_episode=40 if quick else eval_steps,
            env_path=env_path,
            output_report=str(benchmark_path),
            backend=backend,
            device="cpu",
            experiment_root=experiments,
            state_file=state_file,
        )
        summary["stages"]["benchmark"] = benchmark
        summary_path = logs / "workflow_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tracker.add_artifact(summary_path, "workflow_summary")
        tracker.finish(summary)
        telemetry.update_stage("IDLE")
        return summary
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (logs / "workflow_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        tracker.fail(exc)
        telemetry.update_stage("FAILED")
        raise

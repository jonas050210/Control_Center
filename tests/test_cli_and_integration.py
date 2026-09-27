"""Comprehensive CLI entrypoint and integration tests across SandboxAI."""

import json
from pathlib import Path
import pytest
import torch

from bc.infer import run_inference_on_dataset
from bc.models import BCVisionNetwork
from bc.sandbox_runner import run_bc_sandbox
from data_pipeline.stats import inspect_dataset
from data_pipeline.validate import validate_dataset
from data_pipeline.recorder import SessionRecorder
from evaluation.benchmark import run_benchmark


@pytest.fixture(scope="module")
def integrated_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generates a sample dataset session for CLI/tool testing."""
    root = tmp_path_factory.mktemp("cli_test_root")
    rec = SessionRecorder(
        output_dir=root,
        session_id="integrated_session_01",
        target_fps=15.0,
        frame_width=84,
        frame_height=84,
        image_format="jpg",
        is_mock=True,
    )
    rec.record(max_duration=0.5)
    return root / "integrated_session_01"


def test_validate_and_stats_on_integrated_session(integrated_session: Path) -> None:
    # 1. Validation report
    val_report = validate_dataset(integrated_session)
    assert val_report.is_valid
    assert val_report.total_steps > 0
    assert len(val_report.errors) == 0

    # 2. Stats inspection
    stats = inspect_dataset(integrated_session)
    assert "total_steps" in stats
    assert stats["total_steps"] == val_report.total_steps
    assert "mouse_dx_metrics" in stats
    assert "action_stats" in stats
    assert "storage" in stats


def test_bc_inference_on_dataset(integrated_session: Path, tmp_path: Path) -> None:
    # Create a small BC checkpoint
    model = BCVisionNetwork(in_channels=3, num_bins_x=21, num_bins_y=21, latent_dim=64, channel_scales=(8, 16, 16))
    ckpt_path = tmp_path / "bc_test.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "target_h": 84,
                "target_w": 84,
                "num_bins_x": 21,
                "num_bins_y": 21,
                "latent_dim": 64,
                "channel_scales": (8, 16, 16),
                "use_gru": False,
            },
        },
        ckpt_path,
    )

    preds = run_inference_on_dataset(
        checkpoint_path=ckpt_path,
        dataset_path=integrated_session,
        max_samples=5,
        device="cpu",
    )
    assert len(preds) == 5
    assert "step_idx" in preds[0]
    assert "predicted_move_x" in preds[0]
    assert "ground_truth_move_x" in preds[0]


def test_bc_closed_loop_sandbox_runner(tmp_path: Path) -> None:
    # Create dummy BC checkpoint
    model = BCVisionNetwork(in_channels=3, num_bins_x=21, num_bins_y=21, latent_dim=64, channel_scales=(8, 16, 16))
    ckpt_path = tmp_path / "bc_test_runner.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "target_h": 84,
                "target_w": 84,
                "num_bins_x": 21,
                "num_bins_y": 21,
                "latent_dim": 64,
                "channel_scales": (8, 16, 16),
                "use_gru": False,
            },
        },
        ckpt_path,
    )

    metrics = run_bc_sandbox(
        checkpoint_path=str(ckpt_path),
        num_episodes=2,
        max_steps_per_episode=20,
        device="cpu",
    )
    assert metrics["episodes_completed"] == 2
    assert "mean_reward" in metrics
    assert "total_hits" in metrics
    assert "hit_rate_pct" in metrics

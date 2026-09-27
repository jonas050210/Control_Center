"""Tests for Evaluation and Benchmark Harness (Phase 6)."""

from pathlib import Path

import pytest
import torch

from bc.models import BCVisionNetwork
from evaluation.benchmark import evaluate_policy_on_env, run_benchmark


def test_random_policy_evaluation() -> None:
    res = evaluate_policy_on_env(
        policy_type="random",
        policy_obj=None,
        num_episodes=2,
        max_steps_per_episode=20,
    )
    assert res["policy_name"] == "RANDOM"
    assert res["num_episodes"] == 2
    assert "mean_reward" in res
    assert "hit_rate_pct" in res


def test_full_benchmark_smoke(tmp_path: Path) -> None:
    # Create dummy BC checkpoint
    model = BCVisionNetwork(in_channels=3, num_bins_x=21, num_bins_y=21)
    ckpt_path = tmp_path / "bc_dummy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "target_h": 84,
                "target_w": 84,
                "num_bins_x": 21,
                "num_bins_y": 21,
                "latent_dim": 256,
                "channel_scales": (16, 32, 32),
            },
        },
        ckpt_path,
    )

    report_path = tmp_path / "benchmark_report.json"
    results = run_benchmark(
        bc_checkpoint=str(ckpt_path),
        ppo_checkpoint=None,
        num_episodes=2,
        max_steps_per_episode=15,
        output_report=str(report_path),
    )
    assert "random" in results
    assert "bc" in results
    assert report_path.exists()

    import json
    with open(report_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert "random" in saved_data
    assert "bc" in saved_data


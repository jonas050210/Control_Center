"""Tests for PPO Training in Sandbox (Phase 5)."""

from pathlib import Path

import pytest

from rl.train_ppo import train_ppo_sandbox


def test_ppo_sandbox_train_smoke(tmp_path: Path) -> None:
    results = train_ppo_sandbox(
        timesteps=256,
        checkpoint_dir=tmp_path / "checkpoints",
        n_envs=1,
        learning_rate=1e-3,
        n_steps=128,
        batch_size=64,
        device="cpu",
    )
    assert results["timesteps"] == 256
    assert Path(results["checkpoint_path"]).exists()

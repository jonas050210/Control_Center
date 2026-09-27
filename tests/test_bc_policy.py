"""Tests for BCPolicy inference and action mapping (Phase 2)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bc.models import BCVisionNetwork
from bc.policy import BCPolicy
from data_pipeline.schema import ActionState


def test_bc_policy_predict_from_pil(tmp_path: Path) -> None:
    model = BCVisionNetwork(
        in_channels=3,
        num_bins_x=21,
        num_bins_y=21,
        latent_dim=64,
        channel_scales=(8, 16, 16),
    )
    ckpt_path = tmp_path / "test_model.pt"
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
            },
        },
        ckpt_path,
    )

    policy = BCPolicy.load_from_checkpoint(ckpt_path, device="cpu")
    img = Image.new("RGB", (160, 120), color=(100, 150, 200))

    action_state, raw_dict = policy.predict(img)

    assert isinstance(action_state, ActionState)
    assert action_state.move_x in (-1, 0, 1)
    assert action_state.move_y in (-1, 0, 1)
    assert action_state.fire in (0, 1)
    assert 0 <= action_state.mouse_dx_bin < 21
    assert isinstance(action_state.mouse_dx, float)
    assert "move_x" in raw_dict


def test_bc_policy_maps_noncanonical_dataset_bins_to_environment_contract() -> None:
    model = BCVisionNetwork(
        in_channels=3,
        num_bins_x=11,
        num_bins_y=11,
        latent_dim=32,
        channel_scales=(8, 8, 8),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head_mouse_dx.bias[5] = 1.0
        model.head_mouse_dy.bias[5] = 1.0
    policy = BCPolicy(
        model,
        {
            "target_h": 84,
            "target_w": 84,
            "num_bins_x": 11,
            "num_bins_y": 11,
            "latent_dim": 32,
            "channel_scales": (8, 8, 8),
        },
    )
    env_action, raw = policy.predict_env_action(np.zeros((3, 84, 84), dtype=np.uint8))
    assert tuple(env_action[8:]) == (10, 10)
    assert raw["mouse_dx_bin"] == 5
    assert raw["env_mouse_dx_bin"] == 10


def test_bc_policy_predict_from_numpy(tmp_path: Path) -> None:
    model = BCVisionNetwork(in_channels=3, num_bins_x=21, num_bins_y=21)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"target_h": 84, "target_w": 84, "num_bins_x": 21, "num_bins_y": 21},
        },
        ckpt_path,
    )

    policy = BCPolicy.load_from_checkpoint(ckpt_path, device="cpu")
    arr = np.random.randint(0, 256, (3, 84, 84), dtype=np.uint8)

    action_state, _ = policy.predict(arr)
    assert isinstance(action_state, ActionState)

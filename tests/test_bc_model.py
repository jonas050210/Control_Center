"""Tests for Behavioral Cloning Neural Network Models (Phase 2)."""

import pytest
import torch

from bc.models import BCVisionNetwork


def test_bc_vision_network_single_frame_forward() -> None:
    model = BCVisionNetwork(
        in_channels=3,
        num_bins_x=21,
        num_bins_y=21,
        latent_dim=128,
        use_temporal_gru=False,
        channel_scales=(8, 16, 16),
    )

    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 120, 160)
    logits_dict, hx = model(dummy_input)

    assert hx is None
    assert logits_dict["move_x"].shape == (batch_size, 3)
    assert logits_dict["move_y"].shape == (batch_size, 3)
    assert logits_dict["fire"].shape == (batch_size, 2)
    assert logits_dict["ads"].shape == (batch_size, 2)
    assert logits_dict["jump"].shape == (batch_size, 2)
    assert logits_dict["mouse_dx_bin"].shape == (batch_size, 21)
    assert logits_dict["mouse_dy_bin"].shape == (batch_size, 21)
    assert logits_dict["mouse_continuous"].shape == (batch_size, 2)


def test_bc_vision_network_sequence_gru_forward() -> None:
    model = BCVisionNetwork(
        in_channels=3,
        num_bins_x=21,
        num_bins_y=21,
        latent_dim=64,
        use_temporal_gru=True,
        gru_hidden_dim=64,
        channel_scales=(8, 16, 16),
    )

    batch_size = 2
    seq_len = 4
    dummy_seq = torch.randn(batch_size, seq_len, 3, 84, 84)
    logits_dict, new_hx = model(dummy_seq)

    assert new_hx is not None
    assert new_hx.shape == (1, batch_size, 64)
    assert logits_dict["move_x"].shape == (batch_size, 3)
    assert logits_dict["mouse_dx_bin"].shape == (batch_size, 21)


def test_extract_discrete_actions() -> None:
    logits_dict = {
        "move_x": torch.tensor([[5.0, 0.0, 1.0]]),  # argmax 0 -> move_x = -1
        "move_y": torch.tensor([[0.0, 1.0, 5.0]]),  # argmax 2 -> move_y = +1
        "fire": torch.tensor([[0.0, 3.0]]),  # argmax 1 -> fire = 1
        "jump": torch.tensor([[3.0, 0.0]]),  # argmax 0 -> jump = 0
        "mouse_dx_bin": torch.zeros(1, 21),
    }
    logits_dict["mouse_dx_bin"][0, 14] = 10.0  # argmax 14

    actions = BCVisionNetwork.extract_discrete_actions(logits_dict, deterministic=True)
    assert actions["move_x"][0] == -1
    assert actions["move_y"][0] == 1
    assert actions["fire"][0] == 1
    assert actions["jump"][0] == 0
    assert actions["mouse_dx_bin"][0] == 14

"""Tests for Action Space and Mouse Discretization (M1)."""

import pytest

from data_pipeline.actions import MouseBinner, build_action_state
from data_pipeline.schema import ActionState


def test_mouse_binner_symmetric_log() -> None:
    binner = MouseBinner(num_bins=21, strategy="symmetric_log")
    assert binner.num_bins == 21
    assert len(binner.edges) == 22  # num_bins + 1 edges

    # Center deadzone should discretize to center bin (bin 10)
    assert binner.discretize(0.0) == 10
    assert binner.discretize(0.05) == 10
    assert binner.discretize(-0.05) == 10

    # Symmetric behavior
    pos_idx = binner.discretize(5.0)
    neg_idx = binner.discretize(-5.0)
    # Difference from center (10) should be symmetric
    assert (pos_idx - 10) == (10 - neg_idx)
    assert pos_idx > 10
    assert neg_idx < 10

    # Large values should clamp to outer bins
    assert binner.discretize(99999.0) == 20
    assert binner.discretize(-99999.0) == 0


def test_mouse_binner_dequantize() -> None:
    binner = MouseBinner(num_bins=21, strategy="symmetric_log")
    # Center bin representative value should be 0.0
    center_val = binner.dequantize(10)
    assert center_val == 0.0

    # Dequantized positive and negative should have opposing signs
    pos_val = binner.dequantize(15)
    neg_val = binner.dequantize(5)
    assert pos_val > 0
    assert neg_val < 0
    assert abs(pos_val + neg_val) < 1e-4


def test_mouse_binner_uniform() -> None:
    binner = MouseBinner(num_bins=11, strategy="uniform")
    assert binner.num_bins == 11
    assert binner.discretize(0.0) == 5  # center bin of 11 is 5


def test_mouse_binner_fit_quantile() -> None:
    # Synthetic empirical distribution
    samples = [-50, -20, -10, -5, -2, -1, 0, 0, 0, 1, 2, 5, 10, 20, 50] * 20
    binner = MouseBinner.fit_quantile_edges(samples, num_bins=11)
    assert binner.num_bins == 11
    assert len(binner.edges) == 12
    assert binner.discretize(0.0) >= 0


def test_mouse_binner_invalid_bin_count() -> None:
    with pytest.raises(ValueError):
        MouseBinner(num_bins=10)  # Even number of bins is invalid

    with pytest.raises(ValueError):
        MouseBinner(num_bins=1)  # < 3 is invalid


def test_build_action_state_wasd_and_cancellation() -> None:
    binner = MouseBinner(num_bins=21)

    # W only -> move_y = 1
    act = build_action_state(
        active_keys={"w"},
        mouse_buttons={},
        mouse_dx=0.0,
        mouse_dy=0.0,
        binner_x=binner,
        binner_y=binner,
    )
    assert act.move_y == 1
    assert act.move_x == 0

    # W + S -> cancellation -> move_y = 0
    act_cancel = build_action_state(
        active_keys={"w", "s"},
        mouse_buttons={},
        mouse_dx=0.0,
        mouse_dy=0.0,
        binner_x=binner,
        binner_y=binner,
    )
    assert act_cancel.move_y == 0

    # A + D -> cancellation -> move_x = 0
    act_cancel_x = build_action_state(
        active_keys={"a", "d"},
        mouse_buttons={},
        mouse_dx=0.0,
        mouse_dy=0.0,
        binner_x=binner,
        binner_y=binner,
    )
    assert act_cancel_x.move_x == 0

    # Diagonal forward-right (W + D)
    act_diag = build_action_state(
        active_keys={"w", "d", "shift", "space"},
        mouse_buttons={"left": True, "right": False},
        mouse_dx=12.0,
        mouse_dy=-4.0,
        binner_x=binner,
        binner_y=binner,
    )
    assert act_diag.move_x == 1
    assert act_diag.move_y == 1
    assert act_diag.sprint == 1
    assert act_diag.jump == 1
    assert act_diag.fire == 1
    assert act_diag.ads == 0
    assert act_diag.mouse_dx_bin > 10
    assert act_diag.mouse_dy_bin < 10

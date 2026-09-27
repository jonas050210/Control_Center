"""Tests for Dataset Schema and Serialization (M1)."""

import json
import tempfile
from pathlib import Path

import pytest

from data_pipeline.schema import (
    ACTION_SPACE_SPEC,
    ActionState,
    CaptureConfig,
    DatasetMetadata,
    DatasetSample,
    MouseConfig,
    SCHEMA_VERSION,
)


def test_action_state_defaults_and_serialization() -> None:
    act = ActionState()
    assert act.move_x == 0
    assert act.move_y == 0
    assert act.jump == 0
    assert act.crouch == 0
    assert act.sprint == 0
    assert act.reload == 0
    assert act.fire == 0
    assert act.ads == 0
    assert act.mouse_dx == 0.0
    assert act.mouse_dy == 0.0
    assert act.mouse_dx_bin == 0
    assert act.mouse_dy_bin == 0

    d = act.to_dict()
    assert isinstance(d, dict)
    assert d["move_x"] == 0

    act_restored = ActionState.from_dict(d)
    assert act_restored == act


def test_action_state_custom_values() -> None:
    act = ActionState(
        move_x=1,
        move_y=-1,
        jump=1,
        crouch=0,
        sprint=1,
        reload=0,
        fire=1,
        ads=1,
        mouse_dx=14.5,
        mouse_dy=-3.2,
        mouse_dx_bin=16,
        mouse_dy_bin=7,
        active_keys=["d", "s", "shift", "space"],
        mouse_buttons={"left": True, "right": True, "middle": False},
    )
    d = act.to_dict()
    restored = ActionState.from_dict(d)
    assert restored.move_x == 1
    assert restored.move_y == -1
    assert restored.jump == 1
    assert restored.mouse_dx == 14.5
    assert restored.mouse_dy == -3.2
    assert restored.mouse_dx_bin == 16
    assert restored.active_keys == ["d", "s", "shift", "space"]
    assert restored.mouse_buttons["left"] is True


def test_capture_and_mouse_config_serialization() -> None:
    cap = CaptureConfig(
        target_fps=30.0,
        frame_width=320,
        frame_height=240,
        color_mode="RGB",
        image_format="png",
        roi=[10, 20, 800, 600],
    )
    cap_d = cap.to_dict()
    cap_restored = CaptureConfig.from_dict(cap_d)
    assert cap_restored.target_fps == 30.0
    assert cap_restored.frame_width == 320
    assert cap_restored.image_format == "png"
    assert cap_restored.roi == [10, 20, 800, 600]

    mouse_cfg = MouseConfig(
        sensitivity_scale=1.5,
        binning_strategy="uniform",
        num_bins_x=11,
        num_bins_y=11,
    )
    m_d = mouse_cfg.to_dict()
    m_restored = MouseConfig.from_dict(m_d)
    assert m_restored.sensitivity_scale == 1.5
    assert m_restored.binning_strategy == "uniform"
    assert m_restored.num_bins_x == 11


def test_dataset_metadata_save_and_load(tmp_path: Path) -> None:
    meta = DatasetMetadata(
        session_id="test_session_123",
        source="godot_sandbox",
        capture_config=CaptureConfig(target_fps=15.0, frame_width=160, frame_height=120),
        mouse_config=MouseConfig(num_bins_x=21, num_bins_y=21),
    )
    save_file = tmp_path / "metadata.json"
    meta.save(save_file)

    assert save_file.exists()
    loaded_meta = DatasetMetadata.load(save_file)

    assert loaded_meta.session_id == "test_session_123"
    assert loaded_meta.schema_version == SCHEMA_VERSION
    assert loaded_meta.source == "godot_sandbox"
    assert loaded_meta.capture_config.frame_width == 160
    assert loaded_meta.mouse_config.num_bins_x == 21
    assert "actions" in loaded_meta.action_space


def test_dataset_sample_serialization() -> None:
    sample = DatasetSample(
        step_idx=42,
        timestamp=100.52345,
        iso_timestamp="2026-09-27T12:00:00Z",
        dt=0.0667,
        frame_file="frames/frame_00000042.jpg",
        actions=ActionState(move_x=-1, move_y=1, fire=1, mouse_dx=-5.2, mouse_dx_bin=8),
        is_valid=True,
    )
    d = sample.to_dict()
    json_str = json.dumps(d)
    parsed = json.loads(json_str)

    restored = DatasetSample.from_dict(parsed)
    assert restored.step_idx == 42
    assert restored.timestamp == 100.52345
    assert restored.dt == 0.0667
    assert restored.frame_file == "frames/frame_00000042.jpg"
    assert restored.actions.move_x == -1
    assert restored.actions.move_y == 1
    assert restored.actions.fire == 1
    assert restored.actions.mouse_dx == -5.2
    assert restored.actions.mouse_dx_bin == 8


def test_metadata_partial_and_corrupt_data() -> None:
    empty_meta = DatasetMetadata.from_dict({})
    assert empty_meta.schema_version == SCHEMA_VERSION
    assert empty_meta.session_id == "unknown_session"
    assert empty_meta.capture_config.target_fps == 15.0
    assert empty_meta.mouse_config.num_bins_x == 21


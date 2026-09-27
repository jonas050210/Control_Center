"""Tests for Behavioral Cloning Dataset and DataLoaders (Phase 2)."""

from pathlib import Path
from typing import Any

import pytest
import torch

from bc.dataset import GameplayDataset
from data_pipeline.mock import MockScreenCapture
from data_pipeline.recorder import SessionRecorder


@pytest.fixture(scope="module")
def mock_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates two dummy sessions for dataset testing."""
    root = tmp_path_factory.mktemp("test_bc_data")
    for s_idx in range(2):
        rec = SessionRecorder(
            output_dir=root,
            session_id=f"session_test_{s_idx}",
            target_fps=15.0,
            frame_width=160,
            frame_height=120,
            image_format="jpg",
            is_mock=True,
        )
        rec.record(max_duration=0.3)
    return root


def test_gameplay_dataset_loading(mock_dataset_dir: Path) -> None:
    session_dirs = list(mock_dataset_dir.iterdir())
    ds = GameplayDataset(session_dirs, target_size=(120, 160), seq_len=1)

    assert len(ds) > 0
    obs, targets = ds[0]

    # Check observation shape [3, 120, 160]
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (3, 120, 160)
    assert obs.dtype == torch.float32
    assert 0.0 <= obs.min() <= obs.max() <= 1.0

    # Check target dictionary keys
    assert "move_x" in targets
    assert "move_y" in targets
    assert "fire" in targets
    assert "jump" in targets
    assert "mouse_dx_bin" in targets
    assert "mouse_dy_bin" in targets
    assert "mouse_continuous" in targets

    assert targets["move_x"].dtype == torch.long
    assert 0 <= targets["move_x"].item() <= 2
    assert targets["mouse_dx_bin"].dtype == torch.long


def test_gameplay_dataset_sequence_window(mock_dataset_dir: Path) -> None:
    session_dirs = list(mock_dataset_dir.iterdir())
    ds_seq = GameplayDataset(session_dirs, target_size=(120, 160), seq_len=3)

    assert len(ds_seq) > 0
    obs, targets = ds_seq[0]

    # Shape should be [3, 3, 120, 160] -> [T, C, H, W]
    assert obs.shape == (3, 3, 120, 160)


def test_session_level_train_val_split(mock_dataset_dir: Path) -> None:
    train_ds, val_ds = GameplayDataset.create_train_val_split(
        data_root=mock_dataset_dir,
        val_ratio=0.5,
        target_size=(120, 160),
    )
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(train_ds.sessions_data) >= 1
    assert len(val_ds.sessions_data) >= 1

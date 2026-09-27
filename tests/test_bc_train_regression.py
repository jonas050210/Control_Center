"""Regression and edge-case tests for BC training and policy execution."""

from pathlib import Path
import pytest
import torch

from bc.train import train_bc
from bc.policy import BCPolicy
from data_pipeline.recorder import SessionRecorder


def test_bc_train_resume_missing_file_raises_error(tmp_path: Path) -> None:
    rec = SessionRecorder(
        output_dir=tmp_path,
        session_id="mock_session_1",
        target_fps=15.0,
        frame_width=84,
        frame_height=84,
        image_format="jpg",
        is_mock=True,
    )
    rec.record(max_duration=0.3)

    non_existent_ckpt = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError, match="Checkpoint file to resume from not found"):
        train_bc(
            data_dir=tmp_path / "mock_session_1",
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=4,
            resume_path=str(non_existent_ckpt),
            device_str="cpu",
        )


def test_bc_train_and_resume_cycle(tmp_path: Path) -> None:
    rec = SessionRecorder(
        output_dir=tmp_path,
        session_id="mock_session_cycle",
        target_fps=15.0,
        frame_width=84,
        frame_height=84,
        image_format="jpg",
        is_mock=True,
    )
    rec.record(max_duration=0.3)

    ckpt_dir = tmp_path / "checkpoints"
    # Initial 1 epoch
    report1 = train_bc(
        data_dir=tmp_path / "mock_session_cycle",
        checkpoint_dir=ckpt_dir,
        epochs=1,
        batch_size=4,
        device_str="cpu",
    )
    assert report1["epochs_completed"] == 1
    best_ckpt = ckpt_dir / "bc_best.pt"
    assert best_ckpt.exists()

    # Resume for 1 more epoch
    report2 = train_bc(
        data_dir=tmp_path / "mock_session_cycle",
        checkpoint_dir=ckpt_dir,
        epochs=2,
        batch_size=4,
        resume_path=str(best_ckpt),
        device_str="cpu",
    )
    assert report2["epochs_completed"] >= 1

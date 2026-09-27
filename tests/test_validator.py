"""Tests for Dataset Validator and Malformed Sample Detection (M1)."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image

from data_pipeline.actions import MouseBinner
from data_pipeline.schema import (
    ActionState,
    CaptureConfig,
    DatasetMetadata,
    DatasetSample,
    MouseConfig,
)
from data_pipeline.validate import validate_dataset


def create_dummy_dataset(
    dir_path: Path,
    num_samples: int = 5,
    width: int = 160,
    height: int = 120,
    format_ext: str = "jpg",
) -> None:
    """Helper to create a structurally valid dummy dataset."""
    dir_path.mkdir(parents=True, exist_ok=True)
    frames_dir = dir_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    binner = MouseBinner(num_bins=21)
    meta = DatasetMetadata(
        session_id="test_session",
        capture_config=CaptureConfig(
            target_fps=15.0, frame_width=width, frame_height=height, image_format=format_ext
        ),
        mouse_config=MouseConfig(
            num_bins_x=21,
            num_bins_y=21,
            bin_edges_x=binner.edges,
            bin_edges_y=binner.edges,
        ),
    )
    meta.save(dir_path / "metadata.json")

    samples_path = dir_path / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            # Create valid image
            frame_rel = f"frames/frame_{i:04d}.{format_ext}"
            img = Image.new("RGB", (width, height), color=(i * 10, i * 10, i * 10))
            img.save(dir_path / frame_rel)

            sample = DatasetSample(
                step_idx=i,
                timestamp=100.0 + i * 0.0667,
                iso_timestamp="2026-09-27T12:00:00Z",
                dt=0.0667 if i > 0 else 0.0,
                frame_file=frame_rel,
                actions=ActionState(
                    move_x=1 if i % 2 == 0 else 0,
                    move_y=1,
                    fire=1 if i == 2 else 0,
                    mouse_dx=2.0,
                    mouse_dy=-1.0,
                    mouse_dx_bin=12,
                    mouse_dy_bin=8,
                ),
            )
            f.write(json.dumps(sample.to_dict()) + "\n")


def test_validator_passes_on_valid_dataset(tmp_path: Path) -> None:
    ds_dir = tmp_path / "valid_ds"
    create_dummy_dataset(ds_dir, num_samples=5)

    report = validate_dataset(ds_dir)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_steps == 5
    assert report.stats["effective_fps"] > 0


def test_validator_detects_missing_metadata(tmp_path: Path) -> None:
    ds_dir = tmp_path / "missing_meta"
    create_dummy_dataset(ds_dir, num_samples=3)
    (ds_dir / "metadata.json").unlink()

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("metadata.json" in err for err in report.errors)


def test_validator_detects_nonexistent_directory(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path / "does_not_exist")
    assert report.is_valid is False


def test_validator_detects_non_monotonic_timestamps(tmp_path: Path) -> None:
    ds_dir = tmp_path / "broken_time"
    create_dummy_dataset(ds_dir, num_samples=4)

    # Edit samples.jsonl to have a backwards timestamp
    samples_path = ds_dir / "samples.jsonl"
    lines = samples_path.read_text().splitlines()
    s2 = json.loads(lines[2])
    s2["timestamp"] = 99.0  # Before step 0 (100.0)
    lines[2] = json.dumps(s2)
    samples_path.write_text("\n".join(lines) + "\n")

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("Non-monotonic timestamp" in err for err in report.errors)


def test_validator_detects_out_of_order_step_idx(tmp_path: Path) -> None:
    ds_dir = tmp_path / "broken_idx"
    create_dummy_dataset(ds_dir, num_samples=4)

    samples_path = ds_dir / "samples.jsonl"
    lines = samples_path.read_text().splitlines()
    s2 = json.loads(lines[2])
    s2["step_idx"] = 99  # Skip from 1 to 99
    lines[2] = json.dumps(s2)
    samples_path.write_text("\n".join(lines) + "\n")

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("Step index out of order" in err for err in report.errors)


def test_validator_detects_missing_frame_image(tmp_path: Path) -> None:
    ds_dir = tmp_path / "missing_frame"
    create_dummy_dataset(ds_dir, num_samples=3)

    # Delete one frame image
    (ds_dir / "frames" / "frame_0001.jpg").unlink()

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("Referenced frame not found" in err for err in report.errors)


def test_validator_detects_corrupted_frame_image(tmp_path: Path) -> None:
    ds_dir = tmp_path / "corrupted_frame"
    create_dummy_dataset(ds_dir, num_samples=3)

    # Overwrite frame with garbage bytes
    (ds_dir / "frames" / "frame_0001.jpg").write_bytes(b"garbage not an image data")

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("Corrupted frame image" in err for err in report.errors)


def test_validator_detects_frame_resolution_mismatch(tmp_path: Path) -> None:
    ds_dir = tmp_path / "res_mismatch"
    create_dummy_dataset(ds_dir, num_samples=3, width=160, height=120)

    # Replace frame 0 with 320x240 image
    bad_img = Image.new("RGB", (320, 240), color=(255, 0, 0))
    bad_img.save(ds_dir / "frames" / "frame_0000.jpg")

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("Frame dimensions (320, 240) mismatch" in err for err in report.errors)


def test_validator_detects_malformed_json_line(tmp_path: Path) -> None:
    ds_dir = tmp_path / "bad_json"
    create_dummy_dataset(ds_dir, num_samples=3)

    samples_path = ds_dir / "samples.jsonl"
    lines = samples_path.read_text().splitlines()
    lines[1] = "THIS IS NOT JSON {broken"
    samples_path.write_text("\n".join(lines) + "\n")

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("Malformed sample JSON" in err for err in report.errors)


def test_validator_detects_empty_dataset(tmp_path: Path) -> None:
    ds_dir = tmp_path / "empty_ds"
    create_dummy_dataset(ds_dir, num_samples=0)

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("0 valid steps" in err for err in report.errors)


def test_validator_warns_on_large_frame_delay(tmp_path: Path) -> None:
    ds_dir = tmp_path / "lag_spike_ds"
    create_dummy_dataset(ds_dir, num_samples=4)

    # Introduce a 1-second delay in sample 2 (vs 0.0667 target)
    samples_path = ds_dir / "samples.jsonl"
    lines = samples_path.read_text().splitlines()
    s2 = json.loads(lines[2])
    s2["timestamp"] += 1.0
    s2["dt"] = 1.0667
    lines[2] = json.dumps(s2)
    # Also adjust subsequent sample
    s3 = json.loads(lines[3])
    s3["timestamp"] += 1.0
    lines[3] = json.dumps(s3)
    samples_path.write_text("\n".join(lines) + "\n")

    report = validate_dataset(ds_dir)
    assert report.is_valid is True  # Warning, not hard failure
    assert any("Large frame delay" in w for w in report.warnings)



def test_validator_rejects_partial_metadata(tmp_path: Path) -> None:
    ds_dir = tmp_path / "partial_meta"
    create_dummy_dataset(ds_dir, num_samples=1)
    metadata = json.loads((ds_dir / "metadata.json").read_text())
    metadata.pop("capture_config")
    (ds_dir / "metadata.json").write_text(json.dumps(metadata))

    report = validate_dataset(ds_dir)
    assert report.is_valid is False
    assert any("capture_config" in error for error in report.errors)

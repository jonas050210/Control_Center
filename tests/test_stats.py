"""Tests for Dataset Inspection and Statistics Tool (M1)."""

from pathlib import Path

import pytest

from data_pipeline.stats import (
    compute_ascii_histogram,
    format_inspection_report,
    inspect_dataset,
)
from tests.test_validator import create_dummy_dataset


def test_compute_ascii_histogram() -> None:
    bins = [10, 10, 10, 5, 5, 0, 20]
    lines = compute_ascii_histogram(bins, num_bins=21, bar_width=10)
    assert len(lines) == 21
    assert "Bin 10:" in lines[10]
    assert "Bin 05:" in lines[5]


def test_inspect_dataset_and_format(tmp_path: Path) -> None:
    ds_dir = tmp_path / "stats_ds"
    create_dummy_dataset(ds_dir, num_samples=10, width=160, height=120)

    metrics = inspect_dataset(ds_dir)
    assert metrics["session_id"] == "test_session"
    assert metrics["total_steps"] == 10
    assert "effective_fps" in metrics
    assert "action_stats" in metrics
    assert "mouse_dx_metrics" in metrics
    assert "storage" in metrics

    report_str = format_inspection_report(metrics)
    assert "SandboxAI Dataset Inspection Report" in report_str
    assert "Action Frequencies" in report_str
    assert "Mouse dX Bin Distribution" in report_str

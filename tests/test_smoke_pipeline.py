"""End-to-end smoke test for the SandboxAI Data Pipeline."""

from pathlib import Path

from data_pipeline.mock import MockScreenCapture
from data_pipeline.recorder import SessionRecorder
from data_pipeline.stats import inspect_dataset
from data_pipeline.validate import validate_dataset


def test_end_to_end_data_pipeline_smoke(tmp_path: Path) -> None:
    """Full workflow smoke test: record -> validate -> inspect."""
    output_dir = tmp_path / "smoke_dataset"

    recorder = SessionRecorder(
        output_dir=output_dir,
        session_id="smoke_session_001",
        target_fps=15.0,
        frame_width=160,
        frame_height=120,
        image_format="jpg",
        is_mock=True,
    )
    metadata = recorder.record(max_duration=0.4)

    assert recorder.session_dir.exists()
    assert metadata.summary_stats["total_steps"] >= 3

    # Validate dataset
    report = validate_dataset(recorder.session_dir)
    assert report.is_valid is True
    assert len(report.errors) == 0

    # Inspect dataset
    metrics = inspect_dataset(recorder.session_dir)
    assert metrics["session_id"] == "smoke_session_001"
    assert metrics["total_steps"] >= 3
    assert "action_stats" in metrics
    assert "mouse_dx_bins" in metrics

"""Tests for Synthetic / Mock Recorder and End-to-End Pipeline (M1)."""

from pathlib import Path

import pytest
from PIL import Image

from data_pipeline.mock import MockInputGenerator, MockScreenCapture
from data_pipeline.recorder import SessionRecorder
from data_pipeline.schema import MouseConfig
from data_pipeline.validate import validate_dataset


def test_mock_screen_capture() -> None:
    capture = MockScreenCapture(target_width=84, target_height=84)
    capture.start()

    img1, t1 = capture.grab_frame()
    assert isinstance(img1, Image.Image)
    assert img1.size == (84, 84)
    assert img1.mode == "RGB"
    assert t1 > 0

    # Test state update
    capture.update_state(dx=10.0, dy=-5.0, firing=True)
    img2, t2 = capture.grab_frame()
    assert img2.size == (84, 84)
    assert t2 >= t1

    capture.close()


def test_mock_input_generator() -> None:
    gen = MockInputGenerator(seed=123)
    events = gen.generate_events_for_step(t_start=1.0, t_end=1.0667)
    assert len(events) > 0
    # Assert monotonic order
    for i in range(len(events) - 1):
        assert events[i].t <= events[i + 1].t


def test_mock_session_recorder_end_to_end_jpg(tmp_path: Path) -> None:
    recorder = SessionRecorder(
        output_dir=tmp_path / "mock_records",
        target_fps=15.0,
        frame_width=160,
        frame_height=120,
        image_format="jpg",
        is_mock=True,
    )
    metadata = recorder.record(max_duration=0.5)

    assert metadata.summary_stats["total_steps"] >= 5
    assert (recorder.session_dir / "metadata.json").exists()
    assert (recorder.session_dir / "samples.jsonl").exists()
    assert (recorder.session_dir / "frames").exists()

    # Validate output
    report = validate_dataset(recorder.session_dir)
    assert report.is_valid is True
    assert len(report.errors) == 0


def test_mock_session_recorder_end_to_end_png(tmp_path: Path) -> None:
    recorder = SessionRecorder(
        output_dir=tmp_path / "mock_records_png",
        target_fps=20.0,
        frame_width=84,
        frame_height=84,
        image_format="png",
        mouse_config=MouseConfig(num_bins_x=11, num_bins_y=11, binning_strategy="uniform"),
        is_mock=True,
    )
    metadata = recorder.record(max_duration=0.3)

    assert (recorder.session_dir / "metadata.json").exists()
    report = validate_dataset(recorder.session_dir)
    assert report.is_valid is True
    assert metadata.capture_config.frame_width == 84
    assert metadata.capture_config.image_format == "png"
    assert metadata.mouse_config.num_bins_x == 11


def test_input_listener_reports_buffer_overflow():
    from data_pipeline.input_listener import InputListener, RawInputEvent

    listener = InputListener(max_buffer_size=1)
    listener.push_event(RawInputEvent(0.0, "key_down", {"key": "w"}))
    listener.push_event(RawInputEvent(0.1, "key_up", {"key": "w"}))
    assert listener.get_dropped_event_count() == 1
    assert len(listener.drain_events()) == 1

"""Session Recorder for SandboxAI (M1).

Coordinates screen capture, action logging, timestamp synchronization,
and serialized output into versioned datasets.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from data_pipeline.actions import MouseBinner
from data_pipeline.capture import CaptureSource, ScreenCapture
from data_pipeline.input_listener import InputListener, RawInputEvent
from data_pipeline.mock import MockInputGenerator, MockScreenCapture
from data_pipeline.schema import (
    CaptureConfig,
    DatasetMetadata,
    DatasetSample,
    MouseConfig,
    SCHEMA_VERSION,
)
from data_pipeline.sync import ActionSynchronizer
from data_pipeline.ttk_adapter import GameSourceConfig, find_game_window_rect

logger = logging.getLogger("SandboxAI.Recorder")


class SessionRecorder:
    """Manages an active recording session and writes out synchronized dataset files."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "datasets",
        session_id: Optional[str] = None,
        source_name: str = "ttk_testing",
        target_fps: float = 15.0,
        frame_width: int = 160,
        frame_height: int = 120,
        image_format: str = "jpg",
        jpeg_quality: int = 90,
        mouse_config: Optional[MouseConfig] = None,
        capture_source: Optional[CaptureSource] = None,
        input_listener: Optional[InputListener] = None,
        window_title: Optional[str] = None,
        roi: Optional[List[int]] = None,
        is_mock: bool = False,
    ) -> None:
        self.output_root = Path(output_dir)
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"session_{now_str}_{uuid.uuid4().hex[:6]}"
        self.session_dir = self.output_root / self.session_id
        self.frames_dir = self.session_dir / "frames"
        self.samples_file = self.session_dir / "samples.jsonl"
        self.metadata_file = self.session_dir / "metadata.json"

        self.source_name = source_name
        self.is_mock = is_mock

        # Setup configurations
        self.capture_config = CaptureConfig(
            target_fps=float(target_fps),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            color_mode="RGB",
            image_format=image_format.lower(),
            jpeg_quality=int(jpeg_quality),
            window_title=window_title,
            roi=roi,
        )

        self.mouse_config = mouse_config or MouseConfig(
            sensitivity_scale=1.0,
            binning_strategy="symmetric_log",
            num_bins_x=21,
            num_bins_y=21,
        )

        # Generate default bin edges if empty
        binner_x = MouseBinner(
            num_bins=self.mouse_config.num_bins_x,
            strategy=self.mouse_config.binning_strategy,
            custom_edges=self.mouse_config.bin_edges_x if self.mouse_config.bin_edges_x else None,
        )
        binner_y = MouseBinner(
            num_bins=self.mouse_config.num_bins_y,
            strategy=self.mouse_config.binning_strategy,
            custom_edges=self.mouse_config.bin_edges_y if self.mouse_config.bin_edges_y else None,
        )
        self.mouse_config.bin_edges_x = binner_x.edges
        self.mouse_config.bin_edges_y = binner_y.edges

        self.synchronizer = ActionSynchronizer(
            binner_x=binner_x,
            binner_y=binner_y,
            mouse_config=self.mouse_config,
        )

        # Capture and input backends
        self.capture_source = capture_source
        self.input_listener = input_listener
        self._mock_input_gen: Optional[MockInputGenerator] = None

        self._stop_requested = False
        self._total_steps = 0
        self._dropped_frames = 0
        self._start_time_monotonic = 0.0
        self._end_time_monotonic = 0.0

    def _setup_backends(self) -> None:
        """Initializes capture and input sources if not provided."""
        if self.is_mock:
            if self.capture_source is None:
                self.capture_source = MockScreenCapture(
                    target_width=self.capture_config.frame_width,
                    target_height=self.capture_config.frame_height,
                )
            self._mock_input_gen = MockInputGenerator()
        else:
            if self.capture_source is None:
                roi = self.capture_config.roi
                if roi is None and self.capture_config.window_title:
                    detected_roi = find_game_window_rect([self.capture_config.window_title])
                    if detected_roi:
                        roi = list(detected_roi)
                        self.capture_config.roi = roi

                self.capture_source = ScreenCapture(
                    target_width=self.capture_config.frame_width,
                    target_height=self.capture_config.frame_height,
                    roi=roi,
                )

            if self.input_listener is None:
                self.input_listener = InputListener()

    def request_stop(self) -> None:
        """Signals the recording loop to stop gracefully."""
        self._stop_requested = True

    def record(self, max_duration: Optional[float] = None) -> DatasetMetadata:
        """Executes the recording session until max_duration or request_stop."""
        self._setup_backends()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        target_interval = 1.0 / self.capture_config.target_fps
        self._stop_requested = False
        self._total_steps = 0
        self._dropped_frames = 0

        # Save initial metadata stub
        metadata = DatasetMetadata(
            session_id=self.session_id,
            schema_version=SCHEMA_VERSION,
            source="mock_synthetic" if self.is_mock else self.source_name,
            capture_config=self.capture_config,
            mouse_config=self.mouse_config,
        )
        metadata.save(self.metadata_file)

        # Start input listener
        if self.input_listener is not None:
            self.input_listener.start()
        if self.capture_source is not None:
            self.capture_source.start()

        # Handle SIGINT and SIGTERM gracefully
        orig_sigint = signal.getsignal(signal.SIGINT)

        def _sig_handler(sig: Any, frame: Any) -> None:
            self.request_stop()

        try:
            signal.signal(signal.SIGINT, _sig_handler)
        except (ValueError, AttributeError):
            pass

        t_prev_frame = time.perf_counter()
        self._start_time_monotonic = t_prev_frame
        next_target_time = t_prev_frame + target_interval

        with open(self.samples_file, "w", encoding="utf-8") as f_samples:
            try:
                while not self._stop_requested:
                    loop_start = time.perf_counter()
                    if max_duration is not None and (loop_start - self._start_time_monotonic) >= max_duration:
                        break

                    # 1. Grab Frame
                    frame_img, t_frame = self.capture_source.grab_frame()

                    # 2. Collect input events
                    if self.is_mock and self._mock_input_gen is not None:
                        events = self._mock_input_gen.generate_events_for_step(
                            t_start=t_prev_frame,
                            t_end=t_frame,
                        )
                    elif self.input_listener is not None:
                        events = self.input_listener.drain_events()
                    else:
                        events = []

                    # 3. Synchronize Actions
                    frame_filename = f"frames/frame_{self._total_steps:08d}.{self.capture_config.image_format}"
                    sample = self.synchronizer.create_sample(
                        step_idx=self._total_steps,
                        t_curr=t_frame,
                        t_prev=t_prev_frame,
                        frame_file=frame_filename,
                        events=events,
                    )

                    # Update mock visual view state if mock capture
                    if self.is_mock and isinstance(self.capture_source, MockScreenCapture):
                        self.capture_source.update_state(
                            dx=sample.actions.mouse_dx,
                            dy=sample.actions.mouse_dy,
                            firing=(sample.actions.fire == 1),
                        )

                    # 4. Save frame image to disk
                    full_frame_path = self.session_dir / frame_filename
                    if self.capture_config.image_format in ("jpg", "jpeg"):
                        frame_img.save(
                            full_frame_path,
                            format="JPEG",
                            quality=self.capture_config.jpeg_quality,
                        )
                    else:
                        frame_img.save(full_frame_path, format="PNG")

                    # 5. Append sample to jsonl
                    f_samples.write(json.dumps(sample.to_dict()) + "\n")
                    f_samples.flush()

                    self._total_steps += 1
                    t_prev_frame = t_frame

                    # 6. Precise timing throttle to maintain target FPS
                    now = time.perf_counter()
                    sleep_time = next_target_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        next_target_time += target_interval
                    else:
                        # Frame took longer than target interval (frame skip/lag)
                        self._dropped_frames += 1
                        next_target_time = now + target_interval

            finally:
                self._end_time_monotonic = time.perf_counter()
                # Restore original signal handler
                try:
                    signal.signal(signal.SIGINT, orig_sigint)
                except (ValueError, AttributeError):
                    pass

                # Clean up capture and listener backends
                if self.input_listener is not None:
                    self.input_listener.stop()
                if self.capture_source is not None:
                    self.capture_source.close()

        # Calculate final summary statistics
        duration = max(1e-4, self._end_time_monotonic - self._start_time_monotonic)
        effective_fps = self._total_steps / duration if duration > 0 else 0.0

        # Calculate total bytes
        total_bytes = 0
        for p in self.session_dir.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size

        metadata.summary_stats = {
            "duration_seconds": round(duration, 3),
            "total_steps": self._total_steps,
            "effective_fps": round(effective_fps, 2),
            "dropped_frames": self._dropped_frames,
            "total_bytes": total_bytes,
        }
        metadata.save(self.metadata_file)
        return metadata

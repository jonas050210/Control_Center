"""Reliable synchronized screen/action session recorder."""

from __future__ import annotations

import datetime
import json
import os
import signal
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

from data_pipeline.actions import MouseBinner
from data_pipeline.capture import CaptureSource, ScreenCapture
from data_pipeline.input_listener import InputListener
from data_pipeline.mock import MockInputGenerator, MockScreenCapture
from data_pipeline.schema import CaptureConfig, DatasetMetadata, MouseConfig, SCHEMA_VERSION
from data_pipeline.sync import ActionSynchronizer
from data_pipeline.ttk_adapter import find_game_window_rect


class SessionRecorder:
    """Capture frame/action pairs into one crash-detectable session directory."""

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
        overwrite: bool = False,
        telemetry_state_file: Optional[Union[str, Path]] = None,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if image_format.lower() not in {"jpg", "jpeg", "png"}:
            raise ValueError("image_format must be jpg or png")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")
        self.output_root = Path(output_dir)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"session_{stamp}_{uuid.uuid4().hex[:6]}"
        self.session_dir = self.output_root / self.session_id
        self.frames_dir = self.session_dir / "frames"
        self.samples_file = self.session_dir / "samples.jsonl"
        self.metadata_file = self.session_dir / "metadata.json"
        self.marker_file = self.session_dir / ".recording"
        self.source_name = source_name
        self.is_mock = is_mock
        self.overwrite = overwrite
        self.telemetry_state_file = telemetry_state_file
        self.capture_config = CaptureConfig(
            target_fps=float(target_fps),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            image_format="jpg" if image_format.lower() == "jpeg" else image_format.lower(),
            jpeg_quality=int(jpeg_quality),
            window_title=window_title,
            roi=roi,
        )
        self.mouse_config = mouse_config or MouseConfig()
        binner_x = MouseBinner(
            self.mouse_config.num_bins_x,
            self.mouse_config.binning_strategy,
            self.mouse_config.bin_edges_x or None,
        )
        binner_y = MouseBinner(
            self.mouse_config.num_bins_y,
            self.mouse_config.binning_strategy,
            self.mouse_config.bin_edges_y or None,
        )
        self.mouse_config.bin_edges_x = binner_x.edges
        self.mouse_config.bin_edges_y = binner_y.edges
        self.synchronizer = ActionSynchronizer(binner_x, binner_y, self.mouse_config)
        self.capture_source = capture_source
        self.input_listener = input_listener
        self._mock_input_gen: Optional[MockInputGenerator] = None
        self._stop_requested = False
        self._total_steps = 0
        self._dropped_frames = 0
        self._dropped_input_events = 0
        self._start_time_monotonic = 0.0
        self._end_time_monotonic = 0.0

    def _setup_backends(self) -> None:
        if self.is_mock:
            self.capture_source = self.capture_source or MockScreenCapture(
                self.capture_config.frame_width, self.capture_config.frame_height
            )
            self._mock_input_gen = MockInputGenerator()
            return
        if self.capture_source is None:
            roi = self.capture_config.roi
            if roi is None and self.capture_config.window_title:
                found = find_game_window_rect([self.capture_config.window_title])
                if found:
                    roi = list(found)
                    self.capture_config.roi = roi
                elif self.source_name != "desktop":
                    raise RuntimeError(
                        f"Game window containing '{self.capture_config.window_title}' was not found. "
                        "Start the game, use --roi, or explicitly choose --source desktop."
                    )
            self.capture_source = ScreenCapture(
                self.capture_config.frame_width,
                self.capture_config.frame_height,
                roi=roi,
            )
        self.input_listener = self.input_listener or InputListener()

    def request_stop(self) -> None:
        self._stop_requested = True

    def _prepare_directory(self) -> None:
        if self.session_dir.exists() and any(self.session_dir.iterdir()):
            if not self.overwrite:
                raise FileExistsError(
                    f"Session directory is not empty: {self.session_dir}. Use a new session id."
                )
            # The boundary is the explicitly selected session directory, never
            # the dataset root. Clearing it avoids stale orphan frames.
            shutil.rmtree(self.session_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def _save_frame_atomic(self, image: Any, path: Path) -> None:
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        if self.capture_config.image_format == "jpg":
            image.save(temporary, format="JPEG", quality=self.capture_config.jpeg_quality)
        else:
            image.save(temporary, format="PNG")
        os.replace(temporary, path)

    def record(self, max_duration: Optional[float] = None) -> DatasetMetadata:
        if max_duration is not None and max_duration <= 0:
            raise ValueError("max_duration must be positive")
        self._setup_backends()
        self._prepare_directory()
        self.synchronizer.reset()
        self._stop_requested = False
        self._total_steps = self._dropped_frames = self._dropped_input_events = 0
        metadata = DatasetMetadata(
            session_id=self.session_id,
            schema_version=SCHEMA_VERSION,
            source="mock_synthetic" if self.is_mock else self.source_name,
            capture_config=self.capture_config,
            mouse_config=self.mouse_config,
        )
        metadata.summary_stats.update({"status": "recording"})
        metadata.save(self.metadata_file)
        self.marker_file.write_text(
            json.dumps({"pid": os.getpid(), "started_at": metadata.created_at}), encoding="utf-8"
        )
        telemetry = None
        if self.telemetry_state_file:
            from monitoring.state import SystemTelemetry

            telemetry = SystemTelemetry(self.telemetry_state_file)
            telemetry.update_stage("RECORDING")

        target_interval = 1.0 / self.capture_config.target_fps
        listener_started = capture_started = False
        old_sigint = signal.getsignal(signal.SIGINT)

        def request_stop(_signal: Any, _frame: Any) -> None:
            self.request_stop()

        try:
            try:
                signal.signal(signal.SIGINT, request_stop)
            except (ValueError, AttributeError):
                pass
            if self.input_listener is not None:
                self.input_listener.start()
                listener_started = True
            assert self.capture_source is not None
            self.capture_source.start()
            capture_started = True
            previous_frame_time = time.perf_counter()
            self._start_time_monotonic = previous_frame_time
            next_deadline = previous_frame_time

            with self.samples_file.open("w", encoding="utf-8", buffering=1) as samples_handle:
                while not self._stop_requested:
                    now = time.perf_counter()
                    if max_duration is not None and now - self._start_time_monotonic >= max_duration:
                        break
                    if now < next_deadline:
                        time.sleep(next_deadline - now)
                    frame, frame_time = self.capture_source.grab_frame()
                    if self.is_mock and self._mock_input_gen is not None:
                        events = self._mock_input_gen.generate_events_for_step(
                            previous_frame_time, frame_time
                        )
                    elif self.input_listener is not None:
                        events = self.input_listener.drain_events()
                    else:
                        events = []
                    filename = f"frames/frame_{self._total_steps:08d}.{self.capture_config.image_format}"
                    sample = self.synchronizer.create_sample(
                        self._total_steps,
                        frame_time,
                        previous_frame_time,
                        filename,
                        events,
                    )
                    if self.is_mock and isinstance(self.capture_source, MockScreenCapture):
                        self.capture_source.update_state(
                            sample.actions.mouse_dx,
                            sample.actions.mouse_dy,
                            sample.actions.fire == 1,
                        )
                    self._save_frame_atomic(frame, self.session_dir / filename)
                    samples_handle.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
                    self._total_steps += 1
                    previous_frame_time = frame_time
                    next_deadline += target_interval
                    if frame_time > next_deadline + target_interval:
                        missed = int((frame_time - next_deadline) / target_interval)
                        self._dropped_frames += max(1, missed)
                        next_deadline = frame_time + target_interval
                    if telemetry and self._total_steps % max(1, int(self.capture_config.target_fps)) == 0:
                        elapsed = frame_time - self._start_time_monotonic
                        telemetry.update_runtime(
                            self._total_steps / max(1e-9, elapsed),
                            self._total_steps / max(1e-9, elapsed),
                            0.0,
                            sample.actions.to_dict(),
                            {"recorded_steps": self._total_steps},
                        )
            status = "complete"
        except Exception:
            status = "failed"
            raise
        finally:
            self._end_time_monotonic = time.perf_counter()
            try:
                signal.signal(signal.SIGINT, old_sigint)
            except (ValueError, AttributeError):
                pass
            if listener_started and self.input_listener is not None:
                get_dropped = getattr(self.input_listener, "get_dropped_event_count", None)
                if callable(get_dropped):
                    self._dropped_input_events = int(get_dropped())
                self.input_listener.stop()
            if capture_started and self.capture_source is not None:
                self.capture_source.close()
            # Remove the crash marker before calculating the reported footprint;
            # it is an operational lock, not part of the recorded dataset.
            self.marker_file.unlink(missing_ok=True)
            duration = max(0.0, self._end_time_monotonic - self._start_time_monotonic)
            total_bytes = sum(
                path.stat().st_size for path in self.session_dir.rglob("*") if path.is_file()
            )
            metadata.summary_stats = {
                "status": locals().get("status", "failed"),
                "duration_seconds": round(duration, 6),
                "total_steps": self._total_steps,
                "effective_fps": round(self._total_steps / max(1e-9, duration), 3),
                "dropped_frames": self._dropped_frames,
                "dropped_input_events": self._dropped_input_events,
                "total_bytes": total_bytes,
            }
            metadata.save(self.metadata_file)
            self.marker_file.unlink(missing_ok=True)
            if telemetry:
                telemetry.update_dataset_stats(self.output_root)
                telemetry.update_stage("IDLE" if metadata.summary_stats["status"] == "complete" else "FAILED")
        return metadata

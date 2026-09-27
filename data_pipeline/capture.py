"""External Screen Capture engine for SandboxAI (M1).

Captures desktop or specific game window regions, downsamples to the target ML resolution,
and provides precise capture timestamps for ~15 Hz data collection.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False


class CaptureSource(abc.ABC):
    """Abstract interface for screen frame capture sources."""

    @abc.abstractmethod
    def start(self) -> None:
        """Initializes and opens the capture source."""
        pass

    @abc.abstractmethod
    def grab_frame(self) -> Tuple[Image.Image, float]:
        """Grabs a single RGB frame and its precise capture timestamp (monotonic)."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Releases all capture resources."""
        pass

    def __enter__(self) -> CaptureSource:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ScreenCapture(CaptureSource):
    """Captures real screen frames using MSS, cropping and resizing to target dimensions."""

    def __init__(
        self,
        target_width: int = 160,
        target_height: int = 120,
        monitor_index: int = 1,
        roi: Optional[Sequence[int]] = None,  # [left, top, width, height]
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.monitor_index = monitor_index
        self.roi = list(roi) if roi is not None else None
        self._sct: Optional[Any] = None
        self._bbox: Optional[Dict[str, int]] = None

    def start(self) -> None:
        if not MSS_AVAILABLE:
            raise RuntimeError(
                "MSS screen capture library is not available. Please install 'mss'."
            )
        self._sct = mss.mss()
        monitors = self._sct.monitors

        if self.roi is not None:
            # Custom region of interest [left, top, width, height]
            self._bbox = {
                "left": int(self.roi[0]),
                "top": int(self.roi[1]),
                "width": int(self.roi[2]),
                "height": int(self.roi[3]),
            }
        else:
            # Use specified monitor (index 1 is primary monitor in mss)
            mon_idx = min(self.monitor_index, len(monitors) - 1)
            if mon_idx < 1 and len(monitors) > 1:
                mon_idx = 1
            self._bbox = monitors[mon_idx]

    def set_roi(self, roi: Sequence[int]) -> None:
        """Updates the capture bounding box dynamically."""
        self.roi = list(roi)
        self._bbox = {
            "left": int(roi[0]),
            "top": int(roi[1]),
            "width": int(roi[2]),
            "height": int(roi[3]),
        }

    def grab_frame(self) -> Tuple[Image.Image, float]:
        """Grabs the screen within bbox, converts to RGB PIL Image and resizes."""
        if self._sct is None or self._bbox is None:
            self.start()

        # Timestamp at the midpoint of the screen-copy interval. This is a
        # better synchronization estimate than timestamping only before MSS.
        t_before = time.perf_counter()
        raw_sct = self._sct.grab(self._bbox)
        t_capture = (t_before + time.perf_counter()) * 0.5

        # raw_sct.rgb returns raw RGB bytes (BGRA is converted by mss)
        img = Image.frombytes("RGB", raw_sct.size, raw_sct.rgb)

        # Resize to target ML resolution if needed
        if img.size != (self.target_width, self.target_height):
            img = img.resize(
                (self.target_width, self.target_height),
                resample=Image.Resampling.BILINEAR,
            )

        return img, t_capture

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

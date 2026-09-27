"""SandboxAI Dataset Schema (M1).

Defines the versioned data structures for recording, storing, and validating
human FPS gameplay data for behavioral cloning.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

SCHEMA_VERSION: str = "1.0.0"

# Canonical Action Space specification
ACTION_SPACE_SPEC: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "actions": {
        "move_x": {
            "type": "discrete",
            "values": [-1, 0, 1],
            "description": "Lateral movement: -1 = Left (A), 0 = None, +1 = Right (D)",
        },
        "move_y": {
            "type": "discrete",
            "values": [-1, 0, 1],
            "description": "Longitudinal movement: -1 = Backward (S), 0 = None, +1 = Forward (W)",
        },
        "jump": {
            "type": "binary",
            "values": [0, 1],
            "description": "Jump action: 0 = Inactive, 1 = Active (Space)",
        },
        "crouch": {
            "type": "binary",
            "values": [0, 1],
            "description": "Crouch action: 0 = Inactive, 1 = Active (Ctrl / C)",
        },
        "sprint": {
            "type": "binary",
            "values": [0, 1],
            "description": "Sprint action: 0 = Inactive, 1 = Active (Shift)",
        },
        "reload": {
            "type": "binary",
            "values": [0, 1],
            "description": "Reload weapon: 0 = Inactive, 1 = Active (R)",
        },
        "fire": {
            "type": "binary",
            "values": [0, 1],
            "description": "Primary fire: 0 = Inactive, 1 = Active (Left Mouse Button)",
        },
        "ads": {
            "type": "binary",
            "values": [0, 1],
            "description": "Aim Down Sights (ADS): 0 = Inactive, 1 = Active (Right Mouse Button)",
        },
        "mouse_dx": {
            "type": "continuous",
            "unit": "pixels",
            "description": "Continuous horizontal mouse delta accumulated during frame interval",
        },
        "mouse_dy": {
            "type": "continuous",
            "unit": "pixels",
            "description": "Continuous vertical mouse delta accumulated during frame interval",
        },
        "mouse_dx_bin": {
            "type": "discrete",
            "description": "Discretized horizontal mouse rotation bin index in [0, num_bins_x - 1]",
        },
        "mouse_dy_bin": {
            "type": "discrete",
            "description": "Discretized vertical mouse rotation bin index in [0, num_bins_y - 1]",
        },
    },
}


@dataclasses.dataclass
class ActionState:
    """Synchronized player actions for a single timestep."""

    move_x: int = 0  # -1 (A), 0 (none), +1 (D)
    move_y: int = 0  # -1 (S), 0 (none), +1 (W)
    jump: int = 0  # 0 or 1 (Space)
    crouch: int = 0  # 0 or 1 (Ctrl / C)
    sprint: int = 0  # 0 or 1 (Shift)
    reload: int = 0  # 0 or 1 (R)
    fire: int = 0  # 0 or 1 (LMB)
    ads: int = 0  # 0 or 1 (RMB)
    mouse_dx: float = 0.0  # continuous accumulated horizontal delta
    mouse_dy: float = 0.0  # continuous accumulated vertical delta
    mouse_dx_bin: int = 0  # discretized bin index
    mouse_dy_bin: int = 0  # discretized bin index
    wheel_dy: int = 0  # scroll wheel delta
    active_keys: List[str] = dataclasses.field(default_factory=list)
    mouse_buttons: Dict[str, bool] = dataclasses.field(
        default_factory=lambda: {"left": False, "right": False, "middle": False}
    )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionState:
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclasses.dataclass
class CaptureConfig:
    """Configuration used for screen and input capture."""

    target_fps: float = 15.0
    frame_width: int = 160
    frame_height: int = 120
    color_mode: str = "RGB"
    image_format: str = "jpg"
    jpeg_quality: int = 90
    window_title: Optional[str] = None
    roi: Optional[List[int]] = None  # [left, top, width, height]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaptureConfig:
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclasses.dataclass
class MouseConfig:
    """Configuration for mouse sensitivity and discretization binning."""

    sensitivity_scale: float = 1.0
    binning_strategy: str = "symmetric_log"  # "symmetric_log", "quantile", "uniform"
    num_bins_x: int = 21
    num_bins_y: int = 21
    bin_edges_x: List[float] = dataclasses.field(default_factory=list)
    bin_edges_y: List[float] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MouseConfig:
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclasses.dataclass
class DatasetMetadata:
    """Metadata describing a recorded dataset session."""

    session_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    source: str = "ttk_testing"  # "ttk_testing", "godot_sandbox", "mock_synthetic", "desktop"
    platform: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        }
    )
    capture_config: CaptureConfig = dataclasses.field(default_factory=CaptureConfig)
    mouse_config: MouseConfig = dataclasses.field(default_factory=MouseConfig)
    action_space: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(ACTION_SPACE_SPEC)
    )
    summary_stats: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "duration_seconds": 0.0,
            "total_steps": 0,
            "effective_fps": 0.0,
            "dropped_frames": 0,
            "total_bytes": 0,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetMetadata:
        capture_cfg = (
            CaptureConfig.from_dict(data.get("capture_config", {}))
            if isinstance(data.get("capture_config"), dict)
            else CaptureConfig()
        )
        mouse_cfg = (
            MouseConfig.from_dict(data.get("mouse_config", {}))
            if isinstance(data.get("mouse_config"), dict)
            else MouseConfig()
        )
        return cls(
            session_id=data.get("session_id", "unknown_session"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            created_at=data.get("created_at", ""),
            source=data.get("source", "unknown"),
            platform=data.get("platform", {}),
            capture_config=capture_cfg,
            mouse_config=mouse_cfg,
            action_space=data.get("action_space", ACTION_SPACE_SPEC),
            summary_stats=data.get("summary_stats", {}),
        )

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> DatasetMetadata:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclasses.dataclass
class DatasetSample:
    """A single synchronized step in a dataset recording session."""

    step_idx: int
    timestamp: float  # monotonic time (time.perf_counter)
    iso_timestamp: str  # UTC timestamp
    dt: float  # time delta from previous frame (seconds)
    frame_file: str  # relative path to frame image (e.g. "frames/00000000.jpg")
    actions: ActionState
    is_valid: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "timestamp": self.timestamp,
            "iso_timestamp": self.iso_timestamp,
            "dt": self.dt,
            "frame_file": self.frame_file,
            "actions": self.actions.to_dict()
            if isinstance(self.actions, ActionState)
            else self.actions,
            "is_valid": self.is_valid,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetSample:
        action_dict = data.get("actions", {})
        action_state = (
            ActionState.from_dict(action_dict)
            if isinstance(action_dict, dict)
            else ActionState()
        )
        return cls(
            step_idx=int(data["step_idx"]),
            timestamp=float(data["timestamp"]),
            iso_timestamp=str(data.get("iso_timestamp", "")),
            dt=float(data.get("dt", 0.0)),
            frame_file=str(data["frame_file"]),
            actions=action_state,
            is_valid=bool(data.get("is_valid", True)),
            metadata=data.get("metadata", {}),
        )

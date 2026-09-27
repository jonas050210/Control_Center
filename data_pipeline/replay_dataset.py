"""Convert deterministic Godot replay JSON into BC-compatible sessions.

Replay logs are useful for debugging the Godot bridge, but they are not
training data until every action has a matching observation and complete
metadata.  This module keeps the debug JSONL conversion and the stricter BC
session export separate while sharing one frame/action association routine.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from data_pipeline.actions import MouseBinner
from data_pipeline.schema import CaptureConfig, DatasetMetadata, MouseConfig, SCHEMA_VERSION

ACTION_NAMES = [
    "move_x",
    "move_y",
    "jump",
    "crouch",
    "sprint",
    "reload",
    "fire",
    "ads",
    "mouse_dx_bin",
    "mouse_dy_bin",
]


def _action_mapping(raw_action: Any) -> dict[str, Any]:
    """Normalize a replay action without losing partial debug records.

    Godot bridge actions are zero-based MultiDiscrete arrays.  Mapping-based
    actions are treated as dataset/canonical values, which preserves replay
    files produced by external tools.
    """
    if isinstance(raw_action, Sequence) and not isinstance(raw_action, (str, bytes, bytearray)):
        values = list(raw_action)
        result = {name: value for name, value in zip(ACTION_NAMES, values)}
        if len(values) >= 2:
            result["move_x"] = int(values[0]) - 1
            result["move_y"] = int(values[1]) - 1
        return result
    if isinstance(raw_action, Mapping):
        return {str(key): value for key, value in raw_action.items()}
    return {}


def _observation_events(replay: Mapping[str, Any]) -> dict[int, str]:
    observations: dict[int, str] = {}
    for event in replay.get("events", []):
        if not isinstance(event, Mapping) or event.get("type") != "observation":
            continue
        encoded = event.get("observation_png")
        if isinstance(encoded, str) and encoded:
            observations[int(event.get("frame", 0))] = encoded
    return observations


def _observation_for_frame(observations: dict[int, str], frame: int) -> str | None:
    """Find the observation generated for an action, or the latest prior one."""
    if frame in observations:
        return observations[frame]
    prior = [key for key in observations if key <= frame]
    return observations[max(prior)] if prior else None


def replay_to_records(
    replay: dict[str, Any], *, observation_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Convert action events and associate observations by frame number.

    The old implementation associated an observation with the *next* action.
    Godot records the action before the observation for the same frame, which
    shifted every training label by one step.  Frame-based association avoids
    that off-by-one error.
    """
    records: list[dict[str, Any]] = []
    out_dir = Path(observation_dir) if observation_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    observations = _observation_events(replay)
    for event in replay.get("events", []):
        if not isinstance(event, Mapping) or event.get("type") not in {"action", "step"}:
            continue
        frame = int(event.get("frame", 0))
        record: dict[str, Any] = {
            "frame": frame,
            "time_ms": float(event.get("time_ms", 0.0)),
            "action": _action_mapping(event.get("action", {})),
            "event": str(event.get("type", "action")),
        }
        encoded = _observation_for_frame(observations, frame)
        if encoded and out_dir:
            filename = f"observation_{len(records):06d}.png"
            try:
                decoded = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(decoded)) as image:
                    image.verify()
                (out_dir / filename).write_bytes(decoded)
                record["observation_path"] = filename
            except (binascii.Error, ValueError, TypeError, OSError):
                record["observation_error"] = "invalid_base64_or_png"
        records.append(record)
    return records


def convert_replay_file(source: str | Path, destination: str | Path) -> int:
    source_path, destination_path = Path(source), Path(destination)
    replay = json.loads(source_path.read_text(encoding="utf-8"))
    records = replay_to_records(replay, observation_dir=destination_path.parent)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    return len(records)


def convert_directory(source_dir: str | Path, destination_dir: str | Path) -> int:
    total = 0
    for source in sorted(Path(source_dir).glob("**/*.json")):
        total += convert_replay_file(source, Path(destination_dir) / f"{source.stem}.jsonl")
    return total


def _complete_action(raw_action: Any) -> dict[str, Any]:
    """Return all ActionState fields with safe canonical defaults."""
    raw = _action_mapping(raw_action)
    action: dict[str, Any] = {
        "move_x": 0,
        "move_y": 0,
        "jump": 0,
        "crouch": 0,
        "sprint": 0,
        "reload": 0,
        "fire": 0,
        "ads": 0,
        "mouse_dx": 0.0,
        "mouse_dy": 0.0,
        "mouse_dx_bin": 10,
        "mouse_dy_bin": 10,
        "wheel_dy": 0,
        "active_keys": [],
        "mouse_buttons": {"left": False, "right": False, "middle": False},
    }
    action.update({key: value for key, value in raw.items() if key in action})
    action["move_x"] = max(-1, min(1, int(action["move_x"])))
    action["move_y"] = max(-1, min(1, int(action["move_y"])))
    for key in ("jump", "crouch", "sprint", "reload", "fire", "ads"):
        action[key] = int(bool(action[key]))
    action["mouse_dx_bin"] = max(0, min(20, int(action["mouse_dx_bin"])))
    action["mouse_dy_bin"] = max(0, min(20, int(action["mouse_dy_bin"])))
    action["mouse_dx"] = float(action["mouse_dx"])
    action["mouse_dy"] = float(action["mouse_dy"])
    return action


def export_bc_session(source: str | Path, destination: str | Path) -> int:
    """Export a replay as a versioned session, preserving incomplete records.

    Missing observations are retained as invalid samples for auditability, but
    the resulting metadata is marked incomplete so normal training/validation
    refuses it until the replay is repaired.
    """
    source_path, root = Path(source), Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    frames_dir = root / "frames"
    records = replay_to_records(
        json.loads(source_path.read_text(encoding="utf-8")), observation_dir=frames_dir
    )

    width, height = 84, 84
    first_frame = next(
        (frames_dir / record["observation_path"] for record in records if "observation_path" in record),
        None,
    )
    if first_frame and first_frame.is_file():
        with Image.open(first_frame) as image:
            width, height = image.size

    binner = MouseBinner(num_bins=21)
    metadata = DatasetMetadata(
        session_id=root.name or "replay_session",
        schema_version=SCHEMA_VERSION,
        source="godot_sandbox_replay",
        capture_config=CaptureConfig(
            target_fps=20.0,
            frame_width=width,
            frame_height=height,
            image_format="png",
        ),
        mouse_config=MouseConfig(
            num_bins_x=21,
            num_bins_y=21,
            bin_edges_x=binner.edges,
            bin_edges_y=binner.edges,
        ),
    )

    samples: list[dict[str, Any]] = []
    previous_time = 0.0
    all_valid = bool(records)
    for index, record in enumerate(records):
        timestamp = float(record.get("time_ms", 0.0)) / 1000.0
        valid = "observation_path" in record
        all_valid = all_valid and valid
        samples.append(
            {
                "step_idx": index,
                "timestamp": timestamp,
                "iso_timestamp": datetime.datetime.fromtimestamp(
                    timestamp, tz=datetime.timezone.utc
                ).isoformat(),
                "dt": max(0.0, timestamp - previous_time) if index else 0.0,
                "frame_file": f"frames/{record.get('observation_path', f'observation_{index:06d}.png')}",
                "actions": _complete_action(record.get("action", {})),
                "is_valid": valid,
                "metadata": {
                    "event": record.get("event", ""),
                    "source_frame": record.get("frame", 0),
                    "observation_error": record.get("observation_error"),
                },
            }
        )
        previous_time = timestamp

    metadata.summary_stats.update(
        {
            "status": "complete" if all_valid else "incomplete",
            "total_steps": len(samples),
            "duration_seconds": round(previous_time, 6),
            "effective_fps": round(len(samples) / max(previous_time, 1e-9), 3)
            if samples
            else 0.0,
            "dropped_frames": 0,
            "total_bytes": 0,
            "source_replay": str(source_path),
        }
    )
    metadata.save(root / "metadata.json")
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n")
    return len(samples)

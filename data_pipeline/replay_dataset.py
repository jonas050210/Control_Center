"""Convert deterministic Godot replay JSON into BC-compatible sessions."""
from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import Any

ACTION_NAMES = ["move_x", "move_y", "jump", "crouch", "sprint", "reload", "fire", "ads", "mouse_dx_bin", "mouse_dy_bin"]


def replay_to_records(replay: dict[str, Any], *, observation_dir: str | Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pending_png: str | None = None
    out_dir = Path(observation_dir) if observation_dir else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    for event in replay.get("events", []):
        kind = event.get("type")
        if kind == "observation":
            pending_png = event.get("observation_png")
            continue
        if kind not in {"action", "step"}: continue
        raw_action: Any = event.get("action", {})
        if isinstance(raw_action, list): raw_action = {name: value for name, value in zip(ACTION_NAMES, raw_action)}
        record: dict[str, Any] = {"frame": int(event.get("frame", 0)), "time_ms": float(event.get("time_ms", 0.0)), "action": raw_action, "event": kind}
        if pending_png and out_dir:
            filename = f"observation_{len(records):06d}.png"
            try:
                (out_dir / filename).write_bytes(base64.b64decode(pending_png))
                record["observation_path"] = filename
            except (ValueError, TypeError):
                record["observation_error"] = "invalid_base64"
            pending_png = None
        records.append(record)
    return records


def convert_replay_file(source: str | Path, destination: str | Path) -> int:
    source_path, destination_path = Path(source), Path(destination)
    records = replay_to_records(json.loads(source_path.read_text(encoding="utf-8")), observation_dir=destination_path.parent)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(records)


def convert_directory(source_dir: str | Path, destination_dir: str | Path) -> int:
    total = 0
    for source in sorted(Path(source_dir).glob("**/*.json")):
        total += convert_replay_file(source, Path(destination_dir) / f"{source.stem}.jsonl")
    return total


def export_bc_session(source: str | Path, destination: str | Path) -> int:
    source_path, root = Path(source), Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    records = replay_to_records(json.loads(source_path.read_text(encoding="utf-8")), observation_dir=root / "frames")
    (root / "metadata.json").write_text(json.dumps({"schema_version": "1.1.0", "source": "godot_sandbox_replay", "replay": str(source_path), "observation_dir": "frames"}, indent=2), encoding="utf-8")
    samples = []
    for index, record in enumerate(records):
        samples.append({"step_idx": index, "timestamp": record["time_ms"] / 1000.0, "iso_timestamp": "", "dt": 0.0, "frame_file": f"frames/observation_{index:06d}.png", "actions": record.get("action", {}), "is_valid": "observation_path" in record, "metadata": {"event": record.get("event", ""), "source_frame": record.get("frame", 0)}})
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in samples: handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(samples)

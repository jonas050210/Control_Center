"""Convert deterministic Godot replay JSON into behavior-cloning records."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable


def replay_to_records(replay: dict[str, Any], *, observation_dir: str | Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    events = replay.get("events", [])
    for event in events:
        if event.get("type") not in {"action", "step", "shot"}:
            continue
        raw_action = event.get("action", {})
        if isinstance(raw_action, list):
            names = ["move_x", "move_y", "jump", "crouch", "sprint", "reload", "fire", "ads", "mouse_dx_bin", "mouse_dy_bin"]
            raw_action = {name: value for name, value in zip(names, raw_action)}
        record = {"frame": int(event.get("frame", 0)), "time_ms": float(event.get("time_ms", 0.0)), "action": raw_action, "event": event.get("type", "")}
        if "observation" in event: record["observation"] = event["observation"]
        if observation_dir and "observation_file" in event:
            record["observation_path"] = str(Path(observation_dir) / str(event["observation_file"]))
        records.append(record)
    return records


def convert_replay_file(source: str | Path, destination: str | Path) -> int:
    source_path, destination_path = Path(source), Path(destination)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    records = replay_to_records(payload, observation_dir=source_path.parent)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(records)


def convert_directory(source_dir: str | Path, destination_dir: str | Path) -> int:
    total = 0
    destination = Path(destination_dir)
    for source in sorted(Path(source_dir).glob("**/*.json")):
        target = destination / (source.stem + ".jsonl")
        total += convert_replay_file(source, target)
    return total


def export_bc_session(source: str | Path, destination: str | Path) -> int:
    """Write a BC-compatible metadata.json and samples.jsonl session."""
    source_path, root = Path(source), Path(destination)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    records = replay_to_records(payload, observation_dir=source_path.parent)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps({"schema_version": "1.1.0", "source": "godot_sandbox_replay", "replay": str(source_path)}, indent=2), encoding="utf-8")
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(records)

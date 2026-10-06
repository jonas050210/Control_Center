"""Validated JSON import/export for user-authored arena layouts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from env.maps import ArenaMap, ArenaObject, serialize_map


_ALLOWED_OBJECT_TYPES = {"wall", "crate", "barrel", "ramp", "pillar", "shelf", "platform", "rail"}
_MAX_OBJECTS = 500


def map_from_dict(payload: dict[str, Any], *, force_custom_name: bool = False) -> ArenaMap:
    """Validate untrusted map JSON before constructing simulation geometry."""
    if not isinstance(payload, dict):
        raise ValueError("Map document must be a JSON object.")
    try:
        width = float(payload["width"])
        depth = float(payload["depth"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Map document needs numeric width and depth values.") from exc
    if not (math.isfinite(width) and math.isfinite(depth) and 5.0 <= width <= 200.0 and 5.0 <= depth <= 200.0):
        raise ValueError("Map width and depth must be finite values between 5 and 200 metres.")

    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, list) or len(raw_objects) > _MAX_OBJECTS:
        raise ValueError(f"Map objects must be a list with no more than {_MAX_OBJECTS} entries.")
    objects: list[ArenaObject] = []
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, dict):
            raise ValueError(f"Object {index + 1} must be a JSON object.")
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in _ALLOWED_OBJECT_TYPES:
            raise ValueError(f"Object {index + 1} has unsupported type {kind!r}.")
        try:
            coordinates = {
                key: float(raw[key])
                for key in ("x", "y", "z", "width", "height", "depth")
            }
            yaw = float(raw.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Object {index + 1} is missing a numeric dimension or coordinate.") from exc
        if not all(math.isfinite(value) for value in (*coordinates.values(), yaw)):
            raise ValueError(f"Object {index + 1} contains a non-finite number.")
        if min(coordinates["width"], coordinates["height"], coordinates["depth"]) <= 0.0:
            raise ValueError(f"Object {index + 1} dimensions must be greater than zero.")
        if max(coordinates["width"], coordinates["height"], coordinates["depth"]) > 200.0:
            raise ValueError(f"Object {index + 1} dimensions are too large.")
        if abs(coordinates["x"]) > width + coordinates["width"] / 2.0:
            raise ValueError(f"Object {index + 1} is too far outside the map on the X axis.")
        if abs(coordinates["y"]) > depth + coordinates["depth"] / 2.0:
            raise ValueError(f"Object {index + 1} is too far outside the map on the Y axis.")
        if coordinates["z"] < -5.0 or coordinates["z"] + coordinates["height"] > 200.0:
            raise ValueError(f"Object {index + 1} is outside the supported vertical range.")
        name = str(raw.get("name", f"Custom {kind.title()} {index + 1}"))[:100]
        objects.append(ArenaObject(**coordinates, kind=kind, name=name, yaw=yaw))

    raw_spawns = payload.get("spawn_points", [[-width * 0.36, 0.0], [width * 0.36, 0.0]])
    if not isinstance(raw_spawns, (list, tuple)) or len(raw_spawns) != 2:
        raise ValueError("Map needs exactly two spawn points.")
    spawn_points: list[tuple[float, float]] = []
    for index, point in enumerate(raw_spawns):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Spawn point {index + 1} must contain x and y coordinates.")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Spawn point {index + 1} must be numeric.") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Spawn point {index + 1} must be finite.")
        if abs(x) > width / 2.0 - 0.35 or abs(y) > depth / 2.0 - 0.35:
            raise ValueError(f"Spawn point {index + 1} lies outside the playable map bounds.")
        spawn_points.append((x, y))
    if math.dist(spawn_points[0], spawn_points[1]) < 2.0:
        raise ValueError("Spawn points must be at least 2 metres apart.")

    name = "Custom" if force_custom_name else str(payload.get("name", "Custom"))[:80]
    description = str(payload.get("description", "User-authored arena layout."))[:300]
    return ArenaMap(name, width, depth, objects, (spawn_points[0], spawn_points[1]), description)


def map_from_json(data: str | bytes, *, force_custom_name: bool = True) -> ArenaMap:
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("Map file is not valid UTF-8 JSON.") from exc
    return map_from_dict(payload, force_custom_name=force_custom_name)


def save_map(path: str | Path, arena_map: ArenaMap) -> Path:
    """Atomically save a JSON-friendly map document to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(serialize_map(arena_map), indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_map(path: str | Path, *, force_custom_name: bool = True) -> ArenaMap:
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read map file {path}: {exc}") from exc
    return map_from_json(payload, force_custom_name=force_custom_name)

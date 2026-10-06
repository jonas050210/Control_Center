"""JSON scene descriptions for the browser WebGL renderer.

The former Plotly module (``gui/visuals.py``) built figures server-side. This
module emits compact, renderer-agnostic JSON instead: the browser turns it into
instanced meshes, so a 320-prop arena is still a handful of draw calls and the
wire format stays small enough for a 10 Hz poll loop.

Coordinate contract: scenes use the *simulation* frame — ``x``/``y`` span the
ground plane and ``z`` points up. The Three.js client maps
``(x, y, z) -> (x, z, y)`` and rotates yaw around the scene ``y`` axis.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from env.maps import ArenaMap, ArenaObject
from server.config import AGENT_COLORS, OBJECT_COLORS, theme_color


DETAIL_PRESETS = {"Performance": 60, "Balanced": 160, "Ultra": 320}
_DETAIL_PALETTE = ("#324d3b", "#405b46", "#273d31", "#607052", "#54412e")


def _round(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def map_signature(arena_map: ArenaMap) -> tuple[Any, ...]:
    return (
        arena_map.name,
        float(arena_map.width),
        float(arena_map.depth),
        tuple(tuple(map(float, point)) for point in arena_map.spawn_points),
        tuple((float(item.x), float(item.y), float(item.z), float(item.width),
               float(item.height), float(item.depth), item.kind, item.name, float(item.yaw))
              for item in arena_map.objects),
    )


def scene_key(arena_map: ArenaMap, detail_count: int, show_spawns: bool = True) -> str:
    """Stable identifier so the client only refetches static geometry when needed."""
    digest = hashlib.blake2b(repr(map_signature(arena_map)).encode("utf-8"), digest_size=8)
    return f"{arena_map.name}:{detail_count}:{int(bool(show_spawns))}:{digest.hexdigest()}"


def map_key(arena_map: ArenaMap) -> str:
    digest = hashlib.blake2b(repr(map_signature(arena_map)).encode("utf-8"), digest_size=6)
    return f"{arena_map.name}-{arena_map.width:g}x{arena_map.depth:g}-{len(arena_map.objects)}-{digest.hexdigest()}"


def _inside_object(x: float, y: float, item: Sequence[Any], margin: float) -> bool:
    """Test a point against an expanded, yaw-aware object footprint."""
    ox, oy, _oz, width, _height, depth, _kind, _name, yaw = item
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = x - ox, y - oy
    local_x = dx * cosine + dy * sine
    local_y = -dx * sine + dy * cosine
    return abs(local_x) < width / 2 + margin and abs(local_y) < depth / 2 + margin


@lru_cache(maxsize=32)
def ambient_details(signature: tuple[Any, ...], detail_count: int) -> tuple[tuple[Any, ...], ...]:
    """Seed and cache tiny, non-colliding floor details for a stable render."""
    _name, width, depth, spawns, objects = signature
    digest = hashlib.blake2b(repr(signature).encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    points: list[tuple[float, float, float, str, float]] = []
    maximum_attempts = max(100, int(detail_count) * 20)
    attempts = 0
    while len(points) < detail_count and attempts < maximum_attempts:
        attempts += 1
        x = float(rng.uniform(-width * 0.48, width * 0.48))
        y = float(rng.uniform(-depth * 0.48, depth * 0.48))
        if any(math.hypot(x - spawn[0], y - spawn[1]) < 2.0 for spawn in spawns):
            continue
        if any(_inside_object(x, y, item, 0.32) for item in objects):
            continue
        color = _DETAIL_PALETTE[int(rng.integers(0, len(_DETAIL_PALETTE)))]
        size = float(rng.uniform(0.35, 1.05))
        points.append((x, y, float(rng.uniform(0.025, 0.09)), color, size))
    return tuple(points)


@lru_cache(maxsize=32)
def ambient_props(signature: tuple[Any, ...], detail_count: int) -> tuple[ArenaObject, ...]:
    """Place a few decorative barrels/crates near the perimeter, away from lanes."""
    _name, width, depth, spawns, objects = signature
    count = min(24, max(0, int(detail_count) // 16))
    if count == 0:
        return ()
    digest = hashlib.blake2b((repr(signature) + "set-dressing").encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    kinds = ("crate", "barrel", "pillar")
    props: list[ArenaObject] = []
    attempts = 0
    while len(props) < count and attempts < count * 24:
        attempts += 1
        x = float(rng.uniform(-width * 0.43, width * 0.43))
        y = float(rng.uniform(-depth * 0.43, depth * 0.43))
        edge_distance = min(width / 2 - abs(x), depth / 2 - abs(y))
        if edge_distance > min(width, depth) * 0.18:
            continue
        candidate_width = float(rng.uniform(0.65, 1.55))
        candidate_depth = float(rng.uniform(0.65, 1.55))
        if any(math.hypot(x - sx, y - sy) < 4.0 for sx, sy in spawns):
            continue
        if any(_inside_object(x, y, item, max(candidate_width, candidate_depth) * 0.65)
               for item in objects):
            continue
        if any(math.hypot(x - item.x, y - item.y) < 2.2 for item in props):
            continue
        kind = str(rng.choice(kinds))
        height = float(rng.uniform(0.65, 1.8))
        props.append(ArenaObject(
            x=x, y=y, z=0, width=candidate_width, height=height,
            depth=candidate_depth, kind=kind,
            name=f"Set dressing · {kind.title()} {len(props) + 1}",
            yaw=float(rng.uniform(-math.pi, math.pi)),
        ))
    return tuple(props)


def _object_payload(item: ArenaObject, detail: bool = False) -> dict[str, Any]:
    return {
        "x": _round(item.x),
        "y": _round(item.y),
        "z": _round(item.z),
        "w": _round(item.width),
        "h": _round(item.height),
        "d": _round(item.depth),
        "yaw": _round(item.yaw, 4),
        "kind": item.kind,
        "name": item.name or item.kind.title(),
        "color": OBJECT_COLORS.get(item.kind, "#66727c"),
        "detail": bool(detail),
    }


def map_scene(
    arena_map: ArenaMap,
    detail_count: int = DETAIL_PRESETS["Balanced"],
    show_spawns: bool = True,
) -> dict[str, Any]:
    """Build the static geometry payload for one map."""
    detail_count = int(max(0, detail_count))
    signature = map_signature(arena_map)
    dressing = ambient_props(signature, detail_count)
    objects = [_object_payload(item) for item in arena_map.objects]
    objects.extend(_object_payload(item, detail=True) for item in dressing)
    details = ambient_details(signature, detail_count)
    theme = theme_color(arena_map.name)
    grid_step = max(2.0, round(min(arena_map.width, arena_map.depth) / 10.0))
    return {
        "key": scene_key(arena_map, detail_count, show_spawns),
        "map_key": map_key(arena_map),
        "name": arena_map.name,
        "width": _round(arena_map.width),
        "depth": _round(arena_map.depth),
        "description": arena_map.description,
        "theme": theme,
        "floor_color": "#101815",
        "grid_step": grid_step,
        "ring_radius": _round(min(arena_map.width, arena_map.depth) * 0.19),
        "objects": objects,
        "details": {
            "x": [_round(point[0]) for point in details],
            "y": [_round(point[1]) for point in details],
            "z": [_round(point[2]) for point in details],
            "size": [_round(point[4]) for point in details],
            "color": [point[3] for point in details],
        },
        "spawns": [[_round(point[0]), _round(point[1])] for point in arena_map.spawn_points]
        if show_spawns else [],
    }


def agent_payload(agent: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "x": _round(agent["x"]),
        "y": _round(agent["y"]),
        "z": _round(agent["z"]),
        "yaw": _round(agent["yaw"], 4),
        "pitch": _round(agent["pitch"], 4),
        "height": _round(agent["height"]),
        "hp": _round(agent["hp"], 1),
        "alive": bool(agent.get("alive", True)),
        "stance": int(agent.get("stance", 0)),
        "sprinting": bool(agent.get("sprinting", False)),
        "in_air": bool(agent.get("in_air", False)),
        "weapon": agent["weapon"],
        "ammo": int(agent["ammo"]),
        "mag_size": int(agent["mag_size"]),
        "shots_fired": int(agent.get("shots_fired", 0)),
        "bullets_fired": int(agent.get("bullets_fired", 0)),
        "hits": int(agent.get("hits", 0)),
        "headshots": int(agent.get("headshots", 0)),
        "color": AGENT_COLORS[index % len(AGENT_COLORS)],
        "label": f"AGENT {index + 1}",
    }


def match_frame(env: Any) -> dict[str, Any]:
    """Build the per-tick dynamic payload for one live match."""
    snapshot = env.get_snapshot()
    trails = [
        {
            "start": [_round(value) for value in trail.get("start", (0, 0, 0))],
            "end": [_round(value) for value in trail.get("end", (0, 0, 0))],
            "hit": bool(trail.get("hit", False)),
        }
        for trail in snapshot["trails"][-40:]
        if trail.get("start") and trail.get("end")
    ]
    return {
        "agents": [agent_payload(agent, index) for index, agent in enumerate(snapshot["agents"])],
        "trails": trails,
        "elapsed": _round(snapshot["elapsed"], 2),
        "frame": int(snapshot["frame"]),
        "done": bool(snapshot["done"]),
        "map_name": snapshot["map"]["name"],
    }


def format_event(event: dict[str, Any]) -> str:
    """Human-readable kill-feed line for one combat event."""
    if event.get("type") == "kill":
        tag = " [HEADSHOT]" if event.get("headshot") else ""
        return (
            f"Agent {event.get('attacker', '?')} killed Agent {event.get('target', '?')} "
            f"with {event.get('weapon', 'weapon')}{tag} · {float(event.get('distance', 0.0)):.1f}m"
        )
    tag = " HEADSHOT" if event.get("headshot") else ""
    return (
        f"A{event.get('attacker', '?')} hit A{event.get('target', '?')} · "
        f"{float(event.get('damage', 0.0)):.0f} dmg{tag}"
    )


def aim_index_scene(key: str) -> dict[str, Any]:
    """Static geometry of the 3D aim range (nine cells in a 3×3 wall)."""
    objects: list[dict[str, Any]] = []
    wall_y = 8.4
    objects.append({"x": 0.0, "y": wall_y, "z": 0.0, "w": 18.5, "h": 9.8, "d": 0.55,
                    "yaw": 0.0, "kind": "wall", "name": "Armored target wall",
                    "color": OBJECT_COLORS["wall"], "detail": False})
    for side in (-1, 1):
        objects.append({"x": side * 8.4, "y": 0.8, "z": 0.0, "w": 1.2, "h": 2.4, "d": 1.2,
                        "yaw": 0.0, "kind": "pillar", "name": "Range pillar",
                        "color": OBJECT_COLORS["pillar"], "detail": False})
        objects.append({"x": side * 6.8, "y": -3.6, "z": 0.0, "w": 2.7, "h": 1.5, "d": 1.7,
                        "yaw": 0.0, "kind": "crate", "name": "Ammo stack",
                        "color": OBJECT_COLORS["crate"], "detail": False})
        objects.append({"x": side * 5.2, "y": -3.0, "z": 0.0, "w": 2.2, "h": 1.0, "d": 1.4,
                        "yaw": 0.0, "kind": "crate", "name": "Ammo stack",
                        "color": OBJECT_COLORS["crate"], "detail": False})
        objects.append({"x": side * 8.8, "y": -5.2, "z": 0.0, "w": 0.45, "h": 3.8, "d": 0.45,
                        "yaw": 0.0, "kind": "barrel", "name": "Safety light",
                        "color": OBJECT_COLORS["barrel"], "detail": False})
    return {
        "key": key,
        "map_key": key,
        "name": "Aim Lab",
        "width": 20.0,
        "depth": 19.0,
        "description": "Precision range with nine reactive holographic targets.",
        "theme": theme_color("Custom"),
        "floor_color": "#101815",
        "grid_step": 2.0,
        "ring_radius": 3.5,
        "objects": objects,
        "details": {"x": [], "y": [], "z": [], "size": [], "color": []},
        "spawns": [],
        "camera_target": [0.0, 2.4, 0.0],
        "camera_radius": 26.0,
        "camera_phi": 1.12,
        "targets": [
            {
                "cell": index,
                "x": (index % 3 - 1) * 5.0,
                "y": wall_y - 0.6,
                "z": 1.6 + (1 - index // 3) * 2.3,
            }
            for index in range(9)
        ],
    }


def target_states(cell: int, hits: int = 0, misses: int = 0, streak: int = 0) -> list[dict[str, Any]]:
    states = []
    for index in range(9):
        active = index == int(cell)
        states.append({
            "cell": index,
            "active": active,
            "hit": (index + hits) % 3 == 0 and hits > 0,
            "flash": active and streak > 0,
            "missed": active and misses < 0,
        })
    return states


def dodge_arena_scene(key: str, grid: int = 5) -> dict[str, Any]:
    """Static geometry of the dodge-survival floor plus the 5×5 lane grid."""
    cells = []
    spacing = 3.4
    half = (grid - 1) / 2.0
    for row in range(grid):
        for column in range(grid):
            cells.append({
                "column": column,
                "row": row,
                "x": (column - half) * spacing,
                "y": (row - half) * spacing,
            })
    return {
        "key": key,
        "map_key": key,
        "name": "Dodge Grid",
        "width": 24.0,
        "depth": 24.0,
        "description": "Automated turret lanes on a 5×5 evasion grid.",
        "theme": theme_color("Warehouse"),
        "floor_color": "#0e1410",
        "grid_step": 2.0,
        "ring_radius": 4.2,
        "objects": [
            {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0, "h": 0.05, "d": 1.0, "yaw": 0.0,
             "kind": "platform", "name": "Grid deck", "color": OBJECT_COLORS["platform"],
             "detail": True},
        ],
        "details": {"x": [], "y": [], "z": [], "size": [], "color": []},
        "spawns": [],
        "cells": cells,
        "spacing": spacing,
        "camera_target": [0.0, 1.0, 0.0],
        "camera_radius": 27.0,
        "camera_phi": 1.0,
        "turrets": [{"column": column, "x": (column - half) * spacing, "y": -half * spacing - 2.6}
                    for column in range(grid)],
    }

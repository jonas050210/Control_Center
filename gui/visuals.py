"""Efficient, game-ready Plotly 3D scenes for the headless arena dashboard."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go

from env.maps import ArenaMap, ArenaObject
from gui.common import BG, BLUE, GREEN, NEON, RED, YELLOW, dark_layout


OBJECT_COLORS = {
    "wall": "#53626e",
    "crate": "#a56b3b",
    "barrel": "#d29a2e",
    "ramp": "#426e63",
    "platform": "#58666a",
    "pillar": "#71808b",
    "rail": "#9aaab2",
    "shelf": "#687361",
}

_KIND_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
               (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
               (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
_FACE_SHADES = (0.50, 0.63, 1.16, 1.05, 0.78, 0.70,
                0.95, 0.84, 0.82, 0.72, 0.68, 0.58)
DETAIL_PRESETS = {"Performance": 60, "Balanced": 160, "Ultra": 320}


def _shade(color: str, factor: float) -> str:
    """Lighten or darken one hex color for per-face low-poly shading."""
    value = color.lstrip("#")
    channels = [int(value[index:index + 2], 16) for index in (0, 2, 4)]
    adjusted = [min(255, max(0, round(channel * factor))) for channel in channels]
    return "#" + "".join(f"{channel:02x}" for channel in adjusted)


def _object_hover(item: ArenaObject) -> str:
    return (
        f"{item.name or item.kind.title()}<br>Material: {item.kind}<br>"
        f"{item.width:.1f} × {item.depth:.1f} × {item.height:.1f} m"
    )


def _barrel_geometry(item: ArenaObject, sides: int = 10) -> tuple[list[tuple[float, float, float]],
                                                                    list[tuple[int, int, int]],
                                                                    list[float]]:
    """Create a compact, faceted metal drum instead of drawing barrels as boxes."""
    cosine, sine = math.cos(item.yaw), math.sin(item.yaw)
    lower, upper = [], []
    for index in range(sides):
        angle = 2 * math.pi * index / sides
        local_x = item.width * 0.5 * math.cos(angle)
        local_y = item.depth * 0.5 * math.sin(angle)
        x = item.x + local_x * cosine - local_y * sine
        y = item.y + local_x * sine + local_y * cosine
        lower.append((x, y, item.z))
        upper.append((x, y, item.z + item.height))
    points = lower + upper + [
        (item.x, item.y, item.z),
        (item.x, item.y, item.z + item.height),
    ]
    faces: list[tuple[int, int, int]] = []
    shades: list[float] = []
    bottom_center, top_center = sides * 2, sides * 2 + 1
    for index in range(sides):
        following = (index + 1) % sides
        faces.extend((
            (index, following, sides + following),
            (index, sides + following, sides + index),
            (bottom_center, following, index),
            (top_center, sides + index, sides + following),
        ))
        facet_light = 0.82 + 0.24 * math.cos(2 * math.pi * (index + 0.5) / sides)
        shades.extend((facet_light, facet_light * 0.9, 0.48, 1.18))
    return points, faces, shades


def _add_batched_boxes(
    fig: go.Figure,
    objects: Sequence[ArenaObject],
    opacity: float = 0.98,
) -> None:
    """Draw hundreds of cover props in at most one mesh per material kind."""
    groups: dict[str, list[ArenaObject]] = {}
    for item in objects:
        groups.setdefault(item.kind, []).append(item)

    for kind, group in groups.items():
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        ii: list[int] = []
        jj: list[int] = []
        kk: list[int] = []
        face_colors: list[str] = []
        hover_text: list[str] = []
        base_color = OBJECT_COLORS.get(kind, "#66727c")

        for item in group:
            if kind == "barrel":
                points, faces, shades = _barrel_geometry(item)
            else:
                cosine, sine = math.cos(item.yaw), math.sin(item.yaw)
                corners = []
                for local_x, local_y in (
                    (-item.width / 2, -item.depth / 2),
                    (item.width / 2, -item.depth / 2),
                    (item.width / 2, item.depth / 2),
                    (-item.width / 2, item.depth / 2),
                ):
                    corners.append((
                        item.x + local_x * cosine - local_y * sine,
                        item.y + local_x * sine + local_y * cosine,
                    ))
                points = [
                    (corners[0][0], corners[0][1], item.z),
                    (corners[1][0], corners[1][1], item.z),
                    (corners[2][0], corners[2][1], item.z),
                    (corners[3][0], corners[3][1], item.z),
                    (corners[0][0], corners[0][1], item.z + item.height),
                    (corners[1][0], corners[1][1], item.z + item.height),
                    (corners[2][0], corners[2][1], item.z + item.height),
                    (corners[3][0], corners[3][1], item.z + item.height),
                ]
                faces = list(_KIND_FACES)
                shades = list(_FACE_SHADES)

            offset = len(xs)
            xs.extend(point[0] for point in points)
            ys.extend(point[1] for point in points)
            zs.extend(point[2] for point in points)
            hover_text.extend([_object_hover(item)] * len(points))
            for face_index, (a, b, c) in enumerate(faces):
                ii.append(offset + a)
                jj.append(offset + b)
                kk.append(offset + c)
                face_colors.append(_shade(base_color, shades[face_index]))

        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
            facecolor=face_colors,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            flatshading=True,
            lighting={"ambient": 0.62, "diffuse": 0.86, "specular": 0.22,
                      "roughness": 0.82, "fresnel": 0.08},
            lightposition={"x": -80, "y": -120, "z": 200},
            opacity=opacity,
            name=f"{kind.title()} props · {len(group)}",
            showlegend=False,
        ))


def add_box(fig: go.Figure, item: ArenaObject, opacity: float = 0.72) -> None:
    """Compatibility helper to add one detailed, shaded 3D cover object."""
    _add_batched_boxes(fig, [item], opacity=opacity)


def _map_signature(arena_map: ArenaMap) -> tuple[Any, ...]:
    return (
        arena_map.name,
        float(arena_map.width),
        float(arena_map.depth),
        tuple(tuple(map(float, point)) for point in arena_map.spawn_points),
        tuple((float(item.x), float(item.y), float(item.z), float(item.width),
               float(item.height), float(item.depth), item.kind, item.name, float(item.yaw))
              for item in arena_map.objects),
    )


def _inside_object(x: float, y: float, item: Sequence[Any], margin: float) -> bool:
    """Test a point against an expanded, yaw-aware object footprint."""
    ox, oy, _oz, width, _height, depth, _kind, _name, yaw = item
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = x - ox, y - oy
    local_x = dx * cosine + dy * sine
    local_y = -dx * sine + dy * cosine
    return abs(local_x) < width / 2 + margin and abs(local_y) < depth / 2 + margin


@lru_cache(maxsize=32)
def _ambient_details(signature: tuple[Any, ...], detail_count: int) -> tuple[tuple[Any, ...], ...]:
    """Seed and cache tiny, non-colliding floor details for a stable map render."""
    _name, width, depth, spawns, objects = signature
    digest = hashlib.blake2b(repr(signature).encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    palette = ("#324d3b", "#405b46", "#273d31", "#607052", "#54412e")
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
        color = palette[int(rng.integers(0, len(palette)))]
        size = float(rng.uniform(1.2, 3.3))
        points.append((x, y, float(rng.uniform(0.025, 0.09)), color, size))
    return tuple(points)


@lru_cache(maxsize=32)
def _ambient_props(signature: tuple[Any, ...], detail_count: int) -> tuple[ArenaObject, ...]:
    """Place a few decorative barrels/crates near the perimeter, away from combat lanes."""
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


def _add_floor(
    fig: go.Figure,
    width: float,
    depth: float,
    accent: str,
    detail_signature: tuple[Any, ...] | None = None,
    detail_count: int = 0,
) -> None:
    """Add one floor mesh, one batched grid, arena rings, and safe-zone edge."""
    x_min, x_max = -width / 2, width / 2
    y_min, y_max = -depth / 2, depth / 2
    floor_z = -0.08
    fig.add_trace(go.Mesh3d(
        x=[x_min, x_max, x_max, x_min],
        y=[y_min, y_min, y_max, y_max],
        z=[floor_z] * 4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#101815", opacity=1.0,
        flatshading=True, hoverinfo="skip", showlegend=False, name="Armored floor",
        lighting={"ambient": 0.92, "diffuse": 0.24, "specular": 0.04, "roughness": 0.95},
    ))

    grid_step = max(2.0, round(min(width, depth) / 10.0))
    grid_x: list[float | None] = []
    grid_y: list[float | None] = []
    grid_z: list[float | None] = []
    for x in np.arange(math.ceil(x_min / grid_step) * grid_step, x_max + 0.01, grid_step):
        grid_x.extend((float(x), float(x), None))
        grid_y.extend((y_min, y_max, None))
        grid_z.extend((0.005, 0.005, None))
    for y in np.arange(math.ceil(y_min / grid_step) * grid_step, y_max + 0.01, grid_step):
        grid_x.extend((x_min, x_max, None))
        grid_y.extend((float(y), float(y), None))
        grid_z.extend((0.005, 0.005, None))
    fig.add_trace(go.Scatter3d(
        x=grid_x, y=grid_y, z=grid_z, mode="lines",
        line={"color": "#233c30", "width": 1}, opacity=0.56,
        hoverinfo="skip", showlegend=False, name="Floor grid",
    ))

    ring_x: list[float | None] = []
    ring_y: list[float | None] = []
    ring_z: list[float | None] = []
    radius_base = min(width, depth) * 0.19
    angles = np.linspace(0, 2 * math.pi, 48)
    for radius in (radius_base, radius_base * 0.72, radius_base * 0.45):
        ring_x.extend((radius * np.cos(angles)).tolist() + [None])
        ring_y.extend((radius * np.sin(angles)).tolist() + [None])
        ring_z.extend([0.014] * len(angles) + [None])
    fig.add_trace(go.Scatter3d(
        x=ring_x, y=ring_y, z=ring_z, mode="lines",
        line={"color": accent, "width": 2}, opacity=0.22,
        hoverinfo="skip", showlegend=False, name="Painted range markings",
    ))

    border_x = [x_min, x_max, x_max, x_min, x_min, None,
                x_min + 0.7, x_max - 0.7, x_max - 0.7, x_min + 0.7, x_min + 0.7]
    border_y = [y_min, y_min, y_max, y_max, y_min, None,
                y_min + 0.7, y_min + 0.7, y_max - 0.7, y_max - 0.7, y_min + 0.7]
    fig.add_trace(go.Scatter3d(
        x=border_x, y=border_y, z=[0.02] * len(border_x), mode="lines",
        line={"color": accent, "width": 3}, opacity=0.64,
        hoverinfo="skip", showlegend=False, name="Perimeter light strip",
    ))

    if detail_signature is not None and detail_count > 0:
        details = _ambient_details(detail_signature, int(detail_count))
        if details:
            fig.add_trace(go.Scatter3d(
                x=[point[0] for point in details],
                y=[point[1] for point in details],
                z=[point[2] for point in details],
                mode="markers",
                marker={
                    "size": [point[4] for point in details],
                    "color": [point[3] for point in details],
                    "opacity": 0.76,
                    "symbol": "diamond",
                    "line": {"width": 0},
                },
                hoverinfo="skip", showlegend=False, name=f"Scattered floor detail · {len(details)}",
            ))


def _add_beacons(fig: go.Figure, width: float, depth: float, accent: str) -> None:
    """Draw four lightweight perimeter beacons as one line and one point trace."""
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    beacon_x = [-width * 0.46, width * 0.46, width * 0.46, -width * 0.46]
    beacon_y = [-depth * 0.46, -depth * 0.46, depth * 0.46, depth * 0.46]
    beacon_height = min(4.8, max(2.5, min(width, depth) * 0.15))
    for x, y in zip(beacon_x, beacon_y):
        x_values.extend((x, x, None))
        y_values.extend((y, y, None))
        z_values.extend((0.0, beacon_height, None))
    fig.add_trace(go.Scatter3d(
        x=x_values, y=y_values, z=z_values, mode="lines",
        line={"color": accent, "width": 5}, opacity=0.82,
        hoverinfo="skip", showlegend=False, name="Light pylons",
    ))
    fig.add_trace(go.Scatter3d(
        x=beacon_x, y=beacon_y, z=[beacon_height] * 4,
        mode="markers",
        marker={"size": 7, "color": accent, "symbol": "diamond",
                "line": {"color": "#e6fff0", "width": 1}},
        hoverinfo="skip", showlegend=False, name="Beacon lamps",
    ))


def _add_segment_trace(
    fig: go.Figure,
    segments: Sequence[tuple[Sequence[float], Sequence[float]]],
    color: str,
    width: float,
    name: str,
    opacity: float = 0.9,
) -> None:
    """Pack many independent 3D line segments into a single WebGL trace."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in segments:
        xs.extend((float(start[0]), float(end[0]), None))
        ys.extend((float(start[1]), float(end[1]), None))
        zs.extend((float(start[2]), float(end[2]), None))
    if xs:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line={"color": color, "width": width}, opacity=opacity,
            hoverinfo="skip", showlegend=False, name=name,
        ))


def _cylinder_geometry(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    sides: int = 8,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[float]]:
    """Return a low-poly capsule segment oriented between two 3D points."""
    start_point = np.asarray(start, dtype=np.float64)
    end_point = np.asarray(end, dtype=np.float64)
    direction = end_point - start_point
    length = max(float(np.linalg.norm(direction)), 1e-8)
    direction /= length
    reference = np.asarray((0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (0.0, 1.0, 0.0))
    axis_u = np.cross(direction, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-8)
    axis_v = np.cross(direction, axis_u)
    points: list[tuple[float, float, float]] = []
    for center in (start_point, end_point):
        for side in range(sides):
            angle = 2 * math.pi * side / sides
            point = center + radius * (math.cos(angle) * axis_u + math.sin(angle) * axis_v)
            points.append(tuple(float(value) for value in point))
    points.extend((tuple(float(value) for value in start_point),
                   tuple(float(value) for value in end_point)))
    faces: list[tuple[int, int, int]] = []
    shades: list[float] = []
    bottom_center, top_center = sides * 2, sides * 2 + 1
    for side in range(sides):
        following = (side + 1) % sides
        side_light = 0.78 + 0.26 * math.cos(2 * math.pi * (side + 0.5) / sides)
        faces.extend(((side, following, sides + following),
                      (side, sides + following, sides + side),
                      (bottom_center, following, side),
                      (top_center, sides + side, sides + following)))
        shades.extend((side_light, side_light * 0.88, 0.52, 1.13))
    return points, faces, shades


def _sphere_geometry(
    center: Sequence[float],
    radii: tuple[float, float, float],
    segments: int = 10,
    rings: int = 4,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[float]]:
    """Create a faceted UV sphere for a low-poly fighter helmet/head."""
    cx, cy, cz = map(float, center)
    rx, ry, rz = radii
    points = [(cx, cy, cz + rz)]
    for ring in range(1, rings):
        latitude = math.pi * ring / rings
        for segment in range(segments):
            longitude = 2 * math.pi * segment / segments
            points.append((
                cx + rx * math.sin(latitude) * math.cos(longitude),
                cy + ry * math.sin(latitude) * math.sin(longitude),
                cz + rz * math.cos(latitude),
            ))
    bottom_index = len(points)
    points.append((cx, cy, cz - rz))
    faces: list[tuple[int, int, int]] = []
    shades: list[float] = []
    first_ring = 1
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((0, first_ring + segment, first_ring + following))
        shades.append(1.18)
    for ring in range(rings - 2):
        upper = first_ring + ring * segments
        lower = upper + segments
        for segment in range(segments):
            following = (segment + 1) % segments
            faces.extend(((upper + segment, lower + segment, lower + following),
                          (upper + segment, lower + following, upper + following)))
            shades.extend((0.84, 0.70))
    last_ring = first_ring + (rings - 2) * segments
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((bottom_index, last_ring + following, last_ring + segment))
        shades.append(0.55)
    return points, faces, shades


def _add_fighter_mesh(fig: go.Figure, agent: dict[str, Any], color: str, label: str) -> None:
    """Draw a compact, low-poly humanoid and rifle using one mesh per fighter."""
    x, y, z = float(agent["x"]), float(agent["y"]), float(agent.get("z", 0.0))
    yaw = float(agent.get("yaw", 0.0))
    height_scale = max(0.28, min(1.05, float(agent.get("height", 1.8)) / 1.8))
    forward = np.asarray((math.sin(yaw), math.cos(yaw), 0.0))
    side = np.asarray((math.cos(yaw), -math.sin(yaw), 0.0))
    origin = np.asarray((x, y, z))
    parts: list[tuple[list[tuple[float, float, float]],
                      list[tuple[int, int, int]], list[float], str]] = []

    # Two separated legs, a torso, arms, head, and a dark forward-pointing carbine.
    for sign in (-1.0, 1.0):
        hip = origin + side * (0.14 * sign)
        parts.append((*_cylinder_geometry(hip + (0, 0, 0.08), hip + (0, 0, 0.76), 0.105),
                      _shade(color, 0.72)))
    torso_start = origin + np.asarray((0, 0, 0.62))
    torso_end = origin + np.asarray((0, 0, 1.30))
    parts.append((*_cylinder_geometry(torso_start, torso_end, 0.30, sides=10), color))
    for sign in (-1.0, 1.0):
        shoulder = origin + side * (0.27 * sign) + np.asarray((0, 0, 1.17))
        wrist = origin + forward * 0.40 + side * (0.19 * sign) + np.asarray((0, 0, 0.98))
        parts.append((*_cylinder_geometry(shoulder, wrist, 0.105), _shade(color, 0.86)))
    parts.append((*_sphere_geometry(origin + np.asarray((0, 0, 1.52)), (0.20, 0.20, 0.23)),
                  _shade(color, 1.24)))
    rifle_start = origin + forward * 0.30 + np.asarray((0, 0, 1.02))
    rifle_end = origin + forward * 0.91 + np.asarray((0, 0, 1.02))
    parts.append((*_cylinder_geometry(rifle_start, rifle_end, 0.055, sides=6), "#c2d0c9"))

    vertices_x: list[float] = []
    vertices_y: list[float] = []
    vertices_z: list[float] = []
    triangle_i: list[int] = []
    triangle_j: list[int] = []
    triangle_k: list[int] = []
    face_colors: list[str] = []
    for points, faces, shades, material in parts:
        offset = len(vertices_x)
        scaled_points = [
            (point[0], point[1], z + (point[2] - z) * height_scale)
            for point in points
        ]
        vertices_x.extend(point[0] for point in scaled_points)
        vertices_y.extend(point[1] for point in scaled_points)
        vertices_z.extend(point[2] for point in scaled_points)
        for index, (a, b, c) in enumerate(faces):
            triangle_i.append(offset + a)
            triangle_j.append(offset + b)
            triangle_k.append(offset + c)
            face_colors.append(_shade(material, shades[index]))
    fig.add_trace(go.Mesh3d(
        x=vertices_x, y=vertices_y, z=vertices_z,
        i=triangle_i, j=triangle_j, k=triangle_k,
        facecolor=face_colors,
        flatshading=True,
        lighting={"ambient": 0.66, "diffuse": 0.92, "specular": 0.32,
                  "roughness": 0.62, "fresnel": 0.10},
        lightposition={"x": -80, "y": -120, "z": 200},
        hovertemplate=f"{label}<br>{agent.get('weapon', '')}<extra></extra>",
        name=f"{label} low-poly fighter",
        showlegend=False,
    ))


def _set_scene_layout(
    fig: go.Figure,
    arena_map: ArenaMap,
    height: int,
    camera_eye: tuple[float, float, float],
    uirevision: str,
) -> go.Figure:
    maximum_z = max([5.0] + [item.z + item.height + 1.0 for item in arena_map.objects])
    fig.update_layout(
        scene={
            "xaxis": {"title": "X (m)", "range": [-arena_map.width / 2, arena_map.width / 2],
                     "backgroundcolor": BG, "gridcolor": "#263b31", "zerolinecolor": "#43634f"},
            "yaxis": {"title": "Y (m)", "range": [-arena_map.depth / 2, arena_map.depth / 2],
                     "backgroundcolor": BG, "gridcolor": "#263b31", "zerolinecolor": "#43634f"},
            "zaxis": {"title": "Z (m)", "range": [-0.2, maximum_z],
                     "backgroundcolor": BG, "gridcolor": "#263b31", "zerolinecolor": "#43634f"},
            "bgcolor": BG,
            "aspectmode": "data",
            "camera": {
                "eye": {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
        height=height,
        uirevision=uirevision,
        hovermode="closest",
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
    )
    return dark_layout(fig, height=height)


def build_map_figure(
    arena_map: ArenaMap,
    agents: list[dict[str, Any]] | None = None,
    trails: list[dict[str, Any]] | None = None,
    show_spawns: bool = True,
    height: int = 620,
    detail_count: int = 150,
    camera_eye: tuple[float, float, float] = (1.5, 1.6, 1.12),
) -> go.Figure:
    """Render a shaded, detailed 3D map while batching geometry for smooth interaction."""
    fig = go.Figure()
    signature = _map_signature(arena_map)
    theme = {
        "Dust": "#ffb547",
        "Warehouse": "#39dcaa",
        "Highrise": "#42aaff",
        "Arena": "#bd65ff",
        "Sniper Alley": "#ff5e63",
        "Custom": NEON,
    }.get(arena_map.name, NEON)

    dressing = _ambient_props(signature, int(detail_count))
    display_map = arena_map.copy()
    display_map.objects.extend(dressing)
    display_signature = _map_signature(display_map)
    _add_floor(fig, arena_map.width, arena_map.depth, theme, display_signature, detail_count)
    _add_batched_boxes(fig, display_map.objects)
    _add_beacons(fig, arena_map.width, arena_map.depth, theme)

    if show_spawns:
        spawn_x = [point[0] for point in arena_map.spawn_points]
        spawn_y = [point[1] for point in arena_map.spawn_points]
        fig.add_trace(go.Scatter3d(
            x=spawn_x, y=spawn_y, z=[0.16] * len(spawn_x), mode="markers+text",
            marker={"size": 8, "color": BLUE, "symbol": "diamond",
                    "line": {"color": "#e8fbff", "width": 1}},
            text=[f"SPAWN {index + 1}" for index in range(len(spawn_x))],
            textposition="top center", textfont={"color": BLUE, "size": 10},
            name="Spawn points", hovertemplate="Spawn<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        ))

    if agents:
        visible_agents = agents[:2]
        for index, agent in enumerate(visible_agents):
            _add_fighter_mesh(fig, agent, GREEN if index == 0 else RED, f"FIGHTER {index + 1}")
        fig.add_trace(go.Scatter3d(
            x=[float(agent["x"]) for agent in visible_agents],
            y=[float(agent["y"]) for agent in visible_agents],
            z=[float(agent.get("z", 0.0)) + float(agent.get("height", 1.8)) * 0.88
               for agent in visible_agents],
            mode="markers+text",
            marker={"size": [5, 5][:len(visible_agents)], "color": [GREEN, RED][:len(visible_agents)],
                    "symbol": ["circle", "diamond"][:len(visible_agents)],
                    "line": {"color": "#edfff0", "width": 2}},
            text=[f"P{index + 1} · {float(agent.get('hp', 0)):.0f} HP"
                  for index, agent in enumerate(visible_agents)],
            textposition="top center", textfont={"color": [GREEN, RED][:len(visible_agents)],
                                                  "size": 11, "family": "Courier New"},
            customdata=[[agent.get("weapon", ""), agent.get("ammo", 0)] for agent in visible_agents],
            hovertemplate=("%{text}<br>Weapon: %{customdata[0]}<br>Ammo: %{customdata[1]}"
                           "<br>Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>"),
            showlegend=False, name="Fighters",
        ))

        facing_segments = []
        cones_x, cones_y, cones_z, cones_u, cones_v, cones_w = [], [], [], [], [], []
        hp_segments_by_agent: list[list[tuple[Sequence[float], Sequence[float]]]] = [[], []]
        for index, agent in enumerate(visible_agents):
            x, y, z = float(agent["x"]), float(agent["y"]), float(agent.get("z", 0.0))
            height_value = float(agent.get("height", 1.8))
            facing = float(agent.get("yaw", 0.0))
            fx = x + 2.6 * math.sin(facing)
            fy = y + 2.6 * math.cos(facing)
            facing_segments.append(((x, y, z + 0.95), (fx, fy, z + 0.95)))
            cones_x.append(x)
            cones_y.append(y)
            cones_z.append(z + 0.95)
            cones_u.append(fx - x)
            cones_v.append(fy - y)
            cones_w.append(0.0)
            bar_y = y - 1.35
            bar_z = z + height_value + 0.62
            hp_fraction = max(0.0, min(1.0, float(agent.get("hp", 0.0)) / 100.0))
            hp_segments_by_agent[index].append(((x - 1.35, bar_y, bar_z),
                                                (x - 1.35 + 2.7 * hp_fraction, bar_y, bar_z)))
        _add_segment_trace(fig, facing_segments, BLUE, 5, "Fighter facing")
        if cones_x:
            fig.add_trace(go.Cone(
                x=cones_x, y=cones_y, z=cones_z,
                u=cones_u, v=cones_v, w=cones_w,
                anchor="tail", sizemode="absolute", sizeref=0.75,
                colorscale=[[0, BLUE], [1, BLUE]], showscale=False,
                hoverinfo="skip", showlegend=False, name="Aim direction",
            ))
        for index, segments in enumerate(hp_segments_by_agent):
            if segments:
                _add_segment_trace(fig, segments, GREEN if index == 0 else RED,
                                   9, f"Player {index + 1} health")

    if trails:
        hit_segments = []
        miss_segments = []
        for trail in trails[-40:]:
            start, end = trail.get("start"), trail.get("end")
            if not start or not end:
                continue
            (hit_segments if trail.get("hit") else miss_segments).append((start, end))
        _add_segment_trace(fig, miss_segments, "#8ab6cf", 2, "Shot trails", opacity=0.52)
        _add_segment_trace(fig, hit_segments, YELLOW, 4, "Hit sparks", opacity=0.9)

    fig.update_layout(title={"text": f"{arena_map.name.upper()} · {arena_map.width:g} × {arena_map.depth:g} m",
                             "x": 0.02, "xanchor": "left"})
    return _set_scene_layout(fig, arena_map, height, camera_eye,
                             uirevision=f"{arena_map.name}-{arena_map.width:g}-{arena_map.depth:g}")


def build_aim_range_figure(
    target_cell: int,
    hits: int = 0,
    misses: int = 0,
    streak: int = 0,
    detail_count: int = 120,
    height: int = 570,
) -> go.Figure:
    """Build a 3D indoor shooting gallery with nine reactive bullseye targets."""
    wall_y = 8.4
    objects = [
        ArenaObject(0, wall_y, 0, 18.5, 9.8, 0.55, "wall", "Armored target wall"),
        ArenaObject(-8.4, 0.8, 0, 1.2, 2.4, 1.2, "pillar", "Left range pillar"),
        ArenaObject(8.4, 0.8, 0, 1.2, 2.4, 1.2, "pillar", "Right range pillar"),
        ArenaObject(-6.8, -3.6, 0, 2.7, 1.5, 1.7, "crate", "Left ammo stack A"),
        ArenaObject(-5.2, -3.0, 0, 2.2, 1.0, 1.4, "crate", "Left ammo stack B"),
        ArenaObject(6.8, -3.6, 0, 2.7, 1.5, 1.7, "crate", "Right ammo stack A"),
        ArenaObject(5.2, -3.0, 0, 2.2, 1.0, 1.4, "crate", "Right ammo stack B"),
        ArenaObject(-8.8, -5.2, 0, 0.45, 3.8, 0.45, "barrel", "Safety light left"),
        ArenaObject(8.8, -5.2, 0, 0.45, 3.8, 0.45, "barrel", "Safety light right"),
    ]
    arena_map = ArenaMap("Aim Lab", 20, 19, objects, ((0, -7.5), (0, 7.0)),
                         "Precision range with nine reactive holographic targets.")
    fig = build_map_figure(arena_map, show_spawns=False, height=height,
                           detail_count=detail_count, camera_eye=(1.35, -1.75, 1.12))

    ring_x: list[float | None] = []
    ring_y: list[float | None] = []
    ring_z: list[float | None] = []
    active_x: list[float | None] = []
    active_y: list[float | None] = []
    active_z: list[float | None] = []
    center_x, center_y, center_z = [], [], []
    center_colors = []
    center_sizes = []
    target_text = []
    rings = np.linspace(0.0, 2 * math.pi, 32)
    for cell in range(9):
        column = cell % 3
        row = cell // 3
        x = (column - 1) * 5.5
        y = wall_y - 0.31
        z = 8.0 - row * 2.75
        is_active = cell == int(target_cell)
        target_color = NEON if is_active else "#ffb44c"
        for radius in (0.94, 0.66, 0.36, 0.13):
            xs = (x + radius * np.cos(rings)).tolist()
            zs = (z + radius * np.sin(rings)).tolist()
            if is_active:
                active_x.extend(xs + [None])
                active_y.extend([y] * len(xs) + [None])
                active_z.extend(zs + [None])
            else:
                ring_x.extend(xs + [None])
                ring_y.extend([y] * len(xs) + [None])
                ring_z.extend(zs + [None])
        center_x.append(x)
        center_y.append(y - 0.035)
        center_z.append(z)
        center_colors.append(target_color)
        center_sizes.append(14 if is_active else 8)
        target_text.append(f"TARGET {cell + 1}" + (" · ACTIVE" if is_active else ""))

    fig.add_trace(go.Scatter3d(
        x=ring_x, y=ring_y, z=ring_z, mode="lines",
        line={"color": "#ff9b43", "width": 3}, opacity=0.76,
        hoverinfo="skip", showlegend=False, name="Holographic target rings",
    ))
    fig.add_trace(go.Scatter3d(
        x=active_x, y=active_y, z=active_z, mode="lines",
        line={"color": NEON, "width": 6}, opacity=1.0,
        hoverinfo="skip", showlegend=False, name="Active target lock",
    ))
    fig.add_trace(go.Scatter3d(
        x=center_x, y=center_y, z=center_z, mode="markers+text",
        marker={"size": center_sizes, "color": center_colors, "symbol": "circle",
                "line": {"color": "#f5fff6", "width": 1}},
        text=[str(index + 1) for index in range(9)], textposition="middle center",
        textfont={"color": "#08110a", "size": 10, "family": "Courier New"},
        customdata=target_text,
        hovertemplate="%{customdata}<br>3D bullseye<extra></extra>",
        showlegend=False, name="Target cores",
    ))

    origin = (0.0, -6.0, 1.55)
    target_x = (int(target_cell) % 3 - 1) * 5.5
    target_z = 8.0 - (int(target_cell) // 3) * 2.75
    _add_segment_trace(fig, [(origin, (target_x, wall_y - 0.25, target_z))],
                       NEON, 2, "Targeting laser", opacity=0.36)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[-6.0], z=[1.55], mode="markers+text",
        marker={"size": 11, "color": BLUE, "symbol": "diamond",
                "line": {"color": "#efffff", "width": 1}},
        text=["YOU"], textposition="bottom center",
        textfont={"color": BLUE, "size": 10},
        hovertemplate="Firing position<extra></extra>",
        showlegend=False, name="Player firing position",
    ))
    fig.update_layout(title={"text": f"AIM LAB · {hits} HITS · {misses} MISSES · STREAK {streak}",
                             "x": 0.02, "xanchor": "left"})
    fig.update_scenes(uirevision="aim-range")
    return fig


def build_dodge_arena_figure(
    game: dict[str, Any],
    detail_count: int = 100,
    height: int = 570,
) -> go.Figure:
    """Turn the grid survival mini-game into a compact 3D projectile arena."""
    size = 5
    grid_step = 3.0
    arena_extent = 9.0
    objects = [
        ArenaObject(-9.2, 0, 0, 0.45, 3.2, 19.0, "wall", "West blast shield"),
        ArenaObject(9.2, 0, 0, 0.45, 3.2, 19.0, "wall", "East blast shield"),
        ArenaObject(0, 9.2, 0, 19.0, 3.2, 0.45, "wall", "Turret wall"),
        ArenaObject(0, -9.2, 0, 19.0, 3.2, 0.45, "wall", "Safe-zone wall"),
        ArenaObject(-7.6, 6.4, 0, 1.2, 2.2, 1.2, "pillar", "Left turret pylon"),
        ArenaObject(7.6, 6.4, 0, 1.2, 2.2, 1.2, "pillar", "Right turret pylon"),
        ArenaObject(-7.4, -7.4, 0, 1.8, 1.5, 1.8, "crate", "South cover A"),
        ArenaObject(7.4, -7.4, 0, 1.8, 1.5, 1.8, "crate", "South cover B"),
    ]
    arena_map = ArenaMap("Dodge Grid", 20, 20, objects, ((0, -arena_extent), (0, arena_extent)),
                         "Five-lane survival arena with overhead projectile turrets.")
    tick = int(game.get("tick", 0))
    active = bool(game.get("active", False))
    theme = NEON if active else RED
    fig = build_map_figure(arena_map, show_spawns=False, height=height,
                           detail_count=detail_count, camera_eye=(1.55, -1.9, 1.4))

    lane_segments = []
    turret_x = []
    turret_y = []
    turret_z = []
    for lane in range(size):
        world_x = (lane - size // 2) * grid_step
        lane_segments.append(((world_x, -arena_extent + 0.5, 0.04),
                              (world_x, arena_extent - 0.5, 0.04)))
        turret_x.append(world_x)
        turret_y.append(arena_extent - 0.35)
        turret_z.append(2.85)
    _add_segment_trace(fig, lane_segments, "#65806d", 2, "Dodge lanes", opacity=0.52)

    fig.add_trace(go.Cone(
        x=turret_x, y=turret_y, z=turret_z,
        u=[0.0] * size, v=[-1.0] * size, w=[0.0] * size,
        anchor="tail", sizemode="absolute", sizeref=0.5,
        colorscale=[[0, "#ff654d"], [1, "#ff654d"]], showscale=False,
        hoverinfo="skip", showlegend=False, name="Automated turrets",
    ))
    fig.add_trace(go.Scatter3d(
        x=turret_x, y=turret_y, z=turret_z, mode="markers",
        marker={"size": 8, "color": "#ff654d", "symbol": "diamond",
                "line": {"color": "#ffe1c6", "width": 1}},
        hoverinfo="skip", showlegend=False, name="Turret muzzle lights",
    ))

    projectile_x, projectile_y, projectile_z = [], [], []
    projectile_segments = []
    projectiles = game.get("projectiles", [])
    for lane, row in projectiles:
        world_x = (int(lane) - size // 2) * grid_step
        world_y = (size // 2 - int(row)) * grid_step
        world_z = 1.3 + 0.25 * math.sin((tick + int(row)) * 0.65)
        projectile_x.append(world_x)
        projectile_y.append(world_y)
        projectile_z.append(world_z)
        projectile_segments.append(((world_x, world_y + 0.72, world_z),
                                   (world_x, world_y - 0.45, world_z)))
    if projectile_x:
        _add_segment_trace(fig, projectile_segments, "#ff464e", 5, "Incoming energy bolts", opacity=0.95)
        fig.add_trace(go.Scatter3d(
            x=projectile_x, y=projectile_y, z=projectile_z,
            mode="markers",
            marker={"size": 12, "color": "#ff3e55", "symbol": "diamond",
                    "line": {"color": "#fff0c7", "width": 1}},
            hovertemplate="Incoming bolt<extra></extra>",
            showlegend=False, name="Projectile cores",
        ))

    player_x = (int(game["player"][0]) - size // 2) * grid_step
    player_y = (size // 2 - int(game["player"][1])) * grid_step
    player_color = RED if game.get("hit") else GREEN
    fig.add_trace(go.Scatter3d(
        x=[player_x], y=[player_y], z=[1.05], mode="markers+text",
        marker={"size": 18, "color": player_color,
                "symbol": "x" if game.get("hit") else "diamond",
                "line": {"color": "#f2fff5", "width": 2}},
        text=["HIT" if game.get("hit") else "YOU"], textposition="top center",
        textfont={"color": player_color, "size": 12, "family": "Courier New"},
        hovertemplate="Player · lane %{x:.0f}, row %{y:.0f}<extra></extra>",
        showlegend=False, name="Player beacon",
    ))
    fig.add_trace(go.Cone(
        x=[player_x], y=[player_y], z=[1.0],
        u=[0.0], v=[1.4], w=[0.0],
        anchor="tail", sizemode="absolute", sizeref=0.45,
        colorscale=[[0, player_color], [1, player_color]], showscale=False,
        hoverinfo="skip", showlegend=False, name="Player direction",
    ))
    fig.update_layout(title={"text": f"DODGE ARENA · SCORE {game.get('score', 0)} · SALVOS EVADED {game.get('dodged', 0)}",
                             "x": 0.02, "xanchor": "left"})
    fig.update_scenes(uirevision="dodge-arena")
    return fig

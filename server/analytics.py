"""CSV-backed telemetry, zone heatmaps, and resource monitoring.

Replaces the former Plotly/Streamlit stats and heatmap tabs. All numbers are
returned as plain JSON so the browser can draw them on a canvas without pandas.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from env.maps import MAP_NAMES, ArenaMap, create_map
from env.weapons import WEAPON_NAMES
from server.config import LOGS_DIR, OBJECT_COLORS, NEON, theme_color


# Tail-read window for CSV telemetry (4 MB ≈ tens of thousands of rows).
TAIL_READ_BYTES = 4 * 1024 * 1024
# Newest episodes used for the heatmap; older rows stay on disk.
HEATMAP_EVENT_LIMIT = 50_000


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read a runtime CSV defensively; a partially written last line is skipped.

    With ``limit`` the newest rows are kept and only the tail of a large file is
    parsed, so a long training run cannot make a dashboard request read hundreds
    of megabytes from disk.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, str]] = []
    try:
        size = path.stat().st_size
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return []
            if limit is not None and size > TAIL_READ_BYTES:
                handle.seek(max(0, size - TAIL_READ_BYTES))
                handle.readline()  # drop the partial first line of the window
                reader = csv.reader(handle)
            for values in reader:
                # A partially written final line has a different column count.
                if len(values) != len(header):
                    continue
                rows.append(dict(zip(header, values)))
    except (OSError, csv.Error, UnicodeDecodeError):
        return rows
    if limit is not None and len(rows) > limit:
        return rows[-limit:]
    return rows


def _kill_ttks(rows: list[dict[str, str]]) -> list[float]:
    """Time-to-kill values of episodes that ended in a confirmed kill.

    ``ttk`` alone is misleading: the environment logs the match clock for
    time-limit decisions and defeats as well. Rows written before the ``killed``
    column existed fall back to "won and not a draw", which is the closest
    available approximation.
    """
    values: list[float] = []
    for row in rows:
        ttk = _to_float(row.get("ttk"))
        if ttk is None or _to_float(row.get("win")) != 1.0:
            continue
        killed = _to_float(row.get("killed"))
        if killed is None:
            killed = 0.0 if _to_float(row.get("draw")) else 1.0
        if killed == 1.0:
            values.append(ttk)
    return values



def _series(rows: list[dict[str, str]], field: str, scale: float = 1.0) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        parsed = _to_float(row.get(field))
        values.append(None if parsed is None else parsed * scale)
    return values


def training_stats(logs_dir: Path = LOGS_DIR, live: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate training metrics, episode events, and system resources."""
    metrics_rows = read_csv_rows(logs_dir / "training_metrics.csv", limit=4000)
    event_rows = read_csv_rows(logs_dir / "heatmap_events.csv", limit=5000)
    live_metrics = dict(live or {})
    last = metrics_rows[-1] if metrics_rows else {}

    fps_values = [value for value in _series(metrics_rows, "fps") if value is not None]
    avg_fps = float(np.mean(fps_values)) if fps_values else 0.0
    ttk_values = [value for value in _series(event_rows, "ttk") if value is not None]
    kill_ttks = _kill_ttks(event_rows)
    kill_distances = [value for value in _series(event_rows, "avg_kill_distance")
                      if value is not None]
    wins = [value for value in _series(event_rows, "win") if value is not None]
    weapon_counts: dict[str, int] = {}
    opponent_counts: dict[str, int] = {}
    deaths_by_weapon: dict[str, int] = {}
    for row in event_rows:
        weapon = str(row.get("weapon") or "").strip()
        if weapon:
            weapon_counts[weapon] = weapon_counts.get(weapon, 0) + 1
        opponent = str(row.get("opponent_weapon") or "").strip()
        if opponent:
            opponent_counts[opponent] = opponent_counts.get(opponent, 0) + 1
        win_value = _to_float(row.get("win"), 0.0) or 0.0
        if win_value < 0.5 and opponent:
            deaths_by_weapon[opponent] = deaths_by_weapon.get(opponent, 0) + 1

    def live_or_last(field: str, default: float = 0.0) -> float:
        value = _to_float(live_metrics.get(field))
        if value is None:
            value = _to_float(last.get(field), default)
        return float(value if value is not None else default)

    resources = resource_snapshot()
    top_weapon = max(weapon_counts.items(), key=lambda item: item[1])[0] if weapon_counts else "—"
    worst_weapon = max(deaths_by_weapon.items(), key=lambda item: item[1])[0] if deaths_by_weapon else "—"
    return {
        "series": {
            "steps": _series(metrics_rows, "steps"),
            "reward": _series(metrics_rows, "avg_reward"),
            # Both curves are plotted as percent: a win at the time limit is only a
            # HP comparison, the kill rate is the honest number (see ANALYSE.md).
            "win_rate": _series(metrics_rows, "win_rate", 100.0),
            "kill_rate": _series(metrics_rows, "kill_rate", 100.0),
            "fps": _series(metrics_rows, "fps"),
        },
        "histogram": {
            # Kills only – otherwise the "TTK" histogram is really an episode-length plot.
            "ttk": kill_ttks[-2000:],
            "edges": [round(value, 3) for value in np.linspace(
                0.0, max([1.0] + kill_ttks[-2000:]) if kill_ttks else 1.0, 25)],
        },
        "weapons": weapon_counts,
        "opponents": opponent_counts,
        "deaths_by_weapon": deaths_by_weapon,
        "summary": {
            "fps": live_or_last("fps"),
            "avg_fps": avg_fps,
            "steps": int(live_or_last("timesteps") if live_metrics else
                         (_to_float(last.get("steps"), 0.0) or 0.0)),
            "episodes": int(live_or_last("episodes") if live_metrics else
                            (_to_float(last.get("episodes"), 0.0) or 0.0)),
            "win_rate": float(np.mean(wins)) if wins else live_or_last("win_rate"),
            "kill_rate": live_or_last("kill_rate"),
            "avg_reward": live_or_last("avg_reward"),
            "avg_ttk": float(np.mean(kill_ttks)) if kill_ttks
            else (0.0 if event_rows else live_or_last("avg_ttk")),
            "kill_count": len(kill_ttks),
            "accuracy": live_or_last("accuracy"),
            "headshot_pct": live_or_last("headshot_pct"),
            "avg_kill_distance": float(np.mean(kill_distances)) if kill_distances else None,
            "top_weapon": top_weapon,
            "worst_matchup": worst_weapon,
            "training_time": live_or_last("elapsed"),
            "logged_episodes": len(event_rows),
        },
        "resources": resources,
    }


def resource_snapshot() -> dict[str, Any]:
    """CPU and RAM usage for the server process tree (psutil is optional)."""
    snapshot: dict[str, Any] = {"cpu_percent": None, "process_mb": None,
                                "system_ram_percent": None, "children": None}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        children = 0
        try:
            child_processes = process.children(recursive=True)
            children = len(child_processes)
            rss += sum(child.memory_info().rss for child in child_processes)
        except (psutil.Error, OSError):
            pass
        snapshot["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        snapshot["process_mb"] = rss / (1024.0 * 1024.0)
        snapshot["system_ram_percent"] = float(psutil.virtual_memory().percent)
        snapshot["children"] = children
    except Exception:  # noqa: BLE001 - monitoring must never break the API
        pass
    return snapshot


def heatmap_payload(
    map_name: str,
    *,
    arena_map: ArenaMap | None = None,
    weapon: str | None = None,
    mode: str = "Both",
    episode_range: tuple[int, int] | None = None,
    distance_range: tuple[float, float] | None = None,
    logs_dir: Path = LOGS_DIR,
    extra_events: list[dict[str, Any]] | None = None,
    bins: int = 48,
) -> dict[str, Any]:
    """Build the death/safe/kill zone grid plus cover outlines for one map."""
    events: list[dict[str, Any]] = [
        dict(row) for row in read_csv_rows(logs_dir / "heatmap_events.csv", limit=HEATMAP_EVENT_LIMIT)
    ]
    if extra_events:
        events.extend(dict(event) for event in extra_events)
    events = [event for event in events if str(event.get("map", "")) == map_name]

    if weapon and weapon != "All weapons":
        events = [
            event for event in events
            if weapon in {str(event.get("weapon") or ""), str(event.get("opponent_weapon") or "")}
        ]
    if episode_range is not None:
        low, high = episode_range
        events = [event for event in events
                  if low <= (_to_float(event.get("episode"), 0.0) or 0.0) <= high]
    if distance_range is not None:
        low, high = distance_range
        filtered = []
        for event in events:
            distance = _to_float(event.get("distance"))
            if distance is None or low <= distance <= high:
                filtered.append(event)
        events = filtered

    arena_map = arena_map or create_map(map_name)
    x_edges = np.linspace(-arena_map.width / 2, arena_map.width / 2, bins + 1)
    y_edges = np.linspace(-arena_map.depth / 2, arena_map.depth / 2, bins + 1)
    death_grid = np.zeros((bins, bins), dtype=np.float32)
    kill_grid = np.zeros((bins, bins), dtype=np.float32)

    def fill(grid: np.ndarray, x_field: str, y_field: str) -> None:
        x = np.asarray([_to_float(event.get(x_field), math.nan) for event in events], dtype=float)
        y = np.asarray([_to_float(event.get(y_field), math.nan) for event in events], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            counts, _, _ = np.histogram2d(x[valid], y[valid], bins=(x_edges, y_edges))
            grid += counts.astype(np.float32)

    if mode in {"Deaths", "Both"}:
        fill(death_grid, "death_x", "death_y")
    if mode in {"Kills", "Both"}:
        fill(kill_grid, "kill_x", "kill_y")

    net = kill_grid - death_grid
    peak = max(1.0, float(np.max(np.abs(net))))
    net = net / peak
    x_centres = ((x_edges[:-1] + x_edges[1:]) / 2.0).round(2)
    y_centres = ((y_edges[:-1] + y_edges[1:]) / 2.0).round(2)
    hottest_kill = _hottest_zone(kill_grid, x_centres, y_centres)
    hottest_death = _hottest_zone(death_grid, x_centres, y_centres)
    weapon_names = sorted({str(event.get("weapon") or "") for event in events}
                          | {str(event.get("opponent_weapon") or "") for event in events}
                          | set(WEAPON_NAMES))
    weapon_names = [name for name in weapon_names if name]
    episodes = [_to_float(event.get("episode")) for event in events]
    episodes = [value for value in episodes if value is not None]
    return {
        "map": map_name,
        "width": arena_map.width,
        "depth": arena_map.depth,
        "theme": theme_color(map_name),
        "grid": net.T.round(3).tolist(),
        "x": [float(value) for value in x_centres],
        "y": [float(value) for value in y_centres],
        "bins": bins,
        "objects": [
            {
                "x": float(item.x), "y": float(item.y),
                "x0": float(item.x - item.width / 2), "x1": float(item.x + item.width / 2),
                "y0": float(item.y - item.depth / 2), "y1": float(item.y + item.depth / 2),
                "kind": item.kind, "color": OBJECT_COLORS.get(item.kind, "#66727c"),
            }
            for item in arena_map.objects
        ],
        "events": len(events),
        "kills": int(np.sum(kill_grid)),
        "deaths": int(np.sum(death_grid)),
        "hottest_kill": hottest_kill,
        "hottest_death": hottest_death,
        "maps": list(MAP_NAMES),
        "weapons": ["All weapons", *weapon_names],
        "max_episode": int(max(episodes)) if episodes else 0,
        "accent": NEON,
    }


def _hottest_zone(grid: np.ndarray, x_centres: np.ndarray, y_centres: np.ndarray) -> dict[str, Any] | None:
    if not np.any(grid):
        return None
    index_x, index_y = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return {
        "x": float(x_centres[index_x]),
        "y": float(y_centres[index_y]),
        "count": int(grid[index_x, index_y]),
    }


def benchmark_payload(snapshot: dict[str, Any], total: int) -> dict[str, Any]:
    """Add derived fields (best row, matrix shape) to a benchmark snapshot."""
    rows = list(snapshot.get("results", []))
    completed = [row for row in rows if row.get("status") == "complete"]
    best = max(completed, key=lambda row: row.get("steps_per_second", 0.0)) if completed else None
    done = sum(1 for row in rows if row.get("status") in {"complete", "stopped"})
    return {
        "status": snapshot.get("status", "stopped"),
        "results": rows,
        "current": list(snapshot["current"]) if snapshot.get("current") else None,
        "error": snapshot.get("error"),
        "total": int(total),
        "completed": int(done),
        "best": best,
        "progress": min(1.0, done / max(1, total)),
    }

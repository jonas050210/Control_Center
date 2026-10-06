"""Death, kill, and safe-zone density map with event filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from env.maps import MAP_NAMES, create_map
from gui.common import LOGS_DIR, NEON, dark_layout


HEAT_COLORS = [
    [0.0, "#ff0040"],
    [0.48, "#bc173b"],
    [0.5, "#00a833"],
    [0.54, "#00a833"],
    [1.0, "#ffaa00"],
]


def _load_events() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    path = LOGS_DIR / "heatmap_events.csv"
    if path.exists() and path.stat().st_size:
        try:
            frames.append(pd.read_csv(path, on_bad_lines="skip"))
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            pass
    arena_events = st.session_state.get("arena_events", [])
    if arena_events:
        frames.append(pd.DataFrame(arena_events))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _heatmap_figure(arena_map, events: pd.DataFrame, mode: str, height: int = 620) -> go.Figure:
    bins = 48
    x_edges = np.linspace(-arena_map.width / 2, arena_map.width / 2, bins + 1)
    y_edges = np.linspace(-arena_map.depth / 2, arena_map.depth / 2, bins + 1)
    death_grid = np.zeros((bins, bins), dtype=np.float32)
    kill_grid = np.zeros((bins, bins), dtype=np.float32)

    if not events.empty:
        if mode in {"Deaths", "Both"} and {"death_x", "death_y"}.issubset(events.columns):
            x = pd.to_numeric(events["death_x"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(events["death_y"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                death_grid, _, _ = np.histogram2d(x[valid], y[valid], bins=(x_edges, y_edges))
        if mode in {"Kills", "Both"} and {"kill_x", "kill_y"}.issubset(events.columns):
            x = pd.to_numeric(events["kill_x"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(events["kill_y"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                kill_grid, _, _ = np.histogram2d(x[valid], y[valid], bins=(x_edges, y_edges))

    net = kill_grid - death_grid
    peak = max(1.0, float(np.max(np.abs(net))))
    # Empty/low-event cells remain green and represent areas with no recorded deaths.
    net = net / peak
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    fig = go.Figure(go.Heatmap(
        x=x_centres,
        y=y_centres,
        z=net.T,
        zmin=-1,
        zmax=1,
        colorscale=HEAT_COLORS,
        opacity=0.72,
        colorbar={
            "title": {"text": "Death ← 0 → Kill", "font": {"color": NEON}},
            "tickvals": [-1, 0, 1],
            "ticktext": ["Death", "Safe", "Kill"],
            "tickfont": {"color": NEON},
        },
        hovertemplate="X %{x:.1f}m · Y %{y:.1f}m<br>relative zone score %{z:.2f}<extra></extra>",
        name="Event density",
    ))
    for item in arena_map.objects:
        line_color = "#64736a"
        if item.kind in {"crate", "barrel", "pillar", "shelf", "wall"}:
            line_color = "#8d795d"
        fig.add_shape(
            type="rect",
            x0=item.x - item.width / 2,
            x1=item.x + item.width / 2,
            y0=item.y - item.depth / 2,
            y1=item.y + item.depth / 2,
            line={"color": line_color, "width": 1},
            fillcolor="rgba(75,85,78,0.16)",
            layer="above",
        )
    fig.add_shape(type="rect", x0=-arena_map.width / 2, x1=arena_map.width / 2,
                  y0=-arena_map.depth / 2, y1=arena_map.depth / 2,
                  line={"color": NEON, "width": 1.5}, fillcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="X (metres)", range=[-arena_map.width / 2, arena_map.width / 2],
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="Y (metres)", range=[-arena_map.depth / 2, arena_map.depth / 2])
    return dark_layout(fig, f"{arena_map.name.upper()} · DEATH / SAFE / KILL ZONES", height=height)


def render() -> None:
    st.subheader("🔥 HEATMAP · DEATH / SAFE / KILL ZONES")
    st.caption("Zone intensity is collected from completed training episodes and browser-side arena matches.")
    events = _load_events()
    map_name = st.selectbox("MAP", MAP_NAMES, key="heatmap_map")
    filtered = events.copy()
    if not filtered.empty and "map" in filtered.columns:
        filtered = filtered[filtered["map"].astype(str) == map_name]

    filter_cols = st.columns([1.2, 1.3, 1.5, 1])
    with filter_cols[0]:
        weapon_values = ["All weapons"]
        if not filtered.empty:
            for column in ("weapon", "opponent_weapon"):
                if column in filtered.columns:
                    weapon_values.extend(filtered[column].dropna().astype(str).unique().tolist())
        weapon_values = list(dict.fromkeys(weapon_values))
        weapon_filter = st.selectbox("WEAPON", weapon_values, key="heatmap_weapon_filter")
    with filter_cols[1]:
        episode_series = pd.to_numeric(filtered.get("episode", pd.Series(dtype=float)), errors="coerce").dropna()
        max_episode = max(2, int(episode_series.max()) if len(episode_series) else 2)
        episode_range = st.slider("EPISODE RANGE", min_value=1, max_value=max_episode,
                                  value=(1, max_episode), key="heatmap_episode_range")
    with filter_cols[2]:
        distance_range = st.slider("EVENT DISTANCE (m)", min_value=0, max_value=100,
                                   value=(0, 100), key="heatmap_distance_range")
    with filter_cols[3]:
        mode = st.radio("SHOW", ["Both", "Deaths", "Kills"], horizontal=False,
                        key="heatmap_mode")

    if not filtered.empty:
        if weapon_filter != "All weapons":
            weapon_mask = pd.Series(False, index=filtered.index)
            for column in ("weapon", "opponent_weapon"):
                if column in filtered.columns:
                    weapon_mask |= filtered[column].astype(str) == weapon_filter
            filtered = filtered[weapon_mask]
        if "episode" in filtered.columns:
            episode_values = pd.to_numeric(filtered["episode"], errors="coerce")
            filtered = filtered[episode_values.between(episode_range[0], episode_range[1])]
        if "distance" in filtered.columns:
            distances = pd.to_numeric(filtered["distance"], errors="coerce")
            filtered = filtered[distances.between(distance_range[0], distance_range[1]) | distances.isna()]

    arena_map = create_map(map_name)
    stats = st.columns(3)
    stats[0].metric("FILTERED EPISODES", f"{len(filtered):,}")
    stats[1].metric("MAP COVER", f"{arena_map.cover_density:.1f}%")
    stats[2].metric("AVG SIGHTLINE", f"{arena_map.average_sightline:.1f}m")
    st.plotly_chart(_heatmap_figure(arena_map, filtered, mode), use_container_width=True,
                    key="heatmap_plot")
    if filtered.empty:
        st.info("No events match the selected filters yet. Safe floor areas stay green until logged data appears.")

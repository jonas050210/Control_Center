"""Live training metrics and dark-theme Plotly charts."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st

from gui.common import LOGS_DIR, RED, NEON, YELLOW, dark_layout, format_percent


def _read_csv(filename: str) -> pd.DataFrame:
    path = LOGS_DIR / filename
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _draw_reward_chart(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not metrics.empty and {"steps", "avg_reward"}.issubset(metrics.columns):
        x = pd.to_numeric(metrics["steps"], errors="coerce")
        y = pd.to_numeric(metrics["avg_reward"], errors="coerce")
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Episode reward",
                                 line={"color": NEON, "width": 2.5}, fill="tozeroy",
                                 fillcolor="rgba(0,255,65,0.08)"))
    else:
        fig.add_annotation(text="Reward data appears after the first training update.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(title="Timesteps")
    fig.update_yaxes(title="Average reward")
    return dark_layout(fig, "REWARD OVER TIME", height=340)


def _draw_win_chart(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not metrics.empty and {"steps", "win_rate"}.issubset(metrics.columns):
        x = pd.to_numeric(metrics["steps"], errors="coerce")
        y = 100.0 * pd.to_numeric(metrics["win_rate"], errors="coerce")
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Win rate",
                                 line={"color": "#6cff88", "width": 2.5},
                                 marker={"color": NEON, "size": 4}))
    else:
        fig.add_annotation(text="Win-rate data appears after completed episodes.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(title="Timesteps")
    fig.update_yaxes(title="Win rate (%)", range=[0, 100])
    return dark_layout(fig, "WIN-RATE CURVE", height=340)


def _draw_ttk_chart(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not events.empty and "ttk" in events.columns:
        values = pd.to_numeric(events["ttk"], errors="coerce").dropna()
        if len(values):
            fig.add_trace(go.Histogram(x=values, nbinsx=24, name="TTK",
                                       marker={"color": "#00cc33", "line": {"color": NEON, "width": 1}}))
        else:
            fig.add_annotation(text="No completed rounds logged yet.", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False)
    else:
        fig.add_annotation(text="No completed rounds logged yet.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(title="Time to kill (seconds)")
    fig.update_yaxes(title="Rounds")
    return dark_layout(fig, "TTK DISTRIBUTION", height=340)


def _draw_weapon_chart(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not events.empty and "weapon" in events.columns:
        counts = events["weapon"].dropna().astype(str).value_counts()
        if not counts.empty:
            fig.add_trace(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.55,
                marker={"colors": [NEON, "#00aaff", "#ffaa00", RED, "#a855f7", "#55ddaa"],
                        "line": {"color": "#0a0a0a", "width": 2}},
                textfont={"color": "#e6ffe9"},
            ))
    if not fig.data:
        fig.add_annotation(text="Weapon usage is collected after rounds finish.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.08})
    return dark_layout(fig, "WEAPON USAGE", height=340)


def _render_dashboard() -> None:
    st.subheader("📊 TRAINING TELEMETRY · LIVE PERFORMANCE")
    metrics_frame = _read_csv("training_metrics.csv")
    events = _read_csv("heatmap_events.csv")
    job = st.session_state.get("training_job")
    snapshot = job.snapshot() if job is not None else None
    live_metrics = snapshot["metrics"] if snapshot else {}
    last_row = metrics_frame.iloc[-1].to_dict() if not metrics_frame.empty else {}

    current_fps = _finite(live_metrics.get("fps", last_row.get("fps", 0)))
    avg_fps = _finite(pd.to_numeric(metrics_frame.get("fps", pd.Series(dtype=float)), errors="coerce").mean())
    total_steps = int(_finite(live_metrics.get("timesteps", last_row.get("steps", 0))))
    episodes = int(_finite(live_metrics.get("episodes", last_row.get("episodes", 0))))
    win_rate = _finite(live_metrics.get("win_rate", last_row.get("win_rate", 0)))
    avg_reward = _finite(live_metrics.get("avg_reward", last_row.get("avg_reward", 0)))
    avg_ttk = _finite(live_metrics.get("avg_ttk", 0))
    if not events.empty and "ttk" in events.columns:
        event_ttk = pd.to_numeric(events["ttk"], errors="coerce").dropna()
        if len(event_ttk):
            avg_ttk = float(event_ttk.mean())
    headshot_pct = _finite(live_metrics.get("headshot_pct", last_row.get("headshot_pct", 0)))
    accuracy = _finite(live_metrics.get("accuracy", last_row.get("accuracy", 0)))
    kill_distance = 0.0
    if not events.empty and "avg_kill_distance" in events.columns:
        distances = pd.to_numeric(events["avg_kill_distance"], errors="coerce").dropna()
        if len(distances):
            kill_distance = float(distances.mean())
    deaths_by_weapon = "—"
    if not events.empty and {"win", "opponent_weapon"}.issubset(events.columns):
        losses = events[pd.to_numeric(events["win"], errors="coerce").fillna(0) < 0.5]
        counts = losses["opponent_weapon"].dropna().astype(str).value_counts()
        if not counts.empty:
            deaths_by_weapon = f"{counts.index[0]} · {int(counts.iloc[0])}"
    elapsed = _finite(live_metrics.get("elapsed", last_row.get("elapsed", 0)))
    training_time = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"

    top = st.columns(4)
    top[0].metric("CURRENT FPS", f"{current_fps:,.0f}")
    top[1].metric("AVG FPS", f"{avg_fps:,.0f}")
    top[2].metric("TOTAL STEPS", f"{total_steps:,}")
    top[3].metric("TOTAL EPISODES", f"{episodes:,}")
    second = st.columns(4)
    second[0].metric("WIN RATE", format_percent(win_rate))
    second[1].metric("AVG REWARD", f"{avg_reward:.2f}")
    second[2].metric("AVG TTK", f"{avg_ttk:.2f}s")
    second[3].metric("HEADSHOT SHARE", format_percent(headshot_pct))
    third = st.columns(4)
    third[0].metric("ACCURACY", format_percent(accuracy))
    third[1].metric("AVG KILL DISTANCE", f"{kill_distance:.1f}m" if kill_distance else "—")
    third[2].metric("DEATHS BY WEAPON", deaths_by_weapon)
    third[3].metric("TRAINING TIME", training_time)

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.plotly_chart(_draw_reward_chart(metrics_frame), use_container_width=True)
    with chart_b:
        st.plotly_chart(_draw_win_chart(metrics_frame), use_container_width=True)
    chart_c, chart_d = st.columns(2)
    with chart_c:
        st.plotly_chart(_draw_ttk_chart(events), use_container_width=True)
    with chart_d:
        st.plotly_chart(_draw_weapon_chart(events), use_container_width=True)

    st.markdown("#### SYSTEM RESOURCE MONITOR")
    cpu_col, ram_col, status_col = st.columns(3)
    try:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_mb = process.memory_info().rss / (1024 * 1024)
        cpu_col.metric("CPU UTILIZATION", f"{cpu:.0f}%", "system-wide")
        ram_col.metric("PROCESS RAM", f"{process_mb:,.0f} MB", f"{memory.percent:.0f}% system RAM used")
    except (OSError, AttributeError):
        cpu_col.metric("CPU UTILIZATION", "n/a")
        ram_col.metric("PROCESS RAM", "n/a")
    if snapshot:
        status_col.metric("RUN STATUS", snapshot["status"].upper())
    else:
        status_col.metric("RUN STATUS", "IDLE")


def render() -> None:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        live_dashboard = fragment(run_every="2s")(_render_dashboard)
        live_dashboard()
    else:
        _render_dashboard()

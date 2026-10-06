"""Shared paths, plotting defaults, and session helpers for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_MAP_PATH = DATA_DIR / "custom_map.json"

NEON = "#00ff41"
GREEN = "#00cc33"
RED = "#ff0040"
YELLOW = "#ffaa00"
BLUE = "#00aaff"
BG = "#0a0a0a"
PANEL = "#141414"


def dark_layout(fig: Any, title: str | None = None, height: int | None = None) -> Any:
    """Apply the project's Plotly dark theme to a figure."""
    layout: dict[str, Any] = {
        "paper_bgcolor": BG,
        "plot_bgcolor": PANEL,
        "font": {"color": NEON, "family": "Courier New, monospace", "size": 12},
        "margin": {"l": 24, "r": 24, "t": 54 if title else 24, "b": 24},
        "hoverlabel": {"bgcolor": "#101810", "font": {"color": NEON}},
        "legend": {"bgcolor": "rgba(10,10,10,0.65)", "bordercolor": "#174b24", "borderwidth": 1},
    }
    if title:
        layout["title"] = {"text": title, "x": 0.02, "xanchor": "left"}
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#233327", zerolinecolor="#294432", linecolor="#31533a")
    fig.update_yaxes(gridcolor="#233327", zerolinecolor="#294432", linecolor="#31533a")
    return fig


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{100.0 * float(value):.{decimals}f}%"


def csv_path(filename: str) -> Path:
    return LOGS_DIR / filename


def get_custom_map(st: Any) -> Any:
    """Return the browser's custom map, loading the last saved JSON layout once."""
    from env.maps import create_map

    overrides = st.session_state.setdefault("map_preview_overrides", {})
    if "Custom" in overrides:
        return overrides["Custom"]
    if CUSTOM_MAP_PATH.exists():
        try:
            from env.map_io import load_map
            custom_map = load_map(CUSTOM_MAP_PATH, force_custom_name=True)
            st.session_state.pop("custom_map_load_error", None)
        except ValueError as exc:
            custom_map = create_map("Custom")
            st.session_state["custom_map_load_error"] = str(exc)
    else:
        custom_map = create_map("Custom")
    overrides["Custom"] = custom_map
    return custom_map


def ensure_session_defaults(st: Any) -> None:
    """Initialize persistent per-browser state without resetting user choices."""
    defaults = {
        "training_job": None,
        "benchmark_job": None,
        "arena_env": None,
        "arena_messages": [],
        "arena_model_cache": {},
        "arena_running": False,
        "arena_events": [],
        "playground_env": None,
        "playground_signature": None,
        "playground_messages": [],
        "playground_demo_rows": [],
        "aim_game": None,
        "aim_best_score": 0,
        "dodge_game": None,
        "dodge_best_score": 0,
        "map_preview_seed": 2026,
        "map_preview_overrides": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

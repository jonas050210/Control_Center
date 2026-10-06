"""Shared paths, theme constants, and dependency probing for the control center."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import platform
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
WEB_DIR = PROJECT_ROOT / "web"
VENDOR_DIR = WEB_DIR / "vendor"
CUSTOM_MAP_PATH = DATA_DIR / "custom_map.json"
DEMOS_PATH = DATA_DIR / "demos.csv"

APP_NAME = "NEURAL ARENA · Control Center"
APP_VERSION = "3.0"

# Cyberpunk palette shared by the CSS theme and the WebGL renderer.
NEON = "#00ff41"
GREEN = "#00cc33"
RED = "#ff0040"
YELLOW = "#ffaa00"
BLUE = "#00aaff"
BG = "#0a0a0a"
PANEL = "#141414"
GRID = "#233c30"

# Chart colors reused by the canvas charts in the browser.
CHART_COLORS = (NEON, BLUE, YELLOW, RED, "#a855f7", "#55ddaa")

DETAIL_PRESETS = {"Performance": 60, "Balanced": 160, "Ultra": 320}

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

FLOOR_COLORS = {
    "Dust": "#ffb547",
    "Warehouse": "#39dcaa",
    "Highrise": "#42aaff",
    "Arena": "#bd65ff",
    "Sniper Alley": "#ff5e63",
    "Custom": NEON,
}

AGENT_COLORS = (GREEN, RED)


def theme_color(map_name: str) -> str:
    """Return the accent color used for one map's perimeter lights."""
    return FLOOR_COLORS.get(map_name, NEON)


def dark_layout(title: str | None = None) -> dict[str, Any]:
    """Legacy helper kept for API compatibility: chart theming lives in CSS/JS."""
    layout: dict[str, Any] = {"title": title, "background": BG, "accent": NEON}
    return layout


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{100.0 * float(value):.{decimals}f}%"


def relative_path(path: Path) -> str:
    """Return a repository-relative path when possible (tests use temp folders)."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_runtime_dirs() -> None:
    for directory in (MODELS_DIR, LOGS_DIR, DATA_DIR, VENDOR_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def dependency_status() -> dict[str, bool]:
    """Report which optional runtime dependencies are importable."""
    return {
        "numpy": find_spec("numpy") is not None,
        "gymnasium": find_spec("gymnasium") is not None,
        "fastapi": find_spec("fastapi") is not None,
        "uvicorn": find_spec("uvicorn") is not None,
        "psutil": find_spec("psutil") is not None,
        "plotly": find_spec("plotly") is not None,
        "streamlit": find_spec("streamlit") is not None,
        "stable_baselines3": find_spec("stable_baselines3") is not None,
        "torch": find_spec("torch") is not None,
    }


def runtime_info() -> dict[str, Any]:
    deps = dependency_status()
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "training_ready": bool(deps["stable_baselines3"] and deps["torch"]),
        "dependencies": deps,
        "legacy_ui_installed": bool(deps["streamlit"] or deps["plotly"]),
    }

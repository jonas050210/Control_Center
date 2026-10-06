"""NEURAL ARENA cyberpunk control center entry point.

Run from the repository root with:
    streamlit run gui/app.py --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from gui.common import ensure_session_defaults
from gui.tabs import arena, benchmark, heatmap, maps, playground, stats, training, ttk


st.set_page_config(
    page_title="NEURAL ARENA · Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

style_path = Path(__file__).resolve().parent / "style.css"
try:
    css = style_path.read_text(encoding="utf-8")
except OSError:
    css = ""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
ensure_session_defaults(st)

PAGES = {
    "🎮  ARENA": arena.render,
    "🔫  PLAYGROUND": playground.render,
    "🏋️  TRAINING": training.render,
    "📊  STATS": stats.render,
    "🔧  BENCHMARK": benchmark.render,
    "🎯  TTK-TESTER": ttk.render,
    "🗺️  MAPS": maps.render,
    "🔥  HEATMAP": heatmap.render,
}

with st.sidebar:
    st.markdown('<div class="neural-logo">⚡ NEURAL ARENA<br><small>CONTROL CENTER · v2.0</small></div>',
                unsafe_allow_html=True)
    selection = st.radio("NAVIGATION", list(PAGES.keys()), label_visibility="collapsed", key="main_navigation")
    st.markdown("---")
    job = st.session_state.get("training_job")
    if job is not None:
        status = job.snapshot()["status"]
        color_class = "status-good" if status in {"running", "complete"} else (
            "status-warn" if status in {"paused", "starting", "stopping"} else "status-bad"
        )
        st.markdown(f'<div class="neon-card">TRAINING<br><b class="{color_class}">● {status.upper()}</b></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="neon-card">TRAINING<br><b class="status-bad">● STOPPED</b></div>',
                    unsafe_allow_html=True)
    st.caption("HEADLESS SIMULATION · CPU PARALLEL · STREAMLIT / PLOTLY")

st.markdown("# ⚡ NEURAL ARENA", help="3D shooter reinforcement-learning sandbox")
st.markdown("### CYBERNETIC TRAINING SANDBOX  /  HEADLESS MODE ACTIVE")
st.markdown("---")
PAGES[selection]()

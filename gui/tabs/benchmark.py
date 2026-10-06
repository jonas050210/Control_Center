"""Parallel worker/environment benchmark dashboard."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from env.maps import MAP_NAMES
from gui.common import NEON, dark_layout
from training.workers import BenchmarkRunner, valid_benchmark_configurations


def _styled_table(results: list[dict]) -> None:
    if not results:
        st.info("No benchmark results yet.")
        return
    frame = pd.DataFrame(results)
    completed = frame[frame["status"] == "complete"] if "status" in frame else pd.DataFrame()
    best_index = completed["steps_per_second"].astype(float).idxmax() if not completed.empty else None
    display = pd.DataFrame({
        "Workers": frame["workers"],
        "Envs / worker": frame["envs_per_worker"],
        "Total envs": frame.get("total_envs", 0),
        "Steps / s": frame["steps_per_second"].map(lambda value: f"{float(value):,.0f}"),
        "CPU %": frame["cpu_percent"].map(lambda value: f"{float(value):.0f}%"),
        "RAM MB": frame["ram_mb"].map(lambda value: f"{float(value):,.0f}"),
        "Elapsed": frame["seconds"].map(lambda value: f"{float(value):.2f}s"),
        "Status": frame["status"].map(lambda value: "✅ COMPLETE" if value == "complete" else str(value).upper()),
    })
    # Status labels are assigned from the best completed throughput row.
    if best_index is not None:
        display.loc[best_index, "Status"] = "🏆 BEST"
    def highlight(row: pd.Series) -> list[str]:
        if best_index is not None and row.name == best_index:
            return ["background-color: #12351b; color: #00ff41; font-weight: bold"] * len(row)
        return ["background-color: #111411; color: #c2e6c8"] * len(row)
    st.dataframe(display.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)


def _render_live_panel(runner: BenchmarkRunner | None) -> None:
    if runner is None:
        st.info("Run the full benchmark to measure CPU throughput for all valid worker × environment combinations.")
        return
    snapshot = runner.snapshot()
    rows = snapshot["results"]
    total = snapshot["total"]
    completed = sum(row.get("status") in {"complete", "stopped"} for row in rows)
    current = snapshot.get("current")
    if current:
        current_text = f"now testing {current[0]} worker(s) × {current[1]} env(s) per worker"
    else:
        current_text = snapshot["status"].upper()
    st.markdown(f"**BENCHMARK STATUS:** `{current_text}` · {len(rows)}/{total} combinations initialized")
    st.progress(min(1.0, completed / max(1, total)), text=f"Completed {completed} of {total} combinations")
    if snapshot.get("error"):
        st.error(snapshot["error"])
    _styled_table(rows)

    completed_rows = [row for row in rows if row.get("status") == "complete"]
    if completed_rows:
        best = max(completed_rows, key=lambda row: row.get("steps_per_second", 0.0))
        st.success(
            f"Best measured config: **{best['workers']} workers × {best['envs_per_worker']} envs/worker** "
            f"({best['total_envs']} total envs) at **{best['steps_per_second']:,.0f} aggregate steps/s**."
        )
        worker_values = sorted({row["workers"] for row in rows})
        env_values = sorted({row["envs_per_worker"] for row in rows})
        matrix = np.full((len(env_values), len(worker_values)), np.nan, dtype=float)
        for row in completed_rows:
            matrix[env_values.index(row["envs_per_worker"]), worker_values.index(row["workers"])] = row["steps_per_second"]
        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=worker_values,
            y=env_values,
            colorscale=[[0.0, "#103018"], [0.45, "#008f2c"], [1.0, NEON]],
            colorbar={"title": {"text": "steps/s", "font": {"color": NEON}}, "tickfont": {"color": NEON}},
            hovertemplate="Workers: %{x}<br>Envs/worker: %{y}<br>Steps/s: %{z:,.0f}<extra></extra>",
        ))
        fig.update_xaxes(title="Workers")
        fig.update_yaxes(title="Environments per worker")
        st.plotly_chart(dark_layout(fig, "WORKER × ENVIRONMENT THROUGHPUT", height=370),
                        use_container_width=True)
        if st.button("💾 Apply Best Config to Training Tab", key="benchmark_apply"):
            st.session_state["training_workers"] = int(best["workers"])
            st.session_state["training_envs_per_worker"] = int(best["envs_per_worker"])
            st.success("Benchmark values copied to the Training tab controls.")


def render() -> None:
    st.subheader("🔧 CPU BENCHMARK · WORKERS × ENVS")
    configurations = valid_benchmark_configurations()
    st.caption(
        f"Each valid combination runs for 20 seconds after worker startup. "
        f"{len(configurations)} configurations are tested (product capped at 24 environments). "
        f"A full sweep takes about {len(configurations) * 20 / 60:.1f} minutes plus process startup."
    )
    benchmark_deps_ready = find_spec("stable_baselines3") is not None
    if not benchmark_deps_ready:
        st.warning("Benchmark execution requires Stable-Baselines3; install the project requirements to enable it.")
    controls = st.columns([1, 1.4, 1])
    with controls[0]:
        map_name = st.selectbox("BENCHMARK MAP", MAP_NAMES, key="benchmark_map")
    with controls[1]:
        st.caption("Workers: 1, 2, 4, 8, 12, 16 · Envs/worker: 1, 2, 4, 8")
    with controls[2]:
        st.caption("CPU-only random-policy stepping")

    runner: BenchmarkRunner | None = st.session_state.get("benchmark_job")
    running = bool(runner and runner.snapshot()["status"] in {"running", "stopping"})
    start_col, stop_col = st.columns([1, 1])
    with start_col:
        if st.button("🚀 Run Full Benchmark", key="benchmark_start",
                     disabled=running or not benchmark_deps_ready, use_container_width=True):
            runner = BenchmarkRunner(seconds_per_combo=20.0, map_name=map_name)
            st.session_state.benchmark_job = runner
            runner.start()
    with stop_col:
        if st.button("⏹ Stop Benchmark", key="benchmark_stop", disabled=not running,
                     use_container_width=True):
            runner.stop()

    st.markdown("---")
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        live_panel = fragment(run_every="2s")(_render_live_panel)
        live_panel(runner)
    else:
        _render_live_panel(runner)

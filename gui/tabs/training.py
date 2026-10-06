"""Streamlit PPO training controls; all learning runs in a background thread."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import streamlit as st

from env.maps import MAP_NAMES
from gui.common import MODELS_DIR, PROJECT_ROOT, format_percent, get_custom_map
from training.train import TrainingConfig, TrainingController
from training.workers import auto_worker_count, cpu_core_counts, resolve_parallelism


DURATIONS = {
    "10 min": 10 * 60,
    "30 min": 30 * 60,
    "1 hour": 60 * 60,
    "2 hours": 2 * 60 * 60,
    "5 hours": 5 * 60 * 60,
    "Custom": None,
}


def _model_paths() -> list[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(MODELS_DIR.glob("*.zip"), key=lambda path: (path.name != "best_model.zip", path.name))


def _status_class(status: str) -> tuple[str, str]:
    if status == "running":
        return "status-good", "● TRAINING"
    if status in {"starting", "paused", "stopping"}:
        return "status-warn", f"● {status.upper()}"
    if status == "complete":
        return "status-good", "● COMPLETE"
    if status == "error":
        return "status-bad", "● ERROR"
    return "status-bad", "● STOPPED"


def _render_live_panel(job: TrainingController | None) -> None:
    if job is None:
        st.info("Configure a run and press Start. PPO will execute in a worker thread so the dashboard remains interactive.")
        return
    snapshot = job.snapshot()
    metrics = snapshot["metrics"]
    status_class, label = _status_class(snapshot["status"])
    progress = min(1.0, max(0.0, float(metrics.get("progress", 0.0))))
    remaining = max(0.0, job.config.duration_seconds - float(metrics.get("elapsed", 0.0)))
    eta_text = f"{remaining / 60:.1f} min to selected duration" if job.config.duration_seconds else "step target"
    st.markdown(
        f'<div class="neon-card"><b class="{status_class}">{label}</b> &nbsp; · &nbsp; '
        f'{int(metrics.get("effective_envs", 0))} vector envs &nbsp; · &nbsp; ETA {eta_text}</div>',
        unsafe_allow_html=True,
    )
    st.progress(progress, text=f"PPO rollout progress · {progress:.1%}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TOTAL STEPS", f"{int(metrics.get('timesteps', 0)):,}")
    m2.metric("STEPS / SEC", f"{float(metrics.get('fps', 0.0)):,.0f}")
    m3.metric("EPISODES", f"{int(metrics.get('episodes', 0)):,}")
    m4.metric("WIN RATE", format_percent(float(metrics.get("win_rate", 0.0))))
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("AVG EPISODE REWARD", f"{float(metrics.get('avg_reward', 0.0)):.2f}")
    m6.metric("AVG TTK", f"{float(metrics.get('avg_ttk', 0.0)):.2f}s")
    m7.metric("ACCURACY", format_percent(float(metrics.get("accuracy", 0.0))))
    m8.metric("HEADSHOT SHARE", format_percent(float(metrics.get("headshot_pct", 0.0))))
    if snapshot.get("latest_checkpoint"):
        st.caption(f"Latest checkpoint: `{Path(snapshot['latest_checkpoint']).name}`")
    if snapshot.get("error"):
        st.error(snapshot["error"])
    with st.expander("LIVE WORKER LOG", expanded=True):
        log_text = "\n".join(snapshot.get("logs", [])[-35:]) or "Waiting for the first PPO callback…"
        st.code(log_text, language="text", line_numbers=False)


def render() -> None:
    st.subheader("🏋️ PPO TRAINING · CPU WORKER CONTROL")
    physical, logical = cpu_core_counts()
    st.caption(
        f"CPU-only training profile · detected {physical} physical / {logical} logical cores · "
        "headless Gymnasium environments · SB3 VecNormalize"
    )
    training_deps_ready = find_spec("stable_baselines3") is not None and find_spec("torch") is not None
    if not training_deps_ready:
        st.warning("PPO controls are disabled until Stable-Baselines3 and PyTorch are installed with `pip install -r requirements.txt`.")

    col_duration, col_workers, col_envs = st.columns([1.3, 1, 1])
    with col_duration:
        duration_label = st.selectbox("TRAINING DURATION", list(DURATIONS), key="training_duration")
        if duration_label == "Custom":
            custom_minutes = st.number_input("Custom minutes", min_value=1, max_value=7200,
                                             value=45, step=5, key="training_custom_minutes")
            duration_seconds = float(custom_minutes) * 60.0
        else:
            duration_seconds = float(DURATIONS[duration_label] or 0)
    with col_workers:
        default_workers = auto_worker_count()
        if "training_workers" not in st.session_state:
            st.session_state.training_workers = default_workers
        worker_count = st.slider("WORKERS", min_value=1, max_value=20,
                                 key="training_workers", help=f"Auto-detected default: {default_workers}.")
    with col_envs:
        envs_per_worker = st.slider("ENVS PER WORKER", min_value=1, max_value=8,
                                    value=1, key="training_envs_per_worker")

    method = st.radio(
        "TRAINING METHOD",
        ["Imitation → RL", "Pure RL", "Resume Checkpoint"],
        horizontal=True,
        key="training_method",
    )
    checkpoint_path = None
    imitation_path = None
    if method == "Resume Checkpoint":
        checkpoints = _model_paths()
        if checkpoints:
            chosen = st.selectbox("CHECKPOINT TO RESUME", checkpoints,
                                  format_func=lambda value: value.name, key="training_resume_checkpoint")
            checkpoint_path = str(chosen)
        else:
            st.warning("No PPO .zip checkpoint exists in models/ yet. Train a policy or copy a checkpoint there first.")
    elif method == "Imitation → RL":
        imitation_path = str(MODELS_DIR / "behavior_clone.pt")
        st.caption("The starter demonstrations in data/demos.csv are cloned on the training thread if no behavior_clone.pt exists.")

    option_cols = st.columns(4)
    with option_cols[0]:
        curriculum = st.checkbox("4-phase curriculum", value=True, key="training_curriculum")
    with option_cols[1]:
        self_play = st.checkbox("Frozen self-play after Phase 3", value=True, key="training_self_play")
    with option_cols[2]:
        map_name = st.selectbox("TRAINING MAP", MAP_NAMES, key="training_map")
    with option_cols[3]:
        st.metric("PPO DEVICE", "CPU", "parallel env rollouts")

    requested = resolve_parallelism(worker_count, envs_per_worker, max_envs=24)
    if worker_count * envs_per_worker > 24:
        st.warning(
            f"Requested {worker_count * envs_per_worker} environments; this 12-core-oriented profile "
            f"will cap the run at {requested.effective_envs} parallel envs."
        )
    else:
        st.caption(f"Effective vector environments: {requested.effective_envs} (one SubprocVecEnv child per environment).")

    # Use a conservative aggregate throughput estimate as a finite PPO horizon;
    # the wall-clock duration remains the hard stop in the callback.
    estimated_steps = max(50_000, int(duration_seconds * 1_500))
    st.caption(f"Estimated PPO rollout horizon: {estimated_steps:,} aggregate timesteps; duration callback stops at the selected limit.")

    job: TrainingController | None = st.session_state.get("training_job")
    running = bool(job and job.snapshot()["status"] in {"starting", "running", "paused", "stopping"})
    start_col, pause_col, stop_col, save_col = st.columns([1.2, 1, 1, 1.2])
    with start_col:
        if st.button("🚀 Start Training", key="training_start", disabled=running or not training_deps_ready,
                     use_container_width=True):
            if method == "Resume Checkpoint" and not checkpoint_path:
                st.error("Choose a valid checkpoint before resuming.")
            else:
                config = TrainingConfig(
                    duration_seconds=duration_seconds,
                    total_timesteps=estimated_steps,
                    n_workers=worker_count,
                    envs_per_worker=envs_per_worker,
                    map_name=map_name,
                    map_definition=(get_custom_map(st) if map_name == "Custom" else None),
                    curriculum=curriculum,
                    self_play=self_play,
                    method=method,
                    resume_checkpoint=checkpoint_path,
                    imitation_path=imitation_path,
                    max_envs=24,
                    models_dir=MODELS_DIR,
                    logs_dir=PROJECT_ROOT / "logs",
                )
                job = TrainingController(config)
                st.session_state.training_job = job
                job.start()
                st.success("Training thread launched. This tab remains responsive while PPO runs.")
    with pause_col:
        if st.button("⏸ Pause / Resume", key="training_pause", disabled=not running, use_container_width=True):
            if job is not None and job.snapshot()["status"] == "paused":
                job.resume()
            elif job is not None:
                job.pause()
    with stop_col:
        if st.button("⏹ Stop", key="training_stop", disabled=not running, use_container_width=True):
            if job is not None:
                job.stop()
    with save_col:
        if st.button("💾 Save Checkpoint", key="training_save", disabled=not running, use_container_width=True):
            if job is not None:
                job.request_save()

    st.markdown("---")
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        live_panel = fragment(run_every="2s")(_render_live_panel)
        live_panel(job)
    else:
        _render_live_panel(job)

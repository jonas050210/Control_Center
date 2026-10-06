"""Interactive Plotly arena viewer and model-versus-model match controls."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from env.maps import ArenaMap, MAP_NAMES
from env.shooter_env import ShooterEnv
from env.weapons import WEAPON_NAMES
from gui.common import MODELS_DIR, RED, YELLOW, format_percent, get_custom_map
from gui.visuals import DETAIL_PRESETS, build_map_figure


HEURISTIC = "Heuristic AI"


def _model_choices() -> list[str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [HEURISTIC] + [path.name for path in sorted(MODELS_DIR.glob("*.zip"))]


def _load_model(model_name: str) -> tuple[Any | None, Any | None, str | None]:
    if model_name == HEURISTIC:
        return None, None, None
    path = MODELS_DIR / model_name
    if not path.exists():
        return None, None, f"Model file not found: {path.name}"
    cache = st.session_state.setdefault("arena_model_cache", {})
    cache_key = f"{path}:{path.stat().st_mtime_ns}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(path), device="cpu")
        normalizer = None
        sidecar = path.with_name(f"{path.stem}_vecnormalize.pkl")
        if sidecar.exists():
            with sidecar.open("rb") as handle:
                normalizer = pickle.load(handle)
        result = (model, normalizer, None)
    except Exception as exc:
        result = (None, None, f"Could not load {path.name}: {type(exc).__name__}: {exc}")
    cache[cache_key] = result
    return result


def _predict(model_name: str, observation: np.ndarray, agent_index: int, env: ShooterEnv) -> np.ndarray:
    if model_name == HEURISTIC:
        return env.heuristic_action(agent_index)
    model, normalizer, error = _load_model(model_name)
    if model is None:
        if error:
            st.session_state["arena_model_error"] = error
        return env.heuristic_action(agent_index)
    policy_observation = observation
    if normalizer is not None and getattr(normalizer, "obs_rms", None) is not None:
        mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float32)
        variance = np.asarray(normalizer.obs_rms.var, dtype=np.float32)
        policy_observation = np.clip(
            (observation - mean) / np.sqrt(np.maximum(variance, 1e-8) + 1e-8), -10.0, 10.0
        ).astype(np.float32)
    action, _ = model.predict(policy_observation, deterministic=True)
    return np.asarray(action, dtype=np.int64).reshape(-1)


def _create_env(map_name: str | ArenaMap, weapon_a: str, weapon_b: str) -> ShooterEnv:
    env = ShooterEnv(
        map_name=map_name,
        weapon_name=weapon_a,
        opponent_weapon=weapon_b,
        curriculum=False,
        opponent_mode="full",
        frame_skip=4,
        max_episode_seconds=120,
    )
    env.reset()
    return env


def _event_line(event: dict[str, Any]) -> str:
    if event.get("type") == "kill":
        tag = " [HEADSHOT]" if event.get("headshot") else ""
        return (
            f"Agent {event.get('attacker', '?')} killed Agent {event.get('target', '?')} "
            f"with {event.get('weapon', 'weapon')}{tag} · {event.get('distance', 0):.1f}m"
        )
    tag = " HEADSHOT" if event.get("headshot") else ""
    return (
        f"A{event.get('attacker', '?')} hit A{event.get('target', '?')} · "
        f"{event.get('damage', 0):.0f} dmg{tag}"
    )


def _advance(env: ShooterEnv) -> None:
    if env.get_snapshot()["done"]:
        return
    selected_models = st.session_state.get("arena_models", (HEURISTIC, HEURISTIC))
    obs_a = env.get_observation(0)
    obs_b = env.get_observation(1)
    action_a = _predict(selected_models[0], obs_a, 0, env)
    action_b = _predict(selected_models[1], obs_b, 1, env)
    _, _, terminated, truncated, info = env.step_duel(action_a, action_b)
    for event in info.get("combat_events", []):
        st.session_state.arena_messages.append(_event_line(event))
    st.session_state.arena_messages = st.session_state.arena_messages[-80:]
    if terminated or truncated:
        st.session_state.arena_running = False
        metrics = info.get("episode_metrics", {})
        if metrics:
            st.session_state.arena_events.append(metrics)
            st.session_state.arena_events = st.session_state.arena_events[-500:]
            winner = "AGENT 1 WINS" if metrics.get("win", 0) else (
                "DRAW" if metrics.get("draw") else "AGENT 2 WINS"
            )
            st.session_state.arena_messages.append(
                f"MATCH COMPLETE · {winner} · {metrics.get('ttk', 0):.1f}s"
            )


def _render_live_panel() -> None:
    env = st.session_state.get("arena_env")
    if env is None:
        st.info("Create a match to render the arena.")
        return
    if st.session_state.get("arena_running"):
        for _ in range(12):
            if env.get_snapshot()["done"]:
                st.session_state.arena_running = False
                break
            _advance(env)

    snapshot = env.get_snapshot()
    left, right = st.columns([3.2, 1.15], gap="large")
    with left:
        st.plotly_chart(
            build_map_figure(
                env.arena_map,
                agents=snapshot["agents"],
                trails=snapshot["trails"],
                height=680,
                detail_count=st.session_state.get("arena_detail_count", DETAIL_PRESETS["Balanced"]),
            ),
            use_container_width=True,
            key="arena_plot",
        )
    with right:
        st.markdown("#### MATCH TELEMETRY")
        for index, agent in enumerate(snapshot["agents"]):
            label = f"AGENT {index + 1} · {agent['weapon']}"
            st.metric(label, f"{agent['hp']:.0f} HP", f"{agent['ammo']}/{agent['mag_size']} rounds")
            accuracy = agent["hits"] / max(1, agent.get("bullets_fired", agent["shots_fired"]))
            st.caption(f"Shots {agent['shots_fired']} · Hits {agent['hits']} · Accuracy {format_percent(accuracy)}")
        st.caption(f"MATCH TIME · {snapshot['elapsed']:.1f}s  |  SIM FRAME · {snapshot['frame']:,}")
        st.markdown("#### KILL FEED")
        messages = st.session_state.get("arena_messages", [])
        if messages:
            for message in reversed(messages[-12:]):
                is_kill = "killed" in message or "WINS" in message or "DRAW" in message
                color = YELLOW if is_kill else "#a7c9ad"
                st.markdown(f'<div style="color:{color};padding:3px 0;border-bottom:1px solid #1b2c20">{message}</div>',
                            unsafe_allow_html=True)
        else:
            st.caption("No combat events yet.")
        model_error = st.session_state.pop("arena_model_error", None)
        if model_error:
            st.warning(model_error)


def render() -> None:
    st.subheader("🎮 3D ARENA · LIVE MATCH CONTROL")
    st.caption("The scene is simulated in Python and rendered in the browser with Plotly. No display server is used.")

    controls_left, controls_right = st.columns([1.2, 1.0])
    with controls_left:
        map_name = st.selectbox("ARENA MAP", MAP_NAMES, key="arena_map")
        weapons = list(WEAPON_NAMES)
        weapon_cols = st.columns(2)
        with weapon_cols[0]:
            weapon_a = st.selectbox("AGENT 1 WEAPON", weapons, index=0, key="arena_weapon_a")
        with weapon_cols[1]:
            weapon_b = st.selectbox("AGENT 2 WEAPON", weapons, index=2, key="arena_weapon_b")
    with controls_right:
        choices = _model_choices()
        model_cols = st.columns(2)
        with model_cols[0]:
            model_a = st.selectbox("AGENT 1 POLICY", choices, key="arena_model_a")
        with model_cols[1]:
            model_b = st.selectbox("AGENT 2 POLICY", choices, key="arena_model_b")
        st.session_state["arena_models"] = (model_a, model_b)
        visual_quality = st.selectbox(
            "3D DETAIL", tuple(DETAIL_PRESETS), index=1, key="arena_visual_quality"
        )
        st.session_state["arena_detail_count"] = DETAIL_PRESETS[visual_quality]

    custom_layout = get_custom_map(st) if map_name == "Custom" else None
    layout_signature = tuple(
        (item.x, item.y, item.z, item.width, item.height, item.depth, item.kind)
        for item in custom_layout.objects
    ) if custom_layout is not None else ()
    signature = (map_name, weapon_a, weapon_b, layout_signature)
    if st.session_state.get("arena_signature") != signature or st.session_state.get("arena_env") is None:
        old_env = st.session_state.get("arena_env")
        if old_env is not None:
            old_env.close()
        st.session_state.arena_env = _create_env(custom_layout or map_name, weapon_a, weapon_b)
        st.session_state.arena_signature = signature
        st.session_state.arena_messages = []
        st.session_state.arena_running = False

    env: ShooterEnv = st.session_state.arena_env
    b_start, b_step, b_pause, b_reset = st.columns([1.3, 1, 1, 1])
    with b_start:
        if st.button("▶ Start Match", key="arena_start", use_container_width=True):
            if env.get_snapshot()["done"]:
                env.reset()
                st.session_state.arena_messages = []
            st.session_state.arena_running = True
    with b_step:
        if st.button("⏭ Step", key="arena_step", use_container_width=True):
            st.session_state.arena_running = False
            if env.get_snapshot()["done"]:
                env.reset()
                st.session_state.arena_messages = []
            _advance(env)
    with b_pause:
        if st.button("⏸ Pause", key="arena_pause", use_container_width=True):
            st.session_state.arena_running = False
    with b_reset:
        if st.button("🔄 Reset", key="arena_reset", use_container_width=True):
            env.reset()
            st.session_state.arena_messages = []
            st.session_state.arena_running = False

    st.markdown("---")
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        live_panel = fragment(run_every="1s")(_render_live_panel)
        live_panel()
    else:
        _render_live_panel()

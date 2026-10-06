"""Desktop-friendly human shooter and high-detail 3D training mini-games."""

from __future__ import annotations

import csv
import io
import time
from typing import Any

import numpy as np
import streamlit as st

from env.maps import ArenaMap, MAP_NAMES
from env.shooter_env import ACTION_SIZE, OBSERVATION_SIZE, ShooterEnv
from env.weapons import WEAPON_NAMES
from gui.common import MODELS_DIR, YELLOW, format_percent, get_custom_map
from gui.games import aim_tap, new_aim_game, new_dodge_game, step_dodge_game
from gui.tabs.arena import _predict as predict_arena_policy
from gui.visuals import (
    DETAIL_PRESETS,
    build_aim_range_figure,
    build_dodge_arena_figure,
    build_map_figure,
)


PLAY_MODES = ("🔫 Real Shooter", "🎯 Aim Trainer", "⚡ Dodge Survival")
BOT_BEHAVIORS = {"Tactical": "full", "Shooter": "shooter", "Walker": "walker"}


def _make_action(
    *,
    move: int = 0,
    strafe: int = 0,
    yaw: int = 0,
    pitch: int = 0,
    shoot: bool = False,
    jump: bool = False,
    reload: bool = False,
    sprint: bool = False,
    stance: int = 0,
) -> np.ndarray:
    action = np.asarray([1, 1, 1, 1, int(shoot), int(sprint), stance, 1, int(jump), int(reload)],
                        dtype=np.int64)
    action[0] = int(np.clip(move, -1, 1)) + 1
    action[1] = int(np.clip(strafe, -1, 1)) + 1
    action[2] = int(np.clip(yaw, -1, 1)) + 1
    action[3] = int(np.clip(pitch, -1, 1)) + 1
    return action


def _create_match(map_name: str | ArenaMap, player_weapon: str, enemy_weapon: str,
                  ai_mode: str) -> ShooterEnv:
    match = ShooterEnv(
        map_name=map_name,
        weapon_name=player_weapon,
        opponent_weapon=enemy_weapon,
        curriculum=False,
        opponent_mode=ai_mode,
        frame_skip=4,
        max_episode_seconds=120.0,
    )
    match.reset()
    return match


def _human_action(match: ShooterEnv, controls: dict[str, Any], repeat: int = 1) -> None:
    if match.done:
        return
    action = _make_action(
        move=controls.get("move", 0),
        strafe=controls.get("strafe", 0),
        yaw=controls.get("yaw", 0),
        pitch=controls.get("pitch", 0),
        shoot=bool(controls.get("shoot", False)),
        jump=bool(controls.get("jump", False)),
        reload=bool(controls.get("reload", False)),
        sprint=bool(controls.get("sprint", st.session_state.get("human_sprint", False))),
        stance=int(st.session_state.get("human_stance", 0)),
    )
    repeat = max(1, min(35, int(repeat)))
    for _ in range(repeat):
        if match.done:
            break
        observation = match.get_observation(0)
        if st.session_state.get("human_demo_recording", False):
            rows = st.session_state.setdefault("playground_demo_rows", [])
            if len(rows) < 25_000:
                rows.append((observation.tolist(), action.tolist()))
        bot_policy = st.session_state.get("play_ai", "Tactical")
        if bot_policy in BOT_BEHAVIORS:
            _, reward, terminated, truncated, info = match.step(action)
        else:
            bot_action = predict_arena_policy(
                bot_policy, match.get_observation(1), agent_index=1, env=match
            )
            _, reward, terminated, truncated, info = match.step_duel(action, bot_action)
        st.session_state["playground_reward"] = float(st.session_state.get("playground_reward", 0.0)) + reward
        messages = st.session_state.setdefault("playground_messages", [])
        for event in info.get("combat_events", []):
            player_scored = int(event.get("attacker", 0)) == 1
            if event.get("type") == "kill":
                suffix = " [HEADSHOT]" if event.get("headshot") else ""
                outcome = "BOT DOWN" if player_scored else "YOU DOWN"
                messages.append(
                    f"{outcome} · {event.get('weapon', 'weapon')}{suffix} · "
                    f"{event.get('distance', 0):.1f}m"
                )
            elif event.get("type") == "hit":
                suffix = " HEADSHOT" if event.get("headshot") else ""
                outcome = "HIT" if player_scored else "INCOMING HIT"
                messages.append(f"{outcome} · {event.get('damage', 0):.0f} damage{suffix}")
        st.session_state["playground_messages"] = messages[-35:]
        if terminated or truncated:
            metrics = info.get("episode_metrics", {})
            if metrics:
                st.session_state["playground_result"] = metrics
                arena_events = st.session_state.setdefault("arena_events", [])
                arena_events.append(metrics)
                st.session_state["arena_events"] = arena_events[-500:]
                result_text = "YOU WIN" if metrics.get("win", 0) else ("DRAW" if metrics.get("draw") else "BOT WINS")
                messages.append(f"ROUND COMPLETE · {result_text} · {metrics.get('ttk', 0):.1f}s")
            break


def _action_button(container: Any, label: str, key: str,
                   controls: dict[str, Any], repeat: int = 1) -> bool:
    match: ShooterEnv | None = st.session_state.get("playground_env")
    disabled = match is None or match.done
    if container.button(label, key=key, use_container_width=True, disabled=disabled):
        if match is not None:
            _human_action(match, controls, repeat)
        return True
    return False


def _render_live_shooter(detail_count: int) -> None:
    st.markdown("#### LIVE MATCH · ON-SCREEN CONTROLS")
    st.caption(
        "Jede Aktion steuert direkt die Headless-3D-Simulation. Die Szene lässt sich "
        "mit der Maus drehen und zoomen; Buttons bewegen und feuern."
    )
    setting_cols = st.columns(2, gap="small")
    with setting_cols[0]:
        map_name = st.selectbox("MAP", MAP_NAMES, key="play_map")
    with setting_cols[1]:
        player_weapon = st.selectbox("YOUR WEAPON", WEAPON_NAMES, index=0, key="play_weapon")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = sorted(path.name for path in MODELS_DIR.glob("*.zip"))
    bot_options = list(BOT_BEHAVIORS) + model_names
    setting_cols = st.columns(2, gap="small")
    with setting_cols[0]:
        enemy_weapon = st.selectbox("BOT WEAPON", WEAPON_NAMES, index=2, key="play_enemy_weapon")
    with setting_cols[1]:
        ai_label = st.selectbox("BOT BEHAVIOR / PPO MODEL", bot_options, key="play_ai")
    ai_mode = BOT_BEHAVIORS.get(ai_label, "full")
    custom_map = get_custom_map(st) if map_name == "Custom" else None
    layout_signature = (
        custom_map.width,
        custom_map.depth,
        tuple(custom_map.spawn_points),
        tuple((item.x, item.y, item.z, item.width, item.height, item.depth, item.kind, item.name)
              for item in custom_map.objects),
    ) if custom_map is not None else ()
    signature = (map_name, player_weapon, enemy_weapon, ai_label, ai_mode, layout_signature)
    if st.session_state.get("playground_signature") != signature or st.session_state.get("playground_env") is None:
        previous = st.session_state.get("playground_env")
        if previous is not None:
            previous.close()
        st.session_state.playground_env = _create_match(
            custom_map or map_name, player_weapon, enemy_weapon, ai_mode
        )
        st.session_state.playground_signature = signature
        st.session_state.playground_messages = []
        st.session_state.playground_reward = 0.0
        st.session_state.playground_result = None
    match: ShooterEnv = st.session_state.playground_env

    option_cols = st.columns([1.2, 1.8], gap="small")
    with option_cols[0]:
        st.checkbox("Record demos", key="human_demo_recording", value=False)
    with option_cols[1]:
        stance_name = st.radio("STANCE", ["Stand", "Crouch", "Prone"], horizontal=True,
                               key="human_stance_name")
        st.session_state["human_stance"] = {"Stand": 0, "Crouch": 1, "Prone": 2}[stance_name]
    option_cols = st.columns(2, gap="small")
    with option_cols[0]:
        st.checkbox("Sprint", key="human_sprint", value=False)
    with option_cols[1]:
        if st.button("🔄 New Round", key="play_reset", use_container_width=True):
            match.reset()
            st.session_state.playground_messages = []
            st.session_state.playground_reward = 0.0
            st.session_state.playground_result = None

    move_col, look_col, combat_col = st.columns([1, 1, 1.25], gap="large")
    with move_col:
        st.markdown("**MOVE**")
        row = st.columns(3)
        _action_button(row[1], "▲ Forward", "play_forward", {"move": 1}, repeat=4)
        row = st.columns(3)
        _action_button(row[0], "◀ Strafe", "play_strafe_left", {"strafe": -1}, repeat=4)
        _action_button(row[1], "▼ Back", "play_back", {"move": -1}, repeat=4)
        _action_button(row[2], "Strafe ▶", "play_strafe_right", {"strafe": 1}, repeat=4)
        _action_button(st, "🏃 Sprint Forward", "play_sprint_forward",
                       {"move": 1, "sprint": True}, repeat=6)
    with look_col:
        st.markdown("**LOOK / TURN**")
        row = st.columns(3)
        _action_button(row[1], "⬆ Look", "play_look_up", {"pitch": 1}, repeat=1)
        row = st.columns(3)
        _action_button(row[0], "⟲ Left", "play_turn_left", {"yaw": -1}, repeat=1)
        _action_button(row[1], "⬇ Look", "play_look_down", {"pitch": -1}, repeat=1)
        _action_button(row[2], "Right ⟳", "play_turn_right", {"yaw": 1}, repeat=1)
        _action_button(st, "↗ Move + Fire", "play_move_fire",
                       {"move": 1, "shoot": True}, repeat=4)
    with combat_col:
        st.markdown("**COMBAT**")
        _action_button(st, "🔴 FIRE", "play_fire", {"shoot": True}, repeat=8)
        combat_row = st.columns(2)
        _action_button(combat_row[0], "↻ Reload", "play_reload", {"reload": True}, repeat=30)
        _action_button(combat_row[1], "⬆ Jump", "play_jump", {"jump": True}, repeat=1)
        st.caption(
            "One policy action repeats over four 60 Hz physics frames. Repeated taps "
            "advance longer movement or firing bursts."
        )

    snapshot = match.get_snapshot()
    player, bot = snapshot["agents"]
    telemetry = st.columns(2, gap="small")
    telemetry[0].metric("YOUR HEALTH", f"{player['hp']:.0f} HP",
                        f"{player['ammo']}/{player['mag_size']} {player['weapon']}")
    telemetry[1].metric("BOT HEALTH", f"{bot['hp']:.0f} HP",
                        f"{bot['ammo']}/{bot['mag_size']} {bot['weapon']}")
    st.metric("ROUND TIME", f"{snapshot['elapsed']:.1f}s",
              f"Reward {st.session_state.get('playground_reward', 0.0):+.2f}")
    result = st.session_state.get("playground_result")
    if result:
        state = "VICTORY" if result.get("win", 0) else ("DRAW" if result.get("draw") else "DEFEAT")
        st.markdown(f"**ROUND RESULT · {state}**")
    st.plotly_chart(
        build_map_figure(match.arena_map, agents=snapshot["agents"],
                         trails=snapshot["trails"], height=650, detail_count=detail_count),
        use_container_width=True,
        key="playground_shooter_plot",
    )
    st.markdown("**COMBAT FEED**")
    messages = st.session_state.get("playground_messages", [])
    if messages:
        for message in reversed(messages[-8:]):
            color = YELLOW if "DOWN" in message or "ROUND COMPLETE" in message else "#b8d8be"
            st.markdown(f'<div style="color:{color};padding:3px 0;border-bottom:1px solid #1b2c20">{message}</div>',
                        unsafe_allow_html=True)
    else:
        st.caption("No shots fired yet.")
    model_error = st.session_state.pop("arena_model_error", None)
    if model_error:
        st.warning(model_error)

    demo_rows = st.session_state.get("playground_demo_rows", [])
    demo_col, download_col = st.columns([1.5, 1])
    with demo_col:
        st.caption(f"Human imitation samples in this session: {len(demo_rows):,} / 25,000")
    with download_col:
        if demo_rows:
            state_fields = [f"state_{index}" for index in range(OBSERVATION_SIZE)]
            action_fields = [f"action_{index}" for index in range(ACTION_SIZE)]
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer)
            writer.writerow(state_fields + action_fields)
            for observation, action in demo_rows:
                writer.writerow(observation + action)
            st.download_button("⬇ Download imitation CSV", data=buffer.getvalue(),
                               file_name="neural_arena_demos.csv", mime="text/csv",
                               key="playground_demo_export", use_container_width=True)
        else:
            st.caption("Turn on demo recording before firing to build a PPO warm-start dataset.")


def _render_aim_live(detail_count: int) -> None:
    game = st.session_state.get("aim_game")
    if not game:
        st.plotly_chart(
            build_aim_range_figure(4, detail_count=detail_count),
            use_container_width=True,
            key="aim_3d_range",
        )
        st.info("Start a drill to begin.")
        return
    now = time.monotonic()
    if game["active"] and now - float(game["started_at"]) >= float(game["duration"]):
        game["active"] = False
        st.session_state["aim_best_score"] = max(
            int(st.session_state.get("aim_best_score", 0)), int(game["hits"])
        )
    remaining = max(0.0, float(game["duration"]) - (now - float(game["started_at"])))
    accuracy = game["hits"] / max(1, game["hits"] + game["misses"])
    average_reaction = float(np.mean(game["reaction_ms"])) if game["reaction_ms"] else 0.0
    st.plotly_chart(
        build_aim_range_figure(
            int(game["target"]), hits=int(game["hits"]), misses=int(game["misses"]),
            streak=int(game["streak"]), detail_count=detail_count,
        ),
        use_container_width=True,
        key="aim_3d_range",
    )
    metrics = st.columns(4)
    metrics[0].metric("HITS", str(game["hits"]))
    metrics[1].metric("ACCURACY", format_percent(accuracy))
    metrics[2].metric("BEST STREAK", str(game["best_streak"]))
    metrics[3].metric("AVG REACTION", f"{average_reaction:.0f} ms" if average_reaction else "—")
    if game["active"]:
        st.progress(remaining / float(game["duration"]), text=f"{remaining:.1f}s remaining · tap the target")
        st.markdown("### TARGET GRID")
        for row_index in range(3):
            columns = st.columns(3, gap="small")
            for column_index, column in enumerate(columns):
                cell = row_index * 3 + column_index
                label = f"🎯 {cell + 1}" if cell == int(game["target"]) else f"{cell + 1:02d}"
                if column.button(label, key=f"aim_cell_{cell}", use_container_width=True):
                    aim_tap(game, cell)
    else:
        score = int(game["hits"])
        st.success(
            f"DRILL COMPLETE · {score} targets · {game['misses']} misses · "
            f"best streak {game['best_streak']}"
        )
        st.caption(f"Personal best in this session: {st.session_state.get('aim_best_score', score)} hits")


def _render_aim_trainer(detail_count: int) -> None:
    st.markdown("### 🎯 AIM TRAINER · 3D PRECISION RANGE")
    st.caption(
        "Triff das aktive Bullseye in der nummerierten 3D-Schießbahn. Treffer, Quote "
        "und Reaktionszeit werden live gemessen."
    )
    controls = st.columns([1, 1.5, 1])
    with controls[0]:
        duration = st.selectbox("ROUND", [15, 30, 60], index=1, format_func=lambda value: f"{value} seconds",
                                key="aim_duration")
    with controls[1]:
        if st.button("▶ Start / Restart Drill", key="aim_start", use_container_width=True):
            st.session_state.aim_game = new_aim_game(duration)
    with controls[2]:
        st.metric("SESSION BEST", str(st.session_state.get("aim_best_score", 0)))
    st.markdown("---")
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        live = fragment(run_every=1.0)(_render_aim_live)
        live(detail_count)
    else:
        _render_aim_live(detail_count)


def _render_dodge_survival(detail_count: int) -> None:
    st.markdown("### ⚡ DODGE SURVIVAL · EVADE THE INCOMING FIRE")
    st.caption(
        "Move on the 5×5 grid. Every D-pad tap advances the salvo; projectile lanes "
        "tighten as your score grows."
    )
    game = st.session_state.get("dodge_game")
    top = st.columns([1, 1, 1, 1.3])
    with top[0]:
        if st.button("▶ New Run", key="dodge_start", use_container_width=True):
            st.session_state.dodge_game = new_dodge_game()
            game = st.session_state.dodge_game
    with top[1]:
        score = int(game["score"]) if game else 0
        st.metric("SCORE", str(score))
    with top[2]:
        dodged = int(game["dodged"]) if game else 0
        st.metric("SALVOS EVADED", str(dodged))
    with top[3]:
        st.metric("PERSONAL BEST", str(st.session_state.get("dodge_best_score", 0)))

    preview_game = game if game is not None else new_dodge_game(seed=2026)
    st.plotly_chart(
        build_dodge_arena_figure(preview_game, detail_count=detail_count),
        use_container_width=True,
        key="dodge_3d_arena",
    )
    if game:
        if game.get("active"):
            st.markdown("**DODGE PAD**")
            row = st.columns(3)
            if row[1].button("▲", key="dodge_up", use_container_width=True):
                step_dodge_game(game, 0, -1)
            row = st.columns(3)
            if row[0].button("◀", key="dodge_left", use_container_width=True):
                step_dodge_game(game, -1, 0)
            if row[1].button("WAIT", key="dodge_wait", use_container_width=True):
                step_dodge_game(game, 0, 0)
            if row[2].button("▶", key="dodge_right", use_container_width=True):
                step_dodge_game(game, 1, 0)
            row = st.columns(3)
            if row[1].button("▼", key="dodge_down", use_container_width=True):
                step_dodge_game(game, 0, 1)
        else:
            score = int(game["score"])
            st.session_state["dodge_best_score"] = max(
                int(st.session_state.get("dodge_best_score", 0)), score
            )
            st.error(f"RUN OVER · score {score} · dodged {game['dodged']} salvos. Tap New Run to try again.")
    else:
        st.info("Start a run. The first projectile lanes are gentle; incoming fire accelerates as you survive.")


def render() -> None:
    st.subheader("🕹️ PLAYGROUND · HUMAN SKILL + AI TESTS")
    st.caption(
        "A browser-side training range: fight a Gymnasium bot or saved PPO policy, "
        "measure aim reaction, or test dodging. The games run inside the existing dashboard with no separate game process."
    )
    mode_col, quality_col = st.columns([1.25, 1])
    with mode_col:
        selected_mode = st.selectbox("PLAY MODE", PLAY_MODES, key="playground_mode")
    with quality_col:
        quality = st.selectbox("3D DETAIL", tuple(DETAIL_PRESETS), index=1,
                                key="play_visual_quality")
    detail_count = DETAIL_PRESETS[quality]
    st.caption("Zusätzliche Deko wird im 3D-Bild gebündelt und ist rein visuell; die Physik-/Kollisionsobjekte bleiben unverändert.")
    st.markdown("---")
    if selected_mode == "🔫 Real Shooter":
        _render_live_shooter(detail_count)
    elif selected_mode == "🎯 Aim Trainer":
        _render_aim_trainer(detail_count)
    else:
        _render_dodge_survival(detail_count)

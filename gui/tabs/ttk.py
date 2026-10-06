"""Monte-Carlo time-to-kill weapon balance tester and Roblox Lua export."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from env.weapons import WEAPON_NAMES, WeaponSpec, get_weapon
from gui.common import NEON, RED, YELLOW, dark_layout, format_percent


FIELDS = ("damage", "fire_rate", "mag_size", "reload_time", "spread", "range",
          "headshot_multiplier", "pellets")
def _field_defaults(spec: WeaponSpec) -> dict[str, float | int]:
    return {
        "damage": int(spec.damage),
        "fire_rate": float(spec.fire_rate),
        "mag_size": int(spec.mag_size),
        "reload_time": float(spec.reload_time),
        "spread": float(spec.spread),
        "range": float(spec.range),
        "headshot_multiplier": float(spec.headshot_multiplier),
        "pellets": int(spec.pellets),
    }


def _slider_spec(side: str, title: str) -> WeaponSpec:
    keys = {field: st.session_state[f"ttk_{side}_{field}"] for field in FIELDS}
    original = get_weapon(title).spec
    return WeaponSpec(
        name=title,
        damage=float(keys["damage"]),
        fire_rate=float(keys["fire_rate"]),
        mag_size=int(keys["mag_size"]),
        reload_time=float(keys["reload_time"]),
        spread=float(keys["spread"]),
        range=float(keys["range"]),
        recoil=original.recoil,
        projectile_speed=original.projectile_speed,
        headshot_multiplier=float(keys["headshot_multiplier"]),
        pellets=int(keys["pellets"]),
    )


def _preset_buttons(side: str) -> None:
    button_cols = st.columns(3)
    for index, name in enumerate(WEAPON_NAMES):
        with button_cols[index % 3]:
            if st.button(name, key=f"ttk_{side}_preset_{index}", use_container_width=True):
                values = _field_defaults(get_weapon(name).spec)
                for field, value in values.items():
                    st.session_state[f"ttk_{side}_{field}"] = value
                st.session_state[f"ttk_{side}_weapon_name"] = name
                st.session_state[f"ttk_{side}_active_preset"] = name


def _render_weapon_editor(side: str, default_name: str) -> tuple[str, WeaponSpec]:
    title_key = f"ttk_{side}_weapon_name"
    active_key = f"ttk_{side}_active_preset"
    if title_key not in st.session_state:
        st.session_state[title_key] = default_name
    if active_key not in st.session_state:
        st.session_state[active_key] = st.session_state[title_key]
    defaults = _field_defaults(get_weapon(st.session_state[title_key]).spec)
    for field, default in defaults.items():
        key = f"ttk_{side}_{field}"
        if key not in st.session_state:
            st.session_state[key] = default
    _preset_buttons(side)
    selected_name = st.selectbox("PRESET", WEAPON_NAMES, key=title_key)
    if selected_name != st.session_state.get(active_key):
        # Choosing a named preset resets its values; users can then tune sliders.
        for field, value in _field_defaults(get_weapon(selected_name).spec).items():
            st.session_state[f"ttk_{side}_{field}"] = value
        st.session_state[active_key] = selected_name

    st.slider("Damage", min_value=1, max_value=100, key=f"ttk_{side}_damage")
    st.slider("Fire rate (seconds)", min_value=0.01, max_value=2.0, step=0.01,
              key=f"ttk_{side}_fire_rate")
    st.slider("Magazine size", min_value=1, max_value=100, key=f"ttk_{side}_mag_size")
    st.slider("Reload time (seconds)", min_value=0.5, max_value=6.0, step=0.1,
              key=f"ttk_{side}_reload_time")
    st.slider("Spread (degrees)", min_value=0.0, max_value=30.0, step=0.1,
              key=f"ttk_{side}_spread")
    st.slider("Range (metres)", min_value=5.0, max_value=100.0, step=1.0,
              key=f"ttk_{side}_range")
    st.slider("Headshot multiplier", min_value=1.0, max_value=4.0, step=0.1,
              key=f"ttk_{side}_headshot_multiplier")
    return selected_name, _slider_spec(side, selected_name)


def _accuracy_probability(spread_degrees: float, distance: float) -> float:
    if spread_degrees <= 1e-6:
        return 0.995
    horizontal_tolerance = math.degrees(math.atan2(0.38, max(distance, 0.1)))
    vertical_tolerance = math.degrees(math.atan2(0.90, max(distance, 0.1)))
    denominator = math.sqrt(2.0) * spread_degrees
    horizontal_probability = math.erf(horizontal_tolerance / denominator)
    vertical_probability = math.erf(vertical_tolerance / denominator)
    return float(np.clip(horizontal_probability * vertical_probability, 0.0, 0.995))


def _simulate_weapon(spec: WeaponSpec, distance: float, trials: int, rng: np.random.Generator) -> dict:
    trials = max(1, int(trials))
    hit_probability = _accuracy_probability(spec.spread, distance)
    if distance > spec.range:
        hit_probability = 0.0
    ratio = distance / max(0.1, spec.range)
    falloff = max(0.18, 1.0 - 0.72 * ratio ** 1.35) if distance <= spec.range else 0.0
    expected_damage = (spec.damage * falloff * hit_probability
                       * (1.0 + 0.20 * (spec.headshot_multiplier - 1.0)))
    effective_mag_cycle = max(1e-6, spec.mag_size * spec.fire_rate + spec.reload_time)
    sustained_dps = expected_damage * spec.mag_size / effective_mag_cycle
    mean_damage_per_shot = max(0.05, expected_damage)
    max_shots = min(500, max(24, int(math.ceil(100.0 / mean_damage_per_shot * 3.0)) + spec.mag_size))
    damage = np.zeros(trials, dtype=np.float32)
    shots = np.zeros(trials, dtype=np.int32)
    ttk = np.full(trials, np.inf, dtype=np.float32)
    if hit_probability > 0.0:
        remaining = np.ones(trials, dtype=bool)
        pellet_count = max(1, int(spec.pellets))
        pellet_damage = spec.damage / pellet_count * falloff
        for shot_index in range(max_shots):
            active = np.flatnonzero(remaining)
            if active.size == 0:
                break
            shot_time = (shot_index * spec.fire_rate
                         + (shot_index // max(1, spec.mag_size)) * spec.reload_time)
            if shot_time > 120.0:
                break
            hit_count = rng.binomial(pellet_count, hit_probability, size=active.size)
            head_count = rng.binomial(hit_count, 0.20)
            increment = pellet_damage * (
                hit_count + head_count * (spec.headshot_multiplier - 1.0)
            )
            damage[active] += increment.astype(np.float32)
            shots[active] += 1
            killed_now = damage[active] >= 100.0
            if np.any(killed_now):
                killed_indices = active[killed_now]
                # Shot zero is released at t=0; each later magazine follows a reload.
                ttk[killed_indices] = float(shot_time)
                remaining[killed_indices] = False
    killed = np.isfinite(ttk)
    if np.any(killed):
        mean_ttk = float(np.mean(ttk[killed]))
        median_ttk = float(np.median(ttk[killed]))
        mean_shots = float(np.mean(shots[killed]))
    else:
        mean_ttk = float("nan")
        median_ttk = float("nan")
        mean_shots = float("nan")
    return {
        "ttk_samples": ttk,
        "shots_samples": shots,
        "killed": killed,
        "kill_rate": float(np.mean(killed)),
        "mean_ttk": mean_ttk,
        "median_ttk": median_ttk,
        "mean_shots": mean_shots,
        "accuracy": hit_probability,
        "expected_dps": sustained_dps,
        "expected_damage_per_shot": expected_damage,
    }


def simulate_duels(
    spec_a: WeaponSpec,
    spec_b: WeaponSpec,
    distance: float,
    trials: int = 10_000,
    seed: int = 2026,
) -> dict:
    """Run a vectorized simultaneous-fire Monte-Carlo duel simulation."""
    rng = np.random.default_rng(seed)
    a = _simulate_weapon(spec_a, distance, trials, rng)
    b = _simulate_weapon(spec_b, distance, trials, rng)
    a_ttk, b_ttk = a["ttk_samples"], b["ttk_samples"]
    a_kills, b_kills = a["killed"], b["killed"]
    a_wins = a_kills & (~b_kills | (a_ttk < b_ttk))
    b_wins = b_kills & (~a_kills | (b_ttk < a_ttk))
    simultaneous = a_kills & b_kills & np.isclose(a_ttk, b_ttk, atol=1e-6)
    win_rate_a = (np.count_nonzero(a_wins) + 0.5 * np.count_nonzero(simultaneous)) / max(1, trials)
    win_rate_b = (np.count_nonzero(b_wins) + 0.5 * np.count_nonzero(simultaneous)) / max(1, trials)
    draw_rate = max(0.0, 1.0 - win_rate_a - win_rate_b)
    return {
        "weapon_a": a,
        "weapon_b": b,
        "win_rate_a": float(win_rate_a),
        "win_rate_b": float(win_rate_b),
        "draw_rate": float(draw_rate),
        "trials": int(trials),
        "distance": float(distance),
        "seed": int(seed),
    }


def _verdict(result: dict, name_a: str, name_b: str) -> str:
    rate_a, rate_b = result["win_rate_a"], result["win_rate_b"]
    distance = result["distance"]
    if abs(rate_a - rate_b) < 0.06:
        return f"**Balanced at {distance:g}m:** {name_a} and {name_b} are within six win-rate points."
    winner = name_a if rate_a > rate_b else name_b
    margin = abs(rate_a - rate_b)
    if distance <= 15:
        context = "close range"
    elif distance >= 30:
        context = f"{distance:g}m+"
    else:
        context = f"mid range ({distance:g}m)"
    return f"**{winner} is favored at {context}** by {margin:.0%} in this duel sample."


def _roblox_lua(spec_a: WeaponSpec, spec_b: WeaponSpec) -> str:
    def row(spec: WeaponSpec) -> str:
        return (
            f'    ["{spec.name}"] = {{Damage = {spec.damage:g}, FireRate = {spec.fire_rate:g}, '
            f'MagSize = {spec.mag_size}, ReloadTime = {spec.reload_time:g}, '
            f'SpreadDegrees = {spec.spread:g}, Range = {spec.range:g}, '
            f'HeadshotMultiplier = {spec.headshot_multiplier:g}, Pellets = {spec.pellets}}},'
        )
    return "-- NEURAL ARENA balanced weapon values\nreturn {\n" + row(spec_a) + "\n" + row(spec_b) + "\n}"


def _result_figures(result: dict, name_a: str, name_b: str) -> tuple[go.Figure, go.Figure]:
    a, b = result["weapon_a"], result["weapon_b"]
    ttk_fig = go.Figure()
    ttk_values = [a["mean_ttk"] if math.isfinite(a["mean_ttk"]) else 0.0,
                  b["mean_ttk"] if math.isfinite(b["mean_ttk"]) else 0.0]
    ttk_fig.add_trace(go.Bar(
        x=[name_a, name_b], y=ttk_values, marker_color=[NEON, RED],
        text=[f"{value:.2f}s" if value > 0 else "No kill" for value in ttk_values],
        textposition="outside", name="Mean TTK",
        hovertemplate="%{x}<br>Mean TTK: %{y:.2f}s<extra></extra>",
    ))
    ttk_fig.update_yaxes(title="Mean TTK (seconds)", rangemode="tozero")
    ttk_fig.update_layout(showlegend=False)
    ttk_fig = dark_layout(ttk_fig, "AVERAGE TIME TO KILL", height=350)

    win_fig = go.Figure(go.Pie(
        labels=[name_a, name_b, "Draw / timeout"],
        values=[result["win_rate_a"], result["win_rate_b"], result["draw_rate"]],
        hole=0.48,
        marker={"colors": [NEON, RED, "#616a64"], "line": {"color": "#0a0a0a", "width": 2}},
        textinfo="label+percent", textfont={"color": "#e8ffe9"},
    ))
    win_fig = dark_layout(win_fig, "DUEL WIN-RATE", height=350)
    return ttk_fig, win_fig


def render() -> None:
    st.subheader("🎯 TTK TESTER · WEAPON BALANCING LAB")
    st.caption("Monte-Carlo duels include spread, pellet count, headshots, distance falloff, magazines, and reload delays.")
    for side, name in (("a", "Pistol"), ("b", "AK-47")):
        if f"ttk_{side}_weapon_name" not in st.session_state:
            st.session_state[f"ttk_{side}_weapon_name"] = name
        for field, value in _field_defaults(get_weapon(name).spec).items():
            key = f"ttk_{side}_{field}"
            if key not in st.session_state:
                st.session_state[key] = value

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("### WEAPON 1")
        name_a, spec_a = _render_weapon_editor("a", "Pistol")
    with col_b:
        st.markdown("### WEAPON 2")
        name_b, spec_b = _render_weapon_editor("b", "AK-47")

    simulation = st.columns([1.0, 1.0, 1.6])
    with simulation[0]:
        trials = st.slider("SIMULATION COUNT", min_value=1_000, max_value=100_000,
                           value=5_000, step=1_000, key="ttk_trials")
    with simulation[1]:
        distance = st.slider("DUEL DISTANCE (m)", min_value=5, max_value=80,
                             value=20, step=1, key="ttk_distance")
    with simulation[2]:
        run_simulation = st.button("⚔️ Simulate Duels", key="ttk_run", use_container_width=True)

    if run_simulation:
        with st.spinner(f"Simulating {trials:,} weapon duels at {distance} metres…"):
            result = simulate_duels(spec_a, spec_b, distance, trials,
                                    seed=2026 + int(distance) * 31 + int(trials))
        st.session_state.ttk_last_result = {
            "result": result,
            "name_a": name_a,
            "name_b": name_b,
            "spec_a": spec_a,
            "spec_b": spec_b,
        }

    last = st.session_state.get("ttk_last_result")
    if last:
        result = last["result"]
        name_a = last["name_a"]
        name_b = last["name_b"]
        spec_a = last["spec_a"]
        spec_b = last["spec_b"]
        st.markdown("---")
        st.markdown(f"### SIMULATION RESULTS · {result['trials']:,} DUELS · {result['distance']:g}m")
        a, b = result["weapon_a"], result["weapon_b"]
        metrics = st.columns(4)
        metrics[0].metric(f"{name_a} WIN RATE", format_percent(result["win_rate_a"]))
        metrics[1].metric(f"{name_b} WIN RATE", format_percent(result["win_rate_b"]))
        metrics[2].metric(f"{name_a} AVG TTK", f"{a['mean_ttk']:.2f}s" if math.isfinite(a["mean_ttk"]) else "No kills")
        metrics[3].metric(f"{name_b} AVG TTK", f"{b['mean_ttk']:.2f}s" if math.isfinite(b["mean_ttk"]) else "No kills")
        fig_a, fig_b = _result_figures(result, name_a, name_b)
        chart_a, chart_b = st.columns(2)
        with chart_a:
            st.plotly_chart(fig_a, use_container_width=True)
        with chart_b:
            st.plotly_chart(fig_b, use_container_width=True)
        table = []
        for label, spec, sim in ((name_a, spec_a, a), (name_b, spec_b, b)):
            table.append({
                "Weapon": label,
                "Kill rate": format_percent(sim["kill_rate"]),
                "Mean shots to kill": f"{sim['mean_shots']:.1f}" if math.isfinite(sim["mean_shots"]) else "—",
                "Hit probability": format_percent(sim["accuracy"]),
                "Expected DPS": f"{sim['expected_dps']:.1f}",
                "Base damage": f"{spec.damage:g}",
                "Magazine": spec.mag_size,
                "Reload": f"{spec.reload_time:g}s",
            })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown(f"#### VERDICT\n\n{_verdict(result, name_a, name_b)}")
        st.markdown("#### ROBLOX STUDIO EXPORT")
        st.caption("Copy this Lua table into a ModuleScript and connect the values to your weapon configuration.")
        st.code(_roblox_lua(spec_a, spec_b), language="lua")

"""Monte-Carlo time-to-kill weapon balance lab and Roblox Lua export.

Moved out of the former Streamlit tab (``gui/tabs/ttk.py``) so the simulation is
importable by tests and by the FastAPI backend without a UI dependency.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from env.weapons import WeaponSpec, get_weapon


DUEL_FIELDS = ("damage", "fire_rate", "mag_size", "reload_time", "spread", "range",
               "headshot_multiplier", "pellets")

# Slider bounds used by the browser editor.
FIELD_LIMITS: dict[str, tuple[float, float, float]] = {
    "damage": (1.0, 100.0, 1.0),
    "fire_rate": (0.01, 2.0, 0.01),
    "mag_size": (1.0, 100.0, 1.0),
    "reload_time": (0.5, 6.0, 0.1),
    "spread": (0.0, 30.0, 0.1),
    "range": (5.0, 100.0, 1.0),
    "headshot_multiplier": (1.0, 4.0, 0.1),
    "pellets": (1.0, 12.0, 1.0),
}


def field_defaults(spec: WeaponSpec) -> dict[str, float | int]:
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


def spec_from_fields(name: str, fields: dict[str, Any]) -> WeaponSpec:
    """Rebuild a weapon spec from browser slider values, keeping base stats."""
    original = get_weapon(name).spec
    merged = {**field_defaults(original), **{key: fields[key] for key in DUEL_FIELDS if key in fields}}
    return WeaponSpec(
        name=original.name,
        damage=float(merged["damage"]),
        fire_rate=float(merged["fire_rate"]),
        mag_size=int(merged["mag_size"]),
        reload_time=float(merged["reload_time"]),
        spread=float(merged["spread"]),
        range=float(merged["range"]),
        recoil=original.recoil,
        projectile_speed=original.projectile_speed,
        headshot_multiplier=float(merged["headshot_multiplier"]),
        pellets=max(1, int(merged["pellets"])),
    )


def accuracy_probability(spread_degrees: float, distance: float) -> float:
    if spread_degrees <= 1e-6:
        return 0.995
    horizontal_tolerance = math.degrees(math.atan2(0.38, max(distance, 0.1)))
    vertical_tolerance = math.degrees(math.atan2(0.90, max(distance, 0.1)))
    denominator = math.sqrt(2.0) * spread_degrees
    horizontal_probability = math.erf(horizontal_tolerance / denominator)
    vertical_probability = math.erf(vertical_tolerance / denominator)
    return float(np.clip(horizontal_probability * vertical_probability, 0.0, 0.995))


def simulate_weapon(spec: WeaponSpec, distance: float, trials: int,
                    rng: np.random.Generator) -> dict[str, Any]:
    trials = max(1, int(trials))
    hit_probability = accuracy_probability(spec.spread, distance)
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
) -> dict[str, Any]:
    """Run a vectorized simultaneous-fire Monte-Carlo duel simulation."""
    rng = np.random.default_rng(seed)
    a = simulate_weapon(spec_a, distance, trials, rng)
    b = simulate_weapon(spec_b, distance, trials, rng)
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


def verdict(result: dict[str, Any], name_a: str, name_b: str) -> str:
    rate_a, rate_b = result["win_rate_a"], result["win_rate_b"]
    distance = result["distance"]
    if abs(rate_a - rate_b) < 0.06:
        return f"Balanced at {distance:g}m: {name_a} and {name_b} are within six win-rate points."
    winner = name_a if rate_a > rate_b else name_b
    margin = abs(rate_a - rate_b)
    if distance <= 15:
        context = "close range"
    elif distance >= 30:
        context = f"{distance:g}m+"
    else:
        context = f"mid range ({distance:g}m)"
    return f"{winner} is favored at {context} by {margin:.0%} in this duel sample."


def roblox_lua(specs: list[WeaponSpec]) -> str:
    """Export the tuned weapon table as Roblox Lua for a ModuleScript."""
    rows = []
    for spec in specs:
        rows.append(
            f'    ["{spec.name}"] = {{Damage = {spec.damage:g}, FireRate = {spec.fire_rate:g}, '
            f'MagSize = {spec.mag_size}, ReloadTime = {spec.reload_time:g}, '
            f'SpreadDegrees = {spec.spread:g}, Range = {spec.range:g}, '
            f'HeadshotMultiplier = {spec.headshot_multiplier:g}, Pellets = {spec.pellets}}},'
        )
    return "-- NEURAL ARENA balanced weapon values\nreturn {\n" + "\n".join(rows) + "\n}"


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def result_payload(result: dict[str, Any], name_a: str, name_b: str,
                   spec_a: WeaponSpec, spec_b: WeaponSpec) -> dict[str, Any]:
    """Shape a duel result for the JSON API (charts + table + Lua export)."""
    a, b = result["weapon_a"], result["weapon_b"]
    ttk_a, ttk_b = a["ttk_samples"][a["killed"]], b["ttk_samples"][b["killed"]]
    histogram_edges = np.linspace(0.0, float(max(
        0.6,
        min(6.0, max([0.6] + [float(np.max(ttk_a)) if ttk_a.size else 0.0,
                               float(np.max(ttk_b)) if ttk_b.size else 0.0])),
    )), 21)
    return {
        "names": {"a": name_a, "b": name_b},
        "trials": result["trials"],
        "distance": result["distance"],
        "win_rate_a": result["win_rate_a"],
        "win_rate_b": result["win_rate_b"],
        "draw_rate": result["draw_rate"],
        "verdict": verdict(result, name_a, name_b),
        "lua": roblox_lua([spec_a, spec_b]),
        "weapons": {
            "a": {
                "name": name_a,
                "kill_rate": a["kill_rate"],
                "mean_ttk": _finite(a["mean_ttk"]),
                "median_ttk": _finite(a["median_ttk"]),
                "mean_shots": _finite(a["mean_shots"]),
                "accuracy": a["accuracy"],
                "expected_dps": a["expected_dps"],
                "histogram": _histogram(ttk_a, histogram_edges),
                "spec": field_defaults(spec_a),
            },
            "b": {
                "name": name_b,
                "kill_rate": b["kill_rate"],
                "mean_ttk": _finite(b["mean_ttk"]),
                "median_ttk": _finite(b["median_ttk"]),
                "mean_shots": _finite(b["mean_shots"]),
                "accuracy": b["accuracy"],
                "expected_dps": b["expected_dps"],
                "histogram": _histogram(ttk_b, histogram_edges),
                "spec": field_defaults(spec_b),
            },
        },
        "histogram_edges": [float(value) for value in histogram_edges],
    }


def _histogram(samples: np.ndarray, edges: np.ndarray) -> list[int]:
    if samples.size == 0:
        return [0] * (len(edges) - 1)
    counts, _ = np.histogram(samples, bins=edges)
    return [int(value) for value in counts]

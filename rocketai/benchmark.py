"""Messen statt raten: wie viele Schritte pro Sekunde schafft *dieser* Rechner?

Die Zahlen hängen stark von CPU, Kernanzahl und Betriebssystem ab. Deshalb
misst :func:`run_benchmark` auf dem Rechner des Benutzers:

* **Simulation** — mit echten RocketSim-Matches und echter Netz-Abfrage,
  also genau das, was beim Training passiert. Ergebnis: Schritte pro Sekunde
  und der Echtzeit-Faktor (wie viele simulierte Spielminuten pro Minute).
* **Lernschritt** — ein PPO-Update auf einem erfundenen Stapel, um zu sehen,
  ob das Lernen auf der CPU oder einer Grafikkarte schneller ist.

Der Bericht wird auch von ``rocketai benchmark`` und von der Weboberfläche
verwendet.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from .config import TICK_SKIP, TICKS_PER_SECOND, TrainConfig
from .model import ActorCritic
from .ppo import ppo_update
from .rollout import WorkerPool

#: Simulationsschritte pro Sammelaufruf — klein genug für kurze Messungen.
CHUNK_STEPS = 1500


def _worker_config(
    team_size: int,
    hidden_sizes: list[int],
    envs_per_worker: int,
    reward_stage: int = 1,
    obs_extras: bool = True,
) -> dict[str, Any]:
    return {
        "team_size": team_size,
        "reward_stage": reward_stage,
        "episode_seconds": 60.0,
        "no_touch_seconds": 15.0,
        "envs_per_worker": envs_per_worker,
        "hidden_sizes": hidden_sizes,
        "obs_extras": obs_extras,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "teacher_weight": 0.0,
        "teacher_final_weight": 0.0,
        "teacher_opponent_prob": 0.0,
        "past_opponent_prob": 0.0,
    }


def measure_simulation(
    seconds: float = 6.0,
    workers: int = 1,
    envs_per_worker: int = 1,
    team_size: int = 1,
    hidden_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Miss, wie viele Agent-Schritte pro Sekunde simuliert und bewertet werden."""
    hidden_sizes = hidden_sizes or [256, 256, 128]
    config = _worker_config(team_size, hidden_sizes, envs_per_worker)
    model = ActorCritic(hidden_sizes=hidden_sizes)
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    steps = 0
    sim_seconds = 0.0
    pool = WorkerPool(max(1, int(workers)), config, seed=1)
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < seconds:
            batch = pool.collect(weights, CHUNK_STEPS)
            steps += len(batch)
            sim_seconds += float(batch.stats.get("game_seconds") or 0.0)
            if time.perf_counter() - started > seconds + 5:
                break  # ein einzelner very slow chunk darf die Messung nicht sprengen
    finally:
        pool.close()
    elapsed = time.perf_counter() - started
    agents = max(1, 2 * team_size)
    return {
        "workers": max(1, int(workers)),
        "envs_per_worker": envs_per_worker,
        "team_size": team_size,
        "agent_steps": steps,
        "seconds": round(elapsed, 2),
        "steps_per_second": steps / elapsed if elapsed else 0.0,
        "decisions_per_second": steps / agents / elapsed if elapsed else 0.0,
        "realtime_factor": sim_seconds / elapsed if elapsed else 0.0,
    }


def measure_update(
    steps: int = 20_000,
    hidden_sizes: list[int] | None = None,
    device: str = "cpu",
    epochs: int = 3,
    minibatch_size: int = 10_000,
) -> dict[str, Any]:
    """Zeit für einen PPO-Lernschritt auf einem künstlichen Stapel."""
    hidden_sizes = hidden_sizes or [512, 512, 256]
    model = ActorCritic(hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    from .rollout import Batch

    batch = Batch(
        obs=rng.normal(size=(steps, model.obs_size)).astype(np.float32),
        actions=rng.integers(0, 90, size=steps).astype(np.int64),
        log_probs=rng.normal(size=steps).astype(np.float32),
        advantages=rng.normal(size=steps).astype(np.float32),
        returns=rng.normal(size=steps).astype(np.float32),
        values=rng.normal(size=steps).astype(np.float32),
    )
    started = time.perf_counter()
    ppo_update(
        model,
        optimizer,
        batch,
        epochs=epochs,
        minibatch_size=minibatch_size,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    )
    elapsed = time.perf_counter() - started
    return {
        "device": device,
        "steps": steps,
        "epochs": epochs,
        "seconds": round(elapsed, 3),
        "steps_per_second": steps * epochs / elapsed if elapsed else 0.0,
    }


def run_benchmark(
    seconds: float = 6.0,
    workers: int | None = None,
    envs_per_worker: int = 1,
    team_size: int = 1,
    hidden_sizes: list[int] | None = None,
    with_update: bool = True,
) -> dict[str, Any]:
    """Kompletter Bericht: Simulation (und optional Lernschritt) auf diesem Rechner."""
    cores = os.cpu_count() or 2
    workers = workers if workers and workers > 0 else max(1, cores - 1)
    report: dict[str, Any] = {
        "cpu_count": cores,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "scale": [],
        "advice": [],
    }
    for count in sorted({1, max(1, workers // 2), workers}):
        sim = measure_simulation(
            seconds=seconds,
            workers=count,
            envs_per_worker=envs_per_worker,
            team_size=team_size,
            hidden_sizes=hidden_sizes,
        )
        report["scale"].append(sim)
    best = max(report["scale"], key=lambda item: item["steps_per_second"])
    report["best"] = best
    per_worker = best["steps_per_second"] / best["workers"]
    report["steps_per_second_per_worker"] = per_worker
    report["steps_per_day"] = best["steps_per_second"] * 86_400
    if with_update:
        update_steps = min(50_000, max(5_000, best["steps_per_second"]))
        report["update_cpu"] = measure_update(steps=update_steps, hidden_sizes=hidden_sizes)
        if torch.cuda.is_available():
            report["update_cuda"] = measure_update(
                steps=update_steps, hidden_sizes=hidden_sizes, device="cuda"
            )
    report["advice"] = advice(report)
    return report


def advice(report: dict[str, Any]) -> list[str]:
    """Klartext-Empfehlungen aus den Messwerten."""
    tips: list[str] = []
    best = report.get("best") or {}
    scale = report.get("scale") or []
    per_worker = report.get("steps_per_second_per_worker") or 0.0
    if len(scale) >= 2:
        first, last = scale[0], scale[-1]
        gain = (
            (last["steps_per_second"] / first["steps_per_second"])
            / max(1, last["workers"] / first["workers"])
            if first["steps_per_second"]
            else 1.0
        )
        if gain < 0.75:
            tips.append(
                "Mehr Prozesse bringen kaum etwas (Skalierung "
                f"{gain:.2f}×): Der Rechner ist am Anschlag — eher weniger Prozesse "
                "nehmen, sonst wird nur die CPU heiß."
            )
        elif gain > 0.9:
            tips.append("Die Simulation skaliert gut mit mehr Prozessen: n_workers erhöhen lohnt.")
    if best.get("realtime_factor"):
        tips.append(
            f"Die Simulation läuft {best['realtime_factor']:.0f}× schneller als Echtzeit "
            f"({best['steps_per_second']:.0f} Schritte/s)."
        )
    update_cpu = report.get("update_cpu")
    update_cuda = report.get("update_cuda")
    if update_cpu and per_worker:
        share = update_cpu["seconds"] / max(1e-6, update_cpu["steps"] / per_worker)
        if share > 0.35:
            tips.append(
                f"Der Lernschritt ist mit {share:.0%} der Sammelzeit auffällig teuer — "
                "mehr Simulation oder eine Grafikkarte hilft."
            )
    if update_cuda and update_cpu:
        factor = update_cuda["steps_per_second"] / max(1e-6, update_cpu["steps_per_second"])
        if factor > 1.5:
            tips.append(
                f"Das Lernen auf der Grafikkarte ist {factor:.1f}× schneller als auf der CPU."
            )
        else:
            tips.append(
                "Die Grafikkarte bringt beim Lernen kaum etwas: Das Netz ist klein und die "
                "Simulation läuft auf der CPU."
            )
    return tips


def suggested_config(report: dict[str, Any], base: TrainConfig | None = None) -> TrainConfig:
    """Eine TrainConfig, angepasst an die gemessene Geschwindigkeit."""
    config = base or TrainConfig()
    best = report.get("best") or {}
    if best.get("workers"):
        config.n_workers = int(best["workers"])
    if best.get("envs_per_worker"):
        config.envs_per_worker = int(best["envs_per_worker"])
    if report.get("steps_per_second"):
        # Rund 20 Minuten Sammeln pro Lernschritt: groß genug für stabile Updates.
        config.steps_per_iteration = int(
            max(20_000, min(1_000_000, report["steps_per_second"] * 1200))
        )
    return config


def format_report(report: dict[str, Any]) -> str:
    """Lesbarer Text für die Kommandozeile."""
    lines = [
        f"Rechner: {report['cpu_count']} Kerne, PyTorch {report['torch']}, "
        f"Rechnen auf {report['device_name']}",
        "",
        "Simulation (RocketSim + Netz):",
    ]
    for entry in report["scale"]:
        lines.append(
            f"  {entry['workers']} Prozesse × {entry['envs_per_worker']} Spiel(e): "
            f"{entry['steps_per_second']:,.0f} Schritte/s, "
            f"{entry['decisions_per_second']:,.0f} Entscheidungen/s, "
            f"{entry['realtime_factor']:,.0f}× Echtzeit".replace(",", ".")
        )
    best = report.get("best") or {}
    if best:
        lines += [
            "",
            f"Beste Einstellung: {best['workers']} Prozesse × {best['envs_per_worker']} Spiele "
            f"→ {report['steps_per_day'] / 1e6:,.0f} Mio. Schritte pro Tag (24/7)".replace(
                ",", "."
            ),
        ]
    for key, label in (("update_cpu", "Lernschritt (CPU)"), ("update_cuda", "Lernschritt (GPU)")):
        entry = report.get(key)
        if entry:
            lines.append(
                f"{label}: {entry['steps']:,} Schritte in {entry['seconds']:.2f} s "
                f"({entry['steps_per_second']:,.0f} Schritte/s)".replace(",", ".")
            )
    if report.get("advice"):
        lines += ["", "Empfehlungen:"] + [f"  - {tip}" for tip in report["advice"]]
    return "\n".join(lines)


def tick_seconds() -> float:
    """Dauer einer Entscheidung in simulierten Sekunden."""
    return TICK_SKIP / TICKS_PER_SECOND

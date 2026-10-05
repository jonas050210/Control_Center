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
    threads: int = 0,
) -> dict[str, Any]:
    """Zeit für einen PPO-Lernschritt auf einem künstlichen Stapel.

    ``threads`` setzt die Zahl der PyTorch-Threads für die Messung (0 = wie
    jetzt eingestellt). Das ist wichtig, weil im Training genau diese Zahl
    zählt: Während des Lernschritts warten alle Simulationsprozesse, deshalb
    darf der Lernprozess fast alle Kerne nehmen (siehe ``Trainer.train``).
    """
    hidden_sizes = hidden_sizes or [512, 512, 256]
    previous_threads = torch.get_num_threads()
    used_threads = int(threads) if threads else previous_threads
    if threads:
        torch.set_num_threads(used_threads)
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
    try:
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
    finally:
        torch.set_num_threads(previous_threads)
    elapsed = time.perf_counter() - started
    return {
        "device": device,
        "threads": used_threads,
        "steps": steps,
        "epochs": epochs,
        "seconds": round(elapsed, 3),
        "steps_per_second": steps * epochs / elapsed if elapsed else 0.0,
    }


#: Spiele pro Prozess, die getestet werden (wenn der Benutzer nichts vorgibt).
ENVS_SWEEP = (1, 2, 4)


def plan_combos(workers: int, envs_per_worker: int | None = None) -> list[tuple[int, int]]:
    """Messplan als ``(Prozesse, Spiele pro Prozess)``.

    Ohne Vorgabe wird auch *Spiele pro Prozess* getestet. Vorher stand hier
    immer 1 — und die daraus abgeleitete Empfehlung setzte damit einen guten
    Wert (4) wieder auf 1 zurück, obwohl mehr Spiele pro Prozess die
    Spawn-Kosten teilen und die Simulation besser auslasten.
    """
    counts = sorted({max(1, workers // 2), max(1, workers)})
    if envs_per_worker:
        return [(count, int(envs_per_worker)) for count in counts]
    return [(count, envs) for count in counts for envs in ENVS_SWEEP]


def effective_steps_per_second(sim_sps: float, update_sps: float, epochs: int = 3) -> float:
    """Ende-zu-Ende-Tempo: Sammeln und Lernen laufen nacheinander.

    Der Lernprozess sammelt erst (``sim_sps`` Schritte/s), rechnet dann über
    dieselben Daten ``epochs`` mal (``update_sps`` Schritte/s) — und *während*
    dieser Zeit stehen alle Simulationsprozesse still. Ein Stapel von ``S``
    Schritten braucht deshalb ``S / sim_sps + epochs * S / update_sps``
    Sekunden. Genau diese Zahl zählt für jede Zeitangabe im Programm.
    """
    if sim_sps <= 0:
        return 0.0
    if update_sps <= 0:
        return float(sim_sps)
    return 1.0 / (1.0 / sim_sps + max(1, epochs) / update_sps)


def run_benchmark(
    seconds: float = 6.0,
    workers: int | None = None,
    envs_per_worker: int = 0,
    team_size: int = 1,
    hidden_sizes: list[int] | None = None,
    with_update: bool = True,
    epochs: int = 3,
) -> dict[str, Any]:
    """Kompletter Bericht: Simulation, Lernschritt und **Ende-zu-Ende-Tempo**.

    Gemessen wird über mehrere Einstellungen (Prozesse × Spiele pro Prozess,
    ``envs_per_worker=0`` = automatisch), damit die Empfehlung auf echten Zahlen
    steht. Neu ist die letzte Zeile:
    Simulation und Lernschritt laufen nacheinander im selben Prozess, deshalb
    ist das Tempo, das der Benutzer erlebt, immer kleiner als die reine
    Simulationsrate — und *das* ist die Zahl für alle Zeitangaben.
    """
    cores = os.cpu_count() or 2
    workers = workers if workers and workers > 0 else max(1, cores - 1)
    combos = plan_combos(workers, envs_per_worker or None)
    report: dict[str, Any] = {
        "cpu_count": cores,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_threads_recommended": max(1, cores - 1),
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "workers_requested": workers,
        "envs_per_worker_requested": envs_per_worker,
        "combos": [f"{count} x {envs}" for count, envs in combos],
        "scale": [],
        "advice": [],
    }
    for count, envs in combos:
        sim = measure_simulation(
            seconds=seconds,
            workers=count,
            envs_per_worker=envs,
            team_size=team_size,
            hidden_sizes=hidden_sizes,
        )
        report["scale"].append(sim)
    best = max(report["scale"], key=lambda item: item["steps_per_second"])
    report["best"] = best
    report["steps_per_second_per_worker"] = best["steps_per_second"] / best["workers"]
    report["steps_per_day_simulation"] = best["steps_per_second"] * 86_400
    update: dict[str, Any] = {}
    if with_update:
        update_steps = min(50_000, max(5_000, int(best["steps_per_second"])))
        measured = [
            measure_update(
                steps=update_steps,
                hidden_sizes=hidden_sizes,
                epochs=epochs,
                threads=report["torch_threads_recommended"],
            )
        ]
        if torch.cuda.is_available():
            measured.append(
                measure_update(
                    steps=update_steps, hidden_sizes=hidden_sizes, device="cuda", epochs=epochs
                )
            )
        report["update_cpu"] = measured[0]
        if len(measured) > 1:
            report["update_cuda"] = measured[1]
        update = max(measured, key=lambda item: item["steps_per_second"])
    # Ende-zu-Ende: Sammeln + Lernen, wie es im Training wirklich abläuft.
    effective = effective_steps_per_second(
        best["steps_per_second"], update.get("steps_per_second", 0.0), epochs
    )
    report["effective"] = {
        "steps_per_second": effective,
        "epochs": int(epochs) if update else 0,
        "update_device": update.get("device", "unknown"),
        "update_threads": update.get("threads"),
        "simulation_share": (
            (1.0 / best["steps_per_second"])
            / (1.0 / best["steps_per_second"] + max(1, epochs) / update["steps_per_second"])
            if update and best["steps_per_second"]
            else 1.0
        ),
    }
    report["effective"]["update_share"] = 1.0 - report["effective"]["simulation_share"]
    report["steps_per_day"] = effective * 86_400
    report["advice"] = advice(report)
    return report


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def advice(report: dict[str, Any]) -> list[str]:
    """Klartext-Empfehlungen aus den Messwerten."""
    tips: list[str] = []
    best = report.get("best") or {}
    scale = report.get("scale") or []
    if not scale:
        return tips

    def group(envs: int) -> list[dict[str, Any]]:
        return sorted(
            (item for item in scale if item.get("envs_per_worker") == envs),
            key=lambda item: item["workers"],
        )

    # 1) Skalierung über Prozesse — nur Einstellungen mit gleicher Spielzahl
    # vergleichen, sonst vergleicht man Äpfel mit Birnen.
    for envs in sorted({item.get("envs_per_worker") for item in scale}):
        entries = group(envs)  # type: ignore[arg-type]
        if len(entries) < 2 or not entries[0]["steps_per_second"]:
            continue
        first, last = entries[0], entries[-1]
        gain = (last["steps_per_second"] / first["steps_per_second"]) / max(
            1.0, last["workers"] / first["workers"]
        )
        if gain < 0.75:
            tips.append(
                f"Mehr Prozesse bringen kaum etwas (Skalierung {gain:.2f}× mit "
                f"{_plural(envs, 'Spiel', 'Spielen')} pro Prozess): Der Rechner ist am Anschlag "
                "— eher weniger Prozesse nehmen, sonst wird nur die CPU heiß."
            )
        elif gain > 0.9:
            tips.append(
                f"Die Simulation skaliert gut mit mehr Prozessen ({gain:.2f}× pro Kern): "
                "n_workers erhöhen lohnt."
            )
        break

    # 2) Spiele pro Prozess bei gleicher Prozesszahl.
    same_workers = [item for item in scale if item["workers"] == best.get("workers")]
    if len(same_workers) >= 2:
        env_best = max(same_workers, key=lambda item: item["steps_per_second"])
        env_worst = min(same_workers, key=lambda item: item["steps_per_second"])
        if env_worst["steps_per_second"]:
            factor = env_best["steps_per_second"] / env_worst["steps_per_second"]
            if factor > 1.15:
                tips.append(
                    f"{env_best['envs_per_worker']} Spiele pro Prozess sind {factor:.1f}× schneller "
                    f"als {env_worst['envs_per_worker']} — das teilt sich die Startkosten der "
                    "Prozesse."
                )

    if best.get("realtime_factor"):
        tips.append(
            f"Die Simulation läuft {best['realtime_factor']:.0f}× schneller als Echtzeit "
            f"({best['steps_per_second']:.0f} Schritte/s, "
            f"{_plural(best['workers'], 'Prozess', 'Prozesse')} × "
            f"{_plural(best['envs_per_worker'], 'Spiel', 'Spiele')})."
        )

    # 3) Das, was der Benutzer wirklich erlebt: sammeln + lernen nacheinander.
    update_cpu = report.get("update_cpu")
    update_cuda = report.get("update_cuda")
    effective = report.get("effective") or {}
    if effective.get("steps_per_second") and best.get("steps_per_second"):
        share = effective.get("update_share") or 0.0
        tips.append(
            f"Ende-zu-Ende bleiben {effective['steps_per_second']:.0f} Schritte/s "
            f"({report['steps_per_day'] / 1e6:,.0f} Mio. pro Tag): Der Lernschritt belegt "
            f"{share:.0%} der Zeit. Alle Zeitangaben rechnen mit dieser Zahl.".replace(",", ".")
        )
        if share > 0.5:
            if update_cuda and update_cpu:
                factor = update_cuda["steps_per_second"] / max(1e-6, update_cpu["steps_per_second"])
                if factor > 1.05:
                    tips.append(
                        f"Der Lernschritt dominiert. Die Grafikkarte schafft {factor:.1f}× mehr als "
                        "die CPU — device 'auto' nutzt sie automatisch."
                    )
                else:
                    tips.append(
                        "Der Lernschritt dominiert, die Grafikkarte hilft hier kaum (kleines Netz). "
                        "Hilft: weniger Epochen (epochs: 1–2) oder mehr torch_threads."
                    )
            else:
                tips.append(
                    "Der Lernschritt dominiert: Mehr Simulationsprozesse bringen dann nichts. "
                    "Hilft: weniger Epochen (epochs: 1–2), mehr torch_threads oder eine Grafikkarte."
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
    if report.get("torch_threads_recommended"):
        config.torch_threads = int(report["torch_threads_recommended"])
    # Für die Größe eines Sammelstapels zählt das Ende-zu-Ende-Tempo: Rund
    # 20 Minuten sammeln (plus lernen) pro Update ist ein guter Rhythmus.
    speed = (report.get("effective") or {}).get("steps_per_second") or report.get(
        "steps_per_second"
    )
    if speed:
        # Rund 20 Minuten sammeln + lernen pro Update — aber gedeckelt: Der
        # Stapel liegt komplett im Speicher (200 000 Schritte ≈ 150 MB
        # Beobachtungen, bei Nachahmung +90 MB Lehrer-Ziele) und wird pro Runde
        # durch die Pipe zu den Workern geschickt. Mehr lohnt nicht.
        config.steps_per_iteration = int(max(20_000, min(200_000, speed * 1200)))
    return config


def format_report(report: dict[str, Any]) -> str:
    """Lesbarer Text für die Kommandozeile."""
    lines = [
        f"Rechner: {report['cpu_count']} Kerne, PyTorch {report['torch']}, "
        f"Rechnen auf {report['device_name']}",
        f"Lernprozess: {report.get('torch_threads_recommended')} "
        f"{'Thread' if report.get('torch_threads_recommended') == 1 else 'Threads'} empfohlen "
        "(während des Lernens warten die Simulationen)",
        "",
        "Simulation (RocketSim + Netz):",
    ]
    for entry in report["scale"]:
        lines.append(
            f"  {_plural(entry['workers'], 'Prozess', 'Prozesse')} × "
            f"{_plural(entry['envs_per_worker'], 'Spiel', 'Spiele')}: "
            f"{entry['steps_per_second']:,.0f} Schritte/s, "
            f"{entry['decisions_per_second']:,.0f} Entscheidungen/s je Auto, "
            f"{entry['realtime_factor']:,.0f}× Echtzeit".replace(",", ".")
        )
    best = report.get("best") or {}
    if best:
        lines += [
            "",
            f"Beste Einstellung: {_plural(best['workers'], 'Prozess', 'Prozesse')} × "
            f"{_plural(best['envs_per_worker'], 'Spiel', 'Spiele')}",
        ]
    for key, label in (("update_cpu", "Lernschritt (CPU)"), ("update_cuda", "Lernschritt (GPU)")):
        entry = report.get(key)
        if entry:
            lines.append(
                f"{label}: {entry['steps']:,} Schritte × {entry['epochs']} Epochen in "
                f"{entry['seconds']:.2f} s ({entry['steps_per_second']:,.0f} Schritte/s, "
                f"{_plural(int(entry.get('threads') or 0), 'Thread', 'Threads')})".replace(",", ".")
            )
    effective = report.get("effective") or {}
    if effective.get("steps_per_second"):
        lines += [
            "",
            f"Ende-zu-Ende (sammeln + lernen): {effective['steps_per_second']:,.0f} Schritte/s "
            f"→ {report['steps_per_day'] / 1e6:,.0f} Mio. Schritte pro Tag (24/7)".replace(
                ",", "."
            ),
            "  Zeitanteil: Simulation "
            f"{effective.get('simulation_share', 0):.0%}, Lernen {effective.get('update_share', 0):.0%}"
            f" (Lernen auf {effective.get('update_device', '?')}, {effective.get('epochs', 0)} Epochen)",
        ]
    if report.get("advice"):
        lines += ["", "Empfehlungen:"] + [f"  - {tip}" for tip in report["advice"]]
    return "\n".join(lines)


def tick_seconds() -> float:
    """Dauer einer Entscheidung in simulierten Sekunden."""
    return TICK_SKIP / TICKS_PER_SECOND

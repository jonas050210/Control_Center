"""The training loop. Everything the web app shows is read from the run directory:

- ``config.json``        the TrainConfig
- ``status.json``        live state (running / stopping / finished / failed), steps, speed
- ``metrics.jsonl``      one line per PPO iteration
- ``evaluations.jsonl``  one line per evaluation round
- ``checkpoints/``       ``<steps>.pt`` and ``latest.pt``
- ``replays/``           recorded evaluation matches for the arena view
- ``control.json``       written by the UI: ``{"stop": true}`` ends training cleanly
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RunPaths, TrainConfig, run_paths
from .match import evaluate, play_match, save_replay
from .model import ActorCritic, load_checkpoint, model_from_checkpoint, save_checkpoint
from .opponents import PolicyPlayer, make_player
from .ppo import ppo_update
from .rollout import Batch, WorkerPool
from .teacher import TeacherLabeler

EVAL_OPPONENTS = ("chaser", "defender")
REPLAY_SECONDS = 60.0

#: Autopilot curriculum: (from stage, metric, threshold, minimum steps). The
#: average of the last CURRICULUM_WINDOW iterations must reach the threshold.
CURRICULUM = (
    (1, "touches_per_minute", 15.0, 10_000_000),
    (2, "goals_per_minute", 1.0, 50_000_000),
)
CURRICULUM_WINDOW = 20


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def stop_requested(paths: RunPaths) -> bool:
    return bool(read_json(paths.control, {}).get("stop"))


def summarize(batch: Batch, update: dict[str, float]) -> dict[str, Any]:
    stats = batch.stats
    episodes = stats["episodes"]
    agent_steps = max(1, stats["agent_steps"])
    game_seconds = sum(e["seconds"] for e in episodes)
    return {
        "episodes": len(episodes),
        "episode_reward": float(np.mean([e["reward"] for e in episodes])) if episodes else None,
        "episode_seconds": game_seconds / len(episodes) if episodes else None,
        "goals_per_minute": (60 * sum(e["goal"] for e in episodes) / game_seconds)
        if game_seconds
        else None,
        "touches_per_minute": (60 * sum(e["touches"] for e in episodes) / game_seconds)
        if game_seconds
        else None,
        "reward_parts": {k: v / agent_steps for k, v in stats["reward_parts"].items()},
        **past_summary(episodes),
        **teacher_summary(episodes),
        **update,
    }


def past_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Results of matches against older versions (wins count 1, draws ½)."""
    results = [e["vs_past"] for e in episodes if "vs_past" in e]
    if not results:
        return {}
    wins, losses = results.count(1), results.count(-1)
    draws = len(results) - wins - losses
    return {
        "past_games": len(results),
        "past_wins": wins,
        "past_losses": losses,
        "past_win_rate": (wins + 0.5 * draws) / len(results),
    }


def teacher_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Results of training matches against the teacher (wins count 1, draws ½)."""
    results = [e["vs_teacher"] for e in episodes if "vs_teacher" in e]
    if not results:
        return {}
    wins, losses = results.count(1), results.count(-1)
    draws = len(results) - wins - losses
    goals_for = sum(e.get("teacher_goals_for", 0) for e in episodes if "vs_teacher" in e)
    goals_against = sum(e.get("teacher_goals_against", 0) for e in episodes if "vs_teacher" in e)
    return {
        "teacher_games": len(results),
        "teacher_wins": wins,
        "teacher_losses": losses,
        "teacher_win_rate": (wins + 0.5 * draws) / len(results),
        "teacher_goals_for": goals_for,
        "teacher_goals_against": goals_against,
    }


def curriculum_ready(config: TrainConfig, steps: int, recent: list[dict[str, Any]]) -> bool:
    """Should the autopilot move on to the next reward stage?"""
    if not config.auto_curriculum or len(recent) < CURRICULUM_WINDOW:
        return False
    for stage, metric, threshold, min_steps in CURRICULUM:
        if config.reward_stage != stage or steps < min_steps:
            continue
        values = [m.get(metric) for m in recent[-CURRICULUM_WINDOW:]]
        values = [v for v in values if v is not None]
        return bool(values) and float(np.mean(values)) >= threshold
    return False


def prune_checkpoints(paths: RunPaths, keep: int) -> None:
    numbered = sorted(
        (p for p in paths.checkpoints.glob("*.pt") if p.stem.isdigit()), key=lambda p: int(p.stem)
    )
    for old in numbered[:-keep]:
        old.unlink(missing_ok=True)


class Trainer:
    def __init__(self, config: TrainConfig, log: Callable[[str], None] | None = None):
        config.validate()
        self.config = config
        self.paths = run_paths(config.name).ensure()
        self._log_handle = self.paths.log.open("a", encoding="utf-8")
        self._echo = log or print
        self.model = ActorCritic(hidden_sizes=config.hidden_sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.steps = 0
        self.iteration = 0
        self.started = time.time()
        self.rng = np.random.default_rng(config.seed)
        #: frozen older versions used as opponents: (steps, weights)
        self.past_pool: list[tuple[int, dict[str, Any]]] = []
        self.pool_changed = True
        #: set in train() when a teacher is configured
        self.teacher: TeacherLabeler | None = None
        latest = self.paths.checkpoints / "latest.pt"
        if latest.exists():
            self._resume(latest)
            self._load_past_pool()
        config.save(self.paths.config)

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self._log_handle.write(line + "\n")
        self._log_handle.flush()
        self._echo(line)

    def _resume(self, path: Path) -> None:
        payload = load_checkpoint(path)
        if list(payload["hidden_sizes"]) != list(self.config.hidden_sizes):
            raise ValueError(
                f"run {self.config.name!r} already has a {payload['hidden_sizes']} network; "
                "use a new run name for a different size"
            )
        self.model = model_from_checkpoint(payload)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate
        self.steps = int(payload["steps"])
        self.iteration = int(payload.get("extra", {}).get("iteration", 0))
        self.log(f"Fortgesetzt bei {self.steps:,} Schritten aus {path.name}")

    def _load_past_pool(self) -> None:
        numbered = sorted(
            (p for p in self.paths.checkpoints.glob("*.pt") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        for path in numbered[-self.config.past_pool_size :]:
            try:
                payload = load_checkpoint(path)
            except (OSError, ValueError, RuntimeError):
                continue
            if list(payload["hidden_sizes"]) == list(self.config.hidden_sizes):
                self.past_pool.append((int(payload["steps"]), payload["model"]))
        self.pool_changed = True

    def _add_snapshot(self) -> None:
        weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.past_pool.append((self.steps, weights))
        del self.past_pool[: -self.config.past_pool_size]
        self.pool_changed = True

    def status(self, state: str, **extra: Any) -> None:
        write_json(
            self.paths.status,
            {
                "state": state,
                "steps": self.steps,
                "total_steps": self.config.total_steps,
                "iteration": self.iteration,
                "started": self.started,
                "updated": time.time(),
                "pid": os.getpid(),
                **extra,
            },
        )

    def save(self) -> Path:
        extra = {"iteration": self.iteration}
        config = self.config.to_dict()
        target = self.paths.checkpoints / f"{self.steps}.pt"
        save_checkpoint(target, self.model, steps=self.steps, config=config, extra=extra)
        self._add_snapshot()
        save_checkpoint(
            self.paths.checkpoints / "latest.pt",
            self.model,
            steps=self.steps,
            config=config,
            optimizer=self.optimizer,
            extra=extra,
        )
        prune_checkpoints(self.paths, self.config.keep_checkpoints)
        return target

    def attach_teacher(self, batch: Batch, weight: float) -> dict[str, Any]:
        """Ask the teacher about the recorded situations and store its answers."""
        if (
            self.teacher is None
            or weight <= 0
            or batch.teacher_states is None
            or not len(batch.teacher_states)
        ):
            return {}
        states = batch.teacher_states
        rows = batch.teacher_rows
        slots = batch.teacher_slots
        previous = batch.teacher_previous
        wanted = int(self.config.teacher_samples or 0)
        if wanted and len(rows) > wanted:  # keep it balanced across the batch
            pick = np.linspace(0, len(rows) - 1, wanted).round().astype(int)
            states, rows, slots, previous = states[pick], rows[pick], slots[pick], previous[pick]
        started = time.perf_counter()
        answers = self.teacher.targets(states, slots, previous, self.config.teacher_temperature)
        targets = np.zeros((len(batch), 90), dtype=np.float32)
        mask = np.zeros(len(batch), dtype=bool)
        targets[rows] = answers
        mask[rows] = True
        batch.teacher_probs = targets
        batch.teacher_mask = mask
        return {
            "teacher_samples": int(len(rows)),
            "teacher_weight": round(float(weight), 4),
            "teacher_seconds": round(time.perf_counter() - started, 3),
        }

    def run_evaluation(self) -> dict[str, Any]:
        policy = PolicyPlayer(self.model, name=f"{self.config.name}@{self.steps}")
        results = []
        for spec in EVAL_OPPONENTS:
            results.append(
                evaluate(
                    policy,
                    make_player(spec),
                    games=self.config.eval_games,
                    team_size=self.config.team_size,
                    seed=self.config.seed + self.steps % 10_000,
                )
            )
        record = {"steps": self.steps, "time": time.time(), "results": results}
        append_jsonl(self.paths.evaluations, record)
        replay = play_match(
            policy,
            make_player("chaser"),
            team_size=self.config.team_size,
            seconds=REPLAY_SECONDS,
            record=True,
        )
        save_replay(
            self.paths.replays / f"{self.steps}.json",
            replay,
            {"run": self.config.name, "steps": self.steps, "kind": "evaluation"},
        )
        text = ", ".join(
            f"{r['opponent']}: {r['wins']}S/{r['draws']}U/{r['losses']}N ({r['goals_for']}:{r['goals_against']})"
            for r in results
        )
        self.log(f"Bewertung bei {self.steps:,}: {text}")
        return record

    def train(self) -> None:
        config = self.config
        cpu = os.cpu_count() or 2
        torch.set_num_threads(config.torch_threads or cpu)
        workers = config.resolved_workers()
        self.paths.control.unlink(missing_ok=True)
        self.status("starting", message=f"Starte {workers} Simulations-Prozesse")
        self.log(
            f"Training '{config.name}': {config.team_size}v{config.team_size}, Stufe {config.reward_stage}, "
            f"{workers} Prozesse x {config.envs_per_worker} Spiele, Ziel {config.total_steps:,} Schritte"
        )
        pool: WorkerPool | None = None
        next_checkpoint = (
            self.steps // config.checkpoint_every_steps + 1
        ) * config.checkpoint_every_steps
        every_eval = config.eval_every_steps
        next_eval = (self.steps // every_eval + 1) * every_eval if every_eval else None
        state = "finished"
        recent: deque[dict[str, Any]] = deque(maxlen=CURRICULUM_WINDOW)
        if config.past_opponent_prob:
            self.log(
                f"Gegner-Pool: {config.past_opponent_prob:.0%} der Spiele gegen ältere Versionen "
                f"({len(self.past_pool)} geladen, max. {config.past_pool_size})"
            )
        if config.teacher_opponent_prob > 0:
            self.log(
                f"Lehrer als Gegner: {config.teacher_opponent_prob:.0%} der Matches gegen "
                "Nexto (offline, lokal geladen)"
            )
        if config.auto_curriculum:
            self.log(f"Autopilot aktiv: startet bei Stufe {config.reward_stage}")
        if config.teacher_weight_at(self.steps) > 0:
            self.teacher = TeacherLabeler(progress=self.log)
            self.log(
                "Lehrer aktiv: die KI imitiert Nexto mit "
                f"{config.teacher_weight_at(self.steps):.0%} Gewicht und trainiert dann selbst weiter"
            )
        try:
            pool = WorkerPool(workers, config.to_dict(), config.seed + self.iteration)
            while self.steps < config.total_steps:
                if stop_requested(self.paths):
                    state = "stopped"
                    self.log("Stopp angefordert")
                    break
                tick = time.perf_counter()
                weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                past = None
                if self.pool_changed and config.past_opponent_prob and self.past_pool:
                    past = [w for _, w in self.past_pool]
                batch = pool.collect(weights, config.steps_per_iteration, past)
                self.pool_changed = self.pool_changed and past is None and bool(self.past_pool)
                collect_seconds = time.perf_counter() - tick
                teacher_weight = config.teacher_weight_at(self.steps)
                teacher_info = self.attach_teacher(batch, teacher_weight)
                update = ppo_update(
                    self.model,
                    self.optimizer,
                    batch,
                    epochs=config.epochs,
                    minibatch_size=config.minibatch_size,
                    clip_range=config.clip_range,
                    entropy_coef=config.entropy_coef,
                    value_coef=config.value_coef,
                    max_grad_norm=config.max_grad_norm,
                    target_kl=config.target_kl,
                    teacher_coef=teacher_weight if teacher_info else 0.0,
                    rng=self.rng,
                )
                total_seconds = time.perf_counter() - tick
                self.steps += len(batch)
                self.iteration += 1
                metrics = {
                    "iteration": self.iteration,
                    "steps": self.steps,
                    "time": time.time(),
                    "steps_per_second": len(batch) / total_seconds,
                    "collect_seconds": collect_seconds,
                    "update_seconds": total_seconds - collect_seconds,
                    "stage": config.reward_stage,
                    **teacher_info,
                    **summarize(batch, update),
                }
                append_jsonl(self.paths.metrics, metrics)
                self.status("running", steps_per_second=metrics["steps_per_second"])
                touches = metrics["touches_per_minute"]
                self.log(
                    f"#{self.iteration} {self.steps:,} Schritte | {metrics['steps_per_second']:,.0f}/s | "
                    f"Belohnung {metrics['episode_reward'] or 0:.2f} | "
                    f"Ballkontakte/min {touches if touches is not None else 0:.1f}"
                    + (
                        f" | gegen ältere Versionen {metrics['past_win_rate']:.0%}"
                        if "past_win_rate" in metrics
                        else ""
                    )
                    + (
                        f" | gegen den Lehrer {metrics['teacher_win_rate']:.0%} "
                        f"({metrics['teacher_goals_for']}:{metrics['teacher_goals_against']})"
                        if "teacher_win_rate" in metrics
                        else ""
                    )
                    + (
                        f" | Nachahmung {metrics['teacher_weight']:.0%}, Abweichung "
                        f"{metrics['teacher_loss']:.2f}"
                        if "teacher_weight" in metrics
                        else ""
                    )
                )
                recent.append(metrics)
                if curriculum_ready(config, self.steps, list(recent)):
                    config.reward_stage += 1
                    config.save(self.paths.config)
                    self.save()
                    self.log(
                        f"Autopilot: Ziel erreicht – weiter mit Stufe {config.reward_stage} "
                        f"bei {self.steps:,} Schritten"
                    )
                    pool.close()
                    pool = WorkerPool(workers, config.to_dict(), config.seed + self.iteration)
                    self.pool_changed = True
                    recent.clear()
                if self.steps >= next_checkpoint:
                    self.save()
                    next_checkpoint += config.checkpoint_every_steps
                if next_eval is not None and self.steps >= next_eval:
                    self.status("evaluating")
                    self.run_evaluation()
                    next_eval += every_eval
        except KeyboardInterrupt:
            state = "stopped"
            self.log("Abgebrochen (Strg+C)")
        except Exception as error:
            state = "failed"
            self.log(f"Fehler: {error!r}")
            self.status("failed", message=str(error))
            raise
        finally:
            if pool is not None:
                pool.close()
            if self.iteration:
                path = self.save()
                self.log(f"Checkpoint gespeichert: {path.name}")
            if state != "failed":
                self.status(state)
            self.paths.control.unlink(missing_ok=True)
            self.log(
                {"finished": "Training abgeschlossen", "stopped": "Training gestoppt"}.get(
                    state, state
                )
            )
            self._log_handle.close()


def train(config: TrainConfig) -> None:
    Trainer(config).train()

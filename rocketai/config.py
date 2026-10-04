"""Training configuration, presets and the on-disk layout of runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

#: Observation layout shared by training and the real-game bot: DefaultObs
#: padded for up to 3 cars per team, so one network can play 1v1 to 3v3.
OBS_PADDING = 3
OBS_SIZE = 52 + 20 * OBS_PADDING * 2  # 172
#: LookupTableAction size.
N_ACTIONS = 90
#: Physics ticks each decision is held for (120 Hz / 8 = 15 decisions per second).
TICK_SKIP = 8
TICKS_PER_SECOND = 120

RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def runs_root() -> Path:
    """Where runs live; ``ROCKETAI_RUNS`` overrides the default ``<repo>/runs``."""
    return Path(os.environ.get("ROCKETAI_RUNS") or ROOT / "runs")


def tools_root() -> Path:
    return Path(os.environ.get("ROCKETAI_TOOLS") or ROOT / "tools")


@dataclass
class TrainConfig:
    """Everything that defines a training run. Saved as ``config.json``."""

    name: str = "my-bot"
    team_size: int = 1  # cars per team (1 = 1v1, 2 = 2v2, 3 = 3v3)
    reward_stage: int = 1  # 1 = touch the ball, 2 = shoot at goal, 3 = full game
    total_steps: int = 100_000_000  # agent steps
    n_workers: int = 0  # 0 = auto (CPU cores - 1)
    envs_per_worker: int = 4
    steps_per_iteration: int = 50_000  # agent steps collected before each update
    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 3
    minibatch_size: int = 10_000
    max_grad_norm: float = 0.5
    target_kl: float = 0.02  # stop an update early once the policy moved this far (0 = off)
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 256])
    # Opponents: share of training matches against a frozen older version
    past_opponent_prob: float = 0.2
    past_pool_size: int = 5  # how many older versions are kept as opponents
    # Curriculum: move to the next reward stage by itself once the bot is ready
    auto_curriculum: bool = False
    # Teacher: Nexto shows the way (see rocketai/teacher.py). 0 = off.
    teacher_weight: float = 0.0  # how strongly the teacher counts at the start
    teacher_final_weight: float = 0.1  # ... and after the decay
    teacher_decay_steps: int = 50_000_000  # steps until the weight reaches the final value
    teacher_samples: int = 6_000  # teacher answers computed per iteration (0 = every step)
    teacher_opponent_prob: float = 0.0  # share of matches PLAYED AGAINST the teacher
    teacher_temperature: float = 1.0  # 1.0 = as trained; higher = softer advice
    # Episodes
    episode_seconds: float = 300.0  # hard cap per episode (game time)
    no_touch_seconds: float = 30.0  # reset when nobody touches the ball for this long
    # Bookkeeping
    checkpoint_every_steps: int = 2_000_000  # also what the live view follows
    keep_checkpoints: int = 20
    eval_every_steps: int = 5_000_000  # 0 = never
    eval_games: int = 6
    seed: int = 0
    torch_threads: int = 0  # 0 = auto

    def validate(self) -> None:
        problems = []
        if not RUN_NAME_PATTERN.match(self.name):
            problems.append("name: letters, digits, '-', '_' or '.', max 64 characters")
        if self.team_size not in (1, 2, 3):
            problems.append("team_size must be 1, 2 or 3")
        if self.reward_stage not in (1, 2, 3):
            problems.append("reward_stage must be 1, 2 or 3")
        positive = (
            "total_steps",
            "envs_per_worker",
            "steps_per_iteration",
            "epochs",
            "minibatch_size",
            "checkpoint_every_steps",
            "keep_checkpoints",
        )
        for name in positive:
            if getattr(self, name) < 1:
                problems.append(f"{name} must be at least 1")
        if self.n_workers < 0:
            problems.append("n_workers must be 0 (auto) or more")
        if not 0 < self.learning_rate < 1:
            problems.append("learning_rate must be between 0 and 1")
        if not 0 < self.gamma < 1 or not 0 < self.gae_lambda <= 1:
            problems.append("gamma must be in (0, 1) and gae_lambda in (0, 1]")
        if not 0 <= self.past_opponent_prob < 1:
            problems.append("past_opponent_prob must be in [0, 1)")
        if self.past_pool_size < 1:
            problems.append("past_pool_size must be at least 1")
        if not 0 <= self.teacher_weight <= 10 or not 0 <= self.teacher_final_weight <= 10:
            problems.append("teacher_weight and teacher_final_weight must be in [0, 10]")
        if self.teacher_decay_steps < 0:
            problems.append("teacher_decay_steps must be 0 (never decay) or more")
        if not 0 <= self.teacher_opponent_prob < 1:
            problems.append("teacher_opponent_prob must be in [0, 1)")
        if self.teacher_samples < 0:
            problems.append("teacher_samples must be 0 (all) or more")
        if self.teacher_temperature <= 0:
            problems.append("teacher_temperature must be positive")
        if not self.hidden_sizes or any(size < 8 for size in self.hidden_sizes):
            problems.append("hidden_sizes needs at least one layer of 8+ units")
        if self.episode_seconds <= 0 or self.no_touch_seconds <= 0:
            problems.append("episode_seconds and no_touch_seconds must be positive")
        if problems:
            raise ValueError("; ".join(problems))

    def teacher_weight_at(self, steps: int) -> float:
        """How much the teacher still counts after ``steps`` training steps."""
        from .teacher import teacher_weight_at

        return teacher_weight_at(
            steps, self.teacher_weight, self.teacher_final_weight, self.teacher_decay_steps
        )

    def resolved_workers(self) -> int:
        if self.n_workers:
            return self.n_workers
        return max(1, (os.cpu_count() or 2) - 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        config = cls(**{key: value for key, value in data.items() if key in known})
        config.hidden_sizes = [int(size) for size in config.hidden_sizes]
        return config

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


#: Ready-made starting points shown in the UI.
PRESETS: dict[str, dict[str, Any]] = {
    "quick-test": {
        "label": "Schnelltest",
        "description": "2 Minuten zum Ausprobieren, ob alles läuft.",
        "values": {
            "total_steps": 200_000,
            "steps_per_iteration": 20_000,
            "minibatch_size": 5_000,
            "checkpoint_every_steps": 100_000,
            "eval_every_steps": 0,
            "hidden_sizes": [256, 256],
        },
    },
    "autopilot": {
        "label": "Autopilot (empfohlen)",
        "description": "Startet bei Ballkontakt und schaltet selbst zu Toren und "
        "Komplettspiel weiter, sobald die KI so weit ist.",
        "values": {
            "team_size": 1,
            "reward_stage": 1,
            "auto_curriculum": True,
            "total_steps": 500_000_000,
        },
    },
    "student": {
        "label": "Schüler mit Lehrer (stärkste KI)",
        "description": "Spielt jeden vierten Trainings-Match gegen Nexto (Grand Champion), "
        "ahmt ihn zusätzlich nach und trainiert dann selbst weiter.",
        "values": {
            "team_size": 1,
            "reward_stage": 3,
            "teacher_weight": 1.0,
            "teacher_final_weight": 0.1,
            "teacher_decay_steps": 50_000_000,
            "teacher_opponent_prob": 0.25,
            "total_steps": 300_000_000,
        },
    },
    "beginner": {
        "label": "Anfänger 1v1",
        "description": "Lernt zum Ball zu fahren und ihn zu treffen (Stufe 1).",
        "values": {"team_size": 1, "reward_stage": 1, "total_steps": 50_000_000},
    },
    "striker": {
        "label": "Torjäger 1v1",
        "description": "Aufbauend: Ball Richtung Tor und Tore (Stufe 2).",
        "values": {"team_size": 1, "reward_stage": 2, "total_steps": 300_000_000},
    },
    "full-game": {
        "label": "Komplettes Spiel",
        "description": "Alles inklusive Luftspiel und Boost (Stufe 3), sehr lang.",
        "values": {
            "team_size": 1,
            "reward_stage": 3,
            "total_steps": 1_000_000_000,
            "learning_rate": 1e-4,
        },
    },
}


def preset_config(preset: str, **overrides: Any) -> TrainConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    data = TrainConfig().to_dict()
    data.update(PRESETS[preset]["values"])
    data.update({key: value for key, value in overrides.items() if value is not None})
    return TrainConfig.from_dict(data)


@dataclass
class RunPaths:
    """File layout of one run directory."""

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def control(self) -> Path:
        return self.root / "control.json"

    @property
    def log(self) -> Path:
        return self.root / "train.log"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations.jsonl"

    @property
    def replays(self) -> Path:
        return self.root / "replays"

    def ensure(self) -> RunPaths:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.replays.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(name: str) -> RunPaths:
    if not RUN_NAME_PATTERN.match(name):
        raise ValueError(f"invalid run name {name!r}")
    return RunPaths(runs_root() / name)

"""Central, serialisable configuration for SandboxAI experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import os
import shutil
from typing import Any


def find_godot_executable(preferred: str = "godot") -> str:
    """Find a usable Godot executable across PATH, environment variables and common platform locations."""
    if preferred and (shutil.which(preferred) or Path(preferred).is_file()):
        return preferred
    env_path = os.environ.get("GODOT_PATH") or os.environ.get("GODOT_EXECUTABLE")
    if env_path and (shutil.which(env_path) or Path(env_path).is_file()):
        return env_path
    candidates = [
        "godot4",
        "godot",
        "godot.exe",
        "Godot_v4.7.2-stable_linux.x86_64",
        "Godot_v4.7.2-stable_win64.exe",
        "Godot_v4.3-stable_linux.x86_64",
        "Godot_v4.2-stable_linux.x86_64",
        "/usr/local/bin/godot",
        "/usr/bin/godot",
        "C:\\Program Files\\Godot\\godot.exe",
    ]
    for candidate in candidates:
        if shutil.which(candidate) or Path(candidate).is_file():
            return candidate
    return preferred or "godot"


@dataclass
class TrainingConfig:
    environment_count: int = 8
    enemy_count: int = 1
    learning_rate: float = 3e-4
    rollout_length: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Non-zero by default: with MultiDiscrete actions a 0 entropy coefficient
    # lets the policy collapse to a degenerate action (e.g. never shooting)
    # in the first few updates, after which useful behavior can no longer be
    # discovered. 0.01 is a gentle standard value for discrete control.
    entropy_coefficient: float = 0.01
    clip_range: float = 0.2
    total_training_steps: int = 1_000_000
    checkpoint_frequency: int = 100_000
    evaluation_frequency: int = 50_000
    evaluation_episodes: int = 20
    seed: int = 1234
    device: str = "auto"
    curriculum_level: int = 3
    godot_executable: str = "godot"
    project_path: str = ""
    output_root: str = "training"
    run_id: str = ""
    experiment_id: str = ""
    bc_checkpoint: str = ""
    torch_threads: int = 0
    net_arch: tuple[int, int] = (128, 128)
    early_stopping_patience: int = 0
    min_eval_reward: float | None = None
    reward_breakdown_logging: bool = True

    def validate(self) -> "TrainingConfig":
        if self.environment_count < 1:
            raise ValueError("environment_count must be >= 1")
        if self.enemy_count < 1:
            raise ValueError("enemy_count must be >= 1")
        if self.rollout_length < 1:
            raise ValueError("rollout_length must be >= 1")
        if self.batch_size < 1 or self.batch_size > self.rollout_length * self.environment_count:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be in [1, rollout_length * environment_count] "
                f"([1, {self.rollout_length * self.environment_count}])"
            )
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")
        # Negative entropy would actively reward determinism; forbid it so
        # the mistake surfaces at config load instead of as silent policy
        # collapse during training.
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be non-negative")
        if self.total_training_steps < 1:
            raise ValueError("total_training_steps must be positive")
        if self.checkpoint_frequency < 1 or self.evaluation_frequency < 1:
            raise ValueError("checkpoint/evaluation frequency must be positive")
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be >= 1")
        if len(self.net_arch) < 1 or any(size < 1 for size in self.net_arch):
            raise ValueError("net_arch must contain at least one positive layer size")
        if self.torch_threads < 0:
            raise ValueError("torch_threads must be non-negative (0 = engine default)")
        if self.curriculum_level not in range(1, 6):
            raise ValueError("curriculum_level must be between 1 and 5")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        return self

    @property
    def project(self) -> Path:
        if self.project_path:
            return Path(self.project_path).expanduser().resolve()
        # python/sandboxai/config.py -> repository root
        return Path(__file__).resolve().parents[2]

    def resolved_device(self) -> str:
        requested = self.device.lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of auto, cpu, cuda")
        if requested == "cpu":
            return "cpu"
        try:
            import torch  # type: ignore
        except ImportError:
            if requested == "cuda":
                raise RuntimeError("CUDA was requested but PyTorch is not installed")
            return "cpu"
        available = bool(torch.cuda.is_available())
        if requested == "cuda" and not available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"

    def run_directory(self) -> Path:
        import datetime as _datetime

        run_id = self.run_id or _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix = f"{self.experiment_id}_" if self.experiment_id else ""
        return Path(self.output_root).expanduser() / "runs" / f"{prefix}{run_id}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["net_arch"] = list(self.net_arch)
        data["project_path"] = str(self.project)
        data["resolved_device"] = self.resolved_device()
        return data

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        clean = {key: value for key, value in values.items() if key in fields}
        if "net_arch" in clean:
            clean["net_arch"] = tuple(int(value) for value in clean["net_arch"])
        config = cls(**clean)
        return config.validate()

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class BCConfig:
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 3e-4
    validation_fraction: float = 0.1
    hidden_sizes: tuple[int, int] = (128, 128)
    seed: int = 1234
    device: str = "auto"
    checkpoint_frequency: int = 5
    early_stopping_patience: int = 5
    output_root: str = "training"

    def validate(self) -> "BCConfig":
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("BC epochs and batch_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if any(size < 1 for size in self.hidden_sizes):
            raise ValueError("BC hidden sizes must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        return self

    def resolved_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        try:
            import torch  # type: ignore
        except ImportError:
            if self.device == "cuda":
                raise RuntimeError("CUDA was requested but PyTorch is not installed")
            return "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda" if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()) else "cpu"


@dataclass
class EvaluationConfig:
    episodes: int = 20
    seed: int = 9001
    environment_count: int = 1
    output_root: str = "training/evaluations"


@dataclass
class SelfPlayConfig:
    environment_count: int = 8
    seed: int = 1234
    opponent_checkpoint: str = ""
    opponent_pool: list[str] = field(default_factory=list)
    learning_slot: int = 0
    frozen_slot: int = 1

    def validate(self) -> "SelfPlayConfig":
        if self.environment_count < 1:
            raise ValueError("self-play environment_count must be >= 1")
        if self.learning_slot == self.frozen_slot:
            raise ValueError("learning and frozen self-play slots must differ")
        return self

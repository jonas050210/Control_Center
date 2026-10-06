"""CPU-focused reinforcement-learning and imitation utilities."""

from training.train import TrainingConfig, TrainingController
from training.workers import auto_worker_count, resolve_parallelism

__all__ = ["TrainingConfig", "TrainingController", "auto_worker_count", "resolve_parallelism"]

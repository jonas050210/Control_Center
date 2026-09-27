"""SandboxAI Python training toolkit."""

__version__ = "0.2.0"

from .config import BCConfig, EvaluationConfig, SelfPlayConfig, TrainingConfig
from .dataset import DemonstrationDataset, DemonstrationRecorder

__all__ = [
    "BCConfig",
    "EvaluationConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "DemonstrationDataset",
    "DemonstrationRecorder",
]

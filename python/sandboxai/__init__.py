"""SandboxAI Python training toolkit."""

__version__ = "0.3.0"

from .auto_curriculum import AutoCurriculum, CurriculumSchedule
from .conditions import Condition, ConditionSpace, ConditionTracker, generalization_report
from .config import BCConfig, EvaluationConfig, SelfPlayConfig, TrainingConfig
from .dataset import DemonstrationDataset, DemonstrationRecorder
from .league import CheckpointRegistry, League, PolicyRecord

__all__ = [
    "AutoCurriculum",
    "BCConfig",
    "CheckpointRegistry",
    "Condition",
    "ConditionSpace",
    "ConditionTracker",
    "CurriculumSchedule",
    "DemonstrationDataset",
    "DemonstrationRecorder",
    "EvaluationConfig",
    "League",
    "PolicyRecord",
    "SelfPlayConfig",
    "TrainingConfig",
    "generalization_report",
]

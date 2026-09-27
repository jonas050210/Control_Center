"""SandboxAI Data Pipeline (M1).

Human gameplay recording, timestamp synchronization, schema validation,
and statistics inspection for Behavioral Cloning (BC).
"""

from data_pipeline.schema import (
    ACTION_CONTRACT_VERSION,
    ACTION_SPACE_SPEC,
    ActionState,
    CaptureConfig,
    DatasetMetadata,
    DatasetSample,
    MouseConfig,
    SCHEMA_VERSION,
)
from data_pipeline.actions import MouseBinner, build_action_state
from data_pipeline.capture import CaptureSource, ScreenCapture
from data_pipeline.input_listener import InputListener, RawInputEvent
from data_pipeline.mock import MockInputGenerator, MockScreenCapture
from data_pipeline.sync import ActionSynchronizer

__version__ = SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "ACTION_CONTRACT_VERSION",
    "ACTION_SPACE_SPEC",
    "ActionState",
    "CaptureConfig",
    "MouseConfig",
    "DatasetMetadata",
    "DatasetSample",
    "MouseBinner",
    "build_action_state",
    "CaptureSource",
    "ScreenCapture",
    "InputListener",
    "RawInputEvent",
    "MockScreenCapture",
    "MockInputGenerator",
    "ActionSynchronizer",
]

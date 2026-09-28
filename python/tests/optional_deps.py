"""Optional third-party dependency probes shared by the Python test suite.

The SandboxAI Python package has a deliberately tiny hard dependency set
(``numpy`` only). Training/evaluation extras (``torch``, ``gymnasium``,
``stable-baselines3``, ``tensorboard``, ``psutil``) and the GDScript
tooling extra (``gdtoolkit``) are optional, because the repository must
stay testable in a container that cannot install a multi-hundred-megabyte
CUDA wheel.

Tests that genuinely require one of those extras must SKIP when it is
absent rather than fail: a missing optional dependency is an environment
fact, not a regression in SandboxAI. Tests must never be deleted to make
the suite green — see the testing rules in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from importlib.util import find_spec


def has_module(name: str) -> bool:
    """True when ``name`` can be imported without actually importing it."""
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


HAS_NUMPY: bool = has_module("numpy")
HAS_TORCH: bool = has_module("torch")
HAS_GYMNASIUM: bool = has_module("gymnasium")
HAS_SB3: bool = has_module("stable_baselines3")
HAS_TENSORBOARD: bool = has_module("tensorboard")
HAS_PSUTIL: bool = has_module("psutil")
HAS_GDTOOLKIT: bool = has_module("gdtoolkit")

TORCH_REASON = "PyTorch is an optional training extra and is not installed"
GYMNASIUM_REASON = "gymnasium is an optional training extra and is not installed"
SB3_REASON = "stable-baselines3 is an optional training extra and is not installed"
GDTOOLKIT_REASON = "gdtoolkit is an optional GDScript tooling extra and is not installed"

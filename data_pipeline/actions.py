"""Action space definitions, key mappings, and mouse discretization for SandboxAI."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from data_pipeline.schema import ActionState, MouseConfig

# Canonical key bindings
KEY_MAPPINGS = {
    # Lateral movement: Left (-1), Right (+1)
    "a": ("move_x", -1),
    "d": ("move_x", 1),
    # Longitudinal movement: Backward (-1), Forward (+1)
    "s": ("move_y", -1),
    "w": ("move_y", 1),
    # State flags
    "space": ("jump", 1),
    "ctrl": ("crouch", 1),
    "ctrl_l": ("crouch", 1),
    "ctrl_r": ("crouch", 1),
    "c": ("crouch", 1),
    "shift": ("sprint", 1),
    "shift_l": ("sprint", 1),
    "shift_r": ("sprint", 1),
    "r": ("reload", 1),
}


class MouseBinner:
    """Discretizes continuous mouse movements into discrete bin indices and back.

    FPS mouse movements have a heavy-tailed distribution: many small sub-pixel
    and micro adjustments around 0, and occasional large fast flicks.
    Symmetric non-linear / log-scaled or quantile bins provide fine resolution
    near zero while bounding the action space for imitation learning / RL.
    """

    def __init__(
        self,
        num_bins: int = 21,
        strategy: str = "symmetric_log",
        custom_edges: Optional[Sequence[float]] = None,
    ) -> None:
        if num_bins < 3 or num_bins % 2 == 0:
            raise ValueError(
                f"num_bins should be an odd number >= 3 (e.g. 11, 21, 31), got {num_bins}"
            )
        self.num_bins = num_bins
        self.strategy = strategy

        if custom_edges is not None:
            if len(custom_edges) != num_bins + 1:
                raise ValueError(
                    f"custom_edges must contain num_bins + 1 ({num_bins + 1}) values"
                )
            self.edges = [float(edge) for edge in custom_edges]
            if any(not math.isfinite(edge) for edge in self.edges):
                raise ValueError("custom_edges must contain only finite values")
            if any(not self.edges[i] < self.edges[i + 1] for i in range(num_bins)):
                raise ValueError("custom_edges must be strictly increasing")
        elif strategy == "uniform":
            self.edges = self.generate_uniform_edges(num_bins, max_val=100.0)
        elif strategy == "symmetric_log":
            self.edges = self.generate_symmetric_log_edges(num_bins, max_val=150.0)
        else:
            raise ValueError(
                "strategy must be 'symmetric_log' or 'uniform' unless custom_edges are provided"
            )

    @staticmethod
    def generate_symmetric_log_edges(
        num_bins: int = 21, max_val: float = 150.0
    ) -> List[float]:
        """Generates symmetric bin edges with finer resolution near zero.

        For num_bins = 21:
          Bin 10 (center) corresponds to [-0.2, 0.2] (virtually zero).
          Outer bins use finite JSON-safe sentinels around exponential edges.
        """
        half_bins = (num_bins - 1) // 2  # e.g., 10 for 21 bins
        # Positive thresholds: from 0.2 to max_val geometrically/exponentially
        min_pos = 0.2
        ratios = np.geomspace(min_pos, max_val, half_bins)
        positive_edges = [float(r) for r in ratios]

        negative_edges = [-p for p in reversed(positive_edges)]
        # Center threshold: [-min_pos, +min_pos]
        # Full edges list has length num_bins + 1
        # Finite sentinels keep metadata strict JSON while the interior edges
        # still define the two unbounded outer categories in discretize().
        edges = (
            [-1.0e9]
            + negative_edges[:-1]
            + [-min_pos, min_pos]
            + positive_edges[1:]
            + [1.0e9]
        )
        return edges

    @staticmethod
    def generate_uniform_edges(
        num_bins: int = 21, max_val: float = 100.0
    ) -> List[float]:
        """Generates linearly spaced symmetric bin edges."""
        half_bins = (num_bins - 1) // 2
        step = max_val / half_bins
        pos = [step * (i + 1) for i in range(half_bins - 1)]
        neg = [-p for p in reversed(pos)]
        center_deadzone = step * 0.1
        edges = (
            [-1.0e9]
            + neg
            + [-center_deadzone, center_deadzone]
            + pos
            + [1.0e9]
        )
        return edges

    @classmethod
    def fit_quantile_edges(
        cls,
        values: Sequence[float],
        num_bins: int = 21,
    ) -> MouseBinner:
        """Fits quantile-based bin edges from empirical mouse delta observations."""
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < num_bins * 5:
            # Fallback to symmetric log if insufficient data
            return cls(num_bins=num_bins, strategy="symmetric_log")

        quantiles = np.linspace(0, 1, num_bins + 1)
        edges = np.quantile(arr, quantiles).tolist()
        edges[0] = -1.0e9
        edges[-1] = 1.0e9
        # Ensure monotonic uniqueness
        for i in range(1, len(edges) - 1):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-4
        return cls(num_bins=num_bins, strategy="quantile", custom_edges=edges)

    def discretize(self, value: float) -> int:
        """Maps a continuous delta value into a discrete bin index [0, num_bins - 1]."""
        if math.isnan(value):
            value = 0.0
        # np.digitize: index i where edges[i-1] <= value < edges[i]
        # With edges[0] = -inf, np.digitize returns 1 for values in (edges[0], edges[1]]
        # Subtract 1 to get 0-indexed bin [0, num_bins - 1]
        idx = int(np.digitize(value, self.edges[1:-1]))
        return max(0, min(idx, self.num_bins - 1))

    def dequantize(self, bin_idx: int) -> float:
        """Maps a discrete bin index back to its representative continuous center value."""
        bin_idx = max(0, min(bin_idx, self.num_bins - 1))
        left = self.edges[bin_idx]
        right = self.edges[bin_idx + 1]

        # Outer categories are conceptually unbounded even though their stored
        # sentinels are finite for strict JSON compatibility.
        if bin_idx == 0:
            return float(right - 10.0)
        if bin_idx == self.num_bins - 1:
            return float(left + 10.0)

        # If straddling 0, return exactly 0.0
        if left < 0 and right > 0:
            return 0.0

        return float((left + right) / 2.0)


def build_action_state(
    active_keys: Set[str],
    mouse_buttons: Dict[str, bool],
    mouse_dx: float,
    mouse_dy: float,
    binner_x: MouseBinner,
    binner_y: MouseBinner,
    wheel_dy: int = 0,
) -> ActionState:
    """Translates raw active keys and accumulated mouse deltas into a synchronized ActionState."""
    move_x = 0
    move_y = 0
    jump = 0
    crouch = 0
    sprint = 0
    reload_action = 0

    norm_keys = {k.lower().replace("key.", "") for k in active_keys}

    # Movement calculation with cancellation (e.g. W+S -> 0, A+D -> 0)
    if "a" in norm_keys and "d" not in norm_keys:
        move_x = -1
    elif "d" in norm_keys and "a" not in norm_keys:
        move_x = 1

    if "s" in norm_keys and "w" not in norm_keys:
        move_y = -1
    elif "w" in norm_keys and "s" not in norm_keys:
        move_y = 1

    if "space" in norm_keys:
        jump = 1

    if any(k in norm_keys for k in ("c", "ctrl", "ctrl_l", "ctrl_r")):
        crouch = 1

    if any(k in norm_keys for k in ("shift", "shift_l", "shift_r")):
        sprint = 1

    if "r" in norm_keys:
        reload_action = 1

    fire = 1 if mouse_buttons.get("left", False) else 0
    ads = 1 if mouse_buttons.get("right", False) else 0

    dx_bin = binner_x.discretize(mouse_dx)
    dy_bin = binner_y.discretize(mouse_dy)

    return ActionState(
        move_x=move_x,
        move_y=move_y,
        jump=jump,
        crouch=crouch,
        sprint=sprint,
        reload=reload_action,
        fire=fire,
        ads=ads,
        mouse_dx=float(mouse_dx),
        mouse_dy=float(mouse_dy),
        mouse_dx_bin=dx_bin,
        mouse_dy_bin=dy_bin,
        wheel_dy=wheel_dy,
        active_keys=sorted(list(norm_keys)),
        mouse_buttons=dict(mouse_buttons),
    )

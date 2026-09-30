"""Multi-seed experiment management, statistical aggregation and regression testing.

Provides structured multi-seed experiment tracking, statistical distributions
(mean, sample standard deviation, 95% confidence intervals, standard error),
automated comparison against baseline checkpoints/runs, and regression detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Any, Sequence


@dataclass
class ExperimentConfig:
    """Configuration for a multi-seed experiment."""

    name: str
    seeds: list[int] = field(default_factory=lambda: [101, 202, 303, 404, 505])
    total_training_steps: int = 100_000
    environment_count: int = 8
    enemy_count: int = 1
    curriculum_mode: str = "auto"
    curriculum_start_level: int = 1
    learning_rate: float = 3e-4
    net_arch: tuple[int, ...] = (128, 128)
    device: str = "auto"
    output_root: str = "experiments"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def validate(self) -> "ExperimentConfig":
        if not self.name or not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if not self.seeds:
            raise ValueError("seeds list must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds list must contain unique seeds")
        if self.total_training_steps < 1:
            raise ValueError("total_training_steps must be >= 1")
        if self.environment_count < 1:
            raise ValueError("environment_count must be >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["net_arch"] = list(self.net_arch)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "net_arch" in known and isinstance(known["net_arch"], list):
            known["net_arch"] = tuple(known["net_arch"])
        return cls(**known)


def compute_statistics(values: Sequence[float]) -> dict[str, float]:
    """Computes descriptive sample statistics and 95% confidence intervals."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

    clean = [float(v) for v in values if math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

    sorted_vals = sorted(clean)
    mean_val = sum(clean) / n
    median_val = (
        sorted_vals[n // 2]
        if n % 2 != 0
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    )
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in clean) / (n - 1)
        std_val = math.sqrt(max(0.0, variance))
        sem_val = std_val / math.sqrt(n)
        ci_margin = 1.96 * sem_val
    else:
        std_val = 0.0
        sem_val = 0.0
        ci_margin = 0.0

    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "ci95_low": mean_val - ci_margin,
        "ci95_high": mean_val + ci_margin,
    }


def aggregate_seed_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates multiple seed run results into a multi-seed statistical summary."""
    if not runs:
        return {"runs_count": 0, "metrics": {}}

    metric_keys = [
        "win_rate",
        "mean_episode_reward",
        "mean_kills",
        "mean_accuracy",
        "timesteps",
        "episode_length",
        "damage_dealt",
        "damage_taken",
    ]

    extracted: dict[str, list[float]] = {k: [] for k in metric_keys}

    for run in runs:
        for k in metric_keys:
            if k in run:
                val = run[k]
                if isinstance(val, (int, float)) and math.isfinite(val):
                    extracted[k].append(float(val))
            elif "metrics" in run and isinstance(run["metrics"], dict) and k in run["metrics"]:
                val = run["metrics"][k]
                if isinstance(val, (int, float)) and math.isfinite(val):
                    extracted[k].append(float(val))

    metrics_summary = {}
    for k, v in extracted.items():
        if v:
            metrics_summary[k] = compute_statistics(v)

    return {
        "runs_count": len(runs),
        "metrics": metrics_summary,
    }


def _approx_p_value_z(z: float) -> float:
    """Standard normal tail approximation."""
    z_abs = abs(z)
    # Complementary error function approximation
    t = 1.0 / (1.0 + 0.2316419 * z_abs)
    d = 0.3989423 * math.exp(-z_abs * z_abs / 2.0)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return max(0.0, min(1.0, 2.0 * p))


def compare_experiments(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Compares a candidate experiment against a baseline and flags regressions.

    `threshold` is the minimum relative decrease (e.g. 0.05 = 5%) to consider a regression.
    """
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})

    shared_keys = sorted(set(base_metrics.keys()) & set(cand_metrics.keys()))
    comparison: dict[str, Any] = {}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for k in shared_keys:
        b_stat = base_metrics[k]
        c_stat = cand_metrics[k]

        b_mean = float(b_stat.get("mean", 0.0))
        c_mean = float(c_stat.get("mean", 0.0))
        b_std = float(b_stat.get("std", 0.0))
        c_std = float(c_stat.get("std", 0.0))
        b_n = max(1, int(b_stat.get("count", 1)))
        c_n = max(1, int(c_stat.get("count", 1)))

        delta = c_mean - b_mean
        denom = abs(b_mean) if abs(b_mean) > 1e-6 else 1.0
        rel_change = delta / denom

        # Welch's t-statistic for difference of means
        se_diff = math.sqrt((b_std**2 / b_n) + (c_std**2 / c_n))
        z_score = delta / se_diff if se_diff > 1e-6 else 0.0
        p_val = _approx_p_value_z(z_score)

        item = {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "candidate_mean": c_mean,
            "candidate_std": c_std,
            "delta": delta,
            "relative_change": rel_change,
            "p_value": p_val,
        }
        comparison[k] = item

        # Critical metrics where a drop is bad
        if k in {"win_rate", "mean_episode_reward", "mean_accuracy", "mean_kills"}:
            if delta < -0.01 and rel_change <= -threshold:
                regressions.append({"metric": k, **item})
            elif delta > 0.01 and rel_change >= threshold:
                improvements.append({"metric": k, **item})

    status = "passed"
    if regressions:
        status = "regression_warning"
    elif improvements:
        status = "improved"

    return {
        "status": status,
        "threshold": threshold,
        "regressions": regressions,
        "improvements": improvements,
        "comparison": comparison,
    }


def format_experiment_report(
    summary: dict[str, Any],
    comparison: dict[str, Any] | None = None,
) -> str:
    """Formats experiment metrics and comparative analysis as a text report."""
    lines = [
        "=" * 78,
        " SANDBOXAI MULTI-SEED EXPERIMENT REPORT",
        "=" * 78,
        f"Runs Count:       {summary.get('runs_count', 0)}",
    ]

    metrics = summary.get("metrics", {})
    if metrics:
        lines.append("-" * 78)
        lines.append(f"{'Metric':<22}{'Mean':>10}{'Std':>10}{'Median':>10}{'95% CI Range':>22}")
        lines.append("-" * 78)
        for k, stat in metrics.items():
            mean_s = f"{stat.get('mean', 0.0):.3f}"
            std_s = f"{stat.get('std', 0.0):.3f}"
            med_s = f"{stat.get('median', 0.0):.3f}"
            ci_s = f"[{stat.get('ci95_low', 0.0):.2f}, {stat.get('ci95_high', 0.0):.2f}]"
            lines.append(f"{k:<22}{mean_s:>10}{std_s:>10}{med_s:>10}{ci_s:>22}")

    if comparison:
        lines.append("-" * 78)
        lines.append(f"Comparative Status: {comparison.get('status', 'unknown').upper()}")
        lines.append("-" * 78)
        lines.append(f"{'Metric':<22}{'Baseline':>10}{'Candidate':>10}{'Delta':>10}{'Rel Change':>12}{'p-val':>8}")
        lines.append("-" * 78)
        comp_dict = comparison.get("comparison", {})
        for k, row in comp_dict.items():
            b_s = f"{row.get('baseline_mean', 0.0):.3f}"
            c_s = f"{row.get('candidate_mean', 0.0):.3f}"
            d_s = f"{row.get('delta', 0.0):+.3f}"
            rel_s = f"{row.get('relative_change', 0.0):+.1%}"
            p_s = f"{row.get('p_value', 1.0):.3f}"
            lines.append(f"{k:<22}{b_s:>10}{c_s:>10}{d_s:>10}{rel_s:>12}{p_s:>8}")

        if comparison.get("regressions"):
            lines.append("-" * 78)
            lines.append("REGRESSIONS DETECTED:")
            for reg in comparison["regressions"]:
                lines.append(
                    f"  * {reg['metric']}: {reg['baseline_mean']:.3f} -> {reg['candidate_mean']:.3f} "
                    f"({reg['relative_change']:+.1%})"
                )

    lines.append("=" * 78)
    return "\n".join(lines)

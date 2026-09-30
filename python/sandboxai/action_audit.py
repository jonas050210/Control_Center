"""Low-overhead policy-action diagnostics for training and evaluation.

The engine reports successful discharges (and trigger pulls), but that alone
cannot distinguish a policy that selected ``shoot=0`` from a transport or
weapon-path failure. These helpers count the neural policy's MultiDiscrete
outputs *before* they enter the bridge and, at a sparse interval, inspect the
categorical head probabilities without sampling or changing RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .contract import ACTION_NVEC, ACTION_SPEC


ACTION_NAMES: tuple[str, ...] = tuple(component.name for component in ACTION_SPEC)
SHOOT_COMPONENT: int = 4


def _raw_component_probabilities(model: Any, observations: Any) -> list[Any] | None:
    """Returns SB3 MultiCategorical probabilities, or ``None`` if unavailable.

    This is diagnostics-only and intentionally feature-detected so scripted
    and mocked policies keep working. Calling ``get_distribution`` performs a
    deterministic forward pass under ``no_grad``; it does not sample and does
    not advance PyTorch's RNG stream.
    """
    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "obs_to_tensor") or not hasattr(policy, "get_distribution"):
        return None
    try:
        import torch  # type: ignore

        policy.set_training_mode(False)
        tensor, _vectorized = policy.obs_to_tensor(observations)
        with torch.no_grad():
            distribution = policy.get_distribution(tensor)
        categoricals = getattr(distribution, "distribution", None)
        if not isinstance(categoricals, list) or len(categoricals) != len(ACTION_NVEC):
            return None
        return [categorical.probs.detach().cpu().numpy() for categorical in categoricals]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Evaluation must remain compatible with non-SB3 policy handles.
        return None


def component_probabilities(model: Any, observations: Any) -> list[Any] | None:
    """Thread-safe probability inspection when the model exposes a wrapper."""
    synchronized = getattr(model, "component_probabilities", None)
    if callable(synchronized):
        return synchronized(observations)
    return _raw_component_probabilities(model, observations)


def shoot_probability_at(probabilities: list[Any] | None, batch_index: int = 0) -> float | None:
    """Reads P(shoot=1) from a batched MultiCategorical diagnostic."""
    if probabilities is None or len(probabilities) <= SHOOT_COMPONENT:
        return None
    values = probabilities[SHOOT_COMPONENT]
    try:
        if getattr(values, "ndim", 0) == 1:
            return float(values[1])
        return float(values[batch_index][1])
    except (IndexError, TypeError, ValueError):
        return None


@dataclass
class EpisodeActionAudit:
    """Counts one episode's exact policy-side action outputs."""

    counts: list[list[int]] = field(
        default_factory=lambda: [[0 for _ in range(cardinality)] for cardinality in ACTION_NVEC]
    )
    decisions: int = 0
    shoot_probability_sum: float = 0.0
    shoot_probability_samples: int = 0

    def reset(self) -> None:
        self.counts = [[0 for _ in range(cardinality)] for cardinality in ACTION_NVEC]
        self.decisions = 0
        self.shoot_probability_sum = 0.0
        self.shoot_probability_samples = 0

    def record(self, action: Sequence[Any], shoot_probability: float | None = None) -> None:
        values = action.tolist() if hasattr(action, "tolist") else list(action)
        # A single unvectorized SB3 action may retain a leading size-1 axis.
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = list(values[0])
        if len(values) != len(ACTION_NVEC):
            raise ValueError(
                f"policy action has {len(values)} components; expected {len(ACTION_NVEC)}"
            )
        for component, (raw, cardinality) in enumerate(zip(values, ACTION_NVEC)):
            value = int(raw)
            if value < 0 or value >= cardinality:
                raise ValueError(
                    f"policy action component {component} is {value}; expected 0..{cardinality - 1}"
                )
            self.counts[component][value] += 1
        self.decisions += 1
        if shoot_probability is not None:
            self.shoot_probability_sum += float(shoot_probability)
            self.shoot_probability_samples += 1

    def summary(self) -> dict[str, Any]:
        shoot_requests = self.counts[SHOOT_COMPONENT][1]
        result: dict[str, Any] = {
            "policy_action_decisions": self.decisions,
            "policy_action_counts": {
                name: list(self.counts[index]) for index, name in enumerate(ACTION_NAMES)
            },
            "policy_shoot_requests": shoot_requests,
            "policy_shoot_request_rate": (
                shoot_requests / self.decisions if self.decisions else 0.0
            ),
            "policy_shoot_probability_samples": self.shoot_probability_samples,
        }
        if self.shoot_probability_samples:
            result["policy_mean_shoot_probability"] = (
                self.shoot_probability_sum / self.shoot_probability_samples
            )
        return result


def summarize_action_pipeline(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Localizes policy request -> engine trigger -> weapon discharge."""
    decisions = sum(int(row.get("policy_action_decisions", 0)) for row in rows)
    shoot_requests = sum(int(row.get("policy_shoot_requests", 0)) for row in rows)
    probability_samples = sum(
        int(row.get("policy_shoot_probability_samples", 0)) for row in rows
    )
    probability_sum = sum(
        float(row.get("policy_mean_shoot_probability", 0.0))
        * int(row.get("policy_shoot_probability_samples", 0))
        for row in rows
    )
    trigger_reported = bool(rows) and all("trigger_pulls" in row for row in rows)
    trigger_pulls = sum(int(row.get("trigger_pulls", 0)) for row in rows)
    shots_fired = sum(int(row.get("shots_fired", 0)) for row in rows)
    if shoot_requests == 0:
        localization = "policy_argmax_never_requested_shoot"
    elif not trigger_reported:
        localization = "engine_trigger_metrics_unavailable"
    elif trigger_pulls == 0:
        localization = "policy_requests_not_observed_by_engine"
    elif shots_fired == 0:
        localization = "trigger_reached_engine_but_weapon_never_discharged"
    else:
        localization = "weapon_discharged"
    aggregate_counts = {
        name: [
            sum(
                int(row.get("policy_action_counts", {}).get(name, [0] * cardinality)[choice])
                for row in rows
            )
            for choice in range(cardinality)
        ]
        for name, cardinality in zip(ACTION_NAMES, ACTION_NVEC)
    }
    result: dict[str, Any] = {
        "policy_action_decisions": decisions,
        "policy_action_counts": aggregate_counts,
        "policy_shoot_requests": shoot_requests,
        "policy_shoot_request_rate": shoot_requests / decisions if decisions else 0.0,
        "policy_shoot_probability_samples": probability_samples,
        "engine_trigger_pulls_reported": trigger_reported,
        "engine_trigger_pulls": trigger_pulls if trigger_reported else None,
        "engine_shots_fired": shots_fired,
        "localization": localization,
    }
    if probability_samples:
        result["mean_stochastic_shoot_probability"] = probability_sum / probability_samples
    if trigger_reported:
        result["request_delivery_rate"] = (
            trigger_pulls / shoot_requests if shoot_requests else None
        )
        result["request_trigger_difference"] = shoot_requests - trigger_pulls
    result["fire_conversion_rate"] = shots_fired / trigger_pulls if trigger_pulls else None
    return result

"""Configurable, recorded checkpoint-selection rule.

"Best checkpoint" was hard-coded as *strictly higher mean shaped reward*.
That is a defensible default, but it is a research decision, not a law:
shaped reward moves when the reward weights or the curriculum move, so a
`best_eval.zip` carried across such a change is selected by a rule that no
longer means what it meant. Worse, the rule was implicit — nothing in the
run directory said which quantity had been maximised.

This module makes the rule explicit, configurable and *recorded*:

* explicit — :class:`CheckpointSelectionRule` owns metric, direction and
  the minimum improvement that counts as an improvement at all;
* configurable — the rule is built from ``TrainingConfig`` fields, so it
  lands in the run's ``config.json`` snapshot like every other setting;
* recorded — it is serialised into ``evaluations/best.json``, and a resume
  that finds an *incomparable* rule there refuses to inherit the old score
  instead of silently comparing win rate against shaped reward.

Deliberately not included: smoothing, multi-metric scalarisation, Pareto
selection. Those change what "best" means in ways that need their own
evidence; this module only makes the existing decision honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: Selection directions. ``max`` keeps the highest score (reward, win
#: rate, accuracy); ``min`` keeps the lowest (deaths, time-to-kill).
SELECTION_GOALS = ("max", "min")

#: Default metric: the mean shaped episode reward of the *normal*
#: evaluation, i.e. exactly the historical behaviour.
DEFAULT_SELECTION_METRIC = "mean_episode_reward"


def _lookup(summary: Any, metric: str) -> Any:
    """Resolves a dotted path into nested evaluation summaries.

    ``mean_episode_reward`` reads the top level; ``league.mean_score``
    reaches into the checkpoint report sections that ``train_ppo`` mirrors
    into the summary. Missing or non-numeric values return ``None`` rather
    than raising, so a rule pointed at a section that a given run does not
    produce degrades to "no improvement" instead of killing training.
    """
    current: Any = summary
    for part in metric.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


@dataclass(frozen=True)
class CheckpointSelectionRule:
    """Decides whether an evaluation beats the incumbent best checkpoint."""

    metric: str = DEFAULT_SELECTION_METRIC
    goal: str = "max"
    #: Improvement must exceed the incumbent by more than this. ``0.0``
    #: reproduces the historical strict-inequality behaviour.
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if not self.metric or not self.metric.strip():
            raise ValueError("checkpoint selection metric must be a non-empty key")
        if self.goal not in SELECTION_GOALS:
            raise ValueError(
                "unknown checkpoint selection goal %r; expected one of %s"
                % (self.goal, ", ".join(SELECTION_GOALS))
            )
        if self.min_delta < 0.0 or not math.isfinite(self.min_delta):
            raise ValueError("checkpoint selection min_delta must be finite and >= 0")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> CheckpointSelectionRule:
        return cls(
            metric=str(getattr(config, "checkpoint_selection_metric", DEFAULT_SELECTION_METRIC)),
            goal=str(getattr(config, "checkpoint_selection_goal", "max")),
            min_delta=float(getattr(config, "checkpoint_selection_min_delta", 0.0)),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> CheckpointSelectionRule:
        """Rebuilds a rule from a ``best.json`` record.

        ``None`` (a pre-rule ``best.json``) means the historical default,
        which is what those files were in fact selected by.
        """
        if not isinstance(payload, dict):
            return cls()
        return cls(
            metric=str(payload.get("metric", DEFAULT_SELECTION_METRIC)),
            goal=str(payload.get("goal", "max")),
            min_delta=float(payload.get("min_delta", 0.0)),
        )

    # -- scoring -----------------------------------------------------------

    def initial_score(self) -> float:
        return -math.inf if self.goal == "max" else math.inf

    def score(self, summary: Any) -> float | None:
        """Extracts this rule's metric from an evaluation summary."""
        value = _lookup(summary, self.metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def is_improvement(self, candidate: float, incumbent: float) -> bool:
        """True when ``candidate`` beats ``incumbent`` by more than ``min_delta``."""
        if not math.isfinite(candidate):
            return False
        if self.goal == "max":
            if incumbent == -math.inf:
                return True
            return candidate > incumbent + self.min_delta
        if incumbent == math.inf:
            return True
        return candidate < incumbent - self.min_delta

    # -- provenance --------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "goal": self.goal, "min_delta": self.min_delta}

    def matches(self, payload: Any) -> bool:
        """Whether a recorded rule is comparable with this one.

        Used on resume: an incumbent score produced by a different metric
        or direction is not a number this rule may compare against.
        ``min_delta`` is not part of comparability — it only tightens the
        threshold, it does not change the quantity being measured.
        """
        other = self.from_dict(payload)
        return other.metric == self.metric and other.goal == self.goal

    def describe(self) -> str:
        suffix = f" by more than {self.min_delta:g}" if self.min_delta > 0.0 else ""
        direction = "maximise" if self.goal == "max" else "minimise"
        return f"{direction} {self.metric}{suffix}"

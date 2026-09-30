"""Manually measured time-to-kill (TTK) trials: schema, validation, statistics.

Purpose
-------
Human testing is a legitimate, ethical calibration source for the
simulator: a person plays normally, the session is recorded with consent,
and an annotator marks *visible* events (target acquired, first trigger
pull, first damage shown, lethal frame) from the video or ordinary UI. The
resulting numbers say how fast real encounters resolve, which is what the
Godot weapon/reaction parameters should be calibrated against.

This module turns those annotations into a validated, reproducible dataset
and compares them against the simulator's own analytic TTK
(:mod:`sandboxai.weapons`), so "our rifle kills 40% faster than a human
ever does" becomes a number instead of an impression.

Hard boundaries (enforced, not just documented)
-----------------------------------------------
* **Player-perceivable data only.** Field names that imply private or
  server-authoritative state (memory reads, hidden positions, packet
  captures, injected hooks) are rejected by :func:`validate_trial`. If you
  cannot see it on screen or in your own input log, it does not belong in
  a trial.
* **No automation of a third-party game.** Nothing here reads, drives or
  connects to any game. It consumes a JSONL file an annotator wrote.
* **Censoring is data.** Trials where the target escaped, the tester died
  first, or the recording ended are kept with ``outcome != "kill"`` and
  excluded from TTK statistics *while being counted*, because dropping
  them silently biases every aggregate downwards.
* **Uncertainty is mandatory.** Aggregates report n, median, IQM and a
  bootstrap interval; a single "best" number is never produced.

Calibration evidence, not training data
---------------------------------------
A TTK trial is **not** a behavior-cloning transition: it has no 84-float
observation. :class:`TTKDataset` deliberately offers no ``arrays()``.
Demonstrations for BC come from the local Godot recorder
(:mod:`sandboxai.dataset`), which logs the real contract.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "sandboxai.ttk_trials"
SCHEMA_VERSION = 1

## Outcomes. Only "kill" contributes to TTK statistics; everything else is
## censored and reported separately.
OUTCOMES: tuple[str, ...] = ("kill", "target_escaped", "tester_died", "aborted", "no_damage")

## Where the measurement came from. Both are local and consented; the
## distinction matters because simulator trials can be re-run exactly and
## manual observation trials cannot.
SOURCES: tuple[str, ...] = ("local_simulator", "manual_observation")

MOVEMENT_STATES: tuple[str, ...] = ("stationary", "walking", "strafing", "sprinting", "airborne")
HIT_ZONES: tuple[str, ...] = ("body", "head", "limb", "mixed")

## Substrings that indicate the annotation used information a player cannot
## perceive. Presence of any of these in a field name fails validation.
FORBIDDEN_FIELD_MARKERS: tuple[str, ...] = (
    "memory_",
    "process_",
    "packet",
    "server_authoritative",
    "hidden_",
    "injected",
    "hook_",
    "exploit",
    "aimbot",
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "trial_id",
    "source",
    "weapon_profile",
    "distance_m",
    "target_health",
    "outcome",
)


@dataclass
class TTKTrial:
    """One annotated encounter.

    Times are seconds from a single, explicitly recorded start convention
    (``start_convention``), typically "first frame in which the target is
    visible". Only differences between them are used, so any consistent
    origin works — but mixing conventions inside one dataset does not,
    which is why the convention travels with the trial.
    """

    trial_id: str
    source: str
    weapon_profile: str
    distance_m: float
    target_health: float
    outcome: str
    # -- timing (seconds, same origin) ---------------------------------
    acquisition_time: float | None = None
    first_trigger_time: float | None = None
    first_damage_time: float | None = None
    lethal_time: float | None = None
    # -- shot economy ---------------------------------------------------
    shots_fired: int = 0
    shots_hit: int = 0
    # -- conditions -----------------------------------------------------
    movement_state: str = "stationary"
    target_movement_state: str = "stationary"
    hit_zone: str = "body"
    # -- provenance ------------------------------------------------------
    annotator: str = ""
    tester: str = ""
    consent: bool = False
    start_convention: str = "target_first_visible"
    frame_rate: float = 0.0
    session_id: str = ""
    build_version: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # -- derived ---------------------------------------------------------

    @property
    def reaction_time(self) -> float | None:
        """Acquisition -> first trigger pull (human reaction + decision)."""
        if self.acquisition_time is None or self.first_trigger_time is None:
            return None
        return self.first_trigger_time - self.acquisition_time

    @property
    def trigger_to_kill(self) -> float | None:
        """First trigger pull -> lethal frame. The weapon-comparable TTK."""
        if self.first_trigger_time is None or self.lethal_time is None:
            return None
        return self.lethal_time - self.first_trigger_time

    @property
    def damage_to_kill(self) -> float | None:
        """First damage -> lethal frame (excludes the whiffed opening)."""
        if self.first_damage_time is None or self.lethal_time is None:
            return None
        return self.lethal_time - self.first_damage_time

    @property
    def encounter_time(self) -> float | None:
        """Acquisition -> lethal frame: the full encounter cost."""
        if self.acquisition_time is None or self.lethal_time is None:
            return None
        return self.lethal_time - self.acquisition_time

    @property
    def accuracy(self) -> float | None:
        if self.shots_fired <= 0:
            return None
        return self.shots_hit / self.shots_fired

    @property
    def censored(self) -> bool:
        return self.outcome != "kill"

    def condition_key(self, distance_bucket_m: float = 3.0) -> str:
        """Grouping key: weapon, distance band, movement, hit zone."""
        band = int(self.distance_m // max(distance_bucket_m, 0.1))
        low = band * distance_bucket_m
        return (
            f"{self.weapon_profile}|{low:g}-{low + distance_bucket_m:g}m|"
            f"{self.movement_state}|{self.hit_zone}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TTKTrial:
        known = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        return cls(**known)


def _admissibility_problems(values: dict[str, Any], index: int) -> list[str]:
    """Ethics and shape gate: may the record be looked at at all, and is it complete?

    Runs before every other check and short-circuits them, so a record that
    carries non-perceivable data is rejected on that ground alone rather than
    also being picked apart field by field.
    """
    problems = [
        f"trial {index}: field {name!r} implies non-perceivable or "
        "unauthorized data; TTK trials accept player-visible annotation only"
        for name in values
        for marker in FORBIDDEN_FIELD_MARKERS
        if marker in str(name).lower()
    ]
    problems += [
        f"trial {index}: missing required field {name!r}"
        for name in REQUIRED_FIELDS
        if name not in values
    ]
    return problems


def _enum_problems(values: dict[str, Any], index: int) -> list[str]:
    """Categorical fields that must come from a closed vocabulary."""
    checks = (
        ("source", str(values["source"]), SOURCES),
        ("outcome", str(values["outcome"]), OUTCOMES),
        ("movement_state", str(values.get("movement_state", "stationary")), MOVEMENT_STATES),
        ("hit_zone", str(values.get("hit_zone", "body")), HIT_ZONES),
    )
    return [
        f"trial {index}: {name} must be one of {allowed}"
        for name, value, allowed in checks
        if value not in allowed
    ]


def _numeric_problems(values: dict[str, Any], index: int) -> list[str]:
    """Magnitudes and counts, including the hits-vs-shots relationship."""
    problems = [
        f"trial {index}: {name} must be a positive finite number"
        for name in ("distance_m", "target_health")
        if not _is_positive_finite(values[name])
    ]
    problems += [
        f"trial {index}: {name} must be a non-negative integer"
        for name in ("shots_fired", "shots_hit")
        if not _is_non_negative_int(values.get(name, 0))
    ]
    if int(values.get("shots_hit", 0)) > int(values.get("shots_fired", 0)):
        problems.append(f"trial {index}: shots_hit exceeds shots_fired")
    return problems


def _is_positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and value >= 0


def _timeline_problems(values: dict[str, Any], index: int) -> list[str]:
    """The four annotated timestamps: finite, ordered, and consistent with the outcome.

    Only stamps that are present are ordered against each other — an
    annotator who could not see the acquisition moment leaves it out rather
    than guessing, and that must not fail the ordering check.
    """
    present = [
        (name, float(values[name]))
        for name in (
            "acquisition_time",
            "first_trigger_time",
            "first_damage_time",
            "lethal_time",
        )
        if values.get(name) is not None
    ]
    problems = [
        f"trial {index}: {name} is not finite"
        for name, value in present
        if not math.isfinite(value)
    ]
    ordered = [value for _name, value in present if math.isfinite(value)]
    if ordered != sorted(ordered):
        problems.append(
            f"trial {index}: annotated times must be non-decreasing "
            "(acquisition <= first trigger <= first damage <= lethal)"
        )
    is_kill = str(values["outcome"]) == "kill"
    has_lethal = values.get("lethal_time") is not None
    if is_kill and not has_lethal:
        problems.append(f"trial {index}: outcome 'kill' requires a lethal_time")
    if not is_kill and has_lethal:
        problems.append(f"trial {index}: lethal_time is only meaningful for outcome 'kill'")
    return problems


def validate_trial(values: dict[str, Any], index: int = 0) -> list[str]:
    """Problems with one raw trial record; empty list means acceptable."""
    admissibility = _admissibility_problems(values, index)
    if admissibility:
        # Field-level checks below index into `values` unconditionally, which
        # is only safe once the required fields are known to be present.
        return admissibility

    problems = _enum_problems(values, index)
    problems += _numeric_problems(values, index)
    problems += _timeline_problems(values, index)
    if not bool(values.get("consent", False)):
        problems.append(
            f"trial {index}: consent must be explicitly true; unconsented recordings "
            "are not accepted as research data"
        )
    return problems


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile (no numpy dependency)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def interquartile_mean(values: Sequence[float]) -> float:
    """Mean of the middle 50% — robust to annotation outliers.

    Outliers are *kept* in the dataset; this statistic simply does not let
    a single mis-marked frame dominate the reported centre.
    """
    if not values:
        return float("nan")
    if len(values) < 4:
        return statistics.fmean(values)
    ordered = sorted(values)
    low = _percentile(ordered, 0.25)
    high = _percentile(ordered, 0.75)
    middle = [value for value in ordered if low <= value <= high]
    return statistics.fmean(middle or ordered)


def bootstrap_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 1234,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    Deterministic for a seed (uses a seeded LCG rather than the global
    ``random`` module) so a report can be regenerated exactly.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    count = len(values)
    state = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _draw in range(count):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            total += values[(state >> 33) % count]
        means.append(total / count)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return (_percentile(means, tail), _percentile(means, 1.0 - tail))


def summarize_values(values: Sequence[float], seed: int = 1234) -> dict[str, Any]:
    """n / mean / median / IQM / spread / bootstrap interval."""
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0}
    low, high = bootstrap_interval(clean, seed=seed)
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "iqm": interquartile_mean(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "p25": _percentile(clean, 0.25),
        "p75": _percentile(clean, 0.75),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


@dataclass
class TTKDataset:
    trials: list[TTKTrial]
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- io ---------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, strict: bool = True) -> TTKDataset:
        """Reads a JSONL trial file (optional leading metadata object).

        ``strict`` (the default) refuses the whole file when any trial is
        invalid: a partially accepted calibration dataset is worse than no
        dataset, because the rejected trials are exactly the unusual ones.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        metadata: dict[str, Any] = {}
        raw: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                if not raw and value.get("schema") == SCHEMA:
                    metadata = value
                    continue
                raw.append(value)
        problems: list[str] = []
        trials: list[TTKTrial] = []
        for index, values in enumerate(raw):
            issues = validate_trial(values, index)
            if issues:
                problems.extend(issues)
                continue
            trials.append(TTKTrial.from_dict(values))
        if problems and strict:
            raise ValueError("invalid TTK trials:\n  " + "\n  ".join(problems))
        dataset = cls(trials=trials, metadata=metadata)
        dataset.rejected = problems  # type: ignore[attr-defined]
        return dataset

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            **self.metadata,
        }
        with destination.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(header, separators=(",", ":")) + "\n")
            for trial in self.trials:
                stream.write(json.dumps(trial.to_dict(), separators=(",", ":")) + "\n")
        return destination

    # -- analysis ---------------------------------------------------------

    def kills(self) -> list[TTKTrial]:
        return [trial for trial in self.trials if not trial.censored]

    def censoring_report(self) -> dict[str, Any]:
        counts: dict[str, int] = {outcome: 0 for outcome in OUTCOMES}
        for trial in self.trials:
            counts[trial.outcome] = counts.get(trial.outcome, 0) + 1
        total = max(len(self.trials), 1)
        return {
            "trials": len(self.trials),
            "outcomes": counts,
            "censored": len(self.trials) - counts.get("kill", 0),
            "censored_fraction": (len(self.trials) - counts.get("kill", 0)) / total,
        }

    def summary(self, seed: int = 1234) -> dict[str, Any]:
        kills = self.kills()
        return {
            "format": "sandboxai.ttk_summary/v1",
            "censoring": self.censoring_report(),
            "trigger_to_kill": summarize_values(
                [trial.trigger_to_kill for trial in kills if trial.trigger_to_kill is not None],
                seed,
            ),
            "damage_to_kill": summarize_values(
                [trial.damage_to_kill for trial in kills if trial.damage_to_kill is not None],
                seed,
            ),
            "encounter_time": summarize_values(
                [trial.encounter_time for trial in kills if trial.encounter_time is not None],
                seed,
            ),
            # Reaction time is meaningful for censored trials too: the
            # tester still acquired and fired.
            "reaction_time": summarize_values(
                [trial.reaction_time for trial in self.trials if trial.reaction_time is not None],
                seed,
            ),
            "accuracy": summarize_values(
                [trial.accuracy for trial in self.trials if trial.accuracy is not None], seed
            ),
            "shots_to_kill": summarize_values(
                [float(trial.shots_fired) for trial in kills if trial.shots_fired > 0], seed
            ),
            "conditions": sorted({trial.condition_key() for trial in self.trials}),
            "testers": sorted({trial.tester for trial in self.trials if trial.tester}),
            "sources": sorted({trial.source for trial in self.trials}),
        }

    def by_condition(self, distance_bucket_m: float = 3.0, seed: int = 1234) -> dict[str, Any]:
        """Per-condition aggregates. Thin cells stay visible (n is reported)."""
        buckets: dict[str, list[TTKTrial]] = {}
        for trial in self.trials:
            buckets.setdefault(trial.condition_key(distance_bucket_m), []).append(trial)
        report: dict[str, Any] = {}
        for key, trials in sorted(buckets.items()):
            kills = [trial for trial in trials if not trial.censored]
            report[key] = {
                "trials": len(trials),
                "kills": len(kills),
                "censored_fraction": 1.0 - len(kills) / max(len(trials), 1),
                "trigger_to_kill": summarize_values(
                    [t.trigger_to_kill for t in kills if t.trigger_to_kill is not None], seed
                ),
                "accuracy": summarize_values(
                    [t.accuracy for t in trials if t.accuracy is not None], seed
                ),
            }
        return report

    def holdout_split(
        self, fraction: float = 0.25, seed: int = 1234
    ) -> tuple[TTKDataset, TTKDataset]:
        """Split by *tester* (or session) so calibration keeps a real holdout.

        Splitting by trial would let the same person's habits appear on
        both sides, which is the TTK equivalent of the BC episode-leakage
        problem.
        """
        import hashlib

        keys = sorted({trial.tester or trial.session_id or trial.trial_id for trial in self.trials})
        if len(keys) < 2:
            raise ValueError(
                f"a holdout split needs at least two testers/sessions; this dataset has {len(keys)}"
            )
        ranked = sorted(
            keys,
            key=lambda key: hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).hexdigest(),
        )
        holdout_count = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
        holdout_keys = set(ranked[:holdout_count])

        def key_of(trial: TTKTrial) -> str:
            return trial.tester or trial.session_id or trial.trial_id

        calibration = [trial for trial in self.trials if key_of(trial) not in holdout_keys]
        holdout = [trial for trial in self.trials if key_of(trial) in holdout_keys]
        return (
            TTKDataset(calibration, dict(self.metadata)),
            TTKDataset(holdout, dict(self.metadata)),
        )


def compare_with_simulator(
    dataset: TTKDataset,
    profiles: dict[str, Any] | None = None,
    seed: int = 1234,
) -> dict[str, Any]:
    """Measured human TTK vs the simulator's analytic TTK, per weapon.

    The simulator side comes from :mod:`sandboxai.weapons`, which parses
    ``WeaponState.PROFILE_DEFINITIONS`` rather than duplicating it, so this
    comparison cannot drift from the engine's actual numbers.

    A large positive ``human_minus_ideal`` is expected — ideal TTK assumes
    perfect tracking from t=0 — and its *size* is the calibration signal
    (acquisition, reaction, misses, movement). Weapons absent from the
    simulator are reported, not silently skipped.
    """
    from .weapons import load_profiles

    table = profiles if profiles is not None else load_profiles()
    grouped: dict[str, list[TTKTrial]] = {}
    for trial in dataset.kills():
        grouped.setdefault(trial.weapon_profile, []).append(trial)
    rows: dict[str, Any] = {}
    unknown: list[str] = []
    for weapon, trials in sorted(grouped.items()):
        profile = table.get(weapon)
        measured = summarize_values(
            [trial.trigger_to_kill for trial in trials if trial.trigger_to_kill is not None], seed
        )
        if profile is None:
            unknown.append(weapon)
            rows[weapon] = {"measured_trigger_to_kill": measured, "simulator": None}
            continue
        distances = [trial.distance_m for trial in trials]
        mean_distance = statistics.fmean(distances) if distances else 0.0
        health = statistics.fmean([trial.target_health for trial in trials])
        ideal = profile.ttk(mean_distance, health)
        handling = profile.effective_ttk(mean_distance, health)
        rows[weapon] = {
            "trials": len(trials),
            "mean_distance_m": mean_distance,
            "target_health": health,
            "measured_trigger_to_kill": measured,
            "simulator": {
                "ideal_ttk": ideal,
                "handling_aware_ttk": handling,
            },
            "human_minus_ideal": (
                measured["median"] - ideal if measured.get("n") and math.isfinite(ideal) else None
            ),
            "human_minus_handling_aware": (
                measured["median"] - handling
                if measured.get("n") and math.isfinite(handling)
                else None
            ),
        }
    return {
        "format": "sandboxai.ttk_simulator_comparison/v1",
        "weapons": rows,
        "unknown_weapon_profiles": unknown,
        "note": (
            "Ideal/handling-aware simulator TTK assumes a centred, tracked target; "
            "the human median additionally contains acquisition, misses and movement. "
            "Use the difference to calibrate reaction/spread distributions, not to "
            "claim the simulator is wrong."
        ),
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Compact human-readable report."""
    lines = ["TTK trials"]
    censoring = summary.get("censoring", {})
    lines.append(
        f"  trials={censoring.get('trials', 0)} "
        f"censored={censoring.get('censored', 0)} "
        f"({censoring.get('censored_fraction', 0.0):.0%})"
    )
    for name in (
        "reaction_time",
        "trigger_to_kill",
        "damage_to_kill",
        "encounter_time",
        "accuracy",
    ):
        values = summary.get(name, {})
        if not values.get("n"):
            lines.append(f"  {name:<18} n=0")
            continue
        lines.append(
            f"  {name:<18} n={values['n']:<4} median={values['median']:.3f} "
            f"iqm={values['iqm']:.3f} ci95=[{values['bootstrap_ci95_low']:.3f}, "
            f"{values['bootstrap_ci95_high']:.3f}]"
        )
    return "\n".join(lines)


def trials_from_iterable(values: Iterable[dict[str, Any]]) -> list[TTKTrial]:
    """Validates and converts raw dictionaries (for programmatic callers)."""
    problems: list[str] = []
    trials: list[TTKTrial] = []
    for index, value in enumerate(values):
        issues = validate_trial(value, index)
        if issues:
            problems.extend(issues)
            continue
        trials.append(TTKTrial.from_dict(value))
    if problems:
        raise ValueError("invalid TTK trials:\n  " + "\n  ".join(problems))
    return trials

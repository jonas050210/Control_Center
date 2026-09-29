"""Human demonstration storage, validation, statistics and splitting.

The behavior-cloning data pipeline that consumes this module is
    demonstrations -> validation -> statistics -> episode-aware split ->
    BC training -> evaluation -> checkpoint selection
and every stage here exists to make the *first three* steps loud rather
than silent. In particular:

* ``validate()`` rejects malformed, non-finite, wrong-width or
  non-contract observations and undecodable actions before any tensor is
  built.
* ``episode_groups()`` recovers the episode structure the recorders write
  (``episode_id`` + ``environment_id``), which is what makes a
  leakage-free split possible.
* ``split()`` defaults to splitting **by episode**, not by transition.
  Adjacent transitions inside one episode are almost identical, so a
  transition-level split leaks the validation distribution into training
  and reports a validation loss that is mostly memorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

# Kept as a module-level name for backwards compatibility with existing
# imports; the single source of truth is contract.ACTION_NVEC.
ACTION_NVECS = ACTION_NVEC
SCHEMA = "sandboxai.demonstrations"


# Mirrors Action.Discrete in scripts/core/action.gd. Index 10 (JUMP) was
# added with contract v2.
_DISCRETE_MAPPING: dict[int, list[int]] = {
    0: [1, 1, 1, 1, 0, 0],
    1: [2, 1, 1, 1, 0, 0],
    2: [0, 1, 1, 1, 0, 0],
    3: [1, 0, 1, 1, 0, 0],
    4: [1, 2, 1, 1, 0, 0],
    5: [1, 1, 0, 1, 0, 0],
    6: [1, 1, 2, 1, 0, 0],
    7: [1, 1, 1, 2, 0, 0],
    8: [1, 1, 1, 0, 0, 0],
    9: [1, 1, 1, 1, 1, 0],
    10: [1, 1, 1, 1, 0, 1],
}


def _file_fingerprint(path: Path) -> str:
    """Content hash of a dataset file (provenance, not security)."""
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return f"blake2b:{digest.hexdigest()}"


def discrete_to_multidiscrete(value: int) -> list[int]:
    if value not in _DISCRETE_MAPPING:
        maximum = max(_DISCRETE_MAPPING)
        raise ValueError(f"discrete action must be in [0, {maximum}], got {value}")
    return list(_DISCRETE_MAPPING[value])


def action_to_multidiscrete(action: Any) -> list[int]:
    """Normalise a logged Godot Action or a PPO action to nvec values.

    Accepted inputs, all of which are still produced somewhere in the repo:

      * an int discrete action (0-10),
      * the canonical Godot log array: 7 values (contract v1) or 8 values
        (contract v2, with ``jump`` at index 5), using -1/0/+1 axes,
      * a PPO MultiDiscrete action: 5 values (v1) or 6 values (v2), using
        0/1/2 axes.

    The result always has ``len(ACTION_NVEC)`` entries; a v1 input is padded
    with ``jump = 0`` so old recorded demonstrations remain trainable
    against the extended action head instead of silently mis-shaping the
    batch.
    """
    if isinstance(action, (int, float)):
        return discrete_to_multidiscrete(int(action))
    values = list(action)
    if len(values) < 5:
        raise ValueError(f"action needs at least 5 fields, got {len(values)}")
    canonical_log = len(values) >= 7
    first = [int(values[index]) for index in range(4)]
    if canonical_log or any(value < 0 for value in first):
        if not all(-1 <= value <= 1 for value in first):
            raise ValueError(f"invalid canonical action axis values: {first}")
        encoded = [value + 1 for value in first]
    elif all(0 <= value <= 2 for value in first):
        encoded = first
    else:
        raise ValueError(f"invalid action axis values: {first}")
    shoot = int(bool(values[4]))
    if shoot not in (0, 1):
        raise ValueError("shoot action must be binary")
    # Jump lives at index 5 in both the v2 log array and the v2 PPO action;
    # a 7-value v1 log has continuous look deltas there instead, so it is
    # only read when the payload is long enough to be v2.
    jump = 0
    if canonical_log:
        if len(values) >= 8:
            jump = int(bool(values[5]))
    elif len(values) >= 6:
        jump = int(bool(values[5]))
    return encoded + [shoot, jump]


## Split strategies. "episode" keeps whole episodes together (no
## leakage); "transition" is the historical shuffle and is only correct
## for datasets that genuinely have no episode structure.
SPLIT_STRATEGIES: tuple[str, ...] = ("auto", "episode", "transition")


@dataclass
class SplitReport:
    """Exactly how a train/validation split was produced.

    Written next to every BC run so a reported validation number can be
    audited later: which strategy ran, how many episode groups each side
    holds, and the explicit statement that the two sides share none.
    """

    strategy: str
    requested_fraction: float
    train_transitions: int
    validation_transitions: int
    train_groups: int
    validation_groups: int
    shared_groups: int
    seed: int
    degraded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "requested_validation_fraction": self.requested_fraction,
            "train_transitions": self.train_transitions,
            "validation_transitions": self.validation_transitions,
            "train_groups": self.train_groups,
            "validation_groups": self.validation_groups,
            "shared_groups": self.shared_groups,
            "leakage_free": self.shared_groups == 0,
            "seed": self.seed,
            "degraded_reason": self.degraded_reason,
        }


@dataclass
class DemonstrationDataset:
    transitions: list[dict[str, Any]]
    metadata: dict[str, Any]
    ## Populated by load(); empty for in-memory datasets. Recorded in BC
    ## run artifacts so a checkpoint can be traced back to exact bytes.
    fingerprint: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "DemonstrationDataset":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        metadata: dict[str, Any] = {}
        transitions: list[dict[str, Any]] = []
        if source.suffix.lower() == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                metadata = dict(value.get("metadata", {}))
                transitions = list(value.get("transitions", []))
            else:
                transitions = list(value)
        else:
            with source.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not transitions and isinstance(value, dict) and value.get("schema") == SCHEMA:
                        metadata = value
                        continue
                    if not isinstance(value, dict):
                        raise ValueError(f"line {line_number} is not a transition object")
                    transitions.append(value)
        dataset = cls(transitions=transitions, metadata=metadata)
        dataset.fingerprint = _file_fingerprint(source)
        dataset.validate()
        return dataset

    def validate(self, require_contract_width: bool = False) -> None:
        """Rejects anything BC must never silently train on.

        ``require_contract_width`` additionally pins the observation width
        to the live contract (``OBSERVATION_FIELD_COUNT``). It is off by
        default so older/narrower research datasets can still be inspected,
        and on for BC training, where a width mismatch means the
        checkpoint could never be loaded into a policy anyway.
        """
        if self.metadata and self.metadata.get("schema", SCHEMA) not in (SCHEMA, ""):
            raise ValueError(f"unsupported demonstration schema: {self.metadata.get('schema')}")
        if not self.transitions:
            raise ValueError("demonstration dataset contains no transitions")

        expected_dim: int | None = None
        for index, transition in enumerate(self.transitions):
            for field in ("observation", "action", "next_observation", "reward", "done"):
                if field not in transition:
                    raise ValueError(f"transition {index} is missing field {field!r}")
            obs = transition["observation"]
            next_obs = transition["next_observation"]
            if not obs or not next_obs:
                raise ValueError(f"transition {index} has an empty observation")
            if expected_dim is None:
                expected_dim = len(obs)
            if len(obs) != expected_dim or len(next_obs) != expected_dim:
                raise ValueError(f"transition {index} observation dimension mismatch ({len(obs)} vs {expected_dim})")

            # Check for non-finite values
            for val in obs:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"transition {index} observation contains invalid number: {val}")
            for val in next_obs:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"transition {index} next_observation contains invalid number: {val}")

            rew = transition["reward"]
            if not isinstance(rew, (int, float)) or math.isnan(rew) or math.isinf(rew):
                raise ValueError(f"transition {index} reward contains invalid number: {rew}")

            encoded = action_to_multidiscrete(transition["action"])
            for component, (value, size) in enumerate(zip(encoded, ACTION_NVEC)):
                if not 0 <= value < size:
                    raise ValueError(
                        f"transition {index} action component {component} is {value}, "
                        f"outside the contract range [0, {size - 1}]"
                    )
        if require_contract_width and expected_dim != OBSERVATION_FIELD_COUNT:
            raise ValueError(
                f"dataset observations are {expected_dim} floats but the observation "
                f"contract is {OBSERVATION_FIELD_COUNT} floats "
                "(python/sandboxai/contract.py); this dataset cannot train a policy "
                "usable in the current simulator"
            )

    # -- episode structure ---------------------------------------------

    @staticmethod
    def group_key(transition: dict[str, Any], fallback: int) -> str:
        """Stable identity of the episode a transition belongs to.

        Both recorders write ``episode_id`` and ``environment_id``; a
        dataset merged from several sources may also carry ``source`` or
        ``run_id``, which are included so two runs that both start at
        episode 1 are not fused into one group.
        """
        if "episode_key" in transition:
            return str(transition["episode_key"])
        episode_id = transition.get("episode_id")
        if episode_id is None:
            return f"__ungrouped__{fallback}"
        parts = [
            str(transition.get("run_id", transition.get("source", ""))),
            str(transition.get("environment_id", transition.get("env_index", 0))),
            str(episode_id),
        ]
        return "|".join(parts)

    def episode_groups(self) -> dict[str, list[int]]:
        """Episode key -> transition indices, in file order."""
        groups: dict[str, list[int]] = {}
        for index, transition in enumerate(self.transitions):
            groups.setdefault(self.group_key(transition, index), []).append(index)
        return groups

    def has_episode_structure(self) -> bool:
        return any("episode_id" in item or "episode_key" in item for item in self.transitions)

    def episode_boundary_problems(self) -> list[str]:
        """Structural complaints about episode boundaries.

        Reported rather than raised: a truncated recording (no terminal
        transition) is still usable BC data, but training on it while
        believing every episode ended naturally is how silent dataset rot
        starts.
        """
        problems: list[str] = []
        if not self.has_episode_structure():
            problems.append("no episode_id/episode_key on any transition; episodes cannot be separated")
            return problems
        for key, indices in self.episode_groups().items():
            dones = [bool(self.transitions[index].get("done", False)) for index in indices]
            if any(dones[:-1]):
                problems.append(f"episode {key}: done=true before the last transition")
            if not dones[-1]:
                problems.append(f"episode {key}: truncated (last transition is not terminal)")
            steps = [
                self.transitions[index].get("step")
                for index in indices
                if isinstance(self.transitions[index].get("step"), int)
            ]
            if steps and steps != sorted(steps):
                problems.append(f"episode {key}: step numbers are not monotonic")
        return problems

    def duplicate_report(self) -> dict[str, Any]:
        """Exact-duplicate transitions (observation + action + next).

        Duplicates inflate apparent dataset size, bias the loss toward the
        repeated state and — when they straddle a split — leak. Frame
        repeats from a paused recording are the usual source.
        """
        seen: dict[str, list[int]] = {}
        for index, transition in enumerate(self.transitions):
            digest = hashlib.blake2b(
                json.dumps(
                    [
                        [round(float(value), 6) for value in transition["observation"]],
                        action_to_multidiscrete(transition["action"]),
                        [round(float(value), 6) for value in transition["next_observation"]],
                    ],
                    separators=(",", ":"),
                ).encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            seen.setdefault(digest, []).append(index)
        repeated = {digest: rows for digest, rows in seen.items() if len(rows) > 1}
        duplicate_transitions = sum(len(rows) - 1 for rows in repeated.values())
        cross_group = 0
        for rows in repeated.values():
            keys = {self.group_key(self.transitions[index], index) for index in rows}
            if len(keys) > 1:
                cross_group += 1
        return {
            "unique_transitions": len(seen),
            "duplicate_transitions": duplicate_transitions,
            "duplicate_fraction": duplicate_transitions / max(len(self.transitions), 1),
            "duplicated_groups": len(repeated),
            "cross_episode_duplicate_groups": cross_group,
        }

    def statistics(self) -> dict[str, Any]:
        """Dataset-level statistics used for review and BC provenance."""
        groups = self.episode_groups()
        lengths = sorted(len(indices) for indices in groups.values())
        encoded = [action_to_multidiscrete(item["action"]) for item in self.transitions]
        action_histograms: list[dict[int, int]] = []
        for component, size in enumerate(ACTION_NVEC):
            histogram = {value: 0 for value in range(size)}
            for row in encoded:
                histogram[row[component]] += 1
            action_histograms.append(histogram)
        rewards = [float(item["reward"]) for item in self.transitions]
        observation_dim = len(self.transitions[0]["observation"]) if self.transitions else 0
        out_of_range = 0
        minimum = float("inf")
        maximum = float("-inf")
        for item in self.transitions:
            for value in item["observation"]:
                value = float(value)
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                if value < -1.0 or value > 1.0:
                    out_of_range += 1
        total_values = max(len(self.transitions) * max(observation_dim, 1), 1)
        return {
            "transitions": len(self.transitions),
            "observation_dim": observation_dim,
            "contract_observation_dim": OBSERVATION_FIELD_COUNT,
            "matches_contract": observation_dim == OBSERVATION_FIELD_COUNT,
            "episodes": len(groups),
            "has_episode_structure": self.has_episode_structure(),
            "episode_length_min": lengths[0] if lengths else 0,
            "episode_length_median": lengths[len(lengths) // 2] if lengths else 0,
            "episode_length_max": lengths[-1] if lengths else 0,
            "terminal_transitions": sum(bool(item.get("done")) for item in self.transitions),
            "reward_sum": sum(rewards),
            "reward_mean": sum(rewards) / max(len(rewards), 1),
            "reward_min": min(rewards) if rewards else 0.0,
            "reward_max": max(rewards) if rewards else 0.0,
            "action_histograms": [
                {str(key): value for key, value in histogram.items()}
                for histogram in action_histograms
            ],
            # A component that never varies cannot be learned and usually
            # means a broken binding (e.g. the recorder has no jump key).
            "constant_action_components": [
                component
                for component, histogram in enumerate(action_histograms)
                if sum(1 for value in histogram.values() if value > 0) <= 1
            ],
            "observation_min": minimum if self.transitions else 0.0,
            "observation_max": maximum if self.transitions else 0.0,
            "observation_values_outside_contract_range": out_of_range,
            "observation_out_of_range_fraction": out_of_range / total_values,
            "duplicates": self.duplicate_report(),
            "episode_boundary_problems": self.episode_boundary_problems(),
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": SCHEMA,
            "schema_version": 1,
            "observation_dim": len(self.transitions[0]["observation"]) if self.transitions else OBSERVATION_FIELD_COUNT,
            # Matches the Godot recorder: the contract-v2 log array with
            # `jump` at index 5 and the look deltas at 6/7.
            "action_encoding": "[move, strafe, yaw, pitch, shoot, jump, look_delta_x, look_delta_y]",
            "created_unix": time.time(),
            **self.metadata,
        }
        with destination.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, separators=(",", ":")) + "\n")
            for transition in self.transitions:
                stream.write(json.dumps(transition, separators=(",", ":")) + "\n")
        return destination

    def arrays(self):
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("numpy is required to turn demonstrations into training arrays") from exc
        observations = np.asarray([item["observation"] for item in self.transitions], dtype=np.float32)
        next_observations = np.asarray([item["next_observation"] for item in self.transitions], dtype=np.float32)
        actions = np.asarray([action_to_multidiscrete(item["action"]) for item in self.transitions], dtype=np.int64)
        rewards = np.asarray([float(item["reward"]) for item in self.transitions], dtype=np.float32)
        dones = np.asarray([bool(item["done"]) for item in self.transitions], dtype=np.bool_)
        if observations.ndim != 2 or observations.shape[1] != next_observations.shape[1]:
            raise ValueError("observations must be a rectangular 2-D array with matching next_observations")
        return observations, actions, next_observations, rewards, dones

    def split(self, validation_fraction: float, seed: int = 1234, strategy: str = "auto"):
        """Train/validation split; returns ``(train, validation)``.

        ``strategy``:

        * ``"auto"`` (default) — split by episode when the dataset has at
          least two episode groups, otherwise fall back to the transition
          shuffle and record why. This is the leakage-free default for
          every real recording while keeping group-less research datasets
          splittable.
        * ``"episode"`` — always split by episode; raises when the dataset
          has fewer than two groups instead of silently leaking.
        * ``"transition"`` — the historical shuffle. Correct only when the
          rows are genuinely independent.

        Use :meth:`split_with_report` when the provenance matters (BC
        training does).
        """
        result = self.split_with_report(validation_fraction, seed, strategy)
        return result[0], result[1]

    def split_with_report(
        self,
        validation_fraction: float,
        seed: int = 1234,
        strategy: str = "auto",
    ) -> tuple["DemonstrationDataset", "DemonstrationDataset", SplitReport]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split strategy must be one of {SPLIT_STRATEGIES}")
        count = len(self.transitions)
        if count < 2:
            raise ValueError("at least two transitions are needed for a train/validation split")
        groups = self.episode_groups()
        degraded_reason = ""
        effective = strategy
        if strategy == "auto":
            if len(groups) >= 2:
                effective = "episode"
            else:
                effective = "transition"
                degraded_reason = (
                    "dataset exposes %d episode group(s); a leakage-free episode split "
                    "needs at least 2, so the transition shuffle was used. Record "
                    "episode_id/environment_id to remove this fallback." % len(groups)
                )
        if effective == "episode" and len(groups) < 2:
            raise ValueError(
                "episode split requires at least two episode groups; this dataset has "
                f"{len(groups)} (record episode_id/environment_id, or pass "
                "strategy='transition' and accept the leakage)"
            )

        if effective == "episode":
            train_indices, validation_indices = self._episode_split_indices(
                groups, validation_fraction, seed
            )
        else:
            train_indices, validation_indices = self._transition_split_indices(
                count, validation_fraction, seed
            )

        train = [self.transitions[index] for index in train_indices]
        validation = [self.transitions[index] for index in validation_indices]
        train_groups = {self.group_key(self.transitions[i], i) for i in train_indices}
        validation_groups = {self.group_key(self.transitions[i], i) for i in validation_indices}
        report = SplitReport(
            strategy=effective,
            requested_fraction=float(validation_fraction),
            train_transitions=len(train),
            validation_transitions=len(validation),
            train_groups=len(train_groups),
            validation_groups=len(validation_groups),
            shared_groups=len(train_groups & validation_groups),
            seed=int(seed),
            degraded_reason=degraded_reason,
        )
        if effective == "episode" and report.shared_groups:  # pragma: no cover - guard
            raise AssertionError("episode split produced overlapping groups")
        return (
            DemonstrationDataset(train, dict(self.metadata), self.fingerprint),
            DemonstrationDataset(validation, dict(self.metadata), self.fingerprint),
            report,
        )

    def _episode_split_indices(
        self, groups: dict[str, list[int]], validation_fraction: float, seed: int
    ) -> tuple[list[int], list[int]]:
        """Deterministic, order-independent group assignment.

        Groups are ranked by a seeded hash of their *key*, not by their
        position in the file, so appending new episodes never reshuffles
        the existing train/validation assignment and two datasets sharing
        episodes keep them on the same side. Whole groups move together,
        which is the entire point.
        """
        keys = sorted(groups)
        ranked = sorted(
            keys,
            key=lambda key: hashlib.blake2b(
                f"{seed}:{key}".encode("utf-8"), digest_size=8
            ).hexdigest(),
        )
        total = len(self.transitions)
        target = max(1, int(round(total * validation_fraction)))
        validation_keys: list[str] = []
        held = 0
        for key in ranked:
            if held >= target and validation_keys:
                break
            # Never hand every group to validation: training needs data.
            if len(validation_keys) == len(keys) - 1:
                break
            validation_keys.append(key)
            held += len(groups[key])
        validation_set = set(validation_keys)
        validation_indices = [
            index for key in sorted(validation_set) for index in groups[key]
        ]
        train_indices = [
            index for key in sorted(set(keys) - validation_set) for index in groups[key]
        ]
        return sorted(train_indices), sorted(validation_indices)

    @staticmethod
    def _transition_split_indices(
        count: int, validation_fraction: float, seed: int
    ) -> tuple[list[int], list[int]]:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("numpy is required to split demonstrations") from exc
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        validation_count = max(1, int(round(count * validation_fraction)))
        validation_indices = sorted(int(value) for value in order[:validation_count])
        validation_set = set(validation_indices)
        train_indices = [index for index in range(count) if index not in validation_set]
        return train_indices, validation_indices

    def summary(self) -> dict[str, Any]:
        observation_dim = len(self.transitions[0]["observation"]) if self.transitions else 0
        try:
            _observations, actions, _next, rewards, dones = self.arrays()
            action_dim = int(actions.shape[1])
            reward_sum = float(rewards.sum())
            terminal_transitions = int(dones.sum())
        except (RuntimeError, ImportError):
            encoded_actions = [action_to_multidiscrete(item["action"]) for item in self.transitions]
            action_dim = len(encoded_actions[0]) if encoded_actions else 0
            reward_sum = sum(float(item["reward"]) for item in self.transitions)
            terminal_transitions = sum(bool(item["done"]) for item in self.transitions)
        return {
            "transitions": len(self.transitions),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            # Episode identity is (run, environment, episode_id): two
            # environments both on episode 3 are two episodes, which the
            # old `episode_id`-only count silently merged.
            "episodes": len(self.episode_groups()),
            "reward_sum": reward_sum,
            "terminal_transitions": terminal_transitions,
            "metadata": self.metadata,
        }


class DemonstrationRecorder:
    """Portable Python recorder for tests and non-Godot action sources.

    The graphical Godot recorder uses the same JSONL schema. This class is
    useful when a policy wrapper already has observation/action transitions.
    """

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata = {"schema": SCHEMA, **(metadata or {})}
        self.transitions: list[dict[str, Any]] = []
        self.recording = False

    def start(self) -> None:
        self.recording = True

    def stop(self) -> None:
        self.recording = False

    def append(self, observation, action, next_observation, reward, done, **info) -> None:
        if not self.recording:
            return
        self.transitions.append(
            {
                "observation": list(observation),
                "action": list(action) if not isinstance(action, (int, float)) else int(action),
                "next_observation": list(next_observation),
                "reward": float(reward),
                "done": bool(done),
                "timestamp": time.time(),
                **info,
            }
        )

    def save(self, path: str | Path) -> Path:
        return DemonstrationDataset(self.transitions, self.metadata).save(path)

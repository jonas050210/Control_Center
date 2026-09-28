"""Human demonstration storage, validation and inspection."""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class DemonstrationDataset:
    transitions: list[dict[str, Any]]
    metadata: dict[str, Any]

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
        dataset.validate()
        return dataset

    def validate(self) -> None:
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

            action_to_multidiscrete(transition["action"])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": SCHEMA,
            "schema_version": 1,
            "observation_dim": len(self.transitions[0]["observation"]) if self.transitions else OBSERVATION_FIELD_COUNT,
            "action_encoding": "[move, strafe, yaw, pitch, shoot, look_delta_x, look_delta_y]",
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

    def split(self, validation_fraction: float, seed: int = 1234):
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("numpy is required to split demonstrations") from exc
        count = len(self.transitions)
        if count < 2:
            raise ValueError("at least two transitions are needed for a train/validation split")
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        validation_count = max(1, int(round(count * validation_fraction)))
        validation_indices = set(int(value) for value in order[:validation_count])
        train = [item for index, item in enumerate(self.transitions) if index not in validation_indices]
        validation = [item for index, item in enumerate(self.transitions) if index in validation_indices]
        return DemonstrationDataset(train, dict(self.metadata)), DemonstrationDataset(validation, dict(self.metadata))

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
            "episodes": len({item.get("episode_id", index) for index, item in enumerate(self.transitions)}),
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

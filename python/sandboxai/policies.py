"""Multi-policy ("multi-brain") architecture: identity, slots and matchups.

Phase 5. The single most important invariant in this file:

    A POLICY IS A SET OF WEIGHTS. Parallel environments are not brains.

Sixty-four Godot environments feeding one PPO update are one policy. Two
policies are two checkpoints, two parameter tensors, two training
histories. ``assert_independent_weights`` exists to make that statement
testable instead of aspirational, and the roster refuses to register two
policies that point at the same checkpoint file.

The second invariant: **no hard-coded personalities.** There is no
"aggressive" or "defensive" flag anywhere. What differs between Brain A
and Brain B is their weights and the experience that produced them; any
behavioural difference is learned, not configured. The only non-learned
actor is :class:`ScriptedBaseline`, which is explicitly labelled a
baseline and is never trained, never mixed into a learned policy's action
stream, and never described as intelligent.

A *slot* is an agent position in an environment (slot 0 is the historical
single-agent seat). ``SlotAssignment`` maps slots to policies, so
A-vs-B, A-vs-B-vs-A and A-vs-B-vs-C are configuration rather than code.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from .contract import ACTION_NVEC, observation_value


class PolicyRole:
    """What a policy is doing in a particular match/run."""

    LEARNER: str = "learner"  # weights are being updated
    FROZEN: str = "frozen"  # a snapshot; must not change
    BASELINE: str = "baseline"  # scripted, never trained
    EXTERNAL: str = "external"  # a checkpoint from outside this run
    ALL: tuple[str, ...] = (LEARNER, FROZEN, BASELINE, EXTERNAL)


class PolicyError(RuntimeError):
    """A policy was used in a way that would corrupt an experiment."""


@dataclass
class PolicySpec:
    """Declarative identity of one brain.

    ``kind`` distinguishes a neural checkpoint from the scripted baseline;
    it is *not* a behaviour label. Two ``kind="checkpoint"`` policies are
    expected to behave differently, and the reason is their weights.
    """

    policy_id: str
    checkpoint: str = ""
    role: str = PolicyRole.FROZEN
    kind: str = "checkpoint"  # "checkpoint" | "scripted"
    seed: int = 0
    device: str = "cpu"
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if self.role not in PolicyRole.ALL:
            raise ValueError(f"unknown policy role: {self.role!r} (known: {PolicyRole.ALL})")
        if self.kind not in ("checkpoint", "scripted"):
            raise ValueError(f"unknown policy kind: {self.kind!r}")
        if self.kind == "checkpoint" and self.role != PolicyRole.BASELINE and not self.checkpoint:
            raise ValueError(
                f"policy {self.policy_id!r} is a checkpoint policy without a checkpoint"
            )

    @property
    def trainable(self) -> bool:
        return self.role == PolicyRole.LEARNER and self.kind == "checkpoint"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicySpec:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)


class ScriptedBaseline:
    """A deterministic, non-learned reference opponent.

    It is intentionally simple and intentionally *not* good: its job is to
    give "did the policy learn anything at all?" a floor, and to give the
    league a fixed rung that never drifts between runs. It reads the same
    observation vector the policy does — by field name, never by index —
    so it cannot cheat with information a learned policy would not have.

    It is not a personality. It is a control condition.
    """

    policy_id: str = "scripted_baseline"

    def __init__(self, seed: int = 0, aim_tolerance: float = 0.05) -> None:
        self.seed = int(seed)
        self.aim_tolerance = float(aim_tolerance)
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def predict(self, observation: Sequence[float], deterministic: bool = True):
        """Returns ``(action, None)``, matching the SB3 predict signature."""
        # Batched input (a vector env hands over a 2-D array): act per row.
        first = observation[0] if len(observation) and hasattr(observation[0], "__len__") else None
        if first is not None:
            rows = cast("Sequence[Sequence[float]]", observation)
            return [self.predict(row, deterministic)[0] for row in rows], None

        visible = observation_value(observation, "primary_enemy_visible") > 0.5
        bearing = observation_value(observation, "primary_enemy_bearing_norm")
        distance = observation_value(observation, "primary_enemy_distance_norm")
        weapon_ready = observation_value(observation, "weapon_ready") > 0.5
        clearance = observation_value(observation, "agent_forward_clearance_norm")
        sound_bearing = observation_value(observation, "last_sound_bearing_norm")
        sound_loudness = observation_value(observation, "last_sound_loudness")

        # Turn toward the best available information: a live sighting
        # first, otherwise the loudest thing it heard.
        reference = bearing if visible else (sound_bearing if sound_loudness > 0.05 else 0.0)
        if reference > self.aim_tolerance:
            yaw = 2
        elif reference < -self.aim_tolerance:
            yaw = 0
        else:
            yaw = 1

        move = 1
        if visible and distance > 0.2:
            move = 2  # close the distance
        elif not visible and clearance > 0.2:
            move = 2  # otherwise wander forward while there is room
        strafe = 1
        shoot = 1 if (visible and weapon_ready and abs(bearing) <= self.aim_tolerance) else 0
        jump = 0
        if not deterministic:
            strafe = self._rng.choice((0, 1, 2))
        action = [move, strafe, yaw, 1, shoot, jump]
        for index, value in enumerate(action):
            action[index] = max(0, min(int(value), ACTION_NVEC[index] - 1))
        return action, None


class PolicyHandle:
    """A spec plus (lazily) its loaded model.

    Loading is deferred so a roster can be built, serialized and inspected
    without PyTorch installed — the whole registry layer stays testable in
    an environment that cannot install a multi-hundred-megabyte wheel.
    """

    def __init__(self, spec: PolicySpec, model: Any = None) -> None:
        self.spec = spec
        self.model = model

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, device: str | None = None, loader: Any = None) -> Any:
        """Loads the model. ``loader`` is injectable for tests.

        Every call constructs its own model object: two handles must never
        end up sharing one parameter set, which is exactly how a "self-play"
        run silently degenerates into a policy fighting itself.
        """
        if self.spec.kind == "scripted":
            self.model = ScriptedBaseline(seed=self.spec.seed)
            return self.model
        if not self.spec.checkpoint:
            raise PolicyError(f"policy {self.policy_id!r} has no checkpoint to load")
        target_device = device or self.spec.device
        if loader is not None:
            self.model = loader(self.spec.checkpoint, target_device)
            return self.model
        try:
            from stable_baselines3 import PPO  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise PolicyError(
                "loading a checkpoint policy requires stable-baselines3; install the training extra"
            ) from exc
        self.model = PPO.load(Path(self.spec.checkpoint), device=target_device)
        return self.model

    def predict(self, observation, deterministic: bool = True):
        if self.model is None:
            raise PolicyError(f"policy {self.policy_id!r} is not loaded")
        if self.spec.role in (PolicyRole.FROZEN, PolicyRole.BASELINE, PolicyRole.EXTERNAL):
            # Frozen means frozen: prediction is always deterministic and
            # never updates anything.
            deterministic = True
        return self.model.predict(observation, deterministic=deterministic)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PolicyHandle({self.policy_id!r}, role={self.spec.role}, loaded={self.loaded})"


class PolicyRoster:
    """The set of brains available to a run.

    Enforces the two things that make multi-policy results trustworthy:
    unique ids, and no two *distinct* policies sharing a checkpoint file.
    """

    def __init__(self) -> None:
        self._handles: dict[str, PolicyHandle] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._handles)

    def __contains__(self, policy_id: object) -> bool:
        return policy_id in self._handles

    def __iter__(self):
        return (self._handles[policy_id] for policy_id in self._order)

    def add(self, spec: PolicySpec) -> PolicyHandle:
        if spec.policy_id in self._handles:
            raise PolicyError(f"policy id already registered: {spec.policy_id}")
        if spec.checkpoint:
            resolved = str(Path(spec.checkpoint))
            for existing in self._handles.values():
                if existing.spec.checkpoint and str(Path(existing.spec.checkpoint)) == resolved:
                    raise PolicyError(
                        f"policy {spec.policy_id!r} and {existing.policy_id!r} point at the same "
                        f"checkpoint ({resolved}); independent brains need independent weights"
                    )
        handle = PolicyHandle(spec)
        self._handles[spec.policy_id] = handle
        self._order.append(spec.policy_id)
        return handle

    def add_baseline(self, policy_id: str = "scripted_baseline", seed: int = 0) -> PolicyHandle:
        return self.add(
            PolicySpec(policy_id=policy_id, role=PolicyRole.BASELINE, kind="scripted", seed=seed)
        )

    def get(self, policy_id: str) -> PolicyHandle:
        if policy_id not in self._handles:
            raise KeyError(f"unknown policy id: {policy_id}")
        return self._handles[policy_id]

    def ids(self, role: str | None = None) -> list[str]:
        return [
            policy_id
            for policy_id in self._order
            if role is None or self._handles[policy_id].spec.role == role
        ]

    def specs(self) -> list[PolicySpec]:
        return [self._handles[policy_id].spec for policy_id in self._order]

    def load_all(self, device: str = "cpu", loader: Any = None) -> None:
        for handle in self:
            handle.load(device, loader)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"policies": [spec.to_dict() for spec in self.specs()]}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> PolicyRoster:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        roster = cls()
        for entry in payload.get("policies", []):
            roster.add(PolicySpec.from_dict(entry))
        return roster


@dataclass
class SlotAssignment:
    """Which policy controls which agent slot.

    ``slots`` is ordered: index 0 is agent slot 0. The same policy id may
    appear more than once (A vs B vs A) — that is one brain playing two
    seats, which is a legitimate and *different* experiment from two
    brains, and the naming makes which one you ran unambiguous.
    """

    slots: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slots)

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("a slot assignment needs at least one slot")

    @property
    def distinct_policies(self) -> list[str]:
        seen: list[str] = []
        for policy_id in self.slots:
            if policy_id not in seen:
                seen.append(policy_id)
        return seen

    @property
    def label(self) -> str:
        return " vs ".join(self.slots)

    def policy_for(self, slot: int) -> str:
        if not 0 <= slot < len(self.slots):
            raise IndexError(f"slot {slot} out of range (0..{len(self.slots) - 1})")
        return self.slots[slot]

    def slots_for(self, policy_id: str) -> list[int]:
        return [index for index, value in enumerate(self.slots) if value == policy_id]

    def validate(self, roster: PolicyRoster) -> None:
        for policy_id in self.distinct_policies:
            if policy_id not in roster:
                raise PolicyError(f"slot assignment references unknown policy {policy_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"slots": list(self.slots), "label": self.label}


def matchup(*policy_ids: str) -> SlotAssignment:
    """``matchup("brain_a", "brain_b")`` -> A in slot 0, B in slot 1."""
    return SlotAssignment(list(policy_ids))


def standard_matchups(learner: str, opponent: str, third: str = "") -> list[SlotAssignment]:
    """The matchup set Phase 5 calls for, built from policy ids.

    A-vs-B, A-vs-B-vs-A and (when a third brain exists) A-vs-B-vs-C.
    """
    matchups = [matchup(learner, opponent), matchup(learner, opponent, learner)]
    if third:
        matchups.append(matchup(learner, opponent, third))
    return matchups


def _model_parameters(model: Any) -> list[Any]:
    """Best-effort extraction of a model's parameter tensors.

    Supports SB3 (``model.policy.parameters()``), a bare torch module, and
    a duck-typed test double exposing ``parameters()``. Returns [] when the
    object has no parameters at all (the scripted baseline), which callers
    treat as "trivially independent".
    """
    for attribute in ("policy", None):
        target = getattr(model, attribute, None) if attribute else model
        if target is None:
            continue
        parameters = getattr(target, "parameters", None)
        if callable(parameters):
            try:
                return list(parameters())
            except TypeError:  # pragma: no cover - defensive
                continue
    return []


def assert_independent_weights(handles: Iterable[PolicyHandle]) -> dict[str, Any]:
    """Proves that the given loaded policies do not share parameters.

    Checks three things, in increasing strength:

    1. distinct policy ids;
    2. distinct checkpoint paths;
    3. no shared parameter object between any two models (identity, not
       value — two policies may *coincidentally* hold equal weights right
       after a snapshot, but they must never be the same tensor, because
       then training one silently trains the other).

    Returns a report; raises :class:`PolicyError` on a violation.
    """
    handle_list = list(handles)
    ids = [handle.policy_id for handle in handle_list]
    if len(set(ids)) != len(ids):
        raise PolicyError(f"duplicate policy ids in the comparison set: {ids}")

    checkpoints = [handle.spec.checkpoint for handle in handle_list if handle.spec.checkpoint]
    resolved = [str(Path(value)) for value in checkpoints]
    if len(set(resolved)) != len(resolved):
        raise PolicyError(f"two policies share a checkpoint file: {resolved}")

    parameter_ids: dict[int, str] = {}
    shared: list[tuple[str, str]] = []
    checked = 0
    for handle in handle_list:
        if handle.model is None:
            continue
        for parameter in _model_parameters(handle.model):
            checked += 1
            key = id(parameter)
            owner = parameter_ids.get(key)
            if owner is not None and owner != handle.policy_id:
                shared.append((owner, handle.policy_id))
            parameter_ids[key] = handle.policy_id
    if shared:
        raise PolicyError(
            "policies share parameter tensors (they are the same brain): "
            + ", ".join(f"{a} <-> {b}" for a, b in sorted(set(shared)))
        )
    return {
        "policies": ids,
        "checkpoints": resolved,
        "parameters_checked": checked,
        "independent": True,
    }


def weights_differ(model_a: Any, model_b: Any, tolerance: float = 1e-9) -> bool:
    """True when two models' parameter *values* differ somewhere.

    Complements ``assert_independent_weights``: identity proves they are
    separate objects, this proves they are separate *brains* in practice.
    A freshly taken snapshot legitimately returns False.
    """
    left = _model_parameters(model_a)
    right = _model_parameters(model_b)
    if len(left) != len(right):
        return True
    for first, second in zip(left, right):
        values_a = _flatten_parameter(first)
        values_b = _flatten_parameter(second)
        if len(values_a) != len(values_b):
            return True
        for value_a, value_b in zip(values_a, values_b):
            if not math.isclose(value_a, value_b, rel_tol=0.0, abs_tol=tolerance):
                return True
    return False


def _flatten_parameter(parameter: Any) -> list[float]:
    flatten = getattr(parameter, "flatten", None)
    if callable(flatten):
        try:
            flat = flatten()
            tolist = getattr(flat, "tolist", None)
            if callable(tolist):
                return [float(value) for value in _iter_flat(tolist())]
        except (AttributeError, TypeError, ValueError):
            # An array-like whose flatten()/tolist() does not behave; fall
            # through to the generic element walk below.
            pass
    return [float(value) for value in _iter_flat(parameter)]


def _iter_flat(value: Any) -> Iterable[float]:
    if isinstance(value, (int, float)):
        yield float(value)
        return
    try:
        iterator = iter(value)
    except TypeError:
        yield float(value)
        return
    for item in iterator:
        yield from _iter_flat(item)

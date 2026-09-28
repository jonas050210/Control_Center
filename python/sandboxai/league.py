"""Checkpoint registry, opponent sampling and a self-play league.

Phases 10 and 11. The distinction this module exists to enforce:

* **Parallel environments are experience collectors.** Sixty-four envs
  feeding one PPO update are still ONE brain.
* **A policy is a brain.** It has its own id, its own weights file and its
  own training history, and two policies never share parameters.

``CheckpointRegistry`` owns the second thing: every snapshot gets a stable
policy id and a record on disk, so "current vs frozen" and "A vs B" are
matters of registry lookup rather than of remembering which file was which.

``League`` adds opponent sampling and match bookkeeping on top: win/loss
matrices, per-policy records, deterministic tournaments, and an optional
internal Elo. The Elo is explicitly a **research signal**, not a
leaderboard: it is computed from matches this process ran, with a fixed
K-factor, and it is meaningless outside the run that produced it.

Everything is seeded. Two runs with the same seed sample the same
opponents and play the same tournament pairings.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

## Starting rating for a freshly registered policy.
DEFAULT_ELO: float = 1200.0
## Elo K-factor. Small, because these are noisy short matches.
ELO_K: float = 16.0


@dataclass
class PolicyRecord:
    """One independent brain.

    ``parent_id`` records which policy a snapshot was taken from, so a
    league can be read as a family tree instead of a bag of files.
    """

    policy_id: str
    checkpoint: str = ""
    step: int = 0
    parent_id: str = ""
    frozen: bool = True
    elo: float = DEFAULT_ELO
    matches: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches if self.matches else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyRecord":
        known = {key: payload[key] for key in payload if key in cls.__annotations__}
        return cls(**known)


class CheckpointRegistry:
    """Policy ids to checkpoints, persisted as one JSON file.

    Deliberately not a database and not pickled: a training run should be
    inspectable with ``cat``, and a corrupted registry should be fixable in
    a text editor.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: dict[str, PolicyRecord] = {}
        self._order: list[str] = []
        if self.path and self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, policy_id: object) -> bool:
        return policy_id in self._records

    def register(
        self,
        policy_id: str,
        checkpoint: str = "",
        step: int = 0,
        parent_id: str = "",
        frozen: bool = True,
        tags: Sequence[str] | None = None,
    ) -> PolicyRecord:
        if not policy_id:
            raise ValueError("policy_id must not be empty")
        if policy_id in self._records:
            raise ValueError(f"policy id already registered: {policy_id}")
        record = PolicyRecord(
            policy_id=policy_id,
            checkpoint=checkpoint,
            step=step,
            parent_id=parent_id,
            frozen=frozen,
            tags=list(tags or []),
        )
        self._records[policy_id] = record
        self._order.append(policy_id)
        return record

    def snapshot(
        self, parent_id: str, checkpoint: str, step: int, tags: Sequence[str] | None = None
    ) -> PolicyRecord:
        """Freezes the current state of a learning policy as a new opponent.

        The snapshot is a separate policy id with its own weights file: a
        frozen opponent must never be able to change underneath an
        evaluation, which is exactly the bug that makes self-play results
        unreproducible.
        """
        index = sum(1 for record in self._records.values() if record.parent_id == parent_id)
        policy_id = f"{parent_id}@{step}" if step else f"{parent_id}#{index + 1}"
        return self.register(
            policy_id,
            checkpoint=checkpoint,
            step=step,
            parent_id=parent_id,
            frozen=True,
            tags=tags,
        )

    def get(self, policy_id: str) -> PolicyRecord:
        if policy_id not in self._records:
            raise KeyError(f"unknown policy id: {policy_id}")
        return self._records[policy_id]

    def ids(self, include_learning: bool = True) -> list[str]:
        return [
            policy_id
            for policy_id in self._order
            if include_learning or self._records[policy_id].frozen
        ]

    def records(self) -> list[PolicyRecord]:
        return [self._records[policy_id] for policy_id in self._order]

    def latest(self, parent_id: str = "") -> PolicyRecord | None:
        candidates = [
            record
            for record in self.records()
            if not parent_id or record.parent_id == parent_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda record: (record.step, self._order.index(record.policy_id)))

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no registry path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"policies": [record.to_dict() for record in self.records()]}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.path = target
        return target

    def load(self, path: str | Path | None = None) -> None:
        source = Path(path) if path else self.path
        if source is None or not source.exists():
            raise FileNotFoundError(f"registry not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        self._records = {}
        self._order = []
        for entry in payload.get("policies", []):
            record = PolicyRecord.from_dict(entry)
            self._records[record.policy_id] = record
            self._order.append(record.policy_id)
        self.path = source


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expectation of A scoring against B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class League:
    """Opponent sampling and match bookkeeping over a registry.

    Sampling strategies:

    ``uniform``
        Every frozen opponent equally likely. The safe default: it cannot
        collapse onto one opponent and overfit to it.
    ``latest``
        Always the most recent snapshot. Fast progress, classic
        forgetting: the policy stops being able to beat older versions.
    ``prioritized``
        Weighted toward opponents of similar rating. Keeps matches
        informative without abandoning the rest of the league.
    """

    STRATEGIES = ("uniform", "latest", "prioritized")

    def __init__(
        self,
        registry: CheckpointRegistry,
        strategy: str = "uniform",
        seed: int = 0,
        enable_elo: bool = True,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unknown opponent strategy: {strategy}")
        self.registry = registry
        self.strategy = strategy
        self.seed = seed
        self.enable_elo = enable_elo
        self._rng = random.Random(seed)
        self._matches: list[dict[str, Any]] = []
        self._pairs: dict[tuple[str, str], dict[str, int]] = {}

    def reset_sampling(self) -> None:
        """Rewinds the opponent stream. Two runs with the same seed then
        produce the same opponents again."""
        self._rng = random.Random(self.seed)

    def candidates(self, exclude: str = "") -> list[str]:
        return [
            policy_id
            for policy_id in self.registry.ids(include_learning=False)
            if policy_id != exclude
        ]

    def sample_opponent(self, learner_id: str = "") -> str | None:
        pool = self.candidates(exclude=learner_id)
        if not pool:
            return None
        if self.strategy == "latest":
            return pool[-1]
        if self.strategy == "uniform":
            return self._rng.choice(pool)
        # prioritized: weight by rating proximity to the learner.
        learner_elo = (
            self.registry.get(learner_id).elo
            if learner_id in self.registry
            else DEFAULT_ELO
        )
        weights = [
            1.0 / (1.0 + abs(self.registry.get(policy_id).elo - learner_elo) / 100.0)
            for policy_id in pool
        ]
        return self._rng.choices(pool, weights=weights, k=1)[0]

    def record_match(
        self,
        policy_a: str,
        policy_b: str,
        score_a: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Records one match. ``score_a`` is 1 win, 0.5 draw, 0 loss."""
        if score_a < 0.0 or score_a > 1.0:
            raise ValueError("score_a must be in [0, 1]")
        record_a = self.registry.get(policy_a)
        record_b = self.registry.get(policy_b)
        score_b = 1.0 - score_a

        for record, score in ((record_a, score_a), (record_b, score_b)):
            record.matches += 1
            if score > 0.5:
                record.wins += 1
            elif score < 0.5:
                record.losses += 1
            else:
                record.draws += 1

        if self.enable_elo:
            expectation = expected_score(record_a.elo, record_b.elo)
            delta = ELO_K * (score_a - expectation)
            record_a.elo += delta
            record_b.elo -= delta

        pair_key = (policy_a, policy_b)
        pair = self._pairs.setdefault(pair_key, {"wins": 0, "losses": 0, "draws": 0})
        if score_a > 0.5:
            pair["wins"] += 1
        elif score_a < 0.5:
            pair["losses"] += 1
        else:
            pair["draws"] += 1

        entry = {
            "a": policy_a,
            "b": policy_b,
            "score_a": score_a,
            "elo_a": record_a.elo,
            "elo_b": record_b.elo,
            "metadata": dict(metadata or {}),
        }
        self._matches.append(entry)
        return entry

    def head_to_head(self, policy_a: str, policy_b: str) -> dict[str, int]:
        forward = self._pairs.get((policy_a, policy_b), {"wins": 0, "losses": 0, "draws": 0})
        reverse = self._pairs.get((policy_b, policy_a), {"wins": 0, "losses": 0, "draws": 0})
        return {
            "wins": forward["wins"] + reverse["losses"],
            "losses": forward["losses"] + reverse["wins"],
            "draws": forward["draws"] + reverse["draws"],
        }

    @staticmethod
    def tournament_pairings(policy_ids: Sequence[str], rounds: int = 1) -> list[tuple[str, str]]:
        """Every ordered pair, repeated ``rounds`` times.

        Ordered, not unordered: who spawns as which side is not symmetric
        in an FPS, so A-vs-B and B-vs-A are different matches. The order is
        fixed, so a tournament is reproducible without an RNG at all.
        """
        pairings: list[tuple[str, str]] = []
        for _ in range(max(1, rounds)):
            for first in policy_ids:
                for second in policy_ids:
                    if first != second:
                        pairings.append((first, second))
        return pairings

    def standings(self) -> list[dict[str, Any]]:
        rows = [
            {
                "policy_id": record.policy_id,
                "elo": record.elo,
                "matches": record.matches,
                "wins": record.wins,
                "losses": record.losses,
                "draws": record.draws,
                "win_rate": record.win_rate,
                "step": record.step,
            }
            for record in self.registry.records()
        ]
        return sorted(rows, key=lambda row: (-row["elo"], row["policy_id"]))

    def match_log(self) -> list[dict[str, Any]]:
        return list(self._matches)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "elo_enabled": self.enable_elo,
            "policies": len(self.registry),
            "matches": len(self._matches),
            "standings": self.standings(),
        }


def format_standings(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        f"{'policy':<28}{'elo':>8}{'matches':>9}{'W':>5}{'L':>5}{'D':>5}{'win%':>8}",
        "-" * 68,
    ]
    for row in rows:
        lines.append(
            f"{row['policy_id']:<28}{row['elo']:>8.0f}{row['matches']:>9}"
            f"{row['wins']:>5}{row['losses']:>5}{row['draws']:>5}{row['win_rate']:>8.1%}"
        )
    return "\n".join(lines)

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

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    def from_dict(cls, payload: dict[str, Any]) -> PolicyRecord:
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
            record for record in self.records() if not parent_id or record.parent_id == parent_id
        ]
        if not candidates:
            return None
        return max(
            candidates, key=lambda record: (record.step, self._order.index(record.policy_id))
        )

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
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
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
            self.registry.get(learner_id).elo if learner_id in self.registry else DEFAULT_ELO
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


# ===========================================================================
# Phase 6 completion: rich match results, condition breakdowns, frozen
# opponent enforcement, persisted history and deterministic tournaments.
# ===========================================================================


@dataclass
class MatchResult:
    """One completed evaluation match between two policies.

    The outcome is stored as ``score_a`` (1 / 0.5 / 0) *and* as the raw
    per-side statistics, because a win rate alone cannot distinguish "won
    by outplaying" from "won because the opponent timed out". The
    ``truncated`` flag is separate from draw for the same reason
    ``EpisodeState.to_metrics`` keeps timeouts out of ``loss``.
    """

    policy_a: str
    policy_b: str
    score_a: float
    map_id: str = ""
    lighting: str = ""
    scenario: str = ""
    enemy_count: int = 1
    seed: int = 0
    truncated: bool = False
    ## Per-side diagnostics. Keys are free-form, but the league reports on
    ## the ones listed in MATCH_STAT_KEYS.
    stats_a: dict[str, float] = field(default_factory=dict)
    stats_b: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score_a) <= 1.0:
            raise ValueError("score_a must be in [0, 1]")

    @property
    def outcome(self) -> str:
        if self.truncated and self.score_a == 0.5:
            return "truncated"
        if self.score_a > 0.5:
            return "win_a"
        if self.score_a < 0.5:
            return "win_b"
        return "draw"

    @property
    def condition_key(self) -> str:
        """Groups matches played under the same conditions (seed excluded)."""
        return "|".join(
            (
                self.map_id or "-",
                self.lighting or "-",
                self.scenario or "-",
                f"n{self.enemy_count}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MatchResult:
        known = {key: payload[key] for key in payload if key in cls.__annotations__}
        return cls(**known)


## Diagnostics the league summarizes per policy. Mirrors the Phase 3
## metric categories so a league report and a metrics report line up.
MATCH_STAT_KEYS: tuple[str, ...] = (
    "damage_dealt",
    "damage_received",
    "accuracy",
    "survival_time",
    "awareness",
    "positioning",
)


class FrozenOpponentError(RuntimeError):
    """A frozen opponent changed (or was asked to change) mid-evaluation."""


class FrozenOpponentGuard:
    """Detects a frozen checkpoint file changing during an evaluation.

    This is the bug that silently invalidates self-play results: the
    "frozen" opponent points at ``latest.zip``, training overwrites it,
    and half the tournament was played against a different brain than the
    other half. The guard fingerprints (size, mtime_ns) at arm time and
    re-checks on demand. It deliberately does not hash the file — a
    multi-hundred-megabyte checkpoint would be re-read on every match —
    and a size/mtime change is already proof enough that it moved.
    """

    def __init__(self) -> None:
        self._fingerprints: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int]:
        info = path.stat()
        return (info.st_size, info.st_mtime_ns)

    def arm(self, records: Iterable[PolicyRecord]) -> list[str]:
        """Fingerprints every frozen policy that has a file on disk."""
        watched: list[str] = []
        for record in records:
            if not record.frozen or not record.checkpoint:
                continue
            path = Path(record.checkpoint)
            if not path.exists():
                continue
            self._fingerprints[record.policy_id] = self._fingerprint(path)
            watched.append(record.policy_id)
        return watched

    def verify(self, records: Iterable[PolicyRecord]) -> None:
        for record in records:
            expected = self._fingerprints.get(record.policy_id)
            if expected is None:
                continue
            path = Path(record.checkpoint)
            if not path.exists():
                raise FrozenOpponentError(
                    f"frozen opponent {record.policy_id!r} disappeared: {path}"
                )
            if self._fingerprint(path) != expected:
                raise FrozenOpponentError(
                    f"frozen opponent {record.policy_id!r} changed during evaluation: {path}. "
                    "Snapshot opponents to their own file instead of pointing at a live checkpoint."
                )

    @property
    def watched(self) -> list[str]:
        return sorted(self._fingerprints)


class EvaluationLeague(League):
    """League with per-map / per-condition records and persisted history.

    Kept as a subclass rather than folded into ``League`` so the existing
    (tested) sampling/Elo behaviour is untouched: everything added here is
    bookkeeping on top of ``record_match``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.results: list[MatchResult] = []
        self.guard = FrozenOpponentGuard()

    # -- recording ---------------------------------------------------------

    def record_result(self, result: MatchResult) -> dict[str, Any]:
        """Records a full match result (and its Elo/W-L-D side effects)."""
        for policy_id in (result.policy_a, result.policy_b):
            if policy_id not in self.registry:
                raise KeyError(f"unknown policy id: {policy_id}")
        entry = self.record_match(
            result.policy_a,
            result.policy_b,
            result.score_a,
            metadata={
                "map": result.map_id,
                "lighting": result.lighting,
                "scenario": result.scenario,
                "enemy_count": result.enemy_count,
                "seed": result.seed,
                "outcome": result.outcome,
            },
        )
        self.results.append(result)
        return entry

    # -- breakdowns --------------------------------------------------------

    def _rows_for(self, policy_id: str) -> list[tuple[MatchResult, float, dict[str, float]]]:
        rows: list[tuple[MatchResult, float, dict[str, float]]] = []
        for result in self.results:
            if result.policy_a == policy_id:
                rows.append((result, result.score_a, result.stats_a))
            if result.policy_b == policy_id:
                rows.append((result, 1.0 - result.score_a, result.stats_b))
        return rows

    @staticmethod
    def _tally(rows: Sequence[tuple[MatchResult, float, dict[str, float]]]) -> dict[str, Any]:
        wins = sum(1 for _r, score, _s in rows if score > 0.5)
        losses = sum(1 for _r, score, _s in rows if score < 0.5)
        draws = sum(1 for result, score, _s in rows if score == 0.5 and not result.truncated)
        truncations = sum(1 for result, score, _s in rows if score == 0.5 and result.truncated)
        summary: dict[str, Any] = {
            "matches": len(rows),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "truncations": truncations,
            "win_rate": wins / len(rows) if rows else 0.0,
        }
        for key in MATCH_STAT_KEYS:
            values = [float(stats[key]) for _r, _score, stats in rows if key in stats]
            summary[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
        return summary

    def policy_summary(self, policy_id: str) -> dict[str, Any]:
        summary = self._tally(self._rows_for(policy_id))
        summary["policy_id"] = policy_id
        summary["elo"] = (
            self.registry.get(policy_id).elo if policy_id in self.registry else DEFAULT_ELO
        )
        return summary

    def by_map(self, policy_id: str) -> dict[str, dict[str, Any]]:
        return self._group(policy_id, lambda result: result.map_id or "-")

    def by_condition(self, policy_id: str) -> dict[str, dict[str, Any]]:
        return self._group(policy_id, lambda result: result.condition_key)

    def by_lighting(self, policy_id: str) -> dict[str, dict[str, Any]]:
        return self._group(policy_id, lambda result: result.lighting or "-")

    def by_opponent(self, policy_id: str) -> dict[str, dict[str, Any]]:
        def opponent(result: MatchResult) -> str:
            return result.policy_b if result.policy_a == policy_id else result.policy_a

        return self._group(policy_id, opponent)

    def _group(self, policy_id: str, key_of: Any) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[tuple[MatchResult, float, dict[str, float]]]] = {}
        order: list[str] = []
        for row in self._rows_for(policy_id):
            key = str(key_of(row[0]))
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(row)
        return {key: self._tally(buckets[key]) for key in order}

    def full_report(self) -> dict[str, Any]:
        report = self.report()
        report["matches_recorded"] = len(self.results)
        report["elo_note"] = (
            "Elo here is an EXPERIMENTAL research signal computed from the matches this "
            "run played. It is not a measure of intelligence and is not comparable across runs."
        )
        report["policies"] = {
            policy_id: {
                "summary": self.policy_summary(policy_id),
                "by_map": self.by_map(policy_id),
                "by_condition": self.by_condition(policy_id),
                "by_opponent": self.by_opponent(policy_id),
            }
            for policy_id in self.registry.ids()
        }
        return report

    # -- persistence -------------------------------------------------------

    def save_history(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "seed": self.seed,
            "elo_enabled": self.enable_elo,
            "results": [result.to_dict() for result in self.results],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def load_history(self, path: str | Path, replay_into_registry: bool = False) -> int:
        """Loads a saved match history.

        ``replay_into_registry`` re-applies every match to the registry's
        W/L/D and Elo. It defaults to False because the usual case is
        loading a history *alongside* a registry that already contains
        those totals, and applying them twice would double-count.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        self.results = [MatchResult.from_dict(entry) for entry in payload.get("results", [])]
        if replay_into_registry:
            for result in self.results:
                self.record_match(result.policy_a, result.policy_b, result.score_a)
        return len(self.results)

    # -- tournaments -------------------------------------------------------

    def run_tournament(
        self,
        policy_ids: Sequence[str],
        play_match: Any,
        rounds: int = 1,
        conditions: Sequence[dict[str, Any]] | None = None,
        enforce_frozen: bool = True,
    ) -> list[MatchResult]:
        """Plays every ordered pair (optionally under every condition).

        ``play_match(policy_a, policy_b, condition) -> MatchResult`` is the
        caller's business: it may drive Godot, a mock, or a replay. The
        league only guarantees the *schedule* is deterministic (fixed
        pairing order, fixed condition order, seeds derived from the
        league seed) and that frozen opponents did not move underneath it.
        """
        if enforce_frozen:
            self.guard.arm(self.registry.records())
        plan = self.tournament_pairings(policy_ids, rounds)
        condition_list: Sequence[dict[str, Any]] = conditions or [{}]
        played: list[MatchResult] = []
        for index, (policy_a, policy_b) in enumerate(plan):
            for condition_index, condition in enumerate(condition_list):
                setup = dict(condition)
                setup.setdefault("seed", self.seed + index * 7919 + condition_index * 104729)
                result = play_match(policy_a, policy_b, setup)
                if not isinstance(result, MatchResult):
                    raise TypeError("play_match must return a MatchResult")
                self.record_result(result)
                played.append(result)
        if enforce_frozen:
            self.guard.verify(self.registry.records())
        return played


def format_match_history(results: Sequence[MatchResult], limit: int = 20) -> str:
    lines = [
        f"{'a':<22}{'b':<22}{'map':<16}{'light':<12}{'outcome':>10}",
        "-" * 82,
    ]
    for result in list(results)[-limit:]:
        lines.append(
            f"{result.policy_a:<22}{result.policy_b:<22}{(result.map_id or '-'):<16}"
            f"{(result.lighting or '-'):<12}{result.outcome:>10}"
        )
    return "\n".join(lines)

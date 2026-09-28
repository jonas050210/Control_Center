"""Generalization evaluation: does the policy fight, or did it memorize?

Phase 7. ``conditions.py`` already provides the condition space and rolling
per-condition statistics. What was missing is the *experimental design*
around them: an explicit split between what the policy trained on and what
it has never seen, evaluated along four independent axes.

**Maps** — four buckets, in increasing difficulty of transfer:

``known``            maps and layout seeds seen during training
``unseen_seeds``     trained maps, different layout seeds
``unseen_variants``  trained maps, different scenario/spawn variants
``unseen_maps``      maps held out of training entirely

**Conditions** — normal, low light, night, fog, mixed lighting.
**Combat** — 1, 2, 3 and 5+ enemies.
**Scenarios** — open, cover, corridor, ambush, sound, multi-direction,
vertical.

The report is *data*, produced only from episodes that were actually
played: there is no model of expected performance anywhere in this module
and nothing is extrapolated. ``GeneralizationSuite.plan`` produces the
episode list; the caller runs it (against Godot, a mock, or a replay) and
feeds results back through ``record``. Running the episodes is deliberately
not this module's job, which is what keeps it testable without an engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .conditions import LIGHTING_IDS, MAP_IDS, Condition

## Map split buckets, in report order.
MAP_BUCKETS: tuple[str, ...] = ("known", "unseen_seeds", "unseen_variants", "unseen_maps")

## Lighting conditions Phase 7 asks for. "high_contrast" exists in the
## engine too but is not part of the required matrix; it is included only
## when the caller passes it explicitly.
EVALUATION_LIGHTINGS: tuple[str, ...] = ("normal", "low_light", "night", "fog", "mixed")

## Enemy-count buckets. The 5+ bucket is where the observation vector's
## overflow-contact statistics (fields 66-69) are the only information the
## policy has about most of the opposition, so it is the interesting one.
ENEMY_COUNT_BUCKETS: tuple[int, ...] = (1, 2, 3, 5)

## Evaluation axes, in report order. Kept explicit so a report can name an
## axis that produced ZERO episodes (e.g. nothing held out yet) instead of
## silently dropping it -- an empty axis is a finding, not an omission.
EVALUATION_AXES: tuple[str, ...] = ("maps", "conditions", "combat", "scenarios")

## Scenario families, mapped to ScenarioLibrary ids in
## scripts/scenario/scenario_library.gd. A test checks that every id here
## exists in the GDScript source, so a renamed scenario fails loudly.
SCENARIO_FAMILIES: dict[str, str] = {
    "open": "open_arena",
    "cover": "cover_fight",
    "corridor": "corridor_fight",
    "ambush": "ambush",
    "sound": "sound_only",
    "multi_direction": "multi_direction",
    "vertical": "vertical_encounter",
}


@dataclass(frozen=True)
class EvaluationEpisode:
    """One planned evaluation episode, with the axis labels it belongs to."""

    condition: Condition
    map_bucket: str
    axis: str  # "maps" | "conditions" | "combat" | "scenarios"
    bucket: str  # the value along that axis

    def labels(self) -> dict[str, Any]:
        return {
            "map_id": self.condition.map_id,
            "lighting": self.condition.lighting,
            "scenario": self.condition.scenario,
            "enemy_count": self.condition.enemy_count,
            "curriculum_level": self.condition.level,
            "seed": self.condition.seed,
            "map_bucket": self.map_bucket,
            "axis": self.axis,
            "bucket": self.bucket,
        }


@dataclass
class MapSplit:
    """Which maps/seeds the policy trained on, and what is held out.

    Holding maps out is the only honest way to answer "does it generalize
    to a map it has never seen". Everything else (new seeds, new variants)
    measures a weaker, still useful, kind of transfer.
    """

    train_maps: list[str] = field(default_factory=lambda: list(MAP_IDS[:10]))
    holdout_maps: list[str] = field(default_factory=lambda: list(MAP_IDS[10:]))
    train_seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    unseen_seeds: list[int] = field(default_factory=lambda: [9001, 9002, 9003])
    train_scenarios: list[str] = field(default_factory=lambda: ["single_target", "cover_fight"])
    unseen_scenarios: list[str] = field(default_factory=lambda: ["ambush", "multi_direction"])

    def __post_init__(self) -> None:
        overlap = set(self.train_maps) & set(self.holdout_maps)
        if overlap:
            raise ValueError(f"maps cannot be both trained and held out: {sorted(overlap)}")
        if not self.train_maps:
            raise ValueError("train_maps must not be empty")
        seed_overlap = set(self.train_seeds) & set(self.unseen_seeds)
        if seed_overlap:
            raise ValueError(f"seeds cannot be both trained and unseen: {sorted(seed_overlap)}")

    def bucket_of(self, map_id: str, seed: int, scenario: str) -> str:
        """Classifies a condition into one of MAP_BUCKETS."""
        if map_id in self.holdout_maps:
            return "unseen_maps"
        if scenario and scenario in self.unseen_scenarios:
            return "unseen_variants"
        if seed in self.train_seeds:
            return "known"
        return "unseen_seeds"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneralizationSuite:
    """Builds the evaluation plan and aggregates the results per bucket."""

    def __init__(
        self,
        split: MapSplit | None = None,
        lightings: Sequence[str] = EVALUATION_LIGHTINGS,
        enemy_counts: Sequence[int] = ENEMY_COUNT_BUCKETS,
        scenarios: dict[str, str] | None = None,
        level: int = 10,
        episodes_per_cell: int = 2,
        base_seed: int = 70000,
    ) -> None:
        self.split = split or MapSplit()
        self.lightings = list(lightings)
        self.enemy_counts = list(enemy_counts)
        self.scenarios = dict(scenarios or SCENARIO_FAMILIES)
        self.level = int(level)
        self.episodes_per_cell = max(1, int(episodes_per_cell))
        self.base_seed = int(base_seed)
        self._rows: list[dict[str, Any]] = []

    # -- planning ----------------------------------------------------------

    def _seed(self, index: int) -> int:
        # A fixed arithmetic progression rather than an RNG: an evaluation
        # plan must be identical across processes, machines and Python
        # versions, and `random` only guarantees the first two.
        return self.base_seed + index * 613

    def plan(self) -> list[EvaluationEpisode]:
        """The full evaluation plan. Deterministic and order-stable."""
        episodes: list[EvaluationEpisode] = []
        index = 0

        def add(
            map_id: str,
            lighting: str,
            scenario: str,
            enemy_count: int,
            seed: int,
            axis: str,
            bucket: str,
        ) -> None:
            nonlocal index
            condition = Condition(
                map_id=map_id,
                lighting=lighting,
                scenario=scenario,
                enemy_count=enemy_count,
                level=self.level,
                seed=seed,
            )
            episodes.append(
                EvaluationEpisode(
                    condition=condition,
                    map_bucket=self.split.bucket_of(map_id, seed, scenario),
                    axis=axis,
                    bucket=bucket,
                )
            )
            index += 1

        default_scenario = self.split.train_scenarios[0] if self.split.train_scenarios else ""
        default_lighting = self.lightings[0] if self.lightings else "normal"

        # Axis 1: maps (known / unseen seeds / unseen variants / unseen maps)
        for repeat in range(self.episodes_per_cell):
            for map_id in self.split.train_maps:
                seed = self.split.train_seeds[repeat % len(self.split.train_seeds)]
                add(map_id, default_lighting, default_scenario, 1, seed, "maps", "known")
                add(
                    map_id,
                    default_lighting,
                    default_scenario,
                    1,
                    self.split.unseen_seeds[repeat % len(self.split.unseen_seeds)],
                    "maps",
                    "unseen_seeds",
                )
                if self.split.unseen_scenarios:
                    add(
                        map_id,
                        default_lighting,
                        self.split.unseen_scenarios[repeat % len(self.split.unseen_scenarios)],
                        1,
                        self._seed(index),
                        "maps",
                        "unseen_variants",
                    )
            for map_id in self.split.holdout_maps:
                add(map_id, default_lighting, default_scenario, 1, self._seed(index), "maps", "unseen_maps")

        # Axis 2: lighting conditions, on trained maps so the axis is clean
        for repeat in range(self.episodes_per_cell):
            for lighting in self.lightings:
                map_id = self.split.train_maps[repeat % len(self.split.train_maps)]
                add(map_id, lighting, default_scenario, 1, self._seed(index), "conditions", lighting)

        # Axis 3: enemy counts
        for repeat in range(self.episodes_per_cell):
            for enemy_count in self.enemy_counts:
                map_id = self.split.train_maps[repeat % len(self.split.train_maps)]
                add(
                    map_id,
                    default_lighting,
                    default_scenario,
                    enemy_count,
                    self._seed(index),
                    "combat",
                    f"{enemy_count}_enemies",
                )

        # Axis 4: scenario families
        for repeat in range(self.episodes_per_cell):
            for family, scenario_id in self.scenarios.items():
                map_id = self.split.train_maps[repeat % len(self.split.train_maps)]
                add(map_id, default_lighting, scenario_id, 2, self._seed(index), "scenarios", family)

        return episodes

    def plan_size(self) -> int:
        return len(self.plan())

    # -- results -----------------------------------------------------------

    def record(
        self,
        episode: EvaluationEpisode,
        won: bool,
        reward: float,
        steps: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Records the outcome of one planned episode."""
        row = episode.labels()
        row.update(
            {
                "won": bool(won),
                "reward": float(reward),
                "steps": int(steps),
            }
        )
        row.update(extra or {})
        self._rows.append(row)

    def run(self, play_episode: Callable[[EvaluationEpisode], dict[str, Any]]) -> int:
        """Runs the whole plan through a caller-supplied episode runner.

        ``play_episode`` returns a dict with at least ``won``/``reward``/
        ``steps``. Any other key is carried through into the report, so a
        caller that also computes Phase 3 metrics can attach them.
        """
        for episode in self.plan():
            outcome = dict(play_episode(episode))
            self.record(
                episode,
                bool(outcome.pop("won", False)),
                float(outcome.pop("reward", 0.0)),
                int(outcome.pop("steps", 0)),
                outcome,
            )
        return len(self._rows)

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def reset(self) -> None:
        self._rows.clear()

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"episodes": 0, "win_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0}
        return {
            "episodes": len(rows),
            "win_rate": sum(1.0 for row in rows if row["won"]) / len(rows),
            "mean_reward": sum(float(row["reward"]) for row in rows) / len(rows),
            "mean_steps": sum(float(row["steps"]) for row in rows) / len(rows),
        }

    def _by(self, key: str, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        selected = list(rows if rows is not None else self._rows)
        buckets: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for row in selected:
            value = str(row.get(key, ""))
            if value not in buckets:
                buckets[value] = []
                order.append(value)
            buckets[value].append(row)
        return {value: self._summarize(buckets[value]) for value in order}

    def axis_report(self, axis: str) -> dict[str, dict[str, Any]]:
        rows = [row for row in self._rows if row.get("axis") == axis]
        return self._by("bucket", rows)

    def report(self) -> dict[str, Any]:
        """Structured per-condition report. No fabricated numbers."""
        by_map_bucket = self._by("map_bucket")
        known = by_map_bucket.get("known", {}).get("win_rate", 0.0)
        unseen = by_map_bucket.get("unseen_maps", {}).get("win_rate", 0.0)
        return {
            "episodes": len(self._rows),
            "overall": self._summarize(self._rows),
            "map_buckets": {
                bucket: by_map_bucket.get(bucket, self._summarize([])) for bucket in MAP_BUCKETS
            },
            "by_axis": {
                axis: self.axis_report(axis)
                for axis in EVALUATION_AXES
            },
            "by_map": self._by("map_id"),
            "by_lighting": self._by("lighting"),
            "by_enemy_count": self._by("enemy_count"),
            "by_scenario": self._by("scenario"),
            # The single number that matters: how much worse is it on a map
            # it has never seen. Positive = worse on unseen maps.
            "transfer_gap": known - unseen,
            "split": self.split.to_dict(),
        }

    def export(self, directory: str | Path) -> dict[str, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "generalization.json"
        report_path.write_text(json.dumps(self.report(), indent=2, default=str) + "\n", encoding="utf-8")
        csv_path = target / "episodes.csv"
        fields: list[str] = []
        for row in self._rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._rows)
        text_path = target / "generalization.txt"
        text_path.write_text(format_report(self.report()), encoding="utf-8")
        return {"report": report_path, "episodes": csv_path, "text": text_path}


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "SandboxAI generalization report",
        "=" * 72,
        f"episodes: {report['episodes']}   overall win rate: {report['overall']['win_rate']:.1%}",
        f"transfer gap (known - unseen maps): {report['transfer_gap']:+.1%}",
        "",
        f"{'bucket':<24}{'eps':>6}{'win':>9}{'reward':>11}",
        "-" * 72,
    ]
    for bucket, summary in report["map_buckets"].items():
        lines.append(
            f"{bucket:<24}{summary['episodes']:>6}{summary['win_rate']:>9.1%}{summary['mean_reward']:>11.2f}"
        )
    for axis in ("conditions", "combat", "scenarios"):
        lines.append("")
        lines.append(axis.upper())
        for bucket, summary in report["by_axis"].get(axis, {}).items():
            lines.append(
                f"  {bucket:<22}{summary['episodes']:>6}{summary['win_rate']:>9.1%}"
                f"{summary['mean_reward']:>11.2f}"
            )
    return "\n".join(lines) + "\n"


def compare_policies(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Side-by-side of several policies' generalization reports.

    Takes reports rather than running anything, so the comparison cannot
    accidentally evaluate two policies on different episode sets.
    """
    rows: list[dict[str, Any]] = []
    for policy_id, report in reports.items():
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "episodes": report["episodes"],
            "win_rate": report["overall"]["win_rate"],
            "transfer_gap": report["transfer_gap"],
        }
        for bucket in MAP_BUCKETS:
            row[bucket] = report["map_buckets"].get(bucket, {}).get("win_rate", 0.0)
        rows.append(row)
    rows.sort(key=lambda row: (-row["win_rate"], row["policy_id"]))
    return {"policies": rows, "count": len(rows)}


def iter_conditions(episodes: Iterable[EvaluationEpisode]) -> Iterable[Condition]:
    for episode in episodes:
        yield episode.condition

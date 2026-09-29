"""Checkpoint-time evaluation: conditions, generalization, league, metrics.

At each checkpoint boundary the training callback hands the frozen policy
to :func:`run_checkpoint_evaluation`, which produces one machine-readable
report covering everything a researcher needs to compare checkpoints:

* **condition evaluation** — episodes drawn from a per-run frozen
  evaluation distribution (the union of the curriculum ladder, with
  evaluation-only seeds), so every checkpoint is measured on exactly the
  same episodes;
* **generalization evaluation** — the existing ``GeneralizationSuite``
  driven with a ``MapSplit`` derived from the conditions the run ACTUALLY
  trained on, so a training seed/map/scenario can never leak into an
  "unseen" bucket (and vice versa);
* **league evaluation** (optional) — deterministic two-agent matches
  between a frozen snapshot of the checkpoint and older frozen snapshots,
  on the self-play bridge, with frozen-opponent integrity enforced;
* **training metrics** — the interval aggregate of the research metrics
  collected during training, per condition, per environment, pooled.

Nothing here changes PPO: it reads finished checkpoints, runs frozen
policies and writes JSON. The reward stream is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from .conditions import MAP_IDS, Condition, ConditionTracker, generalization_report
from .curriculum_stages import (
    TRAINABLE_MAX_LEVEL,
    WORLD_MIN_LEVEL,
    applied_condition,
    evaluation_distribution,
)
from .generalization import SCENARIO_FAMILIES, GeneralizationSuite, MapSplit
from .pipeline import EVAL_MASTER_SEED_SALT, SkillMetricsSink
from .randomization import EpisodePlan
from .replay import ReplayHeader, ReplayRecorder

## Generalization plans are seeded from this suite base; if any derived
## seed collides with an actually trained seed the base is bumped by this
## stride (deterministically) until the streams are disjoint.
UNSEEN_SEED_BUMP: int = 100_003


# ---------------------------------------------------------------------------
# Plan-driven episode execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedEpisode:
    """One episode to run in the plan executor: setup + report labels."""

    condition: Condition
    layout_seed: int = 0
    labels: dict[str, Any] = field(default_factory=dict)
    record_replay: bool = False

    def payload(self, env_index: int) -> dict[str, Any]:
        condition = self.condition
        return {
            "index": int(env_index),
            "seed": condition.seed,
            "map_id": condition.map_id,
            "scenario": condition.scenario,
            "lighting": condition.lighting,
            "enemy_count": condition.enemy_count,
            "curriculum_level": condition.level,
        }


def plan_of(condition: Condition, layout_seed: int = 0) -> EpisodePlan:
    """Builds a bridge-staged EpisodePlan description for a condition."""
    plan = EpisodePlan(
        index=0,
        condition=condition,
        layout_seed=layout_seed,
        spawn=None,  # spawn detail is derived engine-side from the seed
    )
    return plan


class PlanExecutor:
    """Runs a list of planned episodes through one Godot bridge batch.

    Deterministic exactness: episode ``j`` always runs on environment
    ``j % N``, in ascending ``j`` per environment, with the plan's own
    seed. Scheduling therefore cannot change results; only the wall time
    changes with N. Model predictions are deterministic=True throughout.

    Because every plan fully reconfigures its environment (level, enemy
    count, map, lighting, scenario, weapon profile, then ``reset(seed)``),
    episodes are pure functions of the plan: reusing one executor across
    checkpoint boundaries — or running the same plans on a different
    environment count — yields bit-identical episodes.
    """

    def __init__(self, env_kwargs: dict[str, Any], skill_metrics: bool = True, profiler: Any = None) -> None:
        from .godot_env import GodotBatchClient

        self.profiler = profiler
        self.client = GodotBatchClient(**env_kwargs)
        self.environment_count = self.client.environment_count
        self.skill_metrics_enabled = bool(skill_metrics)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "PlanExecutor":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def run(
        self,
        model: Any,
        plans: Sequence[PlannedEpisode],
        policy_id: str,
        checkpoint: str = "",
        replay_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Runs every planned episode and returns one result row per plan."""
        import numpy as np

        count = self.environment_count
        queues: list[list[tuple[int, PlannedEpisode]]] = [[] for _ in range(count)]
        for position, plan in enumerate(plans):
            queues[position % count].append((position, plan))

        sink = (
            SkillMetricsSink(count, enabled=True, policy_id=policy_id)
            if self.skill_metrics_enabled
            else None
        )
        in_flight: list[PlannedEpisode | None] = [None] * count
        recorders: list[ReplayRecorder | None] = [None] * count
        rows: list[tuple[int, dict[str, Any]]] = []

        initial: list[dict[str, Any]] = []
        for env_index in range(count):
            if queues[env_index]:
                _, plan = queues[env_index].pop(0)
                in_flight[env_index] = plan
                initial.append(plan.payload(env_index))
        if initial:
            self.client.set_episode_plans(initial)
        # One batched reset consumes every staged plan; idle environments
        # get a plain reset and are ignored until a plan reaches them.
        observations, _infos = self.client.reset(None)
        for env_index in range(count):
            plan = in_flight[env_index]
            if plan is not None:
                self._begin(env_index, plan, sink, recorders, policy_id, checkpoint)

        while any(plan is not None for plan in in_flight):
            batch = np.asarray(observations, dtype=np.float32)
            predict_started = time.monotonic() if self.profiler is not None else 0.0
            prediction = model.predict(batch, deterministic=True)
            actions = prediction[0] if isinstance(prediction, tuple) else prediction
            if self.profiler is not None:
                self.profiler.record("predict", time.monotonic() - predict_started)
                step_started = time.monotonic()
            observations, rewards, dones, infos = self.client.step(actions)
            if self.profiler is not None:
                self.profiler.record("env_step", time.monotonic() - step_started)
                self.profiler.add("env_steps", self.environment_count)
            restage: list[dict[str, Any]] = []
            reset_targets: list[int] = []
            for env_index in range(count):
                plan = in_flight[env_index]
                if plan is None:
                    continue
                info = infos[env_index] if env_index < len(infos) else {}
                events = info.get("events", {})
                if sink is not None:
                    sink.record_step(env_index, observations[env_index], actions[env_index], events)
                recorder = recorders[env_index]
                if recorder is not None:
                    recorder.record_step(actions[env_index], float(rewards[env_index]), events=events)
                if not bool(dones[env_index]):
                    continue
                metrics = dict(info.get("metrics", {}))
                rows.append(
                    (
                        self._plan_position(env_index, plan, queues),
                        self._harvest(
                            env_index, plan, metrics, sink, recorders, policy_id, checkpoint, replay_dir
                        ),
                    )
                )
                in_flight[env_index] = None
                if queues[env_index]:
                    _, next_plan = queues[env_index].pop(0)
                    in_flight[env_index] = next_plan
                    restage.append(next_plan.payload(env_index))
                    reset_targets.append(env_index)
            if restage:
                self.client.set_episode_plans(restage)
                for item in self.client.reset_indices(reset_targets, None):
                    index = int(item.get("index", -1))
                    if 0 <= index < count:
                        observations[index] = item.get("observation", observations[index])
                        plan = in_flight[index]
                        if plan is not None:
                            self._begin(index, plan, sink, recorders, policy_id, checkpoint)
        rows.sort(key=lambda pair: pair[0])
        return [row for _position, row in rows]

    # -- per-episode helpers -------------------------------------------------

    def _plan_position(self, env_index: int, plan: PlannedEpisode, queues: Any) -> int:
        # Rows are re-sorted by their plan-list position afterwards.
        return int(plan.labels.get("_position", 0)) if isinstance(plan.labels, dict) else 0

    def _begin(
        self,
        env_index: int,
        plan: PlannedEpisode,
        sink: SkillMetricsSink | None,
        recorders: list[ReplayRecorder | None],
        policy_id: str,
        checkpoint: str,
    ) -> None:
        if sink is not None:
            plan_stub = EpisodePlan(
                index=0,
                condition=plan.condition,
                layout_seed=0,
                spawn=None,
            )
            sink.begin(env_index, plan_stub)
        recorders[env_index] = None
        if plan.record_replay:
            header = ReplayHeader(
                policy_id=policy_id,
                checkpoint=checkpoint,
                environment_index=env_index,
                notes=dict(plan.labels),
                **{
                    "seed": plan.condition.seed,
                    "map_id": plan.condition.map_id,
                    "scenario": plan.condition.scenario,
                    "lighting": plan.condition.lighting,
                    "enemy_count": plan.condition.enemy_count,
                    "curriculum_level": plan.condition.level,
                },
            )
            header.detail = "light"
            recorder = ReplayRecorder(header=header)
            recorder.start(seed=plan.condition.seed)
            recorders[env_index] = recorder

    def _harvest(
        self,
        env_index: int,
        plan: PlannedEpisode,
        metrics: dict[str, Any],
        sink: SkillMetricsSink | None,
        recorders: list[ReplayRecorder | None],
        policy_id: str,
        checkpoint: str,
        replay_dir: str | Path | None,
    ) -> dict[str, Any]:
        labels = dict(plan.labels)
        condition = plan.condition
        row: dict[str, Any] = {
            "labels": labels,
            "condition": condition.to_dict(),
            "seed": condition.seed,
            "environment_index": env_index,
        }
        for key in (
            "episode_reward",
            "episode_length",
            "kills",
            "deaths",
            "damage_dealt",
            "damage_received",
            "survival_time",
            "accuracy",
            "shots_fired",
            "shots_hit",
            "win",
            "loss",
            "truncated",
            "done_reason",
        ):
            if key in metrics:
                row[key] = metrics[key]
        row.setdefault("win", False)
        row.setdefault("episode_reward", 0.0)
        row.setdefault("episode_length", 0)
        summary = sink.finish(env_index, result={"win": bool(row.get("win", False))}) if sink else None
        if summary is not None:
            row["skill"] = summary.get("categories", {})
        recorder = recorders[env_index]
        recorders[env_index] = None
        if recorder is not None and replay_dir is not None:
            recorder.finish({"win": bool(row.get("win", False)), "done_reason": str(row.get("done_reason", ""))})
            tag = labels.get("bucket") or labels.get("axis") or "eval"
            name = f"{tag}_{condition.map_id or 'arena'}_L{condition.level}_s{condition.seed}.jsonl"
            row["replay"] = str(recorder.save(Path(replay_dir) / name))
        return row


# ---------------------------------------------------------------------------
# Generalization split from what the run actually trained on
# ---------------------------------------------------------------------------


def build_map_split(used_conditions: Sequence[dict[str, Any]]) -> MapSplit:
    """Derives a MapSplit from the conditions a run actually applied.

    Honesty rule, in both directions:

    * a (map, seed) pair the training stream used is classified "known" —
      it can never count as unseen evidence;
    * evaluation seeds that collide with a used training seed are banned —
      the caller bumps the suite base seed until the streams are disjoint.
    """
    train_maps = sorted({str(c.get("map_id", "")) for c in used_conditions if c.get("map_id")})
    train_scenarios = sorted({str(c.get("scenario", "")) for c in used_conditions if c.get("scenario")})
    train_seeds = sorted({int(c.get("seed", -1)) for c in used_conditions if int(c.get("seed", -1)) >= 0})
    holdout_maps = [map_id for map_id in MAP_IDS if map_id not in set(train_maps)]
    unseen_scenarios = [
        scenario_id for scenario_id in SCENARIO_FAMILIES.values() if scenario_id not in set(train_scenarios)
    ]
    return MapSplit(
        train_maps=train_maps or ["open_field"],
        holdout_maps=holdout_maps,
        # MapSplit requires non-overlapping seed lists; the caller checks
        # suite seeds against these, never the other way around.
        train_seeds=train_seeds or [-1],
        unseen_seeds=[90_001, 90_002, 90_003],
        train_scenarios=train_scenarios,
        unseen_scenarios=unseen_scenarios[:4],
    )


def suite_seeds_disjoint(suite: GeneralizationSuite, train_seeds: set[int]) -> bool:
    """No UNSEEN bucket may replay a trained seed.

    "Known"-bucket cells are exempt by design: they deliberately re-run
    trained (map, seed) pairs -- that is what "seen evaluation" means. The
    dangerous direction is the opposite one: an unseen-seed / unseen-map /
    unseen-variant cell that reuses a seed the run actually optimized
    against.
    """
    used = set(train_seeds)
    return all(
        episode.condition.seed not in used
        for episode in suite.plan()
        if episode.map_bucket != "known"
    )


# ---------------------------------------------------------------------------
# League evaluation at checkpoint boundaries
# ---------------------------------------------------------------------------


class LeagueIncompatibilityError(RuntimeError):
    """A checkpoint cannot play in this league (different contract)."""


class LeagueRunner:
    """Plays deterministic self-play tournaments between frozen snapshots.

    Snapshots are COPIES (their own files in ``league/policies/``), so a
    frozen opponent can never change underneath an evaluation; the
    ``FrozenOpponentGuard`` double-checks that on every tournament. The
    live training model is only ever *read* (``model.save``), never
    loaded from or modified.
    """

    def __init__(
        self,
        run_dir: str | Path,
        seed: int = 0,
        strategy: str = "uniform",
        device: str = "cpu",
    ) -> None:
        from .league import CheckpointRegistry, EvaluationLeague

        self.directory = Path(run_dir) / "league"
        self.policies_dir = self.directory / "policies"
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = int(seed)
        self.registry = CheckpointRegistry(self.directory / "registry.json")
        self.league = EvaluationLeague(self.registry, strategy=strategy, seed=self.seed, enable_elo=True)
        history = self.directory / "history.json"
        if history.is_file():
            self.league.load_history(history, replay_into_registry=False)
        # The scripted baseline is a permanent, labelled control condition.
        if "scripted_baseline" not in self.registry:
            self.registry.register("scripted_baseline", checkpoint="", frozen=True, tags=["baseline"])

    # -- snapshots ----------------------------------------------------------

    def snapshot_id(self, experiment_id: str, step: int) -> str:
        return f"{experiment_id}@{step}"

    def register_checkpoint(self, model: Any, experiment_id: str, step: int) -> str:
        """Freezes the in-memory checkpoint into its own file + record."""
        policy_id = self.snapshot_id(experiment_id, step)
        if policy_id in self.registry:
            return policy_id
        path = self.policies_dir / f"{experiment_id}@{step}.zip"
        model.save(str(path))
        self.registry.snapshot(experiment_id, checkpoint=str(path), step=step, tags=["checkpoint"])
        self.registry.save()
        return policy_id

    # -- models --------------------------------------------------------------

    def _load_handle(self, policy_id: str, cache: dict[str, Any]) -> Any:
        if policy_id in cache:
            return cache[policy_id]
        from .policies import PolicyHandle, PolicyRole, PolicySpec

        record = self.registry.get(policy_id)
        spec = PolicySpec(
            policy_id=policy_id,
            checkpoint=record.checkpoint,
            role=PolicyRole.BASELINE if "baseline" in record.tags else PolicyRole.FROZEN,
            kind="scripted" if "baseline" in record.tags else "checkpoint",
            device=self.device,
        )
        handle = PolicyHandle(spec)
        handle.load(device=self.device)
        if spec.kind == "checkpoint":
            self._assert_contract(handle.model, record.checkpoint)
        cache[policy_id] = handle
        return handle

    @staticmethod
    def _assert_contract(model: Any, checkpoint: str) -> None:
        from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

        obs_raw = getattr(model.observation_space, "shape", None)
        nvec_raw = getattr(model.action_space, "nvec", None)
        obs_shape = tuple(obs_raw) if obs_raw is not None else ()
        nvec = tuple(int(v) for v in nvec_raw) if nvec_raw is not None else ()
        problems = []
        if obs_shape != (OBSERVATION_FIELD_COUNT,):
            problems.append(f"observation space {obs_shape} != contract ({OBSERVATION_FIELD_COUNT},)")
        if nvec != tuple(ACTION_NVEC):
            problems.append(f"action space {nvec} != contract {tuple(ACTION_NVEC)}")
        if problems:
            raise LeagueIncompatibilityError(
                f"checkpoint {checkpoint!r} cannot play in this league: {'; '.join(problems)}"
            )

    # -- matches ---------------------------------------------------------------

    def evaluate_checkpoint(
        self,
        model: Any,
        experiment_id: str,
        step: int,
        matches: int,
        env_kwargs: dict[str, Any],
        max_opponents: int = 8,
        client_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Registers the checkpoint and plays it against the frozen pool."""
        from .league import MatchResult
        from .policies import assert_independent_weights
        from .self_play import SelfPlayBatchClient, play_self_play_match

        policy_id = self.register_checkpoint(model, experiment_id, step)
        import hashlib

        frozen = self.registry.ids(include_learning=False)
        opponents = [
            pid
            for pid in frozen
            if pid != policy_id and "baseline" not in self.registry.get(pid).tags
        ]
        # Deterministic opponent draw: a hash of (seed, step, policy id).
        # Stable across resumes (no RNG-stream state dependency), but the
        # pool rotates per checkpoint so a policy is not pair-locked to
        # whichever opponent happened to be first alphabetically. Playing
        # newest-first would skew evaluation toward the current week of
        # training; alphabetical would never sample variety at all.
        opponents.sort(
            key=lambda pid: hashlib.sha256(f"{self.seed}:{step}:{pid}".encode()).hexdigest()
        )
        # League matches play both seatings: only matches INVOLVING the new
        # snapshot are played (the league's full tournament_pairings would
        # add opponent-vs-opponent matches that blow the per-checkpoint
        # budget while saying nothing about this checkpoint). The scripted
        # baseline always plays first: it is the labelled control condition.
        budget_opponents = max(0, matches // 2 - 1) if matches > 0 else 0
        opponents = opponents[: min(budget_opponents, max_opponents)]
        field = ["scripted_baseline", *opponents]

        cache: dict[str, Any] = {}
        checked_pairs: set[frozenset] = set()
        factory = client_factory or SelfPlayBatchClient
        client = factory(**{**env_kwargs, "environment_count": 1})
        played: list[MatchResult] = []

        def play_match(policy_a: str, policy_b: str, seed: int) -> MatchResult:
            handle_a = self._load_handle(policy_a, cache)
            handle_b = self._load_handle(policy_b, cache)
            pair = frozenset((policy_a, policy_b))
            if pair not in checked_pairs:
                assert_independent_weights([handle_a, handle_b])
                checked_pairs.add(pair)
            result = play_self_play_match(
                client,
                lambda obs: handle_a.predict(obs)[0],
                lambda obs: handle_b.predict(obs)[0],
                seed=seed,
            )
            stats_keys = ("damage_dealt", "damage_received", "accuracy", "survival_time")
            return MatchResult(
                policy_a=policy_a,
                policy_b=policy_b,
                score_a=result["score_a"],
                map_id="",
                lighting="",
                scenario="self_play",
                enemy_count=1,
                seed=seed,
                truncated=bool(result["truncated"]),
                stats_a={key: float(result["metrics_a"].get(key, 0.0)) for key in stats_keys},
                stats_b={key: float(result["metrics_b"].get(key, 0.0)) for key in stats_keys},
            )

        self.league.guard.arm(self.registry.records())
        try:
            match_index = 0
            for opponent in field:
                # Both seatings, tournament-style derivation (same 7919
                # stride the league's tournament uses) so results reproduce
                # exactly after a resume.
                for policy_a, policy_b in ((policy_id, opponent), (opponent, policy_id)):
                    seed = self.seed + match_index * 7919
                    result = play_match(policy_a, policy_b, seed)
                    self.league.record_result(result)
                    played.append(result)
                    match_index += 1
            self.league.guard.verify(self.registry.records())
        finally:
            client.close()
        self.registry.save()
        self.league.save_history(self.directory / "history.json")
        summary = self.league.policy_summary(policy_id)
        return {
            "policy_id": policy_id,
            "matches": len(played),
            "opponents": [pid for pid in field if pid != policy_id],
            "summary": summary,
            "standings": self.league.standings()[:10],
            "elo_note": (
                "League Elo is an internal research signal from this run's matches; "
                "it is not comparable across runs and is never a training reward."
            ),
        }

    def report(self) -> dict[str, Any]:
        return self.league.full_report()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _condition_eval_plans(config: Any, count: int) -> list[PlannedEpisode]:
    """The frozen per-run seen-distribution evaluation set.

    The distribution is the union of the curriculum ladder with an
    evaluation-only master seed, so the plan LIST is identical at every
    checkpoint (results are comparable across checkpoints) and the seeds
    never collide with the training stream by construction.
    """
    distribution = evaluation_distribution(config.seed + EVAL_MASTER_SEED_SALT)
    plans: list[PlannedEpisode] = []
    for position in range(count):
        plan = distribution.episode_plan(position)
        condition = applied_condition(plan.condition)
        plans.append(
            PlannedEpisode(
                condition=condition,
                layout_seed=plan.layout_seed,
                labels={
                    "eval": "conditions",
                    "condition_key": condition.key,
                    "bucket": "eval_distribution",
                    "_position": position,
                },
            )
        )
    return plans


def _generalization_plans(
    config: Any, pipeline: Any, record_replay: bool
) -> tuple[GeneralizationSuite | None, str, int]:
    """Builds the unseen-distribution suite, or a skip reason.

    Returns ``(suite, reason, trained_level)``; ``suite`` is None when the
    run has nothing map-relevant trained yet.
    """
    used = pipeline.driver.used_conditions if pipeline is not None else []
    train_maps = {str(c.get("map_id", "")) for c in used if c.get("map_id")}
    if not train_maps:
        return None, "no map-conditioned training episodes yet (curriculum below level 5)", 0
    split = build_map_split(used)
    level = pipeline.driver.level if pipeline is not None else WORLD_MIN_LEVEL
    # Run the suite at the run's natural ladder position, clamped to a real
    # world level: below WORLD_MIN_LEVEL every condition resolves onto the
    # same legacy arena (map_id is inert), which would contaminate every
    # map bucket with identical episodes. The trained level is reported
    # separately in the report ("trained_level").
    suite_level = max(WORLD_MIN_LEVEL, min(TRAINABLE_MAX_LEVEL, level + 2))
    suite = GeneralizationSuite(
        split=split,
        level=suite_level,
        episodes_per_cell=config.generalization_episodes_per_cell,
        base_seed=70_000,
    )
    train_seed_set = {int(c.get("seed", -1)) for c in used}
    attempts = 0
    while not suite_seeds_disjoint(suite, train_seed_set) and attempts < 100:
        suite.base_seed += UNSEEN_SEED_BUMP
        attempts += 1
    if not suite_seeds_disjoint(suite, train_seed_set):  # pragma: no cover - defensive
        return None, "could not derive evaluation seeds disjoint from the training stream", level
    return suite, "", level


def run_checkpoint_evaluation(
    model: Any,
    step: int,
    config: Any,
    pipeline: Any,
    output_dir: str | Path,
    device: str = "cpu",
    normal_summary: dict[str, Any] | None = None,
    executor_factory: Callable[..., Any] | None = None,
    client_factory: Callable[..., Any] | None = None,
    executor: Any = None,
    profiler: Any = None,
) -> dict[str, Any]:
    """The full per-checkpoint evaluation battery. Returns the report.

    Cost is bounded by configuration (condition/generalization episode
    counts, league matches); the caller writes the report next to the
    checkpoint's other artifacts.

    ``executor`` (optional) reuses an existing :class:`PlanExecutor`
    (bridge process) across checkpoint boundaries instead of spawning a
    fresh one per call; plans fully reconfigure every environment they
    touch, so results are bit-identical to a fresh executor. The caller
    owns its lifecycle when it passes one (it is NOT closed here).

    ``profiler`` (optional) records ``eval.battery.*`` buckets for the
    training profile: per-section wall time, prediction and environment
    stepping, plus episode counters.
    """
    started = time.monotonic() if profiler is not None else 0.0
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policy_path = destination / "policy.zip"
    model.save(str(policy_path))

    eval_seed = config.seed + EVAL_MASTER_SEED_SALT
    # The battery runs planned episodes whose scheduling is
    # result-invariant (see PlanExecutor), so it may use more bridge
    # environments than the strictly-serial normal evaluation: this
    # amortises per-step prediction and transport overhead across the
    # batch without changing a single episode. evaluation_environment_count
    # stays the knob for the normal evaluation's exact serial semantics.
    battery_env_count = int(getattr(config, "checkpoint_eval_environment_count", 1) or 1)
    env_kwargs = {
        "project_path": str(config.project),
        "godot_executable": config.godot_executable,
        "environment_count": battery_env_count,
        "enemy_count": config.enemy_count,
        "seed": eval_seed,
        "curriculum_level": config.curriculum_level,
    }
    policy_id = f"{pipeline.policy_id if pipeline else 'policy'}@step{step}"

    report: dict[str, Any] = {
        "timesteps": step,
        "policy": str(policy_path),
        "policy_id": policy_id,
        "normal_evaluation": normal_summary,
    }
    record_eval_replays = config.replay_mode == "all"
    record_gen_replays = config.replay_mode in ("all", "evaluation")

    executor_factory = executor_factory or PlanExecutor
    need_executor = config.checkpoint_condition_eval or config.checkpoint_generalization_eval
    owned_executor = executor is None and need_executor
    if executor is None and need_executor:
        # Profiling views are injected by the caller's executor_factory (see
        # the training callback), keeping this factory signature unchanged.
        executor = executor_factory(env_kwargs, skill_metrics=True)
    try:
        if config.checkpoint_condition_eval and executor is not None:
            section_started = time.monotonic() if profiler is not None else 0.0
            plans = _condition_eval_plans(config, config.condition_eval_episodes)
            for plan in plans:
                if record_eval_replays:
                    object.__setattr__(plan, "record_replay", True)
            rows = executor.run(
                model,
                sorted(plans, key=lambda plan: plan.labels["_position"]),
                policy_id=policy_id,
                checkpoint=str(policy_path),
                replay_dir=destination / "replays" if record_eval_replays else None,
            )
            if profiler is not None:
                profiler.record("eval.battery.condition", time.monotonic() - section_started)
                profiler.add("eval.battery.episodes", len(rows))
            tracker = ConditionTracker(window=len(plans) + 1)
            for row in rows:
                condition = Condition(**row["condition"])
                tracker.record(
                    condition,
                    float(row.get("episode_reward", 0.0)),
                    bool(row.get("win", False)),
                    int(row.get("episode_length", 0)),
                )
            cond_report = generalization_report(tracker, top_n=5)
            cond_report["episodes_detail"] = rows
            cond_report["distribution"] = "union_of_curriculum_ladder_eval_seeds"
            report["condition_evaluation"] = cond_report

        if config.checkpoint_generalization_eval and executor is not None:
            section_started = time.monotonic() if profiler is not None else 0.0
            suite, note, trained_level = _generalization_plans(config, pipeline, record_gen_replays)
            if suite is None:
                report["generalization"] = {
                    "skipped": True,
                    "reason": note,
                    "episodes": 0,
                    "trained_level": trained_level,
                }
            else:
                planned: list[PlannedEpisode] = []
                for position, episode in enumerate(suite.plan()):
                    # The suite level is a real world level by construction
                    # (>= WORLD_MIN_LEVEL), so applied_condition is the
                    # identity here except for the multi-enemy clamp; the
                    # labels stay the suite's own (trained) ones.
                    condition = applied_condition(episode.condition)
                    labels = episode.labels()
                    labels["_position"] = position
                    labels["suite_level"] = condition.level
                    planned.append(
                        PlannedEpisode(
                            condition=condition,
                            labels=labels,
                            record_replay=record_gen_replays,
                        )
                    )
                rows = executor.run(
                    model,
                    planned,
                    policy_id=policy_id,
                    checkpoint=str(policy_path),
                    replay_dir=destination / "replays" if record_gen_replays else None,
                )
                if profiler is not None:
                    profiler.record("eval.battery.generalization", time.monotonic() - section_started)
                    profiler.add("eval.battery.episodes", len(rows))
                plan_episodes = suite.plan()
                for position, row in enumerate(rows):
                    extra = {}
                    if row.get("skill"):
                        aim = row["skill"].get("aim", {})
                        extra["aim_accuracy"] = aim.get("accuracy", 0.0)
                    suite.record(
                        plan_episodes[position],
                        won=bool(row.get("win", False)),
                        reward=float(row.get("episode_reward", 0.0)),
                        steps=int(row.get("episode_length", 0)),
                        extra=extra,
                    )
                generalization = suite.report()
                generalization["suite_level"] = suite.level
                generalization["trained_level"] = trained_level
                generalization["episodes_detail"] = rows
                report["generalization"] = generalization
                suite.export(destination)
        elif config.checkpoint_generalization_eval:
            report["generalization"] = {"skipped": True, "reason": "executor unavailable"}
    finally:
        if owned_executor:
            executor.close()

    if config.checkpoint_league_eval and config.league_matches_per_checkpoint > 0:
        section_started = time.monotonic() if profiler is not None else 0.0
        runner = LeagueRunner(pipeline.run_dir if pipeline else destination, seed=config.seed, device=device)
        league_env_kwargs = dict(env_kwargs)
        report["league"] = runner.evaluate_checkpoint(
            model,
            experiment_id=(config.experiment_id or config.run_id or "policy"),
            step=step,
            matches=config.league_matches_per_checkpoint,
            env_kwargs=league_env_kwargs,
            max_opponents=config.league_max_opponents,
            client_factory=client_factory,
        )
        if profiler is not None:
            profiler.record("eval.battery.league", time.monotonic() - section_started)
    else:
        report["league"] = {"enabled": False}

    if pipeline is not None:
        report["training_metrics"] = pipeline.skill_metrics.flush_aggregate()
        report["training_conditions"] = pipeline.driver.tracker.report()
        report["curriculum"] = pipeline.driver.curriculum_snapshot()
        report["replays_saved"] = pipeline.replays.saved
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if profiler is not None:
        profiler.record("eval.battery.total", time.monotonic() - started)
    return report

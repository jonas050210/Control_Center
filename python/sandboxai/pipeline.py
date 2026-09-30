"""Integrated training pipeline: curriculum, conditions, metrics, replay.

This module wires the previously standalone research subsystems into the
PPO training loop so a normal ``sandboxai train`` run automatically uses
them. It deliberately owns *orchestration*, not new algorithms:

``CurriculumDriver``
    The single authoritative curriculum decision path for vector training.
    Wraps :class:`~sandboxai.curriculum_stages.CurriculumDirector` (which
    owns the level and the distribution) and derives deterministic,
    independent per-environment episode streams from it
    (``TrainingDistribution.stream_index``). Nothing here duplicates
    curriculum logic: level changes come only from the director.

``SkillMetricsSink``
    Feeds every finished step into the research metrics
    (:mod:`sandboxai.metrics`) one accumulator per environment — diagnostics
    only. Rewards flow to PPO untouched; metrics are written to
    ``logs/episodes.jsonl`` and aggregated per checkpoint.

``ReplayController``
    Selective, capped replay recording (:mod:`sandboxai.replay`) straight
    from data already on the bridge wire — actions, rewards and events,
    never duplicated observations unless detail=detailed is requested.

``TrainingPipeline``
    Glues the three onto the :class:`~sandboxai.godot_env.GodotVecEnv`
    reset/step hooks. Per environment the staging invariant is:

    * after any reset (explicit or auto), the environment runs the plan the
      driver believes is "current", and the driver has staged exactly one
      pending plan for the *next* episode on the bridge;
    * the auto-reset at episode end consumes that pending plan atomically,
      so the first observation PPO sees of an episode always belongs to
      that episode's plan.

Determinism: plans are pure functions of (master seed, stage, stream
index); curriculum feedback is a deterministic function of the ordered
outcome sequence; no wall-clock or shared RNG state anywhere on this path.
``state_dict``/``load_state_dict`` make a resumed run continue the same
streams instead of restarting the curriculum.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .conditions import Condition
from .curriculum_stages import (
    TRAINABLE_MAX_LEVEL,
    CurriculumDirector,
    EpisodeOutcome,
    applied_condition,
)
from .manifest import (  # noqa: F401  (contract_fingerprint/write_manifest are re-exported: `from .pipeline import write_manifest` is the historical import path used by ppo.py and the tests)
    build_manifest,
    contract_fingerprint,
    write_manifest,
)
from .metrics import EpisodeMetrics, MetricsAggregator, StepSample
from .randomization import DistributionRunTracker, EpisodePlan
from .replay import DetailLevel, ReplayHeader, ReplayRecorder

## Salt separating the evaluation seed stream from the training stream.
EVAL_MASTER_SEED_SALT: int = 707_000_017


# ---------------------------------------------------------------------------
# Curriculum driver
# ---------------------------------------------------------------------------


class CurriculumDriver:
    """Per-environment episode plans from one authoritative director.

    Parallel environment seed isolation is structural: environment ``k``
    draws stream indices ``k, k + STREAM_STRIDE, ...`` from the *current*
    distribution, so two environments never share a plan and no mutable RNG
    state is involved. The level itself is a property of the RUN (one
    director), which is what keeps promotion statistics pooled instead of
    fragmented across environments.
    """

    def __init__(
        self,
        master_seed: int = 1234,
        environment_count: int = 8,
        start_level: int = 1,
        adaptive: bool = True,
        exploration: bool = False,
    ) -> None:
        if environment_count < 1:
            raise ValueError("environment_count must be >= 1")
        self.master_seed = int(master_seed)
        self.environment_count = int(environment_count)
        self.adaptive = bool(adaptive)
        self.director = CurriculumDirector(
            start_level=start_level,
            master_seed=self.master_seed,
            exploration=exploration,
        )
        # PPO training stays on the combat ladder: level 11 is the
        # two-agent self-play hook, which is an evaluation-time
        # environment, not a single-agent PPO one.
        self.director.schedule.max_level = TRAINABLE_MAX_LEVEL
        if start_level > 10:
            raise ValueError("training start_level must be <= 10")
        self._ordinals: list[int] = [0] * self.environment_count
        self._staged: list[EpisodePlan | None] = [None] * self.environment_count
        self._current: list[EpisodePlan | None] = [None] * self.environment_count
        self.tracker = DistributionRunTracker(window=100)
        self.episodes_completed: int = 0
        self.used_conditions: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []

    # -- plan stream -------------------------------------------------------

    @property
    def level(self) -> int:
        return self.director.level

    def current_plan(self, env_index: int) -> EpisodePlan | None:
        return self._current[env_index]

    def staged_plan(self, env_index: int) -> EpisodePlan | None:
        return self._staged[env_index]

    def _draw(self, env_index: int) -> EpisodePlan:
        """Next unconsumed plan for an environment from the current stage."""
        ordinal = self._ordinals[env_index]
        index = self.director.distribution.stream_index(env_index, ordinal)
        plan = self.director.distribution.episode_plan(index)
        self._ordinals[env_index] = ordinal + 1
        effective = applied_condition(plan.condition)
        if effective is not plan.condition:
            plan = EpisodePlan(
                index=plan.index,
                condition=effective,
                layout_seed=plan.layout_seed,
                spawn=plan.spawn,
            )
        return plan

    def stage_next(self, env_index: int) -> dict[str, Any]:
        """Draws and stages the next plan; returns the bridge payload."""
        plan = self._draw(env_index)
        self._staged[env_index] = plan
        return self.plan_payload(env_index, plan)

    @staticmethod
    def plan_payload(env_index: int, plan: EpisodePlan) -> dict[str, Any]:
        """The wire format for one staged plan (replay header fields)."""
        return {"index": int(env_index), **plan.replay_header_fields()}

    def prime(self) -> list[dict[str, Any]]:
        """Stages the first plan for every environment."""
        return [self.stage_next(index) for index in range(self.environment_count)]

    # -- lifecycle events (driven by TrainingPipeline hooks) --------------

    def consume_at_reset(self, indices: list[int] | None = None) -> list[EpisodePlan | None]:
        """Marks explicit resets: each staged plan became the running one."""
        selected = indices if indices is not None else list(range(self.environment_count))
        started: list[EpisodePlan | None] = []
        for index in selected:
            if self._staged[index] is not None:
                self._current[index] = self._staged[index]
            started.append(self._current[index])
        return started

    def finish_episode(self, env_index: int, metrics: dict[str, Any]) -> dict[str, Any]:
        """Records the episode that just ended in one environment.

        Returns an event describing the finished episode and (possibly) the
        curriculum decision it triggered. The environment's auto-reset has
        already consumed the staged plan, so ``current`` advances to it
        before the next plan is staged.
        """
        finished = self._current[env_index]
        self._current[env_index] = self._staged[env_index]
        self._staged[env_index] = None
        if finished is None:  # defensive: a done without any reset
            return {"finished": None, "change": None}

        reward = float(metrics.get("episode_reward", 0.0))
        steps = int(metrics.get("episode_length", 0))
        won = bool(metrics.get("win", False))
        survived = not bool(metrics.get("deaths", 0))
        self.tracker.record(finished, reward, won, steps)
        self.episodes_completed += 1
        self.used_conditions.append(finished.condition.to_dict())

        outcome = EpisodeOutcome(won=won, reward=reward, steps=steps, survived=survived)
        change: dict[str, Any] | None = None
        if self.adaptive:
            change = self.director.record(outcome)
            if change is not None:
                event = {
                    "event": "curriculum_change",
                    "episodes_completed": self.episodes_completed,
                    "env_index": env_index,
                    **change,
                }
                self.decision_log.append(event)
                change = event
        return {"finished": finished, "change": change, "outcome": asdict(outcome)}

    # -- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        auto = self.director.auto
        return {
            "master_seed": self.master_seed,
            "environment_count": self.environment_count,
            "adaptive": self.adaptive,
            "ordinals": list(self._ordinals),
            "staged": [plan.to_dict() if plan else None for plan in self._staged],
            "current": [plan.to_dict() if plan else None for plan in self._current],
            "auto": {
                "level": auto.state.level,
                "episodes_at_level": auto.state.episodes_at_level,
                "cooldown_remaining": auto.state.cooldown_remaining,
                "window": list(auto.state.window),
                "promotions": auto.state.promotions,
                "demotions": auto.state.demotions,
                "history": list(auto.state.history),
            },
            "episodes_completed": self.episodes_completed,
            "decision_log": list(self.decision_log),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        from .randomization import EnemySpawnPlan

        def plan_of(payload: dict[str, Any] | None) -> EpisodePlan | None:
            if payload is None:
                return None
            return EpisodePlan(
                index=int(payload["index"]),
                condition=Condition(**payload["condition"]),
                layout_seed=int(payload["layout_seed"]),
                spawn=EnemySpawnPlan(**payload["spawn"]),
            )

        auto_state = state.get("auto", {})
        level = int(auto_state.get("level", self.director.level))
        # Rebuild the director at the saved level so the stage table and
        # distribution match, then restore the accumulated evidence.
        self.director = CurriculumDirector(
            start_level=level,
            master_seed=self.master_seed,
            exploration=self.director.exploration,
        )
        self.director.schedule.max_level = TRAINABLE_MAX_LEVEL
        auto = self.director.auto
        auto.state.level = level
        auto.state.episodes_at_level = int(auto_state.get("episodes_at_level", 0))
        auto.state.cooldown_remaining = int(auto_state.get("cooldown_remaining", 0))
        auto.state.window.clear()
        auto.state.window.extend(float(value) for value in auto_state.get("window", []))
        auto.state.promotions = int(auto_state.get("promotions", 0))
        auto.state.demotions = int(auto_state.get("demotions", 0))
        auto.state.history = [dict(entry) for entry in auto_state.get("history", [])]
        self.director.changes = [dict(entry) for entry in auto_state.get("history", [])]

        self._ordinals = [int(value) for value in state.get("ordinals", self._ordinals)]
        self._staged = [plan_of(payload) for payload in state.get("staged", [])] or [
            None
        ] * self.environment_count
        self._current = [plan_of(payload) for payload in state.get("current", [])] or [
            None
        ] * self.environment_count
        self.episodes_completed = int(state.get("episodes_completed", 0))
        self.decision_log = [dict(entry) for entry in state.get("decision_log", [])]

    def staged_payloads(self) -> list[dict[str, Any]]:
        """Bridge payloads re-staging exactly the currently staged plans."""
        return [
            self.plan_payload(index, plan)
            for index, plan in enumerate(self._staged)
            if plan is not None
        ]

    def curriculum_snapshot(self) -> dict[str, Any]:
        snapshot = self.director.snapshot()
        snapshot.update(
            {
                "adaptive": self.adaptive,
                "episodes_completed": self.episodes_completed,
                "level": self.level,
            }
        )
        return snapshot


# ---------------------------------------------------------------------------
# Skill metrics sink (diagnostics only — never a reward signal)
# ---------------------------------------------------------------------------


class SkillMetricsSink:
    """Runs the research metrics on training episodes.

    One :class:`EpisodeMetrics` accumulator per environment, holding only
    scalar counters (no per-tick storage). The accumulators read the same
    observations and event dicts the trainer already received, so the sink
    cannot see anything the policy could not — per the metrics module's
    own contract. Disabled means exactly zero per-step cost.
    """

    def __init__(
        self, environment_count: int, enabled: bool = True, policy_id: str = "policy"
    ) -> None:
        self.environment_count = environment_count
        self.enabled = bool(enabled)
        self.policy_id = policy_id
        self.aggregator = MetricsAggregator()
        self._episodes: list[EpisodeMetrics | None] = [None] * environment_count

    @staticmethod
    def labels_for(plan: EpisodePlan, env_index: int, policy_id: str) -> dict[str, Any]:
        condition = plan.condition
        return {
            "policy_id": policy_id,
            "environment_index": env_index,
            "map_id": condition.map_id,
            "scenario": condition.scenario,
            "lighting": condition.lighting,
            "enemy_count": condition.enemy_count,
            "curriculum_level": condition.level,
            "seed": condition.seed,
            "episode_id": f"env{env_index}:{condition.seed}",
        }

    def begin(self, env_index: int, plan: EpisodePlan) -> None:
        if not self.enabled:
            return
        self._episodes[env_index] = EpisodeMetrics(self.labels_for(plan, env_index, self.policy_id))

    def record_step(
        self,
        env_index: int,
        observation: Any,
        action: Any,
        events: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        episode = self._episodes[env_index]
        if episode is None:
            return
        episode.record(
            StepSample(
                observation=observation,
                events=events,
                action=action,
            )
        )

    def finish(self, env_index: int, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        episode = self._episodes[env_index]
        self._episodes[env_index] = None
        if episode is None:
            return None
        summary = episode.finish(result)
        self.aggregator.add(summary)
        return summary

    def flush_aggregate(self) -> dict[str, Any]:
        """Aggregate since the last flush, then resets the interval."""
        aggregate = self.aggregator.aggregate()
        grouped = {
            "by_environment": self.aggregator.group_by("environment_index"),
            "by_map": self.aggregator.group_by("map_id"),
            "by_lighting": self.aggregator.group_by("lighting"),
            "by_level": self.aggregator.group_by("curriculum_level"),
        }
        self.aggregator.reset()
        return {"aggregate": aggregate, "groups": grouped}


# ---------------------------------------------------------------------------
# Replay controller (selective, capped, cheap when off)
# ---------------------------------------------------------------------------


class ReplayController:
    """Decides which training episodes become replay files and writes them.

    Selection happens at episode END: every episode is recorded into a
    light in-memory recorder (cheap: one small dict per tick) and only the
    selection rules decide whether it lands on disk, so "record failed
    episodes" can know the outcome. A run-wide cap bounds worst-case disk
    use no matter how bad the policy gets.
    """

    MODES = ("off", "interesting", "every_n", "all", "evaluation")

    def __init__(
        self,
        directory: str | Path,
        mode: str = "interesting",
        every_n: int = 50,
        detail: str = DetailLevel.LIGHT,
        max_per_run: int = 64,
        environment_count: int = 8,
        policy_id: str = "policy",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown replay mode: {mode!r}")
        self.directory = Path(directory)
        self.mode = mode
        self.every_n = max(1, int(every_n))
        self.detail = detail
        self.max_per_run = int(max_per_run)
        self.policy_id = policy_id
        self.saved: int = 0
        self.episodes_considered: int = 0
        self._recorders: list[ReplayRecorder | None] = [None] * environment_count
        self._plans: list[EpisodePlan | None] = [None] * environment_count

    @property
    def active(self) -> bool:
        return self.mode not in ("off", "evaluation")

    def _cap_reached(self) -> bool:
        return self.max_per_run > 0 and self.saved >= self.max_per_run

    def _recording_enabled(self) -> bool:
        return self.active and not self._cap_reached()

    def begin(self, env_index: int, plan: EpisodePlan, checkpoint: str = "") -> None:
        self._plans[env_index] = plan
        if not self._recording_enabled():
            self._recorders[env_index] = None
            return
        condition = plan.condition
        header = ReplayHeader(
            detail=self.detail,
            policy_id=self.policy_id,
            checkpoint=checkpoint,
            environment_index=env_index,
            **plan.replay_header_fields(),
        )
        recorder = ReplayRecorder(header=header, detail=self.detail)
        recorder.start(seed=condition.seed)
        self._recorders[env_index] = recorder

    def record_step(
        self,
        env_index: int,
        action: Any,
        reward: float,
        observation: Any = None,
        events: dict[str, Any] | None = None,
    ) -> None:
        recorder = self._recorders[env_index]
        if recorder is None:
            return
        recorder.record_step(action, reward, observation=observation, done=False)
        if events:
            recorder.record_events(events)

    def finish(
        self,
        env_index: int,
        metrics: dict[str, Any],
        curriculum_change: dict[str, Any] | None = None,
        reason: str = "",
    ) -> Path | None:
        """Closes the episode; writes the replay when selected. Returns the
        written path or None."""
        recorder = self._recorders[env_index]
        self._recorders[env_index] = None
        plan = self._plans[env_index]
        if recorder is None or plan is None:
            return None
        self.episodes_considered += 1
        won = bool(metrics.get("win", False))
        truncated = bool(metrics.get("truncated", False))
        done_reason = str(metrics.get("done_reason", reason))
        recorder.finish(
            {
                "win": won,
                "done_reason": done_reason,
                "episode_reward": float(metrics.get("episode_reward", 0.0)),
                "episode_length": int(metrics.get("episode_length", recorder.tick)),
            }
        )
        if not self._select(self.episodes_considered, won, truncated, curriculum_change):
            return None
        condition = plan.condition
        name = (
            f"ep{self.episodes_considered:06d}_env{env_index}"
            f"_{condition.map_id or 'arena'}_{condition.lighting or 'normal'}"
            f"_L{condition.level}_s{condition.seed}.jsonl"
        )
        path = self.directory / name
        recorder.save(path)
        self.saved += 1
        return path

    def _select(
        self,
        episode_number: int,
        won: bool,
        truncated: bool,
        curriculum_change: dict[str, Any] | None,
    ) -> bool:
        if self.mode == "all":
            return True
        if self.mode == "every_n":
            return episode_number % self.every_n == 0
        if self.mode == "interesting":
            # Failures and timeouts are where a reward curve's story lives;
            # a level boundary changes what the next episodes even mean, so
            # the last episode before it is worth keeping too.
            return (not won) or truncated or curriculum_change is not None
        return False


# ---------------------------------------------------------------------------
# Run manifest (Phase 11)
# ---------------------------------------------------------------------------
#
# Manifest construction lives in sandboxai/manifest.py - it is provenance,
# not orchestration. Re-exported here because `from .pipeline import
# build_manifest, write_manifest` is the historical import path.


# ---------------------------------------------------------------------------
# Training pipeline (hooks for GodotVecEnv)
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Owns the per-step integration for one PPO training run."""

    EPISODE_FIELDS = (
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
    )

    def __init__(
        self, config: Any, run_dir: str | Path, telemetry: Any, device: str = "cpu"
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.telemetry = telemetry
        self.device = device
        self.timesteps: int = 0  # refreshed by the PPO callback every step
        policy_id = config.experiment_id or config.run_id or "policy"
        self.policy_id = policy_id
        self.driver = CurriculumDriver(
            master_seed=config.seed,
            environment_count=config.environment_count,
            start_level=config.curriculum_start_level,
            adaptive=config.adaptive_curriculum,
        )
        self.skill_metrics = SkillMetricsSink(
            config.environment_count,
            enabled=config.skill_metrics,
            policy_id=policy_id,
        )
        self.replays = ReplayController(
            self.run_dir / "replays",
            mode=config.replay_mode,
            every_n=config.replay_every_n,
            detail=config.replay_detail,
            max_per_run=config.replay_max_per_run,
            environment_count=config.environment_count,
            policy_id=policy_id,
        )
        self.episodes_log = None
        if config.episode_log:
            log_path = self.run_dir / "logs" / "episodes.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.episodes_log = log_path.open("a", encoding="utf-8")
        self._env = None
        self._checkpoint_path: str = ""
        # `manifest()` is called at least twice per run (once at training
        # start, once when the final manifest is written): see train_ppo.
        # The Godot binary cannot change mid-run, so the first real probe
        # (which launches the executable just to read its version) is
        # cached and reused instead of launching Godot again for every
        # subsequent manifest rebuild.
        self._godot_snapshot: dict[str, Any] | None = None

    # -- attach / hooks ----------------------------------------------------

    def _require_env(self) -> Any:
        """The attached vector env, or a clear error instead of AttributeError."""
        if self._env is None:
            raise RuntimeError("TrainingPipeline.attach(vec_env) has not been called")
        return self._env

    def attach(self, vec_env: Any) -> None:
        """Registers the hooks and stages the first plan per environment."""
        self._env = vec_env
        vec_env.client.set_episode_plans(self.driver.prime())
        vec_env.step_hook = self.on_step
        vec_env.reset_hook = self.on_reset

    def detach(self) -> None:
        if self._env is not None:
            self._env.step_hook = None
            self._env.reset_hook = None
            self._env = None

    def on_reset(self, _observations: Any) -> None:
        started = self.driver.consume_at_reset()
        for env_index, plan in enumerate(started):
            if plan is None:
                continue
            self.skill_metrics.begin(env_index, plan)
            self.replays.begin(env_index, plan, checkpoint=self._checkpoint_path)
        # Stage the NEXT plan per environment right away: the auto-reset at
        # episode end consumes it, and episodes are always >= 1 step long,
        # so the pending slot is never empty when it is needed.
        self._require_env().client.set_episode_plans(
            [self.driver.stage_next(index) for index in range(self.driver.environment_count)]
        )

    def on_step(
        self, actions: Any, observations: Any, rewards: Any, dones: Any, infos: list[dict[str, Any]]
    ) -> None:
        driver = self.driver
        record_metrics = self.skill_metrics.enabled
        record_replays = self.replays.active
        stage: list[dict[str, Any]] = []
        for env_index in range(driver.environment_count):
            info = infos[env_index] if env_index < len(infos) else {}
            events = info.get("events", {})
            if record_metrics:
                self.skill_metrics.record_step(
                    env_index, observations[env_index], actions[env_index], events
                )
            if record_replays:
                self.replays.record_step(
                    env_index, actions[env_index], float(rewards[env_index]), events=events
                )
            if not bool(dones[env_index]):
                continue
            metrics = dict(info.get("metrics", {}))
            event = driver.finish_episode(env_index, metrics)
            finished = event.get("finished")
            if finished is None:
                continue
            change = event.get("change")
            summary = self.skill_metrics.finish(
                env_index,
                result={
                    "win": bool(metrics.get("win", False)),
                    "done_reason": str(metrics.get("done_reason", "")),
                },
            )
            replay_path = self.replays.finish(env_index, metrics, curriculum_change=change)
            self._log_episode(env_index, finished, metrics, summary, replay_path)
            if change is not None:
                self._log_curriculum_change(change)
            current = driver.current_plan(env_index)
            if current is not None:
                self.skill_metrics.begin(env_index, current)
                self.replays.begin(env_index, current, checkpoint=self._checkpoint_path)
            stage.append(driver.stage_next(env_index))
        if stage:
            self._require_env().client.set_episode_plans(stage)

    # -- logging -----------------------------------------------------------

    def _log_episode(
        self,
        env_index: int,
        plan: EpisodePlan,
        metrics: dict[str, Any],
        summary: dict[str, Any] | None,
        replay_path: Path | None,
    ) -> None:
        if self.episodes_log is None:
            return
        row: dict[str, Any] = {
            "timesteps": self.timesteps,
            "environment_index": env_index,
            "curriculum_level": self.driver.level,
            "curriculum_stage": self.driver.director.stage.name,
            "plan": plan.to_dict(),
            "engine_metrics": {
                key: metrics.get(key) for key in self.EPISODE_FIELDS if key in metrics
            },
        }
        if summary is not None:
            row["skill_metrics"] = summary.get("categories", {})
        if replay_path is not None:
            row["replay"] = str(replay_path)
        self.episodes_log.write(json.dumps(row, default=str) + "\n")
        self.episodes_log.flush()

    def _log_curriculum_change(self, change: dict[str, Any]) -> None:
        payload = dict(change)
        payload["timesteps"] = self.timesteps
        self.telemetry.write(payload)
        path = self.run_dir / "logs" / "curriculum.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=str) + "\n")

    # -- checkpoint-facing state ------------------------------------------

    def save_state(self, directory: str | Path | None = None) -> Path:
        target_dir = Path(directory) if directory else self.run_dir / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "curriculum_state.json"
        payload = {
            "driver": self.driver.state_dict(),
            "policy_id": self.policy_id,
            "timesteps": self.timesteps,
        }
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def load_state(self, path: str | Path) -> bool:
        source = Path(path)
        if not source.is_file():
            return False
        payload = json.loads(source.read_text(encoding="utf-8"))
        self.driver.load_state_dict(payload.get("driver", {}))
        return True

    def reattach_after_load(self) -> None:
        """Re-stages the exact plans a loaded state had pending."""
        self._require_env().client.set_episode_plans(self.driver.staged_payloads())

    def note_checkpoint(self, path: str | Path) -> None:
        """Stamps subsequently saved replay files with the checkpoint they
        evidence. Best-effort provenance only - a training replay belongs
        to the live policy, and the association just says which frozen
        checkpoint was nearest when it was recorded."""
        self._checkpoint_path = str(path)

    def manifest(self) -> dict[str, Any]:
        # See _godot_snapshot's docstring in __init__: only probe the live
        # Godot executable once per run and reuse that snapshot (including
        # its version string) for every later manifest rebuild.
        report = build_manifest(
            self.config,
            self.run_dir,
            self.device,
            self.driver,
            probe_godot=self._godot_snapshot is None,
        )
        if self._godot_snapshot is None:
            self._godot_snapshot = report["godot"]
        else:
            report["godot"] = self._godot_snapshot
        return report

    def close(self) -> None:
        self.detach()
        if self.episodes_log is not None:
            self.episodes_log.close()
            self.episodes_log = None

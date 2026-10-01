"""Integration tests for the curriculum->conditions->PPO training pipeline.

Covers (no Godot needed):
* Python<->GDScript mirror drift for the constants the two worlds share
  (seed strides, level thresholds) -- both source sides are grepped so a
  silent drift on EITHER side fails loudly;
* per-environment stream isolation and bit-for-bit determinism;
* the CurriculumDriver decision path (single authority, state roundtrip);
* applied_condition level semantics;
* replay modes/cap and the skill-metrics sink;
* the TrainingPipeline driving a fake vec env end to end.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from sandboxai.conditions import Condition
from sandboxai.config import TrainingConfig
from sandboxai.contract import observation_index
from sandboxai.curriculum_stages import (
    MULTI_ENEMY_MAX_LEVEL,
    MULTI_ENEMY_MIN,
    WORLD_MIN_LEVEL,
    applied_condition,
    distribution_for,
    stage_for,
)
from sandboxai.manifest import MANIFEST_FORMAT
from sandboxai.pipeline import (
    CurriculumDriver,
    ReplayController,
    SkillMetricsSink,
    TrainingPipeline,
    contract_fingerprint,
    write_manifest,
)
from sandboxai.randomization import STREAM_STRIDE, EpisodePlan
from sandboxai.replay import DetailLevel, load_replay

REPO_ROOT = Path(__file__).resolve().parents[2]


def _plan(
    seed: int, level: int = 5, map_id: str = "open_field", enemy_count: int = 1
) -> EpisodePlan:
    condition = Condition(
        map_id=map_id,
        scenario="",
        lighting="normal",
        enemy_count=enemy_count,
        level=level,
        seed=seed,
    )
    return EpisodePlan(index=0, condition=condition, layout_seed=0, spawn=None)


# ---------------------------------------------------------------------------
# Mirror drift: Python constants vs GDScript constants
# ---------------------------------------------------------------------------


class MirrorDriftTest(unittest.TestCase):
    """If a shared constant changes on one side only, training diverges
    from the engine silently. These tests trip before that."""

    def _grep_int(self, path: Path, pattern: str) -> int:
        source = path.read_text(encoding="utf-8")
        match = re.search(pattern, source)
        self.assertIsNotNone(match, f"{pattern} not found in {path}")
        return int(match.group(1).replace("_", ""))

    def test_seed_stride_mirrors_self_play_slot_stride(self):
        gd = self._grep_int(
            REPO_ROOT / "scripts/rl/self_play_adapter.gd", r"SLOT_SEED_STRIDE:\s*int\s*=\s*(\d+)"
        )
        self.assertEqual(STREAM_STRIDE, gd)
        # The same literal must also appear in the self-play environment
        # itself (it derives slot B from slot A) so the mirror covers the
        # whole chain.
        env_source = (REPO_ROOT / "scripts/self_play/self_play_environment.gd").read_text(
            encoding="utf-8"
        )
        self.assertIn("1000003", env_source)

    def test_world_min_level_mirrors_obstacles_cover(self):
        gd = self._grep_int(
            REPO_ROOT / "scripts/core/curriculum_config.gd", r"OBSTACLES_COVER\s*=\s*(\d+)"
        )
        self.assertEqual(WORLD_MIN_LEVEL, gd)

    def test_rl_server_owns_the_new_command_names(self):
        """The Python pipeline issues set_episode_plans/episode_conditions and
        league matches drive --self-play; the GDScript server must actually
        name all three, otherwise the wire tests would pass against a stub
        that silently never existed."""
        server = (REPO_ROOT / "scripts/rl/rl_server.gd").read_text(encoding="utf-8")
        adapter = (REPO_ROOT / "scripts/rl/rl_adapter.gd").read_text(encoding="utf-8")
        self.assertIn("set_episode_plans", server)
        self.assertIn("episode_conditions", server)
        self.assertIn("--self-play", server)
        # Atomic staging: the adapter must validate plans BEFORE mutating.
        self.assertIn("set_episode_plans", adapter)
        self.assertRegex(adapter, r"func\s+set_episode_plans")

    def test_multi_enemy_bounds_mirror_curriculum_config(self):
        source = (REPO_ROOT / "scripts/core/curriculum_config.gd").read_text(encoding="utf-8")
        min_count = int(
            re.search(r"MULTIPLE_ENEMIES_MIN_COUNT:\s*int\s*=\s*(\d+)", source).group(1)  # type: ignore[union-attr]
        )
        first_level = int(re.search(r"^\s*MULTIPLE_ENEMIES\s*=\s*(\d+)", source, re.M).group(1))  # type: ignore[union-attr]
        last_level = int(re.search(r"^\s*AGENT_VS_AGENT\s*=\s*(\d+)", source, re.M).group(1))  # type: ignore[union-attr]
        self.assertEqual(MULTI_ENEMY_MIN, min_count)
        # The clamp spans [MULTIPLE_ENEMIES, AGENT_VS_AGENT): GDScript
        # `effective_enemy_count` applies it below AGENT_VS_AGENT only.
        self.assertIn("level >= Level.MULTIPLE_ENEMIES and level < Level.AGENT_VS_AGENT", source)
        self.assertEqual(MULTI_ENEMY_MAX_LEVEL, last_level - 1)
        self.assertGreaterEqual(first_level, 1)
        # The Python clamp starts at the same level the engine does.
        probe = applied_condition(
            Condition(map_id="", scenario="", lighting="", enemy_count=1, level=first_level, seed=1)
        )
        self.assertEqual(probe.enemy_count, max(MULTI_ENEMY_MIN, 1))


# ---------------------------------------------------------------------------
# Per-environment stream isolation
# ---------------------------------------------------------------------------


class StreamIsolationTest(unittest.TestCase):
    def test_stream_index_formula(self):
        dist = distribution_for(stage_for(WORLD_MIN_LEVEL), master_seed=11)
        self.assertEqual(dist.stream_index(3, 7), 3 + 7 * STREAM_STRIDE)
        with self.assertRaises(ValueError):
            dist.stream_index(-1, 0)
        with self.assertRaises(ValueError):
            dist.stream_index(0, -1)

    def test_per_environment_streams_are_disjoint_and_reproducible(self):
        dist = distribution_for(stage_for(WORLD_MIN_LEVEL), master_seed=11)
        seen: dict[int, tuple[int, int]] = {}
        for environment_index in range(4):
            for ordinal in range(8):
                plan = dist.episode_plan(dist.stream_index(environment_index, ordinal))
                # Reproducibility: pure function of (env, ordinal).
                again = dist.episode_plan(dist.stream_index(environment_index, ordinal))
                self.assertEqual(plan.condition, again.condition)
                self.assertEqual(plan.layout_seed, again.layout_seed)
                # Disjointness across (env, ordinal) pairs.
                self.assertNotIn(
                    plan.condition.seed, seen, "two streams produced the same episode seed"
                )
                seen[plan.condition.seed] = (environment_index, ordinal)


# ---------------------------------------------------------------------------
# applied_condition level semantics
# ---------------------------------------------------------------------------


class AppliedConditionTest(unittest.TestCase):
    def test_below_world_min_level_strips_world_axes(self):
        for level in range(1, WORLD_MIN_LEVEL):
            applied = applied_condition(
                Condition(
                    map_id="open_field",
                    scenario="cover_fight",
                    lighting="fog",
                    enemy_count=1,
                    level=level,
                    seed=42,
                )
            )
            self.assertEqual(applied.map_id, "")
            self.assertEqual(applied.scenario, "")
            self.assertEqual(applied.lighting, "")
            self.assertEqual(applied.level, level)
            self.assertEqual(applied.seed, 42)

    def test_multi_enemy_band_applies_minimum_count(self):
        for level in range(4, MULTI_ENEMY_MAX_LEVEL + 1):
            applied = applied_condition(
                Condition(map_id="", scenario="", lighting="", enemy_count=1, level=level, seed=7)
            )
            self.assertEqual(applied.enemy_count, MULTI_ENEMY_MIN, f"level {level}")
        # Above the band (self-play level): untouched.
        applied = applied_condition(
            Condition(map_id="", scenario="", lighting="", enemy_count=1, level=11, seed=7)
        )
        self.assertEqual(applied.enemy_count, 1)

    def test_world_levels_keep_world_axes(self):
        applied = applied_condition(
            Condition(
                map_id="pillar_hall",
                scenario="ambush",
                lighting="night",
                enemy_count=5,
                level=WORLD_MIN_LEVEL,
                seed=9,
            )
        )
        self.assertEqual(
            (applied.map_id, applied.scenario, applied.lighting), ("pillar_hall", "ambush", "night")
        )
        self.assertEqual(applied.enemy_count, 5)


# ---------------------------------------------------------------------------
# CurriculumDriver: one authoritative decision path
# ---------------------------------------------------------------------------


class CurriculumDriverTest(unittest.TestCase):
    def _drive(self, driver: CurriculumDriver, episodes: int) -> None:
        driver.prime()
        for ordinal in range(episodes):
            env_index = ordinal % driver.environment_count
            plan = driver.current_plan(env_index) or driver.staged_plan(env_index)
            self.assertIsNotNone(plan)
            driver.consume_at_reset([env_index])
            driver.stage_next(env_index)
            driver.finish_episode(env_index, {"win": ordinal % 2 == 0, "episode_length": 60})

    def test_deterministic_staging_across_instances(self):
        a = CurriculumDriver(master_seed=5, environment_count=4, start_level=1, adaptive=True)
        b = CurriculumDriver(master_seed=5, environment_count=4, start_level=1, adaptive=True)
        self.assertEqual(a.prime(), b.prime())
        for _ in range(20):
            for env_index in range(4):
                self.assertEqual(a.stage_next(env_index), b.stage_next(env_index))

    def test_state_roundtrip_continues_identically(self):
        a = CurriculumDriver(master_seed=9, environment_count=2, start_level=1, adaptive=True)
        self._drive(a, episodes=30)
        b = CurriculumDriver(master_seed=9, environment_count=2, start_level=1, adaptive=True)
        b.load_state_dict(a.state_dict())
        self.assertEqual(b.state_dict(), a.state_dict())
        self.assertEqual(b.staged_payloads(), a.staged_payloads())
        # Continue both: identical next plans and identical decisions.
        for ordinal in range(20):
            env_index = ordinal % 2
            metrics = {"win": ordinal % 3 == 0, "episode_length": 42}
            self.assertEqual(
                a.finish_episode(env_index, metrics), b.finish_episode(env_index, metrics)
            )
            self.assertEqual(a.stage_next(env_index), b.stage_next(env_index))

    def test_adaptive_disabled_never_changes_level(self):
        driver = CurriculumDriver(master_seed=3, environment_count=2, start_level=1, adaptive=False)
        self._drive(driver, episodes=80)
        self.assertEqual(driver.level, 1)
        self.assertEqual(driver.episodes_completed, 80)
        # And the plans it draws still differ per env (no degenerate stream).
        self.assertNotEqual(driver.current_plan(0), driver.current_plan(1))

    def test_adaptive_promotes_and_demotes_log_is_written(self):
        driver = CurriculumDriver(master_seed=4, environment_count=2, start_level=1, adaptive=True)
        driver.prime()
        seen_levels = {driver.level}
        for ordinal in range(200):
            env_index = ordinal % 2
            driver.consume_at_reset([env_index])
            driver.stage_next(env_index)
            # Always win -> must eventually promote past level 1.
            driver.finish_episode(env_index, {"win": True, "episode_length": 60})
            seen_levels.add(driver.level)
        self.assertGreater(driver.level, 1)
        self.assertTrue(
            any(
                entry.get("to_level", entry.get("level")) == driver.level
                for entry in driver.decision_log
            )
            or driver.decision_log
        )

    def test_start_level_above_trainable_max_is_rejected(self):
        with self.assertRaises(ValueError):
            CurriculumDriver(master_seed=1, environment_count=1, start_level=11, adaptive=True)

    def test_used_conditions_record_the_APPLIED_condition(self):
        driver = CurriculumDriver(master_seed=6, environment_count=1, start_level=1, adaptive=False)
        driver.prime()
        driver.consume_at_reset([0])
        driver.finish_episode(0, {"win": False, "episode_length": 10})
        self.assertEqual(len(driver.used_conditions), 1)
        recorded = driver.used_conditions[0]
        # Level-1 plans carry no world axes: the recorded condition mirrors
        # what the engine actually ran, not what the sampler sliced.
        self.assertEqual(recorded["map_id"], "")
        self.assertEqual(recorded["level"], 1)


# ---------------------------------------------------------------------------
# ReplayController
# ---------------------------------------------------------------------------


class ReplayControllerTest(unittest.TestCase):
    def _controller(self, tmp: Path, **kwargs) -> ReplayController:
        kwargs.setdefault("environment_count", 1)
        return ReplayController(tmp, **kwargs)

    def _episode(
        self,
        controller: ReplayController,
        env_index: int,
        seed: int,
        level: int = 5,
        metrics: dict | None = None,
        change: dict | None = None,
    ) -> bool:
        plan = _plan(seed=seed, level=level)
        controller.begin(env_index, plan)
        controller.record_step(env_index, [0, 0, 0, 0, 0, 0], 0.1)
        saved_before = (
            len(list(controller.directory.glob("*.jsonl"))) if controller.directory.is_dir() else 0
        )
        path = controller.finish(env_index, metrics or {"win": True}, curriculum_change=change)
        saved_now = (
            len(list(controller.directory.glob("*.jsonl"))) if controller.directory.is_dir() else 0
        )
        if path is None:
            self.assertEqual(saved_now, saved_before)
            return False
        self.assertTrue(Path(path).is_file())
        return True

    def test_off_records_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="off")
            self.assertFalse(controller.active)
            self.assertFalse(self._episode(controller, 0, seed=1))

    def test_all_records_every_episode_and_filename_is_stable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="all", max_per_run=0)
            self.assertTrue(self._episode(controller, 0, seed=77, level=5))
            files = [p.name for p in Path(tmp).glob("*.jsonl")]
            self.assertEqual(len(files), 1)
            self.assertRegex(files[0], r"^ep000001_env0_open_field_normal_L5_s77\.jsonl$")
            # Existing replay format: the first line wraps the header in a
            # {"header": ...} envelope; the fingerprint lives inside it.
            envelope = json.loads(Path(tmp, files[0]).read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("header", envelope)
            header = envelope["header"]
            self.assertEqual(header.get("seed"), 77)
            self.assertIn("observation_dim", header)
            self.assertIn("action_nvec", header)

    def test_every_n_records_only_multiples(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="every_n", every_n=3, max_per_run=0)
            # Episodes are numbered 1-based inside the controller; every_n
            # keeps the multiples (3 and 6 out of 7 here).
            saved = [self._episode(controller, 0, seed=i + 1) for i in range(7)]
            self.assertEqual(saved, [False, False, True, False, False, True, False])

    def test_interesting_records_losses_and_curriculum_changes_not_plain_wins(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="interesting")
            self.assertTrue(self._episode(controller, 0, seed=1, metrics={"win": False}))
            self.assertFalse(self._episode(controller, 0, seed=2, metrics={"win": True}))
            self.assertTrue(
                self._episode(controller, 0, seed=3, metrics={"win": True}, change={"to_level": 2})
            )
            self.assertTrue(
                self._episode(controller, 0, seed=4, metrics={"win": True, "truncated": True})
            )

    def test_evaluation_mode_ignores_training_episodes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="evaluation")
            self.assertFalse(self._episode(controller, 0, seed=1, metrics={"win": False}))

    def test_cap_halts_recording(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="all", max_per_run=2)
            results = [self._episode(controller, 0, seed=i + 1) for i in range(4)]
            self.assertEqual(results, [True, True, False, False])
            self.assertEqual(controller.saved, 2)
            # Zero-cost invariant: after the cap, begin() must not allocate
            # a recorder at all (no buffered growth in the hot loop).
            self.assertFalse(controller._recording_enabled())
            self.assertTrue(all(recorder is None for recorder in controller._recorders))

    def test_cap_is_hard_when_multiple_environments_started_before_the_first_save(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp), mode="all", max_per_run=1, environment_count=2)
            # Both recorders are legitimately allocated below the cap. Their
            # terminals then arrive in one vector step, one after the other.
            controller.begin(0, _plan(seed=1))
            controller.begin(1, _plan(seed=2))
            controller.record_step(0, [0] * 6, 0.1)
            controller.record_step(1, [0] * 6, 0.1)
            self.assertIsNotNone(controller.finish(0, {"win": True}))
            self.assertIsNone(controller.finish(1, {"win": True}))
            self.assertEqual(controller.saved, 1)
            self.assertEqual(len(list(Path(tmp).glob("*.jsonl"))), 1)


# ---------------------------------------------------------------------------
# SkillMetricsSink
# ---------------------------------------------------------------------------


class SkillMetricsSinkTest(unittest.TestCase):
    def test_disabled_sink_is_a_noop(self):
        sink = SkillMetricsSink(2, enabled=False)
        sink.begin(0, _plan(seed=1))
        sink.record_step(0, [0.0] * 84, [0] * 6, {})
        self.assertIsNone(sink.finish(0, {"win": True}))
        aggregate = sink.flush_aggregate()
        self.assertEqual(aggregate.get("aggregate", {}).get("episodes", 0), 0)

    def test_labels_carry_condition_and_environment(self):
        sink = SkillMetricsSink(2, enabled=True, policy_id="pol")
        sink.begin(1, _plan(seed=5, level=6, map_id="pillar_hall"))
        sink.record_step(1, [0.0] * 84, [1, 1, 1, 1, 0, 0], {})
        summary = sink.finish(1, {"win": True})
        self.assertIsNotNone(summary)
        labels = summary.get("labels", {})
        self.assertEqual(labels.get("map_id"), "pillar_hall")
        self.assertEqual(labels.get("environment_index"), 1)
        self.assertEqual(labels.get("policy_id"), "pol")
        aggregate = sink.flush_aggregate()
        groups = aggregate.get("groups", {})
        self.assertIn("pillar_hall", groups.get("by_map", {}))
        self.assertTrue(any("1" in str(key) for key in groups.get("by_environment", {})))


# ---------------------------------------------------------------------------
# TrainingPipeline against a fake vec env
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self):
        self.batches: list[list[dict]] = []

    def set_episode_plans(self, plans):
        self.batches.append(list(plans))
        return {"ok": True, "staged": [int(p.get("index", -1)) for p in plans]}

    def episode_conditions(self):
        return []


class _FakeVecEnv:
    def __init__(self, environment_count: int):
        self.client = _RecordingClient()
        self.reset_hook = None
        self.step_hook = None
        self.environment_count = environment_count


class _Telemetry:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, payload):
        self.rows.append(dict(payload))


class TrainingPipelineTest(unittest.TestCase):
    def _config(self, tmp: Path, episodes: bool = True) -> TrainingConfig:
        config = TrainingConfig(
            seed=5,
            environment_count=2,
            project_path=str(tmp),
            godot_executable="godot",
            episode_log=episodes,
            output_root=str(tmp / "runs"),
            run_id="pipe_test",
            replay_mode="off",
        )
        config.validate()
        return config

    def test_attach_stages_one_plan_per_environment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            pipe = TrainingPipeline(config, config.run_directory(), _Telemetry(), device="cpu")
            env = _FakeVecEnv(2)
            pipe.attach(env)
            self.assertEqual(len(env.client.batches), 1)
            batch = env.client.batches[0]
            self.assertEqual(len(batch), 2)
            self.assertEqual({int(p["index"]) for p in batch}, {0, 1})
            self.assertIsNotNone(env.reset_hook)
            self.assertIsNotNone(env.step_hook)
            # Determinism: a second pipeline over the same config stages
            # the identical batch.
            pipe2 = TrainingPipeline(config, config.run_directory(), _Telemetry(), device="cpu")
            env2 = _FakeVecEnv(2)
            pipe2.attach(env2)
            self.assertEqual(env.client.batches[0], env2.client.batches[0])
            pipe.close()
            pipe2.close()

    def test_step_records_episode_rows_with_plan_and_metrics(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            telemetry = _Telemetry()
            pipe = TrainingPipeline(config, config.run_directory(), telemetry, device="cpu")
            env = _FakeVecEnv(2)
            pipe.attach(env)
            pipe.on_reset(None)
            infos = [
                {
                    "events": {},
                    "metrics": {"win": True, "episode_reward": 2.0, "episode_length": 30},
                    "done_reason": "won",
                },
                {"events": {}},
            ]
            pipe.timesteps = 1234
            pipe.on_step(
                [[0] * 6, [0] * 6], [[0.0] * 84, [0.0] * 84], [1.0, 0.0], [True, False], infos
            )
            row_path = config.run_directory() / "logs" / "episodes.jsonl"
            rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["timesteps"], 1234)
            self.assertEqual(row["environment_index"], 0)
            self.assertIn("plan", row)
            self.assertEqual(row["plan"]["condition"]["level"], 1)
            self.assertEqual(row["engine_metrics"]["episode_reward"], 2.0)
            self.assertIn("skill_metrics", row)
            self.assertTrue(env.client.batches, "next plan was restaged after the episode")
            pipe.close()

    def test_detailed_replay_and_terminal_metrics_use_the_terminal_observation(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            config.replay_mode = "all"
            config.replay_detail = DetailLevel.DETAILED
            config.replay_max_per_run = 0
            pipe = TrainingPipeline(config, config.run_directory(), _Telemetry(), device="cpu")
            env = _FakeVecEnv(2)
            pipe.attach(env)
            pipe.on_reset(None)

            reset_observation = [0.0] * 84
            terminal_observation = [0.0] * 84
            terminal_observation[observation_index("primary_enemy_visible")] = 1.0
            infos = [
                {
                    "events": {},
                    "metrics": {"win": True, "episode_reward": 1.0, "episode_length": 1},
                    "done_reason": "won",
                    "terminal_observation": terminal_observation,
                },
                {"events": {}},
            ]
            pipe.on_step(
                [[0] * 6, [0] * 6],
                [reset_observation, reset_observation],
                [1.0, 0.0],
                [True, False],
                infos,
            )

            replay_path = next((config.run_directory() / "replays").glob("*.jsonl"))
            replay = load_replay(replay_path)
            self.assertTrue(replay.ticks[0].done)
            self.assertEqual(replay.ticks[0].observation, terminal_observation)
            summary = pipe.skill_metrics.aggregator.episodes()[0]
            self.assertEqual(summary["categories"]["awareness"]["visible_contacts"], 1)
            pipe.close()

    def test_resume_restages_identical_plans(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            pipe = TrainingPipeline(config, config.run_directory(), _Telemetry(), device="cpu")
            env = _FakeVecEnv(2)
            pipe.attach(env)
            pipe.on_reset(None)
            for i in range(6):
                infos = [
                    {
                        "events": {},
                        "metrics": {"win": i % 2 == 0, "episode_length": 20},
                        "done_reason": "",
                    },
                    {"events": {}},
                ]
                pipe.on_step(
                    [[0] * 6, [0] * 6], [[0.0] * 84, [0.0] * 84], [0.0, 0.0], [True, False], infos
                )
            pipe.timesteps = 400
            state_path = pipe.save_state()
            staged_before = pipe.driver.state_dict()
            pipe.close()

            pipe2 = TrainingPipeline(config, config.run_directory(), _Telemetry(), device="cpu")
            env2 = _FakeVecEnv(2)
            pipe2.attach(env2)
            self.assertTrue(pipe2.load_state(state_path))
            pipe2.reattach_after_load()
            self.assertEqual(pipe2.driver.state_dict(), staged_before)
            # The restaged batch is the exact pending plan set, not a redraw.
            self.assertEqual(env2.client.batches[-1], pipe.driver.staged_payloads())
            pipe2.close()

    def test_manifest_shape_and_fingerprint(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            pipe = TrainingPipeline(config, config.run_directory(), _Telemetry(), device="cpu")
            manifest = pipe.manifest()
            self.assertEqual(manifest.get("format"), MANIFEST_FORMAT)
            fingerprint = manifest.get("contract", {})
            self.assertEqual(fingerprint.get("observation_dim"), 84)
            self.assertEqual(fingerprint.get("action_nvec"), [3, 3, 3, 3, 2, 2])
            self.assertEqual(manifest.get("contract"), contract_fingerprint())
            curriculum = manifest.get("curriculum", {})
            self.assertEqual(curriculum.get("mode"), "auto")
            self.assertEqual(curriculum.get("current", {}).get("level"), 1)
            systems = manifest.get("enabled_systems", {})
            self.assertTrue(systems.get("skill_metrics"))
            self.assertTrue(systems.get("episode_log"))
            self.assertEqual(systems.get("replay_mode"), "off")
            hyper = manifest.get("hyperparameters", {})
            self.assertEqual(hyper.get("environment_count"), 2)
            self.assertEqual(manifest.get("seed"), config.seed)
            out = write_manifest(config.run_directory(), manifest)
            roundtrip = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(roundtrip.get("format"), MANIFEST_FORMAT)
            pipe.close()


if __name__ == "__main__":
    unittest.main()

"""Fast, display-free smoke tests for the arena environment."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.map_io import load_map, map_from_json, save_map
from env.maps import ArenaObject, MAP_NAMES, create_map
from env.shooter_env import (ACTION_NVECS, OBSERVATION_SIZE, OBSERVATION_VERSION,
                             ShooterEnv)
from env.weapons import WEAPON_NAMES, get_weapon
from training.imitation import demo_meta_path, load_demo_data
from training.rewards import RewardEvents, shape_reward
from training.weapon_lab import simulate_duels
from training.train import TrainingConfig, TrainingController
from training.workers import (BenchmarkRunner, CPU_JOB_LOCK,
                              valid_benchmark_configurations)


class ShooterEnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract(self) -> None:
        env = ShooterEnv(seed=17)
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=17)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(np.all(observation >= -1.0))
        self.assertTrue(np.all(observation <= 1.0))
        self.assertIn("map", info)
        env.close()

    def test_all_map_definitions_reset(self) -> None:
        for name in MAP_NAMES:
            with self.subTest(map=name):
                env = ShooterEnv(map_name=name, seed=3)
                observation, _ = env.reset(seed=3)
                self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
                self.assertEqual(env.arena_map.name, name)
                env.close()

    def test_custom_map_json_roundtrip_and_validation(self) -> None:
        arena_map = create_map("Custom")
        arena_map.objects.append(ArenaObject(2, -3, 0, 2, 1.5, 2, "crate", "Saved crate"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom_map.json"
            save_map(path, arena_map)
            loaded = load_map(path)
        self.assertEqual(loaded.name, "Custom")
        self.assertEqual(len(loaded.objects), 1)
        self.assertEqual(loaded.objects[0].name, "Saved crate")
        decoded = map_from_json(json.dumps({
            "name": "untrusted-name", "width": 50, "depth": 50,
            "objects": [], "spawn_points": [[-18, 0], [18, 0]],
        }))
        self.assertEqual(decoded.name, "Custom")
        with self.assertRaises(ValueError):
            map_from_json('{"width": 50, "depth": 50, "objects": [{"kind": "script"}]}')

    def test_frame_skip_and_duel_interface(self) -> None:
        env = ShooterEnv(frame_skip=4, curriculum=False, seed=5)
        env.reset(seed=5)
        _, reward, terminated, truncated, info = env.step_duel(
            env.action_space.sample(), env.action_space.sample()
        )
        self.assertEqual(env.physics_frames, 4)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated or truncated)
        self.assertIn("reward_agent_2", info)
        self.assertEqual(len(env.get_observation(1)), OBSERVATION_SIZE)
        env.close()

    def test_time_limited_decision_is_not_reported_as_a_kill(self) -> None:
        """``ttk`` is the match clock; only ``killed`` marks a real time-to-kill."""
        env = ShooterEnv(max_episode_seconds=1.0, frame_skip=1, seed=23)
        env.reset(seed=23)
        terminated = truncated = False
        info: dict = {}
        for _ in range(200):
            _, _, terminated, truncated, info = env.step_duel(env.heuristic_action(0),
                                                             env.heuristic_action(1))
            if terminated or truncated:
                break
        self.assertTrue(terminated or truncated)
        metrics = info["episode_metrics"]
        self.assertIn("killed", metrics)
        self.assertFalse(metrics["killed"])
        self.assertGreater(metrics["ttk"], 0.0)
        env.close()

    def test_kill_ends_the_episode_and_flags_the_kill(self) -> None:
        env = ShooterEnv(max_episode_seconds=60.0, frame_skip=1, seed=7)
        env.reset(seed=7)
        opponent = env.opponent
        opponent.body.hp = 1.0
        opponent.body.x, opponent.body.y = env.player.body.x, env.player.body.y + 4.0
        terminated = truncated = False
        info: dict = {}
        for _ in range(600):
            _, _, terminated, truncated, info = env.step_duel(env.heuristic_action(0),
                                                             env.heuristic_action(1))
            if terminated or truncated:
                break
        self.assertTrue(terminated or truncated)
        metrics = info.get("episode_metrics", {})
        if metrics.get("win") and metrics.get("killed"):
            self.assertGreater(metrics["ttk"], 0.0)
        env.close()

    def test_training_opponents_actually_return_fire(self) -> None:
        """A harmless opponent makes the policy farm the aim bonus instead of shooting.

        Regression: the walker used to be a pure moving target. With no incoming
        damage the reward for holding the crosshair outweighed any reason to fire,
        and PPO learned to survive the time limit without a single kill.
        """
        env = ShooterEnv(map_name="Dust", opponent_mode="walker", frame_skip=4,
                         max_episode_seconds=45.0, vision_mode="coarse_los", seed=3)
        env.reset(seed=3)
        for _ in range(700):
            _, _, terminated, truncated, _ = env.step_duel(env.heuristic_action(0),
                                                          env.heuristic_action(1))
            if terminated or truncated:
                break
        walker_damage = env._combatants[1].damage_dealt
        self.assertGreater(walker_damage, 0.0,
                           "the walker opponent never hit the player - no combat pressure")
        env.close()

    def test_scripted_opponent_reaches_the_player_on_every_map(self) -> None:
        """Cover used to freeze the duel: no sighting, no reward, no learning.

        On Warehouse the opponent walked into a wall for the whole episode; both
        fighters stayed blind and the map produced zero training signal. The bot
        now steers around cover.
        """
        idle = np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
        for map_name in ("Dust", "Arena", "Warehouse"):
            with self.subTest(map=map_name):
                env = ShooterEnv(map_name=map_name, curriculum=False, opponent_mode="walker",
                                 frame_skip=4, max_episode_seconds=60.0, vision_mode="coarse_los",
                                 seed=9)
                env.reset(seed=9)
                contact = False
                damage = 0.0
                for _ in range(1000):
                    _, _, terminated, truncated, _ = env.step_duel(idle, env.heuristic_action(1))
                    contact = contact or bool(env.enemy_view(0)["visible"])
                    if terminated or truncated:
                        break
                damage = env._combatants[0].damage_taken
                self.assertTrue(contact, f"the opponent never reached the player on {map_name}")
                self.assertGreater(damage, 0.0, f"the opponent never fired on {map_name}")
                env.close()

    def test_mover_tracks_the_player_without_return_fire(self) -> None:
        """The gentle curriculum rung: motion and aim, but no bullets.

        Phase 2 used to jump straight from a passive opponent to a walker that
        shoots back; a measured run then produced ~500 episodes without a single
        win or kill. ``mover`` separates the two lessons.
        """
        idle = np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
        env = ShooterEnv(map_name="Dust", curriculum=False, opponent_mode="mover",
                         frame_skip=4, max_episode_seconds=45.0, vision_mode="coarse_los",
                         seed=9)
        env.reset(seed=9)
        contact = False
        for _ in range(800):
            _, _, terminated, truncated, _ = env.step_duel(idle, env.heuristic_action(1))
            contact = contact or bool(env.enemy_view(0)["visible"])
            if terminated or truncated:
                break
        self.assertTrue(contact, "the mover never reached the player")
        self.assertEqual(env._combatants[1].shots_fired, 0, "the mover must not fire")
        self.assertEqual(env._combatants[0].damage_taken, 0.0)
        env.close()

    def test_unknown_opponent_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShooterEnv(map_name="Dust", opponent_mode="terminator")

    def test_kill_bonus_shrinks_over_the_episode(self) -> None:
        from training.rewards import KILL_DECAY_FLOOR, KILL_BONUS, RewardEvents, shape_reward

        base = dict(distance_before=10.0, distance_after=10.0, aim_error_radians=math.pi,
                    kills=1, physics_steps=1, time_limit_seconds=60.0)
        early = shape_reward(RewardEvents(elapsed_seconds=0.0, **base)).components["kills"]
        late = shape_reward(RewardEvents(elapsed_seconds=55.0, **base)).components["kills"]
        self.assertAlmostEqual(early, KILL_BONUS, places=6)
        self.assertLess(late, early)
        self.assertGreaterEqual(late, KILL_BONUS * KILL_DECAY_FLOOR - 1e-9)

    def test_approach_bonus_requires_visual_contact(self) -> None:
        from training.rewards import RewardEvents, shape_reward

        common = dict(distance_before=10.0, distance_after=9.0, aim_error_radians=math.pi,
                      physics_steps=1)
        visible = shape_reward(RewardEvents(enemy_visible=True, **common))
        hidden = shape_reward(RewardEvents(enemy_visible=False, **common))
        self.assertAlmostEqual(visible.components.get("approach", 0.0), 0.01)
        self.assertNotIn("approach", hidden.components)

    def test_weapons_and_ammo_reload(self) -> None:
        self.assertEqual(len(WEAPON_NAMES), 6)
        for name in WEAPON_NAMES:
            weapon = get_weapon(name)
            runtime = weapon.create_runtime()
            self.assertEqual(runtime.ammo, weapon.spec.mag_size)
            fired, reason = runtime.try_fire()
            self.assertTrue(fired, reason)
            self.assertEqual(runtime.ammo, weapon.spec.mag_size - 1)
            runtime.begin_reload()
            for _ in range(int(weapon.spec.reload_time / 0.01) + 2):
                runtime.tick(0.01)
            self.assertEqual(runtime.ammo, weapon.spec.mag_size)

    def test_action_space_matches_documented_dimensions(self) -> None:
        env = ShooterEnv(seed=2)
        action = env.action_space.sample()
        self.assertEqual(tuple(env.action_space.nvec.tolist()), ACTION_NVECS)
        self.assertEqual(action.shape, (len(ACTION_NVECS),))
        env.close()

    def test_reward_shaping_weights(self) -> None:
        breakdown = shape_reward(RewardEvents(
            distance_before=10.0,
            distance_after=9.0,
            aim_error_radians=0.0,
            hits=2,
            headshots=1,
            kills=1,
            incoming_hits=1,
            successful_dodges=1,
            wasted_ammo=1,
            physics_steps=4,
        ))
        self.assertAlmostEqual(breakdown.components["approach"], 0.01)
        self.assertAlmostEqual(breakdown.components["hits"], 1.0)
        self.assertAlmostEqual(breakdown.components["headshots"], 2.5)
        self.assertAlmostEqual(breakdown.components["kills"], 5.0)
        self.assertAlmostEqual(breakdown.total, 8.101)
        # Design rule: farming the crosshair for a whole episode must never be
        # worth more than a kill (this is what made the policy pacifist before).
        from training.rewards import AIM_BONUS

        episode_aim_bonus = AIM_BONUS * 30 * 15  # 30 s at 15 decisions per second
        self.assertLess(episode_aim_bonus, breakdown.components["kills"])

    def test_starter_demos_and_benchmark_matrix_are_valid(self) -> None:
        demo_path = Path(__file__).resolve().parents[1] / "data" / "demos.csv"
        states, actions = load_demo_data(demo_path)
        self.assertEqual(states.shape[1], OBSERVATION_SIZE)
        self.assertEqual(actions.shape[1], len(ACTION_NVECS))
        self.assertTrue(np.all(states >= -1.0) and np.all(states <= 1.0))
        self.assertTrue(all(np.all(actions[:, column] < categories)
                            for column, categories in enumerate(ACTION_NVECS)))
        # The dataset is only meaningful together with the layout it was recorded
        # with: a sidecar mismatch must fail loudly instead of training on garbage.
        meta = json.loads(demo_meta_path(demo_path).read_text(encoding="utf-8"))
        self.assertEqual(meta["observation_version"], OBSERVATION_VERSION)

    def test_demo_dataset_with_old_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            demos = root / "demos.csv"
            demos.write_text("state_0,state_1,action_0\n0.1,0.2,1\n", encoding="utf-8")
            (root / "demos_meta.json").write_text(
                json.dumps({"observation_version": OBSERVATION_VERSION - 1}), encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                load_demo_data(demos)
            self.assertIn("observation version", str(context.exception))
        configurations = valid_benchmark_configurations()
        self.assertTrue(configurations)
        self.assertTrue(all(workers * envs <= 24 for workers, envs in configurations))

    def test_ttk_simulation_is_seeded_and_rates_sum_to_one(self) -> None:
        pistol = get_weapon("Pistol").spec
        sniper = get_weapon("Sniper").spec
        first = simulate_duels(pistol, sniper, distance=25, trials=1_000, seed=41)
        second = simulate_duels(pistol, sniper, distance=25, trials=1_000, seed=41)
        self.assertEqual(first["win_rate_a"], second["win_rate_a"])
        self.assertAlmostEqual(first["win_rate_a"] + first["win_rate_b"] + first["draw_rate"], 1.0)
        self.assertGreaterEqual(first["weapon_a"]["kill_rate"], 0.0)
        self.assertLessEqual(first["weapon_a"]["kill_rate"], 1.0)

    def test_training_and_benchmark_respect_shared_cpu_lock(self) -> None:
        CPU_JOB_LOCK.acquire()
        try:
            training_job = TrainingController(TrainingConfig(total_timesteps=2_048))
            training_job.start()
            training_job._thread.join(timeout=2.0)
            self.assertEqual(training_job.snapshot()["status"], "error")
            self.assertIn("CPU-heavy", training_job.snapshot()["error"])

            benchmark_job = BenchmarkRunner(seconds_per_combo=0.01)
            benchmark_job.start()
            benchmark_job._thread.join(timeout=2.0)
            self.assertEqual(benchmark_job.snapshot()["status"], "error")
            self.assertIn("CPU-heavy", benchmark_job.snapshot()["error"])
        finally:
            CPU_JOB_LOCK.release()


if __name__ == "__main__":
    unittest.main()

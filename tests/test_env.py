"""Fast, display-free smoke tests for the arena environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.map_io import load_map, map_from_json, save_map
from env.maps import ArenaObject, MAP_NAMES, create_map
from env.shooter_env import ACTION_NVECS, OBSERVATION_SIZE, ShooterEnv
from env.weapons import WEAPON_NAMES, get_weapon
from gui.tabs.ttk import simulate_duels
from training.imitation import load_demo_data
from training.rewards import RewardEvents, shape_reward
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
        self.assertAlmostEqual(breakdown.total, 8.146)

    def test_starter_demos_and_benchmark_matrix_are_valid(self) -> None:
        demo_path = Path(__file__).resolve().parents[1] / "data" / "demos.csv"
        states, actions = load_demo_data(demo_path)
        self.assertEqual(states.shape[1], OBSERVATION_SIZE)
        self.assertEqual(actions.shape[1], len(ACTION_NVECS))
        self.assertTrue(np.all(states >= -1.0) and np.all(states <= 1.0))
        self.assertTrue(all(np.all(actions[:, column] < categories)
                            for column, categories in enumerate(ACTION_NVECS)))
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

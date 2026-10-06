"""Optional end-to-end PPO smoke test (runs when SB3 and PyTorch are installed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from training.train import TrainingConfig, TrainingController


HAS_PPO_DEPENDENCIES = (
    importlib.util.find_spec("stable_baselines3") is not None
    and importlib.util.find_spec("torch") is not None
)


@unittest.skipUnless(HAS_PPO_DEPENDENCIES, "Install Stable-Baselines3 and PyTorch to run PPO integration tests.")
class PPOTrainingSmokeTests(unittest.TestCase):
    def test_curriculum_gate_holds_a_phase_without_crashing(self) -> None:
        """A withheld curriculum phase must be logged, not raise.

        Regression: the hold branch used a clock variable that was defined later
        in the callback, so the run died with UnboundLocalError after 50k steps.
        """
        with tempfile.TemporaryDirectory(prefix="neural-arena-gate-") as temporary_directory:
            root = Path(temporary_directory)
            config = TrainingConfig(
                duration_seconds=0,
                total_timesteps=2_048,
                n_workers=1,
                envs_per_worker=1,
                map_name="Dust",
                curriculum=True,
                curriculum_min_win_rate=1.0,  # unreachable -> the gate must hold
                self_play=False,
                method="Pure RL",
                max_envs=1,
                episode_seconds=20.0,
                models_dir=root / "models",
                logs_dir=root / "logs",
            )
            controller = TrainingController(config)
            controller.start()
            controller._thread.join(timeout=240)
            self.assertFalse(controller.snapshot()["thread_alive"],
                             "curriculum gate smoke test timed out")
            snapshot = controller.snapshot()
            self.assertEqual(snapshot["status"], "complete", snapshot.get("error"))
            self.assertIn("kill_rate", snapshot["metrics"])
            header = (root / "logs" / "training_metrics.csv").read_text(
                encoding="utf-8").splitlines()[0]
            self.assertIn("kill_rate", header)


    def test_cpu_ppo_completes_one_rollout_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="neural-arena-test-") as temporary_directory:
            root = Path(temporary_directory)
            config = TrainingConfig(
                duration_seconds=0,
                total_timesteps=2_048,
                n_workers=1,
                envs_per_worker=1,
                map_name="Dust",
                curriculum=False,
                self_play=False,
                method="Pure RL",
                max_envs=1,
                models_dir=root / "models",
                logs_dir=root / "logs",
            )
            controller = TrainingController(config)
            controller.start()
            controller._thread.join(timeout=180)
            self.assertFalse(controller.snapshot()["thread_alive"], "PPO smoke test exceeded 180 seconds")
            snapshot = controller.snapshot()
            self.assertEqual(snapshot["status"], "complete", snapshot.get("error"))
            self.assertGreaterEqual(snapshot["metrics"]["timesteps"], 2_048)
            self.assertTrue((root / "logs" / "training_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()

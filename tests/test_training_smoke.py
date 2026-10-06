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
    def test_perception_curriculum_and_automatic_evaluation(self) -> None:
        """A finished run must switch perception per phase and verify itself."""
        with tempfile.TemporaryDirectory(prefix="neural-arena-eval-") as temporary_directory:
            root = Path(temporary_directory)
            config = TrainingConfig(
                duration_seconds=0,
                total_timesteps=2_048,
                n_workers=1,
                envs_per_worker=1,
                map_name="Dust",
                curriculum=True,
                vision_curriculum=True,
                vision_mode="coarse_los",
                eval_after_training=True,
                eval_episodes=1,
                eval_bots=("stationary",),
                self_play=False,
                method="Pure RL",
                max_envs=1,
                episode_seconds=20.0,
                models_dir=root / "models",
                logs_dir=root / "logs",
            )
            controller = TrainingController(config)
            controller.start()
            controller._thread.join(timeout=300)
            snapshot = controller.snapshot()
            self.assertFalse(snapshot["thread_alive"], "training + evaluation timed out")
            self.assertEqual(snapshot["status"], "complete", snapshot.get("error"))
            evaluation = snapshot.get("evaluation")
            self.assertIsNotNone(evaluation, "the run did not evaluate its checkpoint")
            self.assertEqual(len(evaluation["runs"]), 1)
            run = evaluation["runs"][0]
            for key in ("win_rate", "kill_rate", "avg_blind_ratio", "episodes"):
                self.assertIn(key, run)
            self.assertTrue((root / "models" / "best_model_eval.json").exists())
            # Perception curriculum: phase 1 must train the forgiving model.
            from training.train import vision_mode_for_phase

            self.assertEqual(vision_mode_for_phase(1), "noisy")
            self.assertEqual(vision_mode_for_phase(4), "coarse_los")

    def test_automatic_evaluation_reports_both_perception_models(self) -> None:
        """The retained model and the target-perception model are both verified."""
        import json

        from training.train import _evaluate_best_checkpoint

        with tempfile.TemporaryDirectory(prefix="neural-arena-target-") as temporary_directory:
            root = Path(temporary_directory)
            config = TrainingConfig(
                duration_seconds=0,
                total_timesteps=1_024,
                n_workers=1,
                envs_per_worker=1,
                map_name="Dust",
                curriculum=True,
                vision_curriculum=True,
                vision_mode="coarse_los",
                eval_after_training=True,
                eval_episodes=1,
                eval_bots=("stationary",),
                self_play=False,
                method="Pure RL",
                max_envs=1,
                episode_seconds=15.0,
                models_dir=root / "models",
                logs_dir=root / "logs",
            )
            controller = TrainingController(config)
            controller.start()
            controller._thread.join(timeout=300)
            self.assertFalse(controller.snapshot()["thread_alive"], "training timed out")

            # Pretend the retained checkpoint came from the forgiving phase 1 and
            # the end state from the target perception - exactly what the vision
            # curriculum produces over a long run.
            models = root / "models"
            (models / "best_model.zip").write_bytes((models / "final_model.zip").read_bytes())
            for name, mode, phase in (("best_model", "noisy", 1), ("final_model", "coarse_los", 4)):
                meta = json.loads((models / f"final_model_meta.json").read_text(encoding="utf-8"))
                meta.update({"vision_mode": mode, "phase": phase})
                (models / f"{name}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
                (models / f"{name}_vecnormalize.pkl").write_bytes(
                    (models / "final_model_vecnormalize.pkl").read_bytes())

            _evaluate_best_checkpoint(config, controller)
            evaluation = controller.snapshot()["evaluation"]
            self.assertIsNotNone(evaluation)
            self.assertEqual(evaluation["model"], "best_model.zip")
            self.assertEqual(evaluation["vision_mode"], "noisy")
            self.assertEqual(evaluation["runs"][0]["bot"], "stationary")
            self.assertIn("target", evaluation)
            self.assertEqual(evaluation["target"]["vision_mode"], "coarse_los")
            self.assertIsNotNone(evaluation["target"]["win_rate_overall"])
            self.assertTrue((models / "best_model_eval.json").exists())

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

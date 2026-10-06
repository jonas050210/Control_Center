"""Evaluation harness: the numbers it reports must match the simulation.

These checks exist because the report is the *only* honest statement about a
checkpoint - if it measures the wrong unit, every claim in the docs is wrong with
it. Found in a verification run: ``avg_seconds`` multiplied the physics frames by
``frame_skip`` a second time and reported episodes four times too long.
"""

from __future__ import annotations

import unittest

from training.evaluation import evaluate


class EpisodeTimingTests(unittest.TestCase):
    """``avg_seconds`` must be the simulated episode length, not a multiple."""

    def test_time_limit_episode_reports_its_configured_length(self) -> None:
        episode_seconds = 3.0
        result = evaluate(model=None, episodes=2, bot="stationary", vision_mode="coarse_los",
                          phase=1, episode_seconds=episode_seconds, seed=7)
        self.assertEqual(result["episodes"], 2)
        # A policy of zeros neither moves nor shoots, so the rounds run into the
        # time limit - and then the measured length must be that limit.
        self.assertAlmostEqual(result["avg_seconds"], episode_seconds, delta=0.2)
        self.assertAlmostEqual(result["avg_frames"] / 60.0, result["avg_seconds"], delta=0.05)
        self.assertEqual(result["kill_wins"], 0)
        self.assertEqual(result["wins"] + result["draws"] + result["losses"], 2)


if __name__ == "__main__":
    unittest.main()

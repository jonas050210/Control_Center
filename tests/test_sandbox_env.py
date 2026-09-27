"""Tests for Tactical FPS Sandbox Environment (Phase 3)."""

import numpy as np
import pytest

from sandbox.env import MockTacticalArenaEnv, make_sandbox_env


def test_sandbox_env_reset_and_observation_shape() -> None:
    env = make_sandbox_env(width=84, height=84, seed=123)
    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 84, 84)
    assert obs.dtype == np.uint8
    assert "targets_hit" in info


def test_sandbox_env_step_movement_and_firing() -> None:
    env = MockTacticalArenaEnv(width=84, height=84, max_steps=50)
    obs, _ = env.reset()

    # Action: move forward (move_y=2), fire=1, look straight (yaw=10, pitch=10)
    act = [1, 2, 1, 10, 10]
    next_obs, reward, terminated, truncated, info = env.step(act)

    assert next_obs.shape == (3, 84, 84)
    assert isinstance(reward, float)
    assert terminated is False
    assert info["step"] == 1


def test_sandbox_env_max_steps_truncation() -> None:
    env = MockTacticalArenaEnv(width=84, height=84, max_steps=10)
    env.reset()

    terminated = truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step([1, 1, 0, 10, 10])

    assert terminated is False
    assert truncated is True

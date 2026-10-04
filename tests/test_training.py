import json

import numpy as np
import torch

from rocketai.config import TrainConfig, run_paths
from rocketai.model import ActorCritic
from rocketai.ppo import ppo_update
from rocketai.rollout import Collector, compute_gae
from rocketai.trainer import CURRICULUM_WINDOW, Trainer, curriculum_ready, past_summary, train


def test_gae_matches_hand_computation():
    rewards = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    values = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    dones = np.array([False, False, False])
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, dones, np.zeros(3, np.float32), 1.0, gamma, lam)
    d2 = 2.0 + gamma * 1.0 - 0.3
    d1 = 0.0 + gamma * 0.3 - 0.4
    d0 = 1.0 + gamma * 0.4 - 0.5
    expected = [d0 + gamma * lam * (d1 + gamma * lam * d2), d1 + gamma * lam * d2, d2]
    assert np.allclose(adv, expected, atol=1e-6)
    assert np.allclose(ret, adv + values)


def test_gae_cuts_at_episode_end():
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.zeros(2, dtype=np.float32)
    # Goal after step 0 (next value 0), timeout after step 1 bootstraps with 5.
    adv, _ = compute_gae(
        rewards, values, np.array([True, True]), np.array([0.0, 5.0], np.float32), 99.0, 0.5, 1.0
    )
    assert np.allclose(adv, [1.0, 1.0 + 0.5 * 5.0])


def _small_config(**changes):
    data = dict(
        name="t",
        team_size=1,
        reward_stage=1,
        envs_per_worker=2,
        n_workers=1,
        steps_per_iteration=600,
        minibatch_size=300,
        hidden_sizes=[32, 32],
        episode_seconds=10,
        no_touch_seconds=5,
        checkpoint_every_steps=600,
        eval_every_steps=0,
        total_steps=1200,
    )
    data.update(changes)
    return TrainConfig(**data)


def test_collector_and_ppo_update():
    config = _small_config()
    collector = Collector(config.to_dict(), seed=0)
    batch = collector.collect(800)
    assert len(batch) >= 800
    assert batch.obs.shape == (len(batch), 172)
    assert np.isfinite(batch.advantages).all()
    assert batch.stats["episodes"], "10 s episodes should finish within 800 agent steps"
    model = collector.model
    stats = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        batch,
        epochs=2,
        minibatch_size=256,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    )
    assert stats["updates"] > 0 and np.isfinite(stats["policy_loss"])


def test_training_run_writes_everything_and_resumes():
    config = _small_config()
    train(config)
    paths = run_paths("t")
    status = json.loads(paths.status.read_text())
    assert status["state"] == "finished" and status["steps"] >= 1200
    assert (paths.checkpoints / "latest.pt").exists()
    lines = paths.metrics.read_text().splitlines()
    assert len(lines) == 2 and "touches_per_minute" in json.loads(lines[-1])
    # Resuming with a higher target continues from the saved step count.
    train(_small_config(total_steps=1800))
    assert json.loads(paths.status.read_text())["steps"] >= 1800
    assert len(paths.metrics.read_text().splitlines()) == 3


def test_resume_rejects_other_network_size():
    train(_small_config(total_steps=600))
    try:
        train(_small_config(hidden_sizes=[64]))
    except ValueError as error:
        assert "network" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_model_sizes_used():
    assert ActorCritic(hidden_sizes=[16]).hidden_sizes == [16]


def test_collector_plays_against_past_versions():
    config = _small_config(past_opponent_prob=0.95)
    collector = Collector(config.to_dict(), seed=1)
    old = ActorCritic(hidden_sizes=config.hidden_sizes).state_dict()
    collector.set_past([old, old])
    collector.opponent = [0 for _ in collector.envs]  # start every match against the pool
    batch = collector.collect(1200)
    # Only blue (learning) cars produce training samples in pool matches.
    assert len(batch) >= 1200
    results = [e["vs_past"] for e in batch.stats["episodes"] if "vs_past" in e]
    assert results and set(results) <= {-1, 0, 1}
    summary = past_summary(batch.stats["episodes"])
    assert 0.0 <= summary["past_win_rate"] <= 1.0 and summary["past_games"] == len(results)


def test_curriculum_switches_only_when_ready():
    config = _small_config(auto_curriculum=True, reward_stage=1)
    good = [{"touches_per_minute": 20.0}] * CURRICULUM_WINDOW
    bad = [{"touches_per_minute": 3.0}] * CURRICULUM_WINDOW
    assert curriculum_ready(config, 20_000_000, good)
    assert not curriculum_ready(config, 20_000_000, bad)
    assert not curriculum_ready(config, 1_000_000, good)  # too early
    assert not curriculum_ready(config, 20_000_000, good[:5])  # not enough history
    config.auto_curriculum = False
    assert not curriculum_ready(config, 20_000_000, good)
    stage2 = _small_config(auto_curriculum=True, reward_stage=2)
    assert curriculum_ready(stage2, 60_000_000, [{"goals_per_minute": 1.2}] * CURRICULUM_WINDOW)
    stage3 = _small_config(auto_curriculum=True, reward_stage=3)
    assert not curriculum_ready(stage3, 10**10, [{"goals_per_minute": 9.0}] * CURRICULUM_WINDOW)


def test_training_builds_opponent_pool():
    train(_small_config(total_steps=1200, checkpoint_every_steps=500, past_pool_size=2))
    trainer = Trainer(_small_config(total_steps=1800, past_pool_size=2))
    assert 1 <= len(trainer.past_pool) <= 2
    trainer._log_handle.close()

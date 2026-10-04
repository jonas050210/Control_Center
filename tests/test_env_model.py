import numpy as np
import torch

from rocketai.config import N_ACTIONS, OBS_SIZE
from rocketai.env import lookup_table, make_env
from rocketai.model import ActorCritic, load_checkpoint, load_policy, save_checkpoint
from rocketai.rewards import COMPONENTS, StagedReward, component_values


def test_env_shapes_all_team_sizes():
    assert lookup_table().shape == (N_ACTIONS, 8)
    for size in (1, 2, 3):
        env = make_env(team_size=size, seed=1)
        obs = env.reset()
        assert len(obs) == 2 * size
        assert all(o.shape == (OBS_SIZE,) for o in obs.values())
        obs, rewards, terminated, truncated = env.step({a: np.array([0]) for a in env.agents})
        assert set(rewards) == set(env.agents)
        env.close()


def test_reward_components_and_parts():
    env = make_env(seed=2)
    env.reset()
    state = env.state
    for agent in env.agents:
        values = component_values(agent, state)
        assert set(values) == set(COMPONENTS)
        assert values["goal"] == 0.0
    reward = StagedReward(stage=2)
    shared = {}
    reward.reset(env.agents, state, shared)
    reward.get_rewards(env.agents, state, {}, {}, shared)
    assert set(shared["reward_parts"]) == set(reward.weights)


def test_model_act_and_checkpoint_roundtrip(tmp_path):
    model = ActorCritic(hidden_sizes=[32, 32])
    obs = np.random.default_rng(0).normal(size=(5, OBS_SIZE)).astype(np.float32)
    actions, logp, values = model.act(obs)
    assert actions.shape == logp.shape == values.shape == (5,)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, steps=123, config={"name": "t"})
    assert load_checkpoint(path)["steps"] == 123
    loaded = load_policy(path)
    with torch.no_grad():
        assert torch.allclose(
            loaded.logits(torch.as_tensor(obs)), model.logits(torch.as_tensor(obs))
        )

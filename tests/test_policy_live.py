"""Der Live-Pfad (``explain=True``) muss dieselben Entscheidungen treffen wie der Bot.

Die Live-Ansicht rechnet ihren eigenen Weg über ``softmax``/``argmax``, das echte
Spiel über ``model.act``. Laufen die beiden auseinander, zeigt die Anzeige etwas
anderes als der Bot fährt — genau das wird hier festgenagelt.
"""

import numpy as np
import pytest

from rocketai.config import OBS_SIZE
from rocketai.model import ActorCritic, save_checkpoint
from rocketai.opponents import PolicyPlayer


@pytest.fixture
def player(tmp_path):
    model = ActorCritic(hidden_sizes=[32, 32])
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, steps=1, config={})
    return PolicyPlayer.from_checkpoint(path)


def _obs(seed: int = 3, agents: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"blue-{i}": rng.normal(size=OBS_SIZE).astype(np.float32) for i in range(agents)}


def test_explain_path_matches_bot_path(player):
    obs = _obs()
    agents = list(obs)
    player.explain = False
    plain = player.act(agents, obs, None)
    player.explain = True
    explained = player.act(agents, obs, None)
    assert plain == explained


def test_explain_info_belongs_to_the_driven_action(player):
    obs = _obs(seed=4)
    agents = list(obs)
    player.explain = True
    actions = player.act(agents, obs, None)
    for agent in agents:
        info = player.last_info[agent]
        assert info["action"] == actions[agent]
        probabilities = [probability for _, probability in info["top"]]
        assert probabilities == sorted(probabilities, reverse=True)
        assert len(info["top"]) == 5
        assert -1.0 <= info["confidence"] <= 1.0
        assert len(info["controls"]) == 8


def test_actions_repeat_exactly(player):
    """Zweimal dieselbe Beobachtung heißt zweimal dieselbe Aktion (beide Pfade)."""
    obs = _obs(seed=5)
    agents = list(obs)
    for explain in (False, True):
        player.explain = explain
        assert player.act(agents, obs, None) == player.act(agents, obs, None)


def test_no_agents_is_allowed(player):
    assert player.act([], {}, None) == {}

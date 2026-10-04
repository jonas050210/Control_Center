import json

from rocketai.match import evaluate, play_match, save_replay
from rocketai.model import ActorCritic
from rocketai.opponents import (
    IDLE_ACTION,
    ChaserBot,
    IdleBot,
    PolicyPlayer,
    ground_index,
    make_player,
)


def test_ground_index_lookup():
    assert ground_index(1, 0) != ground_index(1, 1)
    assert ground_index(0, 0) == IDLE_ACTION


def test_chaser_beats_idle_and_steers_correctly():
    goals_for = goals_against = touches = 0
    for seed in range(3):
        result = play_match(ChaserBot(), IdleBot(), seconds=60, seed=seed)
        goals_for += result.goals_blue
        goals_against += result.goals_orange
        touches += result.touches_blue
    assert touches > 30
    assert goals_for > goals_against


def test_seeded_matches_repeat():
    first = play_match(ChaserBot(), IdleBot(), seconds=20, seed=7)
    second = play_match(ChaserBot(), IdleBot(), seconds=20, seed=7)
    assert first.summary() == second.summary()


def test_replay_recording(tmp_path):
    policy = PolicyPlayer(ActorCritic(hidden_sizes=[16]))
    result = play_match(policy, make_player("chaser"), seconds=10, record=True)
    assert len(result.frames) == 10 * 15 + 1
    t, ball, cars = result.frames[-1]
    assert len(ball) == 3 and len(cars) == 2 and len(cars[0]) == 7
    path = tmp_path / "r.json"
    save_replay(path, result, {"kind": "test"})
    data = json.loads(path.read_text())
    assert data["format"] == "rocketai.replay.v1" and data["meta"]["kind"] == "test"


def test_evaluate_counts_games():
    out = evaluate(make_player("chaser"), make_player("idle"), games=2, seconds=20)
    assert out["wins"] + out["draws"] + out["losses"] == 2
    assert 0 <= out["score"] <= 1

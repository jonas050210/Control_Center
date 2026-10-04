"""Tests für den Lehrer (Nexto).

Sie laufen auch ohne heruntergeladenen Lehrer durch: alles, was die echten
Nexto-Dateien braucht, wird dann übersprungen. Mit geladenem Lehrer prüfen sie
zusätzlich, dass unsere Übersetzung seines Beobachtungsformats exakt stimmt
und dass er im Spiel wirklich stark ist.
"""

import numpy as np
import pytest

from rocketai.config import TrainConfig, preset_config
from rocketai.env import make_env
from rocketai.teacher import (
    FILES,
    NextoTeacher,
    TeacherError,
    TeacherPlayer,
    canonical_players,
    compact_players,
    encode_compact,
    ensure_teacher,
    expand_compacts,
    teacher_ready,
    teacher_weight_at,
)


@pytest.fixture(scope="module")
def teacher():
    if not teacher_ready():
        pytest.skip("Lehrer nicht geladen (python -m rocketai teacher)")
    return NextoTeacher()


def test_download_verification_rejects_wrong_bytes(tmp_path, monkeypatch):
    """Eine veränderte Datei darf nicht geladen werden (Sicherheit)."""
    import rocketai.teacher as module

    monkeypatch.setattr(module, "FETCHERS", (("fake", lambda name: b"not the model"),))
    with pytest.raises(TeacherError) as error:
        ensure_teacher(tmp_path, force=True)
    assert "Sicherheitsgründen" in str(error.value)


def test_download_accepts_correct_bytes(tmp_path, monkeypatch):
    import hashlib

    import rocketai.teacher as module

    monkeypatch.setattr(
        module,
        "FILES",
        {name: hashlib.sha256(f"content of {name}".encode()).hexdigest() for name in FILES},
    )
    monkeypatch.setattr(module, "FETCHERS", (("fake", lambda name: f"content of {name}".encode()),))
    assert ensure_teacher(tmp_path, force=True).is_dir()
    assert teacher_ready(tmp_path)
    assert (tmp_path / "NOTICE.txt").is_file()


def test_weight_schedule():
    assert teacher_weight_at(0, 1.0, 0.1, 100) == pytest.approx(1.0)
    assert teacher_weight_at(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert teacher_weight_at(500, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert teacher_weight_at(50, 0.0, 0.0, 100) == 0.0
    assert teacher_weight_at(50, 0.5, 0.5, 0) == 0.5  # kein Abfall


def test_student_preset_uses_teacher():
    config = preset_config("student", name="x")
    assert config.teacher_opponent_prob > 0
    assert config.teacher_weight > config.teacher_final_weight
    assert config.teacher_weight_at(0) > config.teacher_weight_at(config.total_steps)


def test_config_rejects_bad_teacher_settings():
    with pytest.raises(ValueError):
        TrainConfig(teacher_opponent_prob=1.0).validate()
    with pytest.raises(ValueError):
        TrainConfig(teacher_temperature=0).validate()
    with pytest.raises(ValueError):
        TrainConfig(teacher_weight=-1).validate()


def test_teacher_shared_by_all_players(teacher):
    """Ein Netz, mehrere Spieler-Ansichten: gleiche Antwort, getrennte Erinnerung."""
    first = TeacherPlayer(teacher=teacher)
    second = TeacherPlayer(teacher=teacher)
    assert first.teacher is second.teacher


def test_teacher_player_acts_and_explains(teacher):
    env = make_env(team_size=1, seed=4)
    env.reset()
    state = env.state
    player = TeacherPlayer(teacher=teacher)
    player.explain = True
    agents = list(env.agents)
    actions = player.act(agents, {}, state)
    assert set(actions) == set(agents)
    assert all(0 <= action < 90 for action in actions.values())
    assert set(player.last_info) == set(agents)
    assert 0.0 <= player.last_info[agents[0]]["confidence"] <= 1.0


def test_encoding_matches_nexto_exactly(teacher):
    """Unsere Zahlen müssen Nextos eigener Kodierung entsprechen (nur Rundung)."""
    from rocketai.teacher import encode_reference

    env = make_env(team_size=1, seed=8)
    env.reset()
    state = env.state
    players = canonical_players(state)
    ours = expand_compacts(encode_compact(state, players)[None])[0]
    theirs = encode_reference(state, players)
    assert ours.shape == theirs.shape
    # Quaternionen dürfen sich im Vorzeichen unterscheiden (gleiche Drehung).
    difference = np.abs(ours - theirs).copy()
    for offset in (5, 18):  # Quaternion je Spieler im Kopf-/Auto-Block
        for player in range(2):
            start = 55 + player * 38 + offset
            difference[start : start + 4] = np.minimum(
                difference[start : start + 4],
                np.abs(ours[start : start + 4] + theirs[start : start + 4]),
            )
    assert difference.max() < 1e-4


def test_compact_layout(teacher):
    env = make_env(team_size=2, seed=2)
    env.reset()
    compact = encode_compact(env.state)
    assert compact_players(compact) == 4  # zwei Autos je Team


def test_teacher_much_stronger_than_chaser(teacher):
    """Der Kern des Versprechens: Der Lehrer spielt sichtbar besser."""
    from rocketai.match import play_match
    from rocketai.opponents import make_player

    result = play_match(
        TeacherPlayer(teacher=teacher),
        make_player("chaser"),
        team_size=1,
        seconds=30.0,
    )
    assert result.touches_blue > result.touches_orange
    assert result.goals_blue > result.goals_orange


def test_rollout_collects_teacher_labels(monkeypatch):
    """Der Sammel-Prozess legt die Lehrer-Fragen zum Batch (ohne Netz nötig)."""
    if not teacher_ready():
        pytest.skip("Lehrer nicht geladen")
    from rocketai.rollout import Collector

    config = TrainConfig(
        team_size=1,
        envs_per_worker=1,
        episode_seconds=20.0,
        no_touch_seconds=6.0,
        hidden_sizes=[32],
        steps_per_iteration=200,
        teacher_weight=0.5,
        teacher_samples=40,
    ).to_dict()
    collector = Collector(config, seed=0)
    batch = collector.collect(200)
    assert batch.teacher_states is not None
    assert len(batch.teacher_states) <= 80  # Stichprobe, nicht jeder Schritt
    assert batch.teacher_rows.max() < len(batch)
    assert batch.teacher_previous.shape[1] == 8
    assert batch.teacher_slots.min() >= 0
    assert len(set(batch.teacher_rows.tolist())) == len(batch.teacher_rows)


def test_trainer_attaches_teacher_targets(teacher):
    if not teacher_ready():
        pytest.skip("Lehrer nicht geladen")
    from rocketai.rollout import Batch
    from rocketai.trainer import Trainer

    config = TrainConfig(name="t", teacher_weight=1.0, hidden_sizes=[32])
    trainer = Trainer(config, log=lambda message: None)
    trainer.teacher = type("Fake", (), {"targets": teacher.probs})()
    states = np.stack([encode_compact(_state_of_seed(3))] * 2)
    batch = Batch(
        obs=np.zeros((4, 172), dtype=np.float32),
        actions=np.zeros(4, dtype=np.int64),
        log_probs=np.zeros(4, dtype=np.float32),
        advantages=np.zeros(4, dtype=np.float32),
        returns=np.zeros(4, dtype=np.float32),
        values=np.zeros(4, dtype=np.float32),
        teacher_states=states,
        teacher_rows=np.array([1, 3]),
        teacher_slots=np.zeros(2, dtype=np.int64),
        teacher_previous=np.zeros((2, 8), dtype=np.float32),
    )
    info = trainer.attach_teacher(batch, weight=1.0)
    assert info["teacher_samples"] == 2
    assert batch.teacher_probs.shape == (4, 90)
    assert batch.teacher_mask.tolist() == [False, True, False, True]
    assert np.allclose(batch.teacher_probs[batch.teacher_mask].sum(axis=1), 1.0, atol=1e-4)


def _state_of_seed(seed: int):
    env = make_env(team_size=1, seed=seed)
    env.reset()
    return env.state

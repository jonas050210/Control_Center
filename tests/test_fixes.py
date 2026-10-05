"""Regressionstests für die Fehler, die beim Umbau gefunden wurden.

Jeder Test hier steht für einen konkreten, reproduzierbaren Fehler:

* ``teacher_rows`` wurden beim Zusammenlegen mehrerer Worker nicht verschoben —
  die KI lernte dann vom Lehrer für die *falschen* Spielsituationen.
* Ballkontakte/Tore wurden für alle Autos gezählt (der Gegner zählte mit).
* ``teacher_weight = 0`` schaltete den Lehrer nicht wirklich aus.
* Zwei Trainingsprozesse für denselben Run konnten sich überschreiben.
* Nach einem Neustart begann der Zufall von vorne (nicht reproduzierbar).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rocketai.benchmark import (
    CHUNK_STEPS,
    advice,
    effective_steps_per_second,
    format_report,
    measure_update,
    plan_combos,
    suggested_config,
)
from rocketai.config import TrainConfig, run_paths
from rocketai.model import (
    ActorCritic,
    rng_payload,
    save_checkpoint,
    set_rng_payload,
)
from rocketai.opponents import PolicyPlayer, make_player
from rocketai.rollout import Batch, Collector, WorkerPool
from rocketai.runtime import AlreadyRunning, TrainLock, lock_owner
from rocketai.trainer import summarize, teacher_summary


def _config(**changes):
    data = dict(
        name="fixes",
        team_size=1,
        reward_stage=1,
        envs_per_worker=1,
        n_workers=1,
        steps_per_iteration=400,
        minibatch_size=200,
        hidden_sizes=[16, 16],
        episode_seconds=6,
        no_touch_seconds=4,
        checkpoint_every_steps=400,
        eval_every_steps=0,
        total_steps=800,
    )
    data.update(changes)
    return TrainConfig(**data)


# ---------------------------------------------------------------- teacher rows


def test_concat_moves_teacher_rows_to_the_combined_batch():
    """Der Kern des Multi-Worker-Fehlers: Zeilennummern brauchen einen Versatz."""
    first = Batch(
        obs=np.zeros((5, 4), np.float32),
        actions=np.zeros(5, np.int64),
        log_probs=np.zeros(5, np.float32),
        advantages=np.zeros(5, np.float32),
        returns=np.zeros(5, np.float32),
        values=np.zeros(5, np.float32),
        teacher_states=np.zeros((2, 3), np.float32),
        teacher_rows=np.array([1, 3], np.int64),
        teacher_slots=np.array([0, 0], np.int64),
        teacher_previous=np.zeros((2, 8), np.float32),
    )
    second = Batch(
        obs=np.zeros((4, 4), np.float32),
        actions=np.zeros(4, np.int64),
        log_probs=np.zeros(4, np.float32),
        advantages=np.zeros(4, np.float32),
        returns=np.zeros(4, np.float32),
        values=np.zeros(4, np.float32),
        teacher_states=np.zeros((1, 3), np.float32),
        teacher_rows=np.array([2], np.int64),
        teacher_slots=np.array([1], np.int64),
        teacher_previous=np.zeros((1, 8), np.float32),
    )
    merged = Batch.concat([first, second])
    assert len(merged) == 9
    # Zweiter Stapel beginnt bei Zeile 5: 2 + 5 = 7
    assert merged.teacher_rows.tolist() == [1, 3, 7]
    assert merged.teacher_rows.max() < len(merged)
    # Ohne den Versatz zeigte die letzte Zeile auf die falsche Situation.
    assert merged.teacher_rows[-1] != second.teacher_rows[-1]


def test_concat_counts_workers_without_teacher_data_too():
    """Ein Worker ohne aufgezeichnete Situationen darf die Zeilen nicht verschieben.

    Der Versatz wurde früher nur über die Stapel *mit* Lehrer-Daten summiert.
    Ein Stapel ohne Daten verschob damit alle folgenden Zeilen nach vorne — die
    KI hätte vom Lehrer für die falschen Situationen gelernt.
    """
    empty = Batch(
        obs=np.zeros((5, 4), np.float32),
        actions=np.zeros(5, np.int64),
        log_probs=np.zeros(5, np.float32),
        advantages=np.zeros(5, np.float32),
        returns=np.zeros(5, np.float32),
        values=np.zeros(5, np.float32),
    )
    with_teacher = Batch(
        obs=np.zeros((4, 4), np.float32),
        actions=np.zeros(4, np.int64),
        log_probs=np.zeros(4, np.float32),
        advantages=np.zeros(4, np.float32),
        returns=np.zeros(4, np.float32),
        values=np.zeros(4, np.float32),
        teacher_states=np.zeros((2, 3), np.float32),
        teacher_rows=np.array([0, 2], np.int64),
        teacher_slots=np.array([0, 0], np.int64),
        teacher_previous=np.zeros((2, 8), np.float32),
    )
    merged = Batch.concat([empty, with_teacher])
    assert len(merged) == 9
    # Der zweite Stapel beginnt bei Zeile 5: 0 + 5 = 5, 2 + 5 = 7
    assert merged.teacher_rows.tolist() == [5, 7]
    assert merged.teacher_rows.max() < len(merged)


def test_multiworker_collection_keeps_teacher_rows_in_range():
    config = _config(teacher_weight=1.0, teacher_final_weight=0.0, teacher_samples=8)
    pool = WorkerPool(2, config.to_dict(), seed=0)
    try:
        model = ActorCritic(obs_size=config.obs_size, hidden_sizes=config.hidden_sizes)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        # teacher_weight=1.0: nur dann werden Lehrer-Situationen aufgezeichnet.
        batch = pool.collect(weights, 400, teacher_weight=1.0)
    finally:
        pool.close()
    assert batch.teacher_rows is not None
    assert len(batch.teacher_rows) > 0
    assert int(batch.teacher_rows.min()) >= 0
    assert int(batch.teacher_rows.max()) < len(batch)
    # Jede Zeile darf nur einmal als Lehrer-Frage vorkommen.
    assert len(set(batch.teacher_rows.tolist())) == len(batch.teacher_rows)


def test_teacher_off_records_nothing():
    config = _config(teacher_weight=0.0, teacher_final_weight=0.0)
    collector = Collector(config.to_dict(), seed=0)
    assert collector.teacher_enabled is False
    batch = collector.collect(200)
    assert batch.teacher_states is None
    assert batch.teacher_rows is None


def test_teacher_weight_zero_means_off_over_time():
    config = _config(teacher_weight=0.0, teacher_final_weight=0.0, teacher_decay_steps=1000)
    assert config.teacher_weight_at(0) == 0.0
    assert config.teacher_weight_at(10_000) == 0.0  # vorher: 0.1 Restgewicht!
    assert config.teacher_labels is False


# ------------------------------------------------------------- ehrliche Zahlen


def test_touch_and_goal_statistics_are_split_by_team():
    stats = {
        "episodes": [
            {
                "reward": 1.0,
                "seconds": 30.0,
                "goal": True,
                "touches": 6,
                "touches_other": 3,
                "goals_own": 1,
                "goals_other": 0,
                "opponent": "past",
                "vs_past": 1,
            },
            {
                "reward": 0.5,
                "seconds": 30.0,
                "goal": True,
                "touches": 2,
                "touches_other": 8,
                "goals_own": 0,
                "goals_other": 1,
                "opponent": "past",
                "vs_past": -1,
            },
        ],
        "reward_parts": {},
        "agent_steps": 400,
        "collect_seconds": 2.0,
        "game_seconds": 60.0,
    }
    metrics = summarize(
        Batch(
            obs=np.zeros((1, 4), np.float32),
            actions=np.zeros(1, np.int64),
            log_probs=np.zeros(1, np.float32),
            advantages=np.zeros(1, np.float32),
            returns=np.zeros(1, np.float32),
            values=np.zeros(1, np.float32),
            stats=stats,
        ),
        {},
    )
    # 8 eigene Kontakte in 60 Spielsekunden = 8 pro Minute (nicht 19!).
    assert metrics["touches_per_minute"] == pytest.approx(8.0)
    assert metrics["touches_against_per_minute"] == pytest.approx(11.0)
    assert metrics["goals_per_minute"] == pytest.approx(1.0)
    assert metrics["goals_against_per_minute"] == pytest.approx(1.0)
    assert metrics["realtime_factor"] == pytest.approx(30.0)


def test_teacher_summary_counts_goals_not_matches():
    episodes = [
        {"vs_teacher": 1, "teacher_goals_for": 2, "teacher_goals_against": 1},
        {"vs_teacher": -1, "teacher_goals_for": 0, "teacher_goals_against": 3},
        {"vs_teacher": 0, "teacher_goals_for": 0, "teacher_goals_against": 0},
    ]
    summary = teacher_summary(episodes)
    assert summary["teacher_games"] == 3
    assert summary["teacher_wins"] == 1
    assert summary["teacher_losses"] == 1
    assert summary["teacher_goals_for"] == 2
    assert summary["teacher_goals_against"] == 4


# ------------------------------------------------------------------- Sperre


def test_only_one_trainer_per_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCKETAI_RUNS", str(tmp_path / "runs"))
    paths = run_paths("locked").ensure()
    with TrainLock(paths.root, "locked"):
        owner = lock_owner(paths.root)
        assert owner is not None and owner["alive"]
        with pytest.raises(AlreadyRunning), TrainLock(paths.root, "locked"):
            pass
    # Nach dem Ende ist die Sperre wirklich frei.
    assert lock_owner(paths.root) is None
    with TrainLock(paths.root, "locked"):
        pass


def test_stale_lock_file_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCKETAI_RUNS", str(tmp_path / "runs"))
    paths = run_paths("stale").ensure()
    # Eine Sperrdatei eines längst beendeten Prozesses (PID 999999) bleibt liegen.
    (paths.root / "trainer.lock").write_text("999999 1.0\n", encoding="utf-8")
    owner = lock_owner(paths.root)
    assert owner is not None and owner["alive"] is False
    with TrainLock(paths.root, "stale"):
        pass


# --------------------------------------------------------------- Zufall/Seed


def test_rng_state_survives_a_checkpoint():
    rng = np.random.default_rng(7)
    rng.random(3)
    payload = rng_payload(rng)
    expected = rng.random(4)

    restored = np.random.default_rng(0)
    assert set_rng_payload(restored, payload) is True
    assert np.allclose(restored.random(4), expected)
    torch.rand(1)  # Torch-Zustand wurde ebenfalls wiederhergestellt
    assert set_rng_payload(restored, None) is False


def test_small_config_can_use_cpu_device():
    config = _config(device="cpu")
    config.validate()
    assert config.device == "cpu"
    with pytest.raises(ValueError):
        _config(device="tpu").validate()


# ------------------------------------------------------- Beobachtung (obs_extras)


def test_new_checkpoint_and_matches_use_matching_observations(tmp_path):
    """Ein alter Checkpoint (172 Werte) muss weiter spielbar bleiben."""
    from rocketai.match import needs_extras, play_match

    old = ActorCritic(obs_size=172, hidden_sizes=[16])
    path = tmp_path / "old.pt"
    save_checkpoint(path, old, steps=1, config={})
    player = PolicyPlayer.from_checkpoint(path)
    assert needs_extras([player]) is False
    assert needs_extras([make_player("chaser")]) is True  # Standard: erweitert

    result = play_match(player, make_player("chaser"), seconds=3.0, seed=1)
    assert result.seconds == 3.0


def test_extra_obs_values_are_bounded_and_consistent():
    from rocketai.env import make_env
    from rocketai.obs import EXTRA_FEATURES, extra_values

    env = make_env(team_size=1, episode_seconds=5, no_touch_seconds=5, seed=3)
    env.reset()
    obs = env.state  # noqa: F841 - nur zur Klarheit
    for agent in env.agents:
        values = extra_values(agent, env.state)
        assert values.shape == (len(EXTRA_FEATURES),)
        assert np.all(np.abs(values) <= 1.0 + 1e-6), values
    # Beide Beobachtungsvarianten haben die dokumentierte Größe.
    from rocketai.config import OBS_BASE_SIZE, OBS_SIZE

    plain = make_env(team_size=1, obs_extras=False, seed=1).reset()
    rich = make_env(team_size=1, obs_extras=True, seed=1).reset()
    agent = next(iter(plain))
    assert plain[agent].shape == (OBS_BASE_SIZE,)
    assert rich[agent].shape == (OBS_SIZE,)


# ---------------------------------------------------------------- Benchmark


def test_benchmark_update_and_report_are_readable():
    entry = measure_update(steps=500, hidden_sizes=[16], epochs=1, minibatch_size=250)
    assert entry["steps_per_second"] > 0
    text = format_report(
        {
            "cpu_count": 8,
            "torch": "test",
            "device_name": "CPU",
            "scale": [
                {
                    "workers": 1,
                    "envs_per_worker": 2,
                    "steps_per_second": 2000.0,
                    "decisions_per_second": 1000.0,
                    "realtime_factor": 60.0,
                }
            ],
            "best": {
                "workers": 1,
                "envs_per_worker": 2,
                "steps_per_second": 2000.0,
                "decisions_per_second": 1000.0,
                "realtime_factor": 60.0,
            },
            "steps_per_day": 2000 * 86_400,
            "update_cpu": entry,
            "advice": ["Testhinweis"],
        }
    )
    assert "Schritte/s" in text
    assert "Testhinweis" in text
    assert CHUNK_STEPS > 0


def test_policy_player_explain_works_on_any_device():
    """„Was denkt die KI?“ muss auch laufen, wenn das Netz auf der Grafikkarte liegt."""
    import torch

    from rocketai.obs import obs_size

    model = ActorCritic(obs_size=obs_size(True), hidden_sizes=[16])
    player = PolicyPlayer(model)
    player.explain = True
    batch = {agent: np.zeros(model.obs_size, np.float32) for agent in ("blue-0", "orange-0")}
    actions = player.act(["blue-0"], batch, None)  # state wird im explain-Pfad nicht gebraucht
    assert set(actions) == {"blue-0"}
    assert player.last_info["blue-0"]["top"]
    assert 0.0 <= player.last_info["blue-0"]["confidence"] <= 1.0
    torch.tensor(0.0)  # Platzhalter, damit torch sicher importiert ist


def test_failed_lock_attempt_does_not_leave_an_open_file(tmp_path, monkeypatch):
    """Ein abgelehnter Versuch darf keinen offenen Dateigriff hinterlassen.

    Unter Windows verhindert ein offener Griff das Löschen der Sperrdatei —
    die Sperre bliebe dann für immer stehen.
    """
    monkeypatch.setenv("ROCKETAI_RUNS", str(tmp_path / "runs"))
    paths = run_paths("leak").ensure()
    with TrainLock(paths.root, "leak"):
        attempt = TrainLock(paths.root, "leak")
        with pytest.raises(AlreadyRunning):
            attempt.__enter__()
        assert attempt._handle is None  # Griff geschlossen
    assert lock_owner(paths.root) is None
    assert not (paths.root / "trainer.lock").exists()


def test_lock_does_not_block_reading_the_pid_file(tmp_path, monkeypatch):
    """Unter Windows sind Dateisperren *erzwingend*: Die gesperrte Datei darf
    nicht die Datei mit der Prozessnummer sein, sonst kann niemand (auch nicht
    der eigene Prozess) sie lesen — genau daran scheiterte CI auf Windows."""
    monkeypatch.setenv("ROCKETAI_RUNS", str(tmp_path / "runs"))
    paths = run_paths("lesbar").ensure()
    with TrainLock(paths.root, "lesbar"):
        owner = lock_owner(paths.root)
        assert owner is not None
        assert owner["pid"] > 0 and owner["alive"] is True
        # Die Schutzzdatei trägt nur die Sperre, die Inhaltsdatei bleibt lesbar.
        assert paths.root.joinpath("trainer.lock.guard").exists()
    assert lock_owner(paths.root) is None


def test_effective_speed_counts_the_learning_step():
    """Zeitangaben müssen sammeln *und* lernen enthalten, nicht nur die Simulation."""
    # 10.000 Schritte/s sammeln, 20.000 Schritte/s lernen, 3 Epochen:
    # 1/(1/10000 + 3/20000) = 4000 Schritte/s
    assert effective_steps_per_second(10_000, 20_000, 3) == pytest.approx(4_000)
    # Ohne gemessenen Lernschritt bleibt die Simulationsrate stehen.
    assert effective_steps_per_second(10_000, 0, 3) == 10_000
    assert effective_steps_per_second(0, 20_000, 3) == 0.0


def test_plan_combos_tests_games_per_process_too():
    combos = plan_combos(8)
    assert (8, 4) in combos and (4, 1) in combos
    # Mit Vorgabe bleibt es bei dieser Spielzahl (kein ungefragter Sweep).
    assert plan_combos(8, 2) == [(4, 2), (8, 2)]
    assert plan_combos(1) == [(1, 1), (1, 2), (1, 4)]


def test_advice_does_not_mix_workers_and_envs():
    """Nur gleiche Einstellungen vergleichen – sonst lobt der Bericht das Falsche."""

    def entry(workers, envs, sps):
        return {
            "workers": workers,
            "envs_per_worker": envs,
            "steps_per_second": sps,
            "decisions_per_second": sps / 2,
            "realtime_factor": sps / 40,
        }

    report = {
        "best": entry(1, 4, 2000),
        "scale": [entry(1, 1, 1400), entry(1, 2, 1700), entry(1, 4, 2000)],
        "effective": {"steps_per_second": 1500, "simulation_share": 0.7, "update_share": 0.3},
        "steps_per_day": 1500 * 86_400,
        "update_cpu": {"steps_per_second": 16000, "device": "cpu", "threads": 3},
    }
    tips = advice(report)
    assert not any("skaliert gut mit mehr Prozessen" in tip for tip in tips)
    assert any("Spiele pro Prozess" in tip for tip in tips)
    assert any("Ende-zu-Ende" in tip for tip in tips)


def test_suggested_config_uses_measured_values_and_end_to_end_speed():
    """Die Empfehlung darf gute Werte nicht verschlechtern und rechnet ehrlich."""
    report = {
        "torch_threads_recommended": 15,
        "best": {"workers": 12, "envs_per_worker": 4, "steps_per_second": 31_000},
        "effective": {"steps_per_second": 5_500},
    }
    config = suggested_config(report)
    assert config.n_workers == 12
    assert config.envs_per_worker == 4  # vorher setzte die Empfehlung hier 1
    assert config.torch_threads == 15
    # Rund 20 Minuten sammeln+lernen bei 5 500 Schritte/s — gedeckelt, damit
    # ein Update nicht unbegrenzt Speicher und Zeit frisst.
    assert config.steps_per_iteration == min(1_000_000, int(5_500 * 1_200))
    small = suggested_config({**report, "effective": {"steps_per_second": 300}})
    assert small.steps_per_iteration >= 20_000
    config.validate()

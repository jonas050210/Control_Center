import pytest
from fastapi.testclient import TestClient

from rocketai import server
from rocketai.config import TrainConfig, run_paths
from rocketai.match import play_match, save_replay
from rocketai.model import ActorCritic, save_checkpoint
from rocketai.opponents import make_player


@pytest.fixture
def client(monkeypatch):
    started = []
    monkeypatch.setattr(server.STATE, "start_training", lambda name: started.append(name))
    test_client = TestClient(server.create_app())
    test_client.started = started
    return test_client


def make_run(name="demo"):
    config = TrainConfig(name=name, hidden_sizes=[16])
    paths = run_paths(name).ensure()
    config.save(paths.config)
    save_checkpoint(
        paths.checkpoints / "1000.pt",
        ActorCritic(hidden_sizes=[16]),
        steps=1000,
        config=config.to_dict(),
    )
    (paths.metrics).write_text('{"steps": 1000, "touches_per_minute": 1.5}\n')
    return paths


def test_static_and_overview(client):
    assert "RocketAI" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    data = client.get("/api/overview").json()
    assert data["runs"] == [] and data["play"]["state"] == "idle"


def test_create_run_validates_and_starts(client):
    assert (
        client.post("/api/runs", json={"name": "bad name", "preset": "beginner"}).status_code == 400
    )
    response = client.post(
        "/api/runs", json={"name": "fresh", "preset": "quick-test", "overrides": {"team_size": 2}}
    )
    assert response.status_code == 201
    assert response.json()["config"]["team_size"] == 2
    assert client.started == ["fresh"]
    assert client.post("/api/runs", json={"name": "fresh"}).status_code == 409


def test_run_detail_metrics_stop_delete(client):
    make_run()
    detail = client.get("/api/runs/demo").json()
    assert detail["checkpoints"][0]["steps"] == 1000
    assert client.get("/api/runs/demo/metrics").json()[0]["touches_per_minute"] == 1.5
    client.post("/api/runs/demo/stop")
    assert run_paths("demo").control.exists()
    assert client.get("/api/runs/missing").status_code == 404
    assert client.delete("/api/runs/demo").status_code == 200
    assert not run_paths("demo").root.exists()


def test_checkpoint_specs_cannot_escape(client):
    make_run()
    for spec in ("../demo/1000.pt", "demo/../../x.pt", "demo/1000.txt", "demo\\1000.pt"):
        assert client.post("/api/matches", json={"blue": spec, "orange": "idle"}).status_code in (
            400,
            404,
        )


def test_match_job_and_replay(client):
    make_run()
    job = client.post(
        "/api/matches", json={"blue": "demo/1000.pt", "orange": "chaser", "seconds": 10}
    ).json()
    server.STATE.executor.submit(lambda: None).result()  # wait for the queue
    info = client.get(f"/api/jobs/{job['id']}").json()
    assert info["state"] == "done", info
    replay = client.get(f"/api/replays/{info['result']['replay']}").json()
    assert replay["frames"] and client.get("/api/replays").json()[0]["blue"] == "demo@1000"


def test_replay_listing_reads_meta(client):
    paths = make_run()
    result = play_match(make_player("chaser"), make_player("idle"), seconds=5, record=True)
    save_replay(paths.replays / "1000.json", result, {"kind": "evaluation"})
    item = client.get("/api/replays").json()[0]
    assert item["kind"] == "evaluation" and item["blue"] == "chaser"


def test_play_and_system_endpoints(client):
    make_run()
    info = client.get("/api/play").json()
    assert set(info["modes"]) == {"psyonix", "human", "bot", "self"}
    assert client.post("/api/play/start", json={"checkpoint": "demo/nope.pt"}).status_code == 404
    checks = client.get("/api/system").json()["checks"]
    assert any(c["key"] == "torch" and c["ok"] for c in checks)


def test_observation_endpoint_describes_what_the_ai_sees(client):
    from rocketai.config import OBS_BASE_SIZE, OBS_SIZE
    from rocketai.obs import EXTRA_FEATURES

    data = client.get("/api/obs").json()
    assert data["size"] == OBS_SIZE
    assert data["base_size"] == OBS_BASE_SIZE
    assert data["extra_size"] == len(EXTRA_FEATURES)
    assert [item["key"] for item in data["extras"]] == list(EXTRA_FEATURES)
    assert all(item["label"] for item in data["extras"])
    assert data["decisions_per_second"] == pytest.approx(15.0)


def test_system_endpoint_includes_observation_and_benchmark(client):
    data = client.get("/api/system").json()
    assert data["observation"]["size"] > data["observation"]["base_size"]
    assert "benchmark" in data


def test_config_check_reports_problems_and_hints(client):
    ok = client.post("/api/config/check", json={"preset": "quick-test", "overrides": {"name": "x"}})
    assert ok.status_code == 200
    body = ok.json()
    assert body["ok"] is True and body["problems"] == []

    bad = client.post(
        "/api/config/check",
        json={"preset": "quick-test", "overrides": {"name": "x", "team_size": 9}},
    )
    body = bad.json()
    assert body["ok"] is False and any("team_size" in problem for problem in body["problems"])

    hinted = client.post(
        "/api/config/check",
        json={"preset": "quick-test", "overrides": {"name": "x", "n_workers": 999}},
    )
    assert hinted.json()["ok"] is True
    assert hinted.json()["hints"]


def test_run_summary_shows_hints_and_separate_team_stats(client):
    paths = make_run("mit-hinweisen")
    config = TrainConfig.load(paths.config)
    config.n_workers = 9999  # nicht falsch, aber ein Hinweis wert
    config.save(paths.config)
    (paths.metrics).write_text(
        '{"steps": 1000, "touches_per_minute": 2.0, "touches_against_per_minute": 9.0,'
        ' "goals_per_minute": 0.1, "goals_against_per_minute": 0.7, "realtime_factor": 90.0}\n'
    )
    run = client.get("/api/runs/mit-hinweisen").json()
    assert run["hints"]
    assert run["last"]["touches_against_per_minute"] == 9.0
    assert run["last"]["realtime_factor"] == 90.0

"""Live matches, the Rocket League check and the brain explanations."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rocketai import rlcheck, server
from rocketai.opponents import ACTION_LABELS, explain_decision


@pytest.fixture
def client():
    yield TestClient(server.create_app())
    from rocketai.live import LIVE

    LIVE.stop()


def test_action_labels_unique_and_readable():
    assert len(ACTION_LABELS) == 90
    assert len(set(ACTION_LABELS)) == 90
    assert "Gas · Boost" in ACTION_LABELS
    assert any(label.startswith("Flip vorwärts") for label in ACTION_LABELS)


def test_explain_decision_shape():
    probs = np.full(90, 1 / 90)
    info = explain_decision(probs, 1.5, 3)
    assert info["action"] == 3 and len(info["controls"]) == 8 and len(info["top"]) == 5
    assert info["confidence"] == pytest.approx(0.0, abs=1e-6)
    sure = np.zeros(90)
    sure[7] = 1.0
    assert explain_decision(sure, 0.0, 7)["confidence"] == pytest.approx(1.0)


VDF = r"""
"libraryfolders"
{
  "0" { "path" "C:\\Program Files (x86)\\Steam" "apps" { "228980" "1" } }
  "1" { "path" "D:\\SteamLibrary" "apps" { "252950" "1" } }
}
"""


def test_parse_steam_files():
    assert rlcheck.parse_vdf_paths(VDF) == ["C:\\Program Files (x86)\\Steam", "D:\\SteamLibrary"]
    acf = '"AppState" { "appid" "252950" "installdir" "rocketleague" }'
    assert rlcheck.parse_acf_installdir(acf) == "rocketleague"
    assert rlcheck.parse_acf_installdir("{}") is None


def test_find_steam_install(tmp_path):
    library = tmp_path / "lib"
    (tmp_path / "steamapps").mkdir()
    (tmp_path / "steamapps" / "libraryfolders.vdf").write_text(f'"path" "{library}"')
    (library / "steamapps").mkdir(parents=True)
    (library / "steamapps" / "appmanifest_252950.acf").write_text('"installdir" "rocketleague"')
    exe = library / "steamapps" / "common" / "rocketleague" / rlcheck.EXE_RELATIVE
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert rlcheck.find_steam_install(tmp_path) == exe


def test_find_epic_install(tmp_path):
    game = tmp_path / "rocketleague"
    exe = game / rlcheck.EXE_RELATIVE
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    manifests = tmp_path / "Manifests"
    manifests.mkdir()
    (manifests / "a.item").write_text(json.dumps({"AppName": "Fortnite", "InstallLocation": "x"}))
    (manifests / "b.item").write_text(
        json.dumps({"AppName": "Sugar", "InstallLocation": str(game)})
    )
    (manifests / "broken.item").write_text("{nope")
    assert rlcheck.find_epic_install(manifests) == exe


def test_classify_and_summarize():
    assert (
        rlcheck.classify_process("RocketLeague.exe -rlbot RLBot_ControllerURL=127.0.0.1:23233")
        == "rlbot"
    )
    assert rlcheck.classify_process('"C:\\RL\\RocketLeague.exe" -EpicPortal') == "normal"
    assert rlcheck.classify_process(None) == "normal"

    base = dict(platform="win32", supported=True, install="C:/RL/RocketLeague.exe", store="Steam")
    assert rlcheck.summarize(rlcheck.RLStatus(**base, game="not_running")).ready
    normal = rlcheck.summarize(rlcheck.RLStatus(**base, game="normal"))
    assert not normal.ready and "schließen" in normal.verdict
    missing = rlcheck.summarize(
        rlcheck.RLStatus(platform="win32", supported=True, game="not_running")
    )
    assert not missing.ready and "nicht gefunden" in missing.verdict
    linux = rlcheck.summarize(rlcheck.RLStatus(platform="linux", supported=False))
    assert not linux.ready and linux.steps[0]["state"] == "bad"


def test_live_session_with_scripted_bots(client):
    response = client.post(
        "/api/live/start", json={"blue": "chaser", "orange": "defender", "speed": 8}
    )
    assert response.status_code == 200, response.text
    deadline = time.time() + 20
    state = {}
    while time.time() < deadline:
        state = client.get("/api/live?since=0").json()
        if len(state["frames"]) >= 10:
            break
        time.sleep(0.2)
    assert state["active"] and len(state["frames"]) >= 10
    frame = state["frames"][-1]["f"]
    assert len(frame[1]) == 3 and len(frame[2]) == 2 and len(frame[2][0]) == 9
    assert isinstance(frame[3], int)
    assert client.post("/api/live/control", json={"paused": True}).json()["paused"] is True
    stopped = client.post("/api/live/stop").json()
    assert stopped["active"] is False


def test_live_rejects_bad_specs(client):
    assert client.post("/api/live/start", json={"blue": "../etc/passwd"}).status_code == 400
    assert client.post("/api/live/start", json={"blue": "run:missing"}).status_code == 400
    assert (
        client.post("/api/live/start", json={"blue": "chaser", "team_size": 5}).status_code == 400
    )


def test_field_and_rocketleague_endpoints(client):
    field = client.get("/api/field").json()
    assert len(field["pads"]) == 34 and sum(p[2] for p in field["pads"]) == 6
    rl = client.get("/api/rocketleague").json()
    assert {"supported", "ready", "verdict", "steps"} <= rl.keys()
    assert client.get("/api/snapshot").json()["live"] in (True, False)
    for path in (
        "/js/core.js",
        "/js/field.js",
        "/js/field3d.js",
        "/js/stage.js",
        "/vendor/three.min.js",
    ):
        assert client.get(path).status_code == 200, path


def test_live_ai_vs_ai_reports_every_brain(client):
    from test_training import _small_config

    from rocketai.trainer import train

    train(_small_config(total_steps=600))
    response = client.post("/api/live/start", json={"blue": "run:t", "orange": "run:t", "speed": 8})
    assert response.status_code == 200, response.text
    deadline = time.time() + 30
    frames = []
    while time.time() < deadline and len(frames) < 5:
        frames = client.get("/api/live?since=0").json()["frames"]
        time.sleep(0.2)
    brains = frames[-1]["brains"]
    assert [b["car"] for b in brains] == [0, 1] and [b["team"] for b in brains] == [0, 1]
    assert len(brains[0]["controls"]) == 8 and brains[0]["top"]

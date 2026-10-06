"""End-to-end API tests for the FastAPI backend that replaced Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server import state as state_module
from server.app import create_app
from server.config import dependency_status


class ApiTestCase(unittest.TestCase):
    """Share one app instance but isolate filesystem writes into a temp folder."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="neural-arena-api-test-")
        cls.temp_root = Path(cls._temporary.name)
        cls._patchers = [
            patch.object(state_module, "CUSTOM_MAP_PATH", cls.temp_root / "custom_map.json"),
            patch.object(state_module, "DEMOS_PATH", cls.temp_root / "demos.csv"),
        ]
        for patcher in cls._patchers:
            patcher.start()
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # pragma: no cover - dev dependency
            for patcher in cls._patchers:
                patcher.stop()
            raise unittest.SkipTest(f"httpx/TestClient unavailable: {exc}") from exc
        cls.client = TestClient(create_app())

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in cls._patchers:
            patcher.stop()
        cls._temporary.cleanup()

    def setUp(self) -> None:
        state_module.STATE.map_overrides.clear()
        state_module.STATE.arena = state_module.ArenaSession()
        state_module.STATE.playground = state_module.PlaygroundSession()
        state_module.STATE.aim_game = None
        state_module.STATE.dodge_game = None

    # ------------------------------------------------------------------ base
    def test_health_and_meta_contract(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        payload = health.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("python", payload)
        self.assertIn("dependencies", payload)

        meta = self.client.get("/api/meta").json()
        self.assertEqual(len(meta["maps"]), 6)
        self.assertEqual(len(meta["weapons"]), 6)
        self.assertEqual(meta["observation_size"], 31)
        self.assertEqual(meta["action_size"], 10)
        # Perception contract: the frontend renders the "what does the AI see"
        # box and the vision dropdown straight from this metadata.
        self.assertEqual(meta["observation_version"], 2)
        self.assertEqual(len(meta["observation_fields"]), meta["observation_size"])
        self.assertEqual(meta["vision_modes"],
                         ["exact", "noisy", "coarse", "coarse_los"])
        self.assertEqual(meta["default_vision_mode"], "coarse_los")
        self.assertIn("Heuristic AI", meta["models"])
        self.assertEqual(meta["max_envs"], 24)
        self.assertGreater(meta["cpu"]["logical"], 0)
        # The legacy UI must not be required anywhere in the metadata contract.
        self.assertNotIn("streamlit", json.dumps(meta["paths"]).lower())

    def test_static_frontend_is_served(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("NEURAL ARENA", index.text)
        self.assertIn("three", index.text.lower())
        for asset in ("/css/style.css", "/js/app.js", "/js/panels/arena.js", "/favicon.svg"):
            with self.subTest(asset=asset):
                self.assertEqual(self.client.get(asset).status_code, 200)

    def test_unknown_map_is_rejected(self) -> None:
        self.assertEqual(self.client.get("/api/maps/NoSuchMap/scene").status_code, 404)

    # ----------------------------------------------------------------- arena
    def test_arena_config_step_run_and_reset(self) -> None:
        config = self.client.post("/api/arena/config", json={
            "map": "Dust", "weapon_a": "Shotgun", "weapon_b": "Sniper",
            "model_a": "Heuristic AI", "model_b": "Heuristic AI", "detail": "Performance",
        }).json()
        self.assertEqual(len(config["frame"]["agents"]), 2)
        self.assertIn("objects", config["static"])
        scene_key = config["scene_key"]
        self.assertTrue(scene_key.startswith("Dust:60:"))

        stepped = self.client.post("/api/arena/step", json={"steps": 5}).json()
        self.assertGreater(stepped["frame"]["frame"], 0)
        self.assertEqual(stepped["frame"]["agents"][0]["weapon"], "Shotgun")
        self.assertEqual(stepped["frame"]["agents"][1]["weapon"], "Sniper")
        self.assertEqual(stepped["scene_key"], scene_key)
        self.assertIsInstance(stepped["new_messages"], list)

        running = self.client.post("/api/arena/run", json={"running": True}).json()
        self.assertTrue(running["running"])
        stopped = self.client.post("/api/arena/run", json={"running": False}).json()
        self.assertFalse(stopped["running"])
        reset = self.client.post("/api/arena/reset").json()
        self.assertFalse(reset["running"])
        self.assertEqual(reset["messages"], [])

        # A different map must invalidate the cached scene key.
        other = self.client.post("/api/arena/config", json={"map": "Warehouse"}).json()
        self.assertNotEqual(other["scene_key"], scene_key)

    def test_arena_rejects_unknown_policy_without_crashing(self) -> None:
        config = self.client.post("/api/arena/config", json={
            "map": "Dust", "model_a": "does_not_exist.zip",
        }).json()
        self.assertEqual(config["models"]["a"], "Heuristic AI")

    def test_arena_perception_contract_and_vision_switch(self) -> None:
        # Default vision mode is the honest one; the arena starts behind cover.
        config = self.client.post("/api/arena/config", json={"map": "Arena"}).json()
        self.assertEqual(config["vision"], "coarse_los")
        perception = config["perception"]
        self.assertEqual(perception["vision_mode"], "coarse_los")
        self.assertEqual(perception["field_of_view_degrees"], 120.0)
        player = perception["player"]
        for key in ("visible", "bearing_sin", "bearing_cos", "distance_band", "hp_band",
                    "seconds_since_seen", "blind_steps", "visible_steps",
                    "memory", "memory_sin", "memory_cos"):
            with self.subTest(key=key):
                self.assertIn(key, player)
        # Sector quantisation: bearing and memory are unit vectors or exactly zero.
        for sin_key, cos_key in (("bearing_sin", "bearing_cos"), ("memory_sin", "memory_cos")):
            length = player[sin_key] ** 2 + player[cos_key] ** 2
            with self.subTest(vector=sin_key):
                self.assertAlmostEqual(length, 1.0 if length > 0.5 else 0.0, places=5)

        # Standalone perception endpoint mirrors the config payload.
        standalone = self.client.get("/api/arena/perception").json()
        self.assertEqual(standalone["vision_mode"], "coarse_los")
        self.assertIn("player", standalone)

        # Switching to the legacy exact mode is reported everywhere.
        exact = self.client.post("/api/arena/config", json={"vision": "exact"}).json()
        self.assertEqual(exact["vision"], "exact")
        self.assertEqual(exact["perception"]["vision_mode"], "exact")
        self.assertEqual(self.client.get("/api/arena/perception").json()["vision_mode"], "exact")

        # Unknown values are rejected instead of silently falling back.
        bad = self.client.post("/api/arena/config", json={"vision": "xray"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("vision", json.dumps(bad.json()).lower())

        # Stepping returns a fresh perception snapshot with monotone counters.
        self.client.post("/api/arena/config", json={"vision": "coarse_los"})
        stepped = self.client.post("/api/arena/step", json={"steps": 12}).json()
        self.assertIn("perception", stepped)
        stepped_player = stepped["perception"]["player"]
        self.assertGreaterEqual(stepped_player["blind_steps"] + stepped_player["visible_steps"],
                                player["blind_steps"] + player["visible_steps"])

    def test_playground_accepts_vision_mode(self) -> None:
        payload = self.client.post("/api/playground/config", json={
            "map": "Dust", "vision": "noisy",
        }).json()
        self.assertEqual(payload["vision"], "noisy")
        self.assertEqual(self.client.post(
            "/api/playground/config", json={"vision": "nonsense"}).status_code, 400)
        self.client.post("/api/playground/config", json={"vision": "coarse_los"})

    # ------------------------------------------------------------ playground
    def test_playground_actions_recording_and_demo_export(self) -> None:
        config = self.client.post("/api/playground/config", json={
            "map": "Arena", "weapon": "Pistol", "enemy_weapon": "AK-47",
            "bot": "Tactical", "detail": "Performance",
        }).json()
        self.assertEqual(config["demo_count"], 0)
        self.assertTrue(config["scene_key"].startswith("Arena:"))

        response = self.client.post("/api/playground/action", json={
            "press": "fire", "recording": True, "stance": 1, "sprint": False,
        }).json()
        self.assertGreater(response["demo_count"], 0)
        self.assertIn("frame", response)
        self.assertEqual(len(response["frame"]["agents"]), 2)

        unknown = self.client.post("/api/playground/action", json={"press": "teleport"})
        self.assertEqual(unknown.status_code, 400)

        csv_response = self.client.get("/api/playground/demos.csv")
        self.assertEqual(csv_response.status_code, 200)
        header = csv_response.text.splitlines()[0].split(",")
        self.assertEqual(len(header), 41)  # 31 state + 10 action columns
        self.assertTrue(header[0].startswith("state_"))

        saved = self.client.post("/api/playground/demos/save").json()
        self.assertGreater(saved["saved"], 0)
        self.assertTrue(state_module.DEMOS_PATH.exists())
        # The exported demos must carry the observation layout they were recorded
        # with, otherwise imitation training would silently learn from stale columns.
        from env.shooter_env import OBSERVATION_VERSION
        from training.imitation import demo_meta_path

        meta = json.loads(demo_meta_path(state_module.DEMOS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(meta["observation_version"], OBSERVATION_VERSION)
        afterwards = self.client.post("/api/playground/action",
                                      json={"press": "wait", "recording": False}).json()
        self.assertEqual(afterwards["demo_count"], 0)
        self.assertFalse(afterwards["recording"])

        reset = self.client.post("/api/playground/reset").json()
        self.assertEqual(reset["reward"], 0.0)
        self.assertEqual(reset["messages"], [])

    def test_playground_accepts_controls_payload(self) -> None:
        self.client.post("/api/playground/config", json={"map": "Dust"})
        response = self.client.post("/api/playground/action", json={
            "controls": {"move": 1, "shoot": True}, "repeat": 2,
        }).json()
        self.assertIn("frame", response)
        self.assertGreater(response["frame"]["frame"], 0)

    # -------------------------------------------------------------- minigames
    def test_aim_drill_endpoints(self) -> None:
        scene = self.client.get("/api/aim/scene").json()
        self.assertEqual(len(scene["static"]["targets"]), 9)
        self.assertIsNone(scene["game"])

        started = self.client.post("/api/aim/start", json={"duration_seconds": 20}).json()
        target = started["game"]["target"]
        self.assertTrue(started["game"]["active"])

        hit = self.client.post("/api/aim/tap", json={"cell": target}).json()
        self.assertEqual(hit["game"]["hits"], 1)
        # The drill moves the target after a hit, so the "miss" cell has to be
        # taken from the *new* target - otherwise the tap can hit by chance and
        # the assertion fails intermittently (measured: 1 in 5 runs).
        current_target = hit["game"]["target"]
        self.assertNotEqual(current_target, target)
        miss = self.client.post("/api/aim/tap", json={"cell": (current_target + 1) % 9}).json()
        self.assertEqual(miss["game"]["misses"], 1)
        self.assertEqual(self.client.post("/api/aim/tap", json={}).status_code, 422)

    def test_aim_drill_without_start_returns_error(self) -> None:
        state_module.STATE.aim_game = None
        self.assertEqual(self.client.post("/api/aim/tap", json={"cell": 0}).status_code, 400)

    def test_dodge_endpoints(self) -> None:
        scene = self.client.get("/api/dodge/scene").json()
        self.assertEqual(len(scene["static"]["cells"]), 25)
        started = self.client.post("/api/dodge/start").json()
        self.assertEqual(started["game"]["tick"], 0)
        stepped = self.client.post("/api/dodge/step", json={"dx": -1, "dy": 0}).json()
        self.assertEqual(stepped["game"]["tick"], 1)
        self.assertLessEqual(stepped["game"]["player"][0], 2)

    # ------------------------------------------------------------------- maps
    def test_map_scene_randomize_and_reset(self) -> None:
        scene = self.client.get("/api/maps/Dust/scene?detail=60").json()
        self.assertEqual(scene["stats"]["name"], "Dust")
        self.assertGreater(scene["stats"]["objects"], 0)
        self.assertEqual(len(scene["details"]["x"]), 60)
        self.assertTrue(scene["overridden"] is False)

        randomized = self.client.post("/api/maps/Dust/randomize").json()
        self.assertTrue(randomized["overridden"])
        self.assertNotEqual(randomized["key"], scene["key"])
        restored = self.client.post("/api/maps/Dust/reset").json()
        self.assertFalse(restored["overridden"])

    def test_custom_map_editor_roundtrip(self) -> None:
        added = self.client.post("/api/maps/custom/objects", json={
            "kind": "crate", "x": 4.0, "y": -5.5, "width": 2.0, "height": 1.5, "depth": 3.0,
        }).json()
        self.assertEqual(added["stats"]["objects"], 1)
        self.assertTrue(state_module.CUSTOM_MAP_PATH.exists())

        exported = self.client.get("/api/maps/custom/export")
        self.assertEqual(exported.status_code, 200)
        payload = json.loads(exported.text)
        self.assertEqual(len(payload["objects"]), 1)

        imported = self.client.post("/api/maps/custom/import", content=exported.content,
                                    headers={"content-type": "application/json"})
        self.assertEqual(imported.status_code, 200)
        self.assertEqual(imported.json()["stats"]["objects"], 1)

        removed = self.client.delete("/api/maps/custom/objects/0").json()
        self.assertEqual(removed["stats"]["objects"], 0)

        invalid = self.client.post("/api/maps/custom/import", content=b'{"objects":[{"kind":"script"}]}',
                                   headers={"content-type": "application/json"})
        self.assertEqual(invalid.status_code, 400)

    def test_custom_map_scene_uses_saved_layout(self) -> None:
        self.client.post("/api/maps/custom/objects", json={"kind": "pillar", "x": 1.0, "y": 2.0})
        scene = self.client.get("/api/maps/Custom/scene?detail=60").json()
        names = [item["name"] for item in scene["objects"]]
        self.assertTrue(any("Pillar" in name or "pillar" in name for name in names))

    # --------------------------------------------------------------------- ttk
    def test_ttk_defaults_and_simulation(self) -> None:
        defaults = self.client.get("/api/ttk").json()
        self.assertEqual(defaults["weapons"][0], "Pistol")
        self.assertIn("limits", defaults)
        self.assertIn("damage", defaults["specs"]["AK-47"])

        result = self.client.post("/api/ttk/simulate", json={
            "weapon_a": "Pistol", "weapon_b": "Sniper", "trials": 2_000, "distance": 30,
        }).json()
        self.assertAlmostEqual(result["win_rate_a"] + result["win_rate_b"] + result["draw_rate"], 1.0,
                               places=6)
        self.assertEqual(len(result["histogram_edges"]), 21)
        self.assertIn("Pistol", result["lua"])
        self.assertTrue(result["verdict"])

    # ------------------------------------------------------------ analytics
    def test_heatmap_payload_is_json_serializable(self) -> None:
        payload = self.client.post("/api/heatmap", json={
            "map": "Dust", "mode": "Both", "episode_range": [1, 10], "distance_range": [0, 100],
        }).json()
        self.assertEqual(payload["bins"], 48)
        self.assertEqual(len(payload["grid"]), 48)
        self.assertEqual(len(payload["grid"][0]), 48)
        self.assertIsInstance(payload["grid"][0][0], float)
        self.assertIn("All weapons", payload["weapons"])
        self.assertTrue(payload["objects"])

    def test_stats_payload_contract(self) -> None:
        payload = self.client.get("/api/stats").json()
        for key in ("series", "histogram", "summary", "resources", "weapons"):
            self.assertIn(key, payload)
        for key in ("steps", "reward", "win_rate", "fps"):
            self.assertIn(key, payload["series"])
        self.assertIn("status", payload)

    def test_benchmark_snapshot_without_run(self) -> None:
        payload = self.client.get("/api/benchmark").json()
        self.assertEqual(payload["status"], "stopped")
        self.assertEqual(payload["results"], [])
        self.assertGreater(payload["total"], 0)
        self.assertEqual(payload["completed"], 0)

    def test_training_status_defaults_to_stopped(self) -> None:
        state_module.STATE.training_job = None
        payload = self.client.get("/api/training/status").json()
        self.assertEqual(payload["status"], "stopped")
        self.assertIsNone(payload["config"])
        self.assertEqual(self.client.post("/api/training/bogus").status_code, 404)

    def test_training_status_reports_the_stored_evaluation(self) -> None:
        """A restart must not hide the verified report that is on disk."""
        report = {"model": "best_model.zip", "phase": 1, "vision_mode": "coarse_los",
                  "episodes_per_matchup": 8,
                  "runs": [{"bot": "stationary", "win_rate": 1.0, "kill_rate": 0.5}],
                  "win_rate_overall": 1.0, "kill_rate_overall": 0.5}
        with tempfile.TemporaryDirectory(prefix="neural-arena-eval-") as temporary:
            models = Path(temporary)
            state_module.STATE.training_job = None
            with patch.object(state_module, "MODELS_DIR", models):
                # No report on disk yet -> the panel says "noch keine Bewertung".
                self.assertIsNone(self.client.get("/api/training/status").json()["evaluation"])
                (models / "best_model_eval.json").write_text(json.dumps(report), encoding="utf-8")
                payload = self.client.get("/api/training/status").json()
            self.assertEqual(payload["status"], "stopped")
            self.assertEqual(payload["evaluation"]["model"], "best_model.zip")
            self.assertEqual(payload["evaluation"]["runs"][0]["bot"], "stationary")

            # A broken file must not break the endpoint either.
            (models / "best_model_eval.json").write_text("{not json", encoding="utf-8")
            with patch.object(state_module, "MODELS_DIR", models):
                self.assertIsNone(self.client.get("/api/training/status").json()["evaluation"])

    def test_training_start_requires_ppo_dependencies(self) -> None:
        dependencies = dependency_status()
        if dependencies["stable_baselines3"] and dependencies["torch"]:
            self.skipTest("PPO dependencies are installed; starting a real run is out of scope here.")
        response = self.client.post("/api/training/start", json={"duration_minutes": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("install.py", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

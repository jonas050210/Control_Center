"""In-process application state shared by the REST endpoints.

Everything the old Streamlit app kept in ``st.session_state`` lives here now:
the live Arena match, the Playground duel, the mini-game runs, the map lab
overrides, the training controller, and the benchmark runner. A single
re-entrant lock serializes mutations because FastAPI runs sync endpoints in a
thread pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

from env.map_io import map_from_json, save_map, load_map
from env.maps import ArenaMap, ArenaObject, MAP_NAMES, create_map, randomize_cover, serialize_map
from env.shooter_env import ACTION_SIZE, OBSERVATION_SIZE, ShooterEnv
from env.weapons import WEAPON_NAMES, get_weapon
from server import analytics, scene
from server.actions import BOT_BEHAVIORS, HEURISTIC, PLAY_ACTIONS, action_from_controls
from server.config import (
    CUSTOM_MAP_PATH,
    DATA_DIR,
    DEMOS_PATH,
    DETAIL_PRESETS,
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    dependency_status,
    ensure_runtime_dirs,
    relative_path,
    runtime_info,
)
from server.minigames import (
    aim_tap,
    new_aim_game,
    new_dodge_game,
    public_aim_game,
    step_dodge_game,
)
from server.policies import PolicyCache, policy_choices

MAX_RECORDED_DEMOS = 25_000
MAX_ARENA_EVENTS = 500
ARENA_EVENT_FEED = 80


@dataclass
class ArenaSession:
    env: ShooterEnv | None = None
    signature: tuple[Any, ...] | None = None
    scene_key: str | None = None
    models: tuple[str, str] = (HEURISTIC, HEURISTIC)
    detail_count: int = DETAIL_PRESETS["Balanced"]
    running: bool = False
    messages: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PlaygroundSession:
    env: ShooterEnv | None = None
    signature: tuple[Any, ...] | None = None
    scene_key: str | None = None
    bot_label: str = "Tactical"
    detail_count: int = DETAIL_PRESETS["Balanced"]
    messages: list[str] = field(default_factory=list)
    reward: float = 0.0
    result: dict[str, Any] | None = None
    recording: bool = False
    demo_rows: list[tuple[list[float], list[int]]] = field(default_factory=list)
    stance: int = 0
    sprint: bool = False


class ServerState:
    """Owns every mutable session object of the control center."""

    def __init__(self) -> None:
        ensure_runtime_dirs()
        self.lock = threading.RLock()
        self.arena = ArenaSession()
        self.playground = PlaygroundSession()
        self.aim_game: dict[str, Any] | None = None
        self.aim_best = 0
        self.dodge_game: dict[str, Any] | None = None
        self.dodge_best = 0
        self.map_overrides: dict[str, ArenaMap] = {}
        self.map_seed = 2026
        self.notices: list[str] = []
        self.policies = PolicyCache()
        self.training_job: Any | None = None
        self.benchmark_runner: Any | None = None
        self.ttk_state: dict[str, Any] = {
            "a": {"weapon": "Pistol"},
            "b": {"weapon": "AK-47"},
            "trials": 5000,
            "distance": 20,
        }
        self._custom_map_error: str | None = None

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict[str, Any]:
        from training.workers import auto_worker_count, cpu_core_counts, valid_benchmark_configurations

        physical, logical = cpu_core_counts()
        deps = dependency_status()
        return {
            "app": runtime_info(),
            "maps": list(MAP_NAMES),
            "weapons": list(WEAPON_NAMES),
            "detail_presets": DETAIL_PRESETS,
            "models": policy_choices(MODELS_DIR),
            "heuristic": HEURISTIC,
            "bot_behaviors": list(BOT_BEHAVIORS),
            "play_actions": sorted(PLAY_ACTIONS),
            "action_size": ACTION_SIZE,
            "observation_size": OBSERVATION_SIZE,
            "cpu": {"physical": physical, "logical": logical,
                    "auto_workers": auto_worker_count()},
            "benchmark_configs": [list(item) for item in valid_benchmark_configurations()],
            "training_ready": bool(deps["stable_baselines3"] and deps["torch"]),
            "dependencies": deps,
            "max_envs": 24,
            "custom_map_path": relative_path(CUSTOM_MAP_PATH),
            "demo_count": len(self.playground.demo_rows),
            "paths": {
                "models": relative_path(MODELS_DIR),
                "logs": relative_path(LOGS_DIR),
                "data": relative_path(DATA_DIR),
            },
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", **runtime_info(),
                "training_status": self._training_status(),
                "benchmark_status": self._benchmark_status()}

    # ------------------------------------------------------------------ maps
    def custom_map(self) -> ArenaMap:
        """Return the Custom layout, loading the saved JSON once per process."""
        if "Custom" in self.map_overrides:
            return self.map_overrides["Custom"]
        if CUSTOM_MAP_PATH.exists():
            try:
                loaded = load_map(CUSTOM_MAP_PATH, force_custom_name=True)
                self._custom_map_error = None
            except ValueError as exc:
                loaded = create_map("Custom")
                self._custom_map_error = f"Could not read data/custom_map.json: {exc}"
        else:
            loaded = create_map("Custom")
        self.map_overrides["Custom"] = loaded
        return loaded

    def map_for(self, name: str) -> ArenaMap:
        if name in self.map_overrides:
            return self.map_overrides[name]
        if name == "Custom":
            return self.custom_map()
        return create_map(name)

    def map_scene(self, name: str, detail_count: int, show_spawns: bool = True) -> dict[str, Any]:
        arena_map = self.map_for(name)
        payload = scene.map_scene(arena_map, detail_count, show_spawns)
        payload["stats"] = map_stats(arena_map)
        payload["breakdown"] = map_breakdown(arena_map)
        payload["load_error"] = self._custom_map_error if name == "Custom" else None
        payload["overridden"] = name in self.map_overrides
        payload["serialized"] = serialize_map(arena_map)
        return payload

    def randomize_map(self, name: str) -> dict[str, Any]:
        arena_map = self.map_for(name)
        self.map_seed += 1
        randomized = randomize_cover(arena_map, seed=self.map_seed)
        self.map_overrides[name] = randomized
        if name == "Custom":
            self._save_custom_map(randomized, "Randomized Custom layout")
        return self.map_scene(name, self.arena.detail_count)

    def reset_map(self, name: str) -> dict[str, Any]:
        self.map_overrides.pop(name, None)
        if name == "Custom":
            try:
                CUSTOM_MAP_PATH.unlink(missing_ok=True)
                self._custom_map_error = None
                self.notice("Custom layout reset to an empty 50 × 50 m map.")
            except OSError as exc:
                self._custom_map_error = f"Saved Custom map could not be removed: {exc}"
        return self.map_scene(name, self.arena.detail_count)

    def add_custom_object(self, payload: dict[str, Any]) -> dict[str, Any]:
        arena_map = self.custom_map().copy()
        arena_map.objects.append(ArenaObject(
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            z=float(payload.get("z", 0.0)),
            width=float(payload.get("width", 2.0)),
            height=float(payload.get("height", 1.5)),
            depth=float(payload.get("depth", 2.0)),
            kind=str(payload.get("kind", "crate")),
            name=str(payload.get("name") or f"Custom {str(payload.get('kind', 'crate')).title()}"),
            yaw=float(payload.get("yaw", 0.0)),
        ))
        self.map_overrides["Custom"] = arena_map
        self._save_custom_map(arena_map, "Custom layout")
        return self.map_scene("Custom", self.arena.detail_count)

    def delete_custom_object(self, index: int) -> dict[str, Any]:
        arena_map = self.custom_map().copy()
        if 0 <= index < len(arena_map.objects):
            removed = arena_map.objects.pop(index)
            self.notice(f"Removed {removed.name or removed.kind}.")
        self.map_overrides["Custom"] = arena_map
        self._save_custom_map(arena_map, "Custom layout")
        return self.map_scene("Custom", self.arena.detail_count)

    def import_custom_map(self, raw: bytes) -> dict[str, Any]:
        imported = map_from_json(raw, force_custom_name=True)
        self.map_overrides["Custom"] = imported
        self._save_custom_map(imported, "Imported Custom map")
        return self.map_scene("Custom", self.arena.detail_count)

    def export_custom_map(self) -> str:
        arena_map = self.custom_map()
        return json.dumps(serialize_map(arena_map), indent=2, ensure_ascii=False, allow_nan=False)

    def _save_custom_map(self, arena_map: ArenaMap, label: str) -> None:
        try:
            save_map(CUSTOM_MAP_PATH, arena_map)
            self._custom_map_error = None
            self.notice(f"{label} saved to data/custom_map.json.")
        except (OSError, ValueError) as exc:
            self._custom_map_error = f"{label} could not be saved: {exc}"

    def notice(self, message: str) -> None:
        self.notices.append(message)
        self.notices = self.notices[-20:]

    def pop_notices(self) -> list[str]:
        notices, self.notices = self.notices, []
        return notices

    # ----------------------------------------------------------------- arena
    def arena_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            map_name = str(payload.get("map", MAP_NAMES[0]))
            if map_name not in MAP_NAMES:
                raise ValueError(f"Unknown map {map_name!r}")
            weapon_a = self._canonical_weapon(payload.get("weapon_a", WEAPON_NAMES[0]))
            weapon_b = self._canonical_weapon(payload.get("weapon_b", WEAPON_NAMES[0]))
            model_options = policy_choices(MODELS_DIR)
            model_a = str(payload.get("model_a", HEURISTIC))
            model_b = str(payload.get("model_b", HEURISTIC))
            if model_a not in model_options:
                model_a = HEURISTIC
            if model_b not in model_options:
                model_b = HEURISTIC
            detail_name = str(payload.get("detail", "Balanced"))
            detail_count = DETAIL_PRESETS.get(detail_name, DETAIL_PRESETS["Balanced"])

            arena_map = self.map_for(map_name)
            layout_signature = _layout_signature(arena_map)
            signature = (map_name, weapon_a, weapon_b, layout_signature, detail_count)
            session = self.arena
            session.models = (model_a, model_b)
            session.detail_count = detail_count
            if session.signature != signature or session.env is None:
                if session.env is not None:
                    session.env.close()
                session.env = ShooterEnv(
                    map_name=arena_map,
                    weapon_name=weapon_a,
                    opponent_weapon=weapon_b,
                    curriculum=False,
                    opponent_mode="full",
                    frame_skip=4,
                    max_episode_seconds=120.0,
                )
                session.env.reset()
                session.signature = signature
                session.messages = []
                session.running = False
                session.scene_key = None
            static = scene.map_scene(arena_map, detail_count, show_spawns=True)
            session.scene_key = static["key"]
            return {
                "static": static,
                "scene_key": static["key"],
                "frame": scene.match_frame(session.env),
                "running": session.running,
                "messages": session.messages[-ARENA_EVENT_FEED:],
                "models": {"a": model_a, "b": model_b},
                "load_errors": self._arena_model_errors(),
            }

    def arena_set_running(self, running: bool) -> dict[str, Any]:
        with self.lock:
            session = self.arena
            if session.env is None:
                raise ValueError("Configure the arena before starting a match")
            if running and session.env.get_snapshot()["done"]:
                session.env.reset()
                session.messages = []
            session.running = bool(running)
            return {"running": session.running, "frame": scene.match_frame(session.env),
                    "messages": session.messages[-ARENA_EVENT_FEED:]}

    def arena_reset(self) -> dict[str, Any]:
        with self.lock:
            session = self.arena
            if session.env is None:
                raise ValueError("Configure the arena before resetting")
            session.env.reset()
            session.messages = []
            session.running = False
            return {"running": False, "frame": scene.match_frame(session.env), "messages": []}

    def arena_step(self, steps: int = 1) -> dict[str, Any]:
        """Advance the match, returning only the dynamic frame payload."""
        with self.lock:
            session = self.arena
            if session.env is None:
                raise ValueError("Configure the arena before stepping")
            env = session.env
            requested = max(0, min(60, int(steps)))
            new_messages: list[str] = []
            errors: list[str] = []
            for _ in range(requested):
                if env.get_snapshot()["done"]:
                    session.running = False
                    break
                new_messages.extend(self._arena_advance(env, errors))
                if env.get_snapshot()["done"]:
                    session.running = False
                    break
            return {
                "frame": scene.match_frame(env),
                "scene_key": session.scene_key,
                "running": session.running,
                "new_messages": new_messages,
                "messages": session.messages[-ARENA_EVENT_FEED:],
                "events_logged": len(session.events),
                "load_errors": errors,
            }

    def _arena_advance(self, env: ShooterEnv, errors: list[str]) -> list[str]:
        session = self.arena
        messages: list[str] = []
        model_a, model_b = session.models
        observation_a = env.get_observation(0)
        observation_b = env.get_observation(1)
        action_a, error_a = self.policies.predict(model_a, observation_a, 0, env, MODELS_DIR)
        action_b, error_b = self.policies.predict(model_b, observation_b, 1, env, MODELS_DIR)
        for error in (error_a, error_b):
            if error and error not in errors:
                errors.append(error)
        _, _, terminated, truncated, info = env.step_duel(action_a, action_b)
        for event in info.get("combat_events", []):
            messages.append(scene.format_event(event))
        if terminated or truncated:
            metrics = info.get("episode_metrics", {})
            if metrics:
                session.events.append(metrics)
                session.events = session.events[-MAX_ARENA_EVENTS:]
                winner = "AGENT 1 WINS" if metrics.get("win", 0) else (
                    "DRAW" if metrics.get("draw") else "AGENT 2 WINS"
                )
                messages.append(
                    f"MATCH COMPLETE · {winner} · {float(metrics.get('ttk', 0.0)):.1f}s"
                )
            session.running = False
        session.messages.extend(messages)
        session.messages = session.messages[-ARENA_EVENT_FEED:]
        return messages

    def arena_state(self) -> dict[str, Any]:
        with self.lock:
            session = self.arena
            if session.env is None:
                return {"configured": False}
            return {
                "configured": True,
                "frame": scene.match_frame(session.env),
                "scene_key": session.scene_key,
                "running": session.running,
                "messages": session.messages[-ARENA_EVENT_FEED:],
                "events_logged": len(session.events),
            }

    def _arena_model_errors(self) -> list[str]:
        errors = []
        for model_name in self.arena.models:
            _, _, error = self.policies.load(model_name, MODELS_DIR)
            if error:
                errors.append(error)
        return errors

    # ------------------------------------------------------------ playground
    def playground_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            map_name = str(payload.get("map", MAP_NAMES[0]))
            if map_name not in MAP_NAMES:
                raise ValueError(f"Unknown map {map_name!r}")
            player_weapon = self._canonical_weapon(payload.get("weapon", WEAPON_NAMES[0]))
            enemy_weapon = self._canonical_weapon(payload.get("enemy_weapon", WEAPON_NAMES[2]))
            bot_label = str(payload.get("bot", "Tactical"))
            options = list(BOT_BEHAVIORS) + [name for name in policy_choices(MODELS_DIR)
                                            if name != HEURISTIC]
            if bot_label not in options:
                bot_label = "Tactical"
            ai_mode = BOT_BEHAVIORS.get(bot_label, "full")
            detail_name = str(payload.get("detail", "Balanced"))
            detail_count = DETAIL_PRESETS.get(detail_name, DETAIL_PRESETS["Balanced"])

            arena_map = self.map_for(map_name)
            signature = (map_name, player_weapon, enemy_weapon, bot_label, _layout_signature(arena_map),
                         detail_count)
            session = self.playground
            session.bot_label = bot_label
            session.detail_count = detail_count
            if session.signature != signature or session.env is None:
                if session.env is not None:
                    session.env.close()
                session.env = ShooterEnv(
                    map_name=arena_map,
                    weapon_name=player_weapon,
                    opponent_weapon=enemy_weapon,
                    curriculum=False,
                    opponent_mode=ai_mode,
                    frame_skip=4,
                    max_episode_seconds=120.0,
                )
                session.env.reset()
                session.signature = signature
                session.messages = []
                session.reward = 0.0
                session.result = None
                session.scene_key = None
            static = scene.map_scene(arena_map, detail_count, show_spawns=True)
            session.scene_key = static["key"]
            return {
                "static": static,
                "scene_key": static["key"],
                "frame": scene.match_frame(session.env),
                "bot": bot_label,
                "recording": session.recording,
                "demo_count": len(session.demo_rows),
                "stance": session.stance,
                "sprint": session.sprint,
            }

    def playground_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one D-pad press (or several repeats) to the human match."""
        with self.lock:
            session = self.playground
            if session.env is None:
                raise ValueError("Configure the playground before sending an action")
            press = payload.get("press")
            controls: dict[str, Any]
            default_repeat = 1
            if press is not None:
                if press not in PLAY_ACTIONS:
                    raise ValueError(f"Unknown press {press!r}")
                controls = dict(PLAY_ACTIONS[press])
                default_repeat = int(controls.pop("repeat", 1))
            else:
                controls = dict(payload.get("controls") or {})
            if "stance" in payload:
                session.stance = max(0, min(2, int(payload["stance"])))
            if "sprint" in payload:
                session.sprint = bool(payload["sprint"])
            if "recording" in payload:
                session.recording = bool(payload["recording"])
            repeat = int(payload.get("repeat", default_repeat))
            repeat = max(1, min(35, repeat))
            return self._playground_steps(controls, repeat)

    def _playground_steps(self, controls: dict[str, Any], repeat: int) -> dict[str, Any]:
        session = self.playground
        env = session.env
        assert env is not None
        messages: list[str] = []
        action = action_from_controls(controls, stance=session.stance, sprint=session.sprint)
        load_errors: list[str] = []
        for _ in range(repeat):
            if env.done:
                break
            observation = env.get_observation(0)
            if session.recording and len(session.demo_rows) < MAX_RECORDED_DEMOS:
                session.demo_rows.append((observation.tolist(), action.tolist()))
            if session.bot_label in BOT_BEHAVIORS:
                _, reward, terminated, truncated, info = env.step(action)
            else:
                bot_action, error = self.policies.predict(
                    session.bot_label, env.get_observation(1), 1, env, MODELS_DIR
                )
                if error and error not in load_errors:
                    load_errors.append(error)
                _, reward, terminated, truncated, info = env.step_duel(action, bot_action)
            session.reward += float(reward)
            messages.extend(self._playground_events(info, controller=False))
            if terminated or truncated:
                metrics = info.get("episode_metrics", {})
                if metrics:
                    session.result = metrics
                    self.arena.events.append(metrics)
                    self.arena.events = self.arena.events[-MAX_ARENA_EVENTS:]
                    result_text = "YOU WIN" if metrics.get("win", 0) else (
                        "DRAW" if metrics.get("draw") else "BOT WINS"
                    )
                    messages.append(
                        f"ROUND COMPLETE · {result_text} · {float(metrics.get('ttk', 0.0)):.1f}s"
                    )
                break
        session.messages.extend(messages)
        session.messages = session.messages[-35:]
        return {
            "frame": scene.match_frame(env),
            "scene_key": session.scene_key,
            "new_messages": messages,
            "messages": session.messages[-10:],
            "reward": session.reward,
            "result": session.result,
            "recording": session.recording,
            "demo_count": len(session.demo_rows),
            "load_errors": load_errors,
        }

    def _playground_events(self, info: dict[str, Any], controller: bool) -> list[str]:
        messages: list[str] = []
        for event in info.get("combat_events", []):
            player_scored = int(event.get("attacker", 0)) == 1
            if event.get("type") == "kill":
                suffix = " [HEADSHOT]" if event.get("headshot") else ""
                outcome = "BOT DOWN" if player_scored else "YOU DOWN"
                messages.append(
                    f"{outcome} · {event.get('weapon', 'weapon')}{suffix} · "
                    f"{float(event.get('distance', 0.0)):.1f}m"
                )
            elif event.get("type") == "hit":
                suffix = " HEADSHOT" if event.get("headshot") else ""
                outcome = "HIT" if player_scored else "INCOMING HIT"
                messages.append(f"{outcome} · {float(event.get('damage', 0.0)):.0f} damage{suffix}")
        return messages

    def playground_reset(self) -> dict[str, Any]:
        with self.lock:
            session = self.playground
            if session.env is None:
                raise ValueError("Configure the playground before resetting")
            session.env.reset()
            session.messages = []
            session.reward = 0.0
            session.result = None
            return {"frame": scene.match_frame(session.env), "reward": 0.0, "messages": []}

    def playground_recording(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            session = self.playground
            if "recording" in payload:
                session.recording = bool(payload["recording"])
            if payload.get("clear"):
                session.demo_rows = []
            return {"recording": session.recording, "demo_count": len(session.demo_rows)}

    def demos_csv(self) -> str:
        """Serialize the recorded human demonstrations to the training CSV schema."""
        state_fields = [f"state_{index}" for index in range(OBSERVATION_SIZE)]
        action_fields = [f"action_{index}" for index in range(ACTION_SIZE)]
        lines = [",".join(state_fields + action_fields)]
        for observation, action in self.playground.demo_rows:
            lines.append(",".join(
                [f"{float(value):.5f}" for value in observation]
                + [str(int(value)) for value in action]
            ))
        return "\n".join(lines) + "\n"

    def save_demos(self) -> dict[str, Any]:
        """Append the recorded demonstrations to data/demos.csv for imitation learning."""
        with self.lock:
            rows = list(self.playground.demo_rows)
            if not rows:
                return {"saved": 0, "path": relative_path(DEMOS_PATH)}
            DEMOS_PATH.parent.mkdir(parents=True, exist_ok=True)
            state_fields = [f"state_{index}" for index in range(OBSERVATION_SIZE)]
            action_fields = [f"action_{index}" for index in range(ACTION_SIZE)]
            header_needed = not DEMOS_PATH.exists() or DEMOS_PATH.stat().st_size == 0
            with DEMOS_PATH.open("a", newline="", encoding="utf-8") as handle:
                if header_needed:
                    handle.write(",".join(state_fields + action_fields) + "\n")
                for observation, action in rows:
                    handle.write(",".join(
                        [f"{float(value):.5f}" for value in observation]
                        + [str(int(value)) for value in action]
                    ) + "\n")
            self.playground.demo_rows = []
            self.notice(f"Added {len(rows):,} human demonstrations to data/demos.csv.")
            return {"saved": len(rows), "path": relative_path(DEMOS_PATH)}

    # -------------------------------------------------------------- minigames
    def aim_start(self, duration_seconds: float = 30.0) -> dict[str, Any]:
        with self.lock:
            self.aim_game = new_aim_game(duration_seconds)
            return self.aim_state()

    def aim_tap(self, cell: int) -> dict[str, Any]:
        with self.lock:
            if self.aim_game is None:
                raise ValueError("Start a drill before tapping a target")
            aim_tap(self.aim_game, int(cell))
            public = public_aim_game(self.aim_game)
            assert public is not None
            if not public["active"]:
                self.aim_best = max(self.aim_best, int(public["hits"]))
            return {"game": public, "best": int(self.aim_best)}

    def aim_state(self) -> dict[str, Any]:
        with self.lock:
            public = public_aim_game(self.aim_game)
            if public is not None and not public["active"]:
                self.aim_best = max(self.aim_best, int(public["hits"]))
            return {"game": public, "best": int(self.aim_best)}

    def dodge_start(self) -> dict[str, Any]:
        with self.lock:
            self.dodge_game = new_dodge_game()
            return self.dodge_state()

    def dodge_step(self, dx: int = 0, dy: int = 0) -> dict[str, Any]:
        with self.lock:
            if self.dodge_game is None:
                raise ValueError("Start a run before moving")
            step_dodge_game(self.dodge_game, int(dx), int(dy))
            if not self.dodge_game["active"]:
                self.dodge_best = max(self.dodge_best, int(self.dodge_game["score"]))
            return self.dodge_state()

    def dodge_state(self) -> dict[str, Any]:
        with self.lock:
            return {"game": self.dodge_game, "best": int(self.dodge_best)}

    # --------------------------------------------------------------- training
    def training_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        deps = dependency_status()
        if not (deps["stable_baselines3"] and deps["torch"]):
            raise ValueError(
                "PPO training needs Stable-Baselines3 and PyTorch. Run install.py to add them."
            )
        with self.lock:
            job = self.training_job
            if job is not None and job.snapshot()["status"] in {"starting", "running", "paused", "stopping"}:
                raise ValueError("A training run is already active. Stop it first.")
            from training.train import TrainingConfig, TrainingController

            duration_minutes = float(payload.get("duration_minutes", 10.0))
            duration_seconds = max(0.0, duration_minutes * 60.0)
            workers = max(1, int(payload.get("workers", 4)))
            envs_per_worker = max(1, int(payload.get("envs_per_worker", 1)))
            map_name = str(payload.get("map", "Dust"))
            if map_name not in MAP_NAMES:
                map_name = "Dust"
            method = str(payload.get("method", "Pure RL"))
            resume_checkpoint = payload.get("resume_checkpoint")
            if method == "Resume Checkpoint":
                if not resume_checkpoint:
                    raise ValueError("Choose a checkpoint before resuming.")
                if not (MODELS_DIR / str(resume_checkpoint)).exists():
                    raise ValueError(f"Checkpoint {resume_checkpoint} does not exist in models/.")
                resume_checkpoint = str(MODELS_DIR / str(resume_checkpoint))
            imitation_path = str(MODELS_DIR / "behavior_clone.pt") if method.startswith("Imitation") else None
            estimated_steps = max(50_000, int(duration_seconds * 1_500))
            config = TrainingConfig(
                duration_seconds=duration_seconds,
                total_timesteps=estimated_steps,
                n_workers=workers,
                envs_per_worker=envs_per_worker,
                map_name=map_name,
                map_definition=(self.custom_map() if map_name == "Custom" else None),
                curriculum=bool(payload.get("curriculum", True)),
                self_play=bool(payload.get("self_play", True)),
                method=method,
                resume_checkpoint=resume_checkpoint,
                imitation_path=imitation_path,
                max_envs=int(payload.get("max_envs", 24)),
                models_dir=MODELS_DIR,
                logs_dir=LOGS_DIR,
            )
            self.training_job = TrainingController(config)
            self.training_job.start()
            return {"started": True, "run": self.training_snapshot(),
                    "estimated_steps": estimated_steps}

    def training_snapshot(self) -> dict[str, Any]:
        with self.lock:
            job = self.training_job
            if job is None:
                return {"status": "stopped", "metrics": {}, "logs": [], "error": None,
                        "latest_checkpoint": None, "thread_alive": False,
                        "config": None}
            snapshot = job.snapshot()
            config = job.config
            snapshot["config"] = {
                "duration_seconds": config.duration_seconds,
                "total_timesteps": config.total_timesteps,
                "n_workers": config.n_workers,
                "envs_per_worker": config.envs_per_worker,
                "map_name": config.map_name,
                "curriculum": config.curriculum,
                "self_play": config.self_play,
                "method": config.method,
            }
            return snapshot

    def _training_status(self) -> str:
        job = self.training_job
        return job.snapshot()["status"] if job is not None else "stopped"

    def training_command(self, command: str) -> dict[str, Any]:
        with self.lock:
            job = self.training_job
            if job is None:
                raise ValueError("No training run has been started yet.")
            if command == "pause":
                job.pause()
            elif command == "resume":
                job.resume()
            elif command == "stop":
                job.stop()
            elif command == "save":
                job.request_save()
            else:
                raise ValueError(f"Unknown training command {command!r}")
            return self.training_snapshot()

    def stats(self) -> dict[str, Any]:
        with self.lock:
            live = self.training_snapshot().get("metrics", {})
            payload = analytics.training_stats(LOGS_DIR, live=live)
            payload["status"] = self._training_status()
            payload["latest_checkpoint"] = self.training_snapshot().get("latest_checkpoint")
        return payload

    def heatmap(self, query: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            map_name = str(query.get("map", "Dust"))
            if map_name not in MAP_NAMES:
                map_name = "Dust"
            episode_range = None
            raw_episodes = query.get("episode_range")
            if raw_episodes:
                episode_range = (int(raw_episodes[0]), int(raw_episodes[1]))
            distance_range = None
            raw_distance = query.get("distance_range")
            if raw_distance:
                distance_range = (float(raw_distance[0]), float(raw_distance[1]))
            return analytics.heatmap_payload(
                map_name,
                arena_map=self._static_map(map_name),
                weapon=query.get("weapon") or None,
                mode=str(query.get("mode", "Both")),
                episode_range=episode_range,
                distance_range=distance_range,
                extra_events=self.arena.events,
            )

    def _static_map(self, map_name: str) -> ArenaMap:
        """Heatmaps always use the canonical layout, not lab overrides."""
        return create_map(map_name)

    # -------------------------------------------------------------- benchmark
    def benchmark_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        deps = dependency_status()
        if not deps["stable_baselines3"]:
            raise ValueError("Benchmarking needs Stable-Baselines3. Run install.py to add it.")
        with self.lock:
            runner = self.benchmark_runner
            if runner is not None and runner.snapshot()["status"] in {"running", "stopping"}:
                raise ValueError("A benchmark is already running.")
            from training.workers import BenchmarkRunner, valid_benchmark_configurations

            map_name = str(payload.get("map", "Dust"))
            if map_name not in MAP_NAMES:
                map_name = "Dust"
            seconds = float(payload.get("seconds_per_combo", 20.0))
            self.benchmark_runner = BenchmarkRunner(seconds_per_combo=seconds, map_name=map_name)
            self.benchmark_runner.start()
            return self.benchmark_snapshot()

    def benchmark_snapshot(self) -> dict[str, Any]:
        from training.workers import valid_benchmark_configurations

        with self.lock:
            total = len(valid_benchmark_configurations())
            runner = self.benchmark_runner
            if runner is None:
                return analytics.benchmark_payload(
                    {"status": "stopped", "results": [], "current": None, "error": None}, total
                )
            return analytics.benchmark_payload(runner.snapshot(), total)

    def _benchmark_status(self) -> str:
        runner = self.benchmark_runner
        return runner.snapshot()["status"] if runner is not None else "stopped"

    def benchmark_stop(self) -> dict[str, Any]:
        with self.lock:
            runner = self.benchmark_runner
            if runner is None:
                raise ValueError("No benchmark has been started yet.")
            runner.stop()
            return self.benchmark_snapshot()

    # -------------------------------------------------------------------- ttk
    def ttk_defaults(self) -> dict[str, Any]:
        from training.weapon_lab import FIELD_LIMITS, field_defaults

        return {
            "weapons": list(WEAPON_NAMES),
            "specs": {name: field_defaults(get_weapon(name).spec) for name in WEAPON_NAMES},
            "limits": FIELD_LIMITS,
            "fields": list(FIELD_LIMITS),
            "state": self.ttk_state,
        }

    def ttk_simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        from training.weapon_lab import result_payload, simulate_duels, spec_from_fields

        with self.lock:
            weapon_a = self._canonical_weapon(payload.get("weapon_a", "Pistol"))
            weapon_b = self._canonical_weapon(payload.get("weapon_b", "AK-47"))
            fields_a = dict(payload.get("fields_a") or {})
            fields_b = dict(payload.get("fields_b") or {})
            trials = max(1_000, min(100_000, int(payload.get("trials", 5_000))))
            distance = max(5.0, min(80.0, float(payload.get("distance", 20.0))))
            spec_a = spec_from_fields(weapon_a, fields_a)
            spec_b = spec_from_fields(weapon_b, fields_b)
            result = simulate_duels(spec_a, spec_b, distance, trials,
                                    seed=2026 + int(distance) * 31 + trials)
            self.ttk_state = {
                "a": {"weapon": weapon_a, "fields": fields_a or None},
                "b": {"weapon": weapon_b, "fields": fields_b or None},
                "trials": trials,
                "distance": distance,
            }
            return result_payload(result, weapon_a, weapon_b, spec_a, spec_b)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _canonical_weapon(name: Any) -> str:
        candidate = str(name)
        for weapon in WEAPON_NAMES:
            if weapon.lower() == candidate.lower():
                return weapon
        return WEAPON_NAMES[0]


def _layout_signature(arena_map: ArenaMap) -> tuple[Any, ...]:
    return (
        float(arena_map.width),
        float(arena_map.depth),
        tuple(tuple(float(value) for value in point) for point in arena_map.spawn_points),
        tuple((float(item.x), float(item.y), float(item.z), float(item.width), float(item.height),
               float(item.depth), item.kind, item.name, float(item.yaw))
              for item in arena_map.objects),
    )


def map_stats(arena_map: ArenaMap) -> dict[str, Any]:
    return {
        "name": arena_map.name,
        "width": arena_map.width,
        "depth": arena_map.depth,
        "objects": len(arena_map.objects),
        "cover_density": arena_map.cover_density,
        "average_sightline": arena_map.average_sightline,
        "description": arena_map.description,
        "spawns": [list(point) for point in arena_map.spawn_points],
    }


def map_breakdown(arena_map: ArenaMap) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for item in arena_map.objects:
        breakdown[item.kind] = breakdown.get(item.kind, 0) + 1
    return dict(sorted(breakdown.items()))


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


STATE = ServerState()

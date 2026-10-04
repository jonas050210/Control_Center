"""Local web app: JSON API + the static single-page UI in ``rocketai/web``.

Training runs as a separate process (``python -m rocketai train --resume NAME``)
so it survives page reloads and can use all CPU cores; the server just reads
the run directory. Short jobs (evaluations, recorded matches) run in a
single background thread.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .config import (
    PRESETS,
    ROOT,
    RUN_NAME_PATTERN,
    TrainConfig,
    preset_config,
    run_paths,
    runs_root,
)
from .play import LAUNCHERS, MODES, SKILLS, PlaySession, PlaySettings, server_path
from .trainer import read_json, write_json

WEB_DIR = Path(__file__).parent / "web"
MATCHES_DIR_NAME = "_matches"  # replays of matches started from the Arena page
STALE_AFTER_SECONDS = 180


# ----------------------------------------------------------------- helpers


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a line being written right now
    return rows


def _downsample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    picked = [rows[int(i * step)] for i in range(limit - 1)]
    return picked + [rows[-1]]


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 200 * lines))
        text = handle.read().decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def _checkpoints(name: str) -> list[dict[str, Any]]:
    folder = run_paths(name).checkpoints
    items = []
    for path in folder.glob("*.pt") if folder.exists() else []:
        if not path.stem.isdigit():
            continue
        stat = path.stat()
        items.append(
            {
                "file": path.name,
                "steps": int(path.stem),
                "size": stat.st_size,
                "created": stat.st_mtime,
            }
        )
    return sorted(items, key=lambda item: item["steps"])


def resolve_checkpoint(spec: str) -> Path:
    """``<run>/<file>.pt`` inside the runs folder (never an arbitrary path)."""
    run, _, file = spec.partition("/")
    if not RUN_NAME_PATTERN.match(run) or not file.endswith(".pt") or "/" in file or "\\" in file:
        raise HTTPException(400, f"Ungültiger Checkpoint {spec!r}")
    path = run_paths(run).checkpoints / file
    if not path.is_file():
        raise HTTPException(404, f"Checkpoint {spec} nicht gefunden")
    return path


def player_from_spec(spec: str) -> Any:
    from .opponents import SCRIPTED_BOTS, PolicyPlayer, make_player

    if spec in SCRIPTED_BOTS:
        return make_player(spec)
    path = resolve_checkpoint(spec)
    player = PolicyPlayer.from_checkpoint(path)
    player.name = f"{spec.split('/')[0]}@{path.stem}"
    return player


# ----------------------------------------------------------------- state


@dataclass
class Job:
    id: str
    kind: str
    title: str
    state: str = "queued"  # queued, running, done, failed
    created: float = field(default_factory=time.time)
    result: Any = None
    error: str = ""

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "state": self.state,
            "created": self.created,
            "result": self.result,
            "error": self.error,
        }


class AppState:
    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen] = {}
        self.jobs: dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rocketai-job")
        self.play = PlaySession()
        self.lock = threading.Lock()

    # Training processes -------------------------------------------------
    def start_training(self, name: str) -> None:
        with self.lock:
            process = self.processes.get(name)
            if process is not None and process.poll() is None:
                raise HTTPException(409, f"'{name}' trainiert bereits")
            paths = run_paths(name)
            paths.root.mkdir(parents=True, exist_ok=True)
            paths.control.unlink(missing_ok=True)
            write_json(paths.status, {"state": "starting", "steps": 0, "updated": time.time()})
            out = (paths.root / "process.log").open("ab")
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            )
            self.processes[name] = subprocess.Popen(
                [sys.executable, "-m", "rocketai", "train", "--resume", name],
                cwd=ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            out.close()

    def run_state(self, name: str, status: dict[str, Any]) -> str:
        state = status.get("state", "new")
        process = self.processes.get(name)
        if process is not None:
            code = process.poll()
            if code is None:
                return "stopping" if run_paths(name).control.exists() else state
            if state in ("starting", "running", "evaluating"):
                return "failed" if code else "stopped"
            return state
        # Not started by this server (or the server restarted): judge by heartbeat.
        heartbeat_age = time.time() - status.get("updated", 0)
        if state in ("starting", "running", "evaluating") and heartbeat_age > STALE_AFTER_SECONDS:
            return "interrupted"
        return state

    # Jobs ----------------------------------------------------------------
    def submit(self, kind: str, title: str, fn: Any) -> Job:
        job = Job(uuid.uuid4().hex[:10], kind, title)
        self.jobs[job.id] = job

        def runner() -> None:
            job.state = "running"
            try:
                job.result = fn()
                job.state = "done"
            except Exception as error:
                job.error = f"{type(error).__name__}: {error}"
                job.state = "failed"

        self.executor.submit(runner)
        # Keep the list short.
        if len(self.jobs) > 50:
            for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-50]:
                self.jobs.pop(old.id, None)
        return job


STATE = AppState()


def run_summary(name: str) -> dict[str, Any]:
    paths = run_paths(name)
    config = read_json(paths.config, {})
    status = read_json(paths.status, {})
    metrics = _read_jsonl(paths.metrics)
    evaluations = _read_jsonl(paths.evaluations)
    last = metrics[-1] if metrics else {}
    return {
        "name": name,
        "config": config,
        "status": {**status, "state": STATE.run_state(name, status)},
        "last": {
            key: last.get(key)
            for key in (
                "steps",
                "steps_per_second",
                "episode_reward",
                "touches_per_minute",
                "goals_per_minute",
                "time",
            )
        },
        "checkpoints": len(_checkpoints(name)),
        "evaluation": evaluations[-1] if evaluations else None,
        "modified": paths.root.stat().st_mtime if paths.root.exists() else 0,
    }


def list_runs() -> list[dict[str, Any]]:
    root = runs_root()
    if not root.exists():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir() and RUN_NAME_PATTERN.match(p.name)]
    runs = [run_summary(name) for name in names if (root / name / "config.json").exists()]
    return sorted(runs, key=lambda run: run["modified"], reverse=True)


def list_replays() -> list[dict[str, Any]]:
    root = runs_root()
    items = []
    for path in root.glob("*/replays/*.json") if root.exists() else []:
        run = path.parent.parent.name
        try:
            with path.open(encoding="utf-8") as handle:
                head = handle.read(4096)
            meta_start = head.find('"meta":')
            meta = {}
            if meta_start != -1:
                # meta comes before frames and is small; parse just that object
                depth, end = 0, None
                for index in range(meta_start + 7, len(head)):
                    char = head[index]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            end = index + 1
                            break
                if end:
                    meta = json.loads(head[meta_start + 7 : end])
        except (OSError, ValueError):
            meta = {}
        items.append(
            {
                "id": f"{run}/{path.name}",
                "run": run,
                "file": path.name,
                "created": path.stat().st_mtime,
                "blue": meta.get("blue"),
                "orange": meta.get("orange"),
                "goals_blue": meta.get("goals_blue"),
                "goals_orange": meta.get("goals_orange"),
                "kind": meta.get("kind", "match"),
            }
        )
    return sorted(items, key=lambda item: item["created"], reverse=True)


# ----------------------------------------------------------------- API models


class NewRun(BaseModel):
    name: str
    preset: str = "beginner"
    overrides: dict[str, Any] = {}


class Resume(BaseModel):
    total_steps: int | None = None


class EvalRequest(BaseModel):
    checkpoint: str  # run/file.pt
    opponents: list[str] = ["chaser", "defender"]
    games: int = 6


class MatchRequest(BaseModel):
    blue: str
    orange: str
    team_size: int = 1
    seconds: float = 120.0


class PlayRequest(BaseModel):
    checkpoint: str  # run/file.pt
    mode: str = "psyonix"
    team_size: int = 1
    skill: str = "rookie"
    launcher: str = "epic"
    opponent_bot: str = ""


# ----------------------------------------------------------------- app


class LiveRequest(BaseModel):
    blue: str
    orange: str = "chaser"
    team_size: int = 1
    speed: float = 1.0
    match_seconds: float = 300.0


class LiveControl(BaseModel):
    paused: bool | None = None
    speed: float | None = None


_RL_CACHE: dict[str, Any] = {"time": 0.0, "data": None}


def rocket_league_status(refresh: bool = False) -> dict[str, Any]:
    """Cached (5 s) because the process query starts PowerShell on Windows."""
    from .rlcheck import check

    if refresh or _RL_CACHE["data"] is None or time.time() - _RL_CACHE["time"] > 5:
        _RL_CACHE["data"] = check().to_dict()
        _RL_CACHE["time"] = time.time()
    return _RL_CACHE["data"]


def create_app() -> FastAPI:
    app = FastAPI(title="RocketAI", version=__version__, docs_url="/api/docs", redoc_url=None)

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        runs = list_runs()
        return {
            "version": __version__,
            "runs": runs,
            "active": [
                r["name"]
                for r in runs
                if r["status"]["state"] in ("starting", "running", "evaluating", "stopping")
            ],
            "jobs": [
                job.info()
                for job in sorted(STATE.jobs.values(), key=lambda j: j.created, reverse=True)
            ],
            "play": STATE.play.info(),
            "replays": list_replays()[:5],
        }

    @app.get("/api/presets")
    def presets() -> dict[str, Any]:
        return {
            "presets": PRESETS,
            "defaults": TrainConfig().to_dict(),
            "cpu_count": os.cpu_count() or 1,
        }

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return list_runs()

    def _existing(name: str) -> Any:
        try:
            paths = run_paths(name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not paths.config.exists():
            raise HTTPException(404, f"Run '{name}' nicht gefunden")
        return paths

    @app.post("/api/runs", status_code=201)
    def create_run(request: NewRun) -> dict[str, Any]:
        try:
            config = preset_config(request.preset, name=request.name, **request.overrides)
            config.validate()
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        paths = run_paths(config.name)
        if paths.config.exists():
            raise HTTPException(409, f"Es gibt schon einen Run '{config.name}'")
        paths.ensure()
        config.save(paths.config)
        STATE.start_training(config.name)
        return run_summary(config.name)

    @app.get("/api/runs/{name}")
    def run_detail(name: str) -> dict[str, Any]:
        paths = _existing(name)
        return {
            **run_summary(name),
            "checkpoints": _checkpoints(name),
            "evaluations": _read_jsonl(paths.evaluations),
        }

    @app.get("/api/runs/{name}/metrics")
    def run_metrics(name: str, limit: int = 600) -> list[dict[str, Any]]:
        paths = _existing(name)
        return _downsample(_read_jsonl(paths.metrics), max(10, min(limit, 5000)))

    @app.get("/api/runs/{name}/log")
    def run_log(name: str, lines: int = 200) -> dict[str, Any]:
        paths = _existing(name)
        return {
            "train": _tail(paths.log, max(1, min(lines, 2000))),
            "process": _tail(paths.root / "process.log", 60),
        }

    @app.post("/api/runs/{name}/resume")
    def resume_run(name: str, request: Resume) -> dict[str, Any]:
        paths = _existing(name)
        if request.total_steps:
            config = TrainConfig.load(paths.config)
            config.total_steps = request.total_steps
            config.save(paths.config)
        STATE.start_training(name)
        return run_summary(name)

    @app.post("/api/runs/{name}/stop")
    def stop_run(name: str) -> dict[str, Any]:
        paths = _existing(name)
        write_json(paths.control, {"stop": True, "time": time.time()})
        return run_summary(name)

    @app.delete("/api/runs/{name}")
    def delete_run(name: str) -> dict[str, str]:
        paths = _existing(name)
        if run_summary(name)["status"]["state"] in (
            "starting",
            "running",
            "evaluating",
            "stopping",
        ):
            raise HTTPException(409, "Erst das Training stoppen")
        STATE.processes.pop(name, None)
        shutil.rmtree(paths.root)
        return {"deleted": name}

    @app.post("/api/evaluate", status_code=202)
    def evaluate_checkpoint(request: EvalRequest) -> dict[str, Any]:
        from .match import evaluate
        from .opponents import make_player
        from .trainer import append_jsonl

        path = resolve_checkpoint(request.checkpoint)
        games = max(1, min(request.games, 50))
        run = request.checkpoint.split("/")[0]

        def job() -> dict[str, Any]:
            policy = player_from_spec(request.checkpoint)
            team_size = int(read_json(run_paths(run).config, {}).get("team_size", 1))
            results = [
                evaluate(
                    policy,
                    make_player(spec) if "/" not in spec else player_from_spec(spec),
                    games=games,
                    team_size=team_size,
                )
                for spec in request.opponents
            ]
            record = {
                "steps": int(path.stem),
                "time": time.time(),
                "results": results,
                "manual": True,
            }
            append_jsonl(run_paths(run).evaluations, record)
            return record

        return STATE.submit("evaluate", f"Bewertung {request.checkpoint}", job).info()

    @app.post("/api/matches", status_code=202)
    def record_match(request: MatchRequest) -> dict[str, Any]:
        from .match import play_match, save_replay

        if request.team_size not in (1, 2, 3):
            raise HTTPException(400, "team_size muss 1, 2 oder 3 sein")
        seconds = max(10.0, min(request.seconds, 600.0))
        blue, orange = player_from_spec(request.blue), player_from_spec(request.orange)

        def job() -> dict[str, Any]:
            result = play_match(
                blue, orange, team_size=request.team_size, seconds=seconds, record=True
            )
            file = f"{time.strftime('%Y%m%d-%H%M%S')}.json"
            save_replay(
                runs_root() / MATCHES_DIR_NAME / "replays" / file, result, {"kind": "match"}
            )
            return {"replay": f"{MATCHES_DIR_NAME}/{file}", **result.summary()}

        return STATE.submit("match", f"{blue.name} gegen {orange.name}", job).info()

    @app.get("/api/jobs/{job_id}")
    def job_info(job_id: str) -> dict[str, Any]:
        job = STATE.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unbekannter Auftrag")
        return job.info()

    @app.get("/api/replays")
    def replays() -> list[dict[str, Any]]:
        return list_replays()

    @app.get("/api/replays/{run}/{file}")
    def replay(run: str, file: str) -> FileResponse:
        if not (RUN_NAME_PATTERN.match(run) or run == MATCHES_DIR_NAME) or not file.endswith(
            ".json"
        ):
            raise HTTPException(400, "Ungültiges Replay")
        path = runs_root() / run / "replays" / Path(file).name
        if not path.is_file():
            raise HTTPException(404, "Replay nicht gefunden")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/opponents")
    def opponents() -> dict[str, Any]:
        from .opponents import SCRIPTED_BOTS

        checkpoints = [
            {"id": f"{run['name']}/{c['file']}", "run": run["name"], "steps": c["steps"]}
            for run in list_runs()
            for c in _checkpoints(run["name"])
        ]
        return {
            "scripted": [{"id": key, "label": label} for key, (label, _) in SCRIPTED_BOTS.items()],
            "checkpoints": checkpoints,
        }

    @app.get("/api/play")
    def play_info() -> dict[str, Any]:
        return {
            **STATE.play.info(),
            "modes": MODES,
            "skills": SKILLS,
            "launchers": LAUNCHERS,
            "server_installed": server_path().exists(),
            "windows": sys.platform == "win32",
        }

    @app.post("/api/play/start")
    def play_start(request: PlayRequest) -> dict[str, Any]:
        path = resolve_checkpoint(request.checkpoint)
        settings = PlaySettings(
            checkpoint=str(path),
            mode=request.mode,
            team_size=request.team_size,
            skill=request.skill,
            launcher=request.launcher,
            opponent_bot=request.opponent_bot,
        )
        try:
            STATE.play.start(settings)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error
        return STATE.play.info()

    @app.post("/api/play/stop")
    def play_stop() -> dict[str, Any]:
        STATE.play.stop()
        return STATE.play.info()

    @app.get("/api/rocketleague")
    def rocketleague(refresh: bool = False) -> dict[str, Any]:
        return rocket_league_status(refresh)

    @app.get("/api/field")
    def field_info() -> dict[str, Any]:
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS

        return {
            "pads": [[int(x), int(y), 1 if z > 72 else 0] for x, y, z in BOOST_LOCATIONS],
            "size": {"x": 4096, "y": 5120, "z": 2044},
            "goal": {"width": 1786, "height": 642.5, "depth": 880},
        }

    @app.get("/api/live")
    def live_state(since: int = -1) -> dict[str, Any]:
        from .live import LIVE

        return LIVE.state(since)

    @app.post("/api/live/start")
    def live_start(request: LiveRequest) -> dict[str, Any]:
        from .live import LIVE, LiveSettings, SpecError

        try:
            return LIVE.start(LiveSettings(**request.model_dump()))
        except SpecError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/live/control")
    def live_control(request: LiveControl) -> dict[str, Any]:
        from .live import LIVE, SpecError

        try:
            return LIVE.control(paused=request.paused, speed=request.speed)
        except SpecError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/live/stop")
    def live_stop() -> dict[str, Any]:
        from .live import LIVE

        LIVE.stop()
        return LIVE.state()

    @app.get("/api/snapshot")
    def snapshot() -> dict[str, Any]:
        """Same data as the event stream, for clients behind proxies that buffer SSE."""
        return json.loads(_snapshot())

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        from .doctor import run_checks

        return {"checks": run_checks(), "runs_folder": str(runs_root()), "version": __version__}

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        """Server-sent events: a compact snapshot whenever something changes."""

        async def stream():
            previous = None
            while True:
                snapshot = await asyncio.to_thread(_snapshot)
                if snapshot != previous:
                    yield f"data: {snapshot}\n\n"
                    previous = snapshot
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
    return app


def _snapshot() -> str:
    runs = [
        {
            "name": r["name"],
            "state": r["status"]["state"],
            "steps": r["status"].get("steps", 0),
            "total": r["status"].get("total_steps") or r["config"].get("total_steps"),
            "sps": r["status"].get("steps_per_second"),
        }
        for r in list_runs()
    ]
    jobs = [{"id": j.id, "state": j.state} for j in STATE.jobs.values()]
    from .live import LIVE

    live = LIVE.session.running if LIVE.session is not None else False
    return json.dumps(
        {"runs": runs, "jobs": jobs, "play": STATE.play.state, "live": live}, sort_keys=True
    )


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"RocketAI läuft auf {url}  (Beenden mit Strg+C)")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")

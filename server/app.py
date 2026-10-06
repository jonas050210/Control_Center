"""FastAPI application powering the NEURAL ARENA control center.

Run it with ``python start.py`` (which also opens the browser) or directly:

    uvicorn server.app:app --host 127.0.0.1 --port 8501

The former Streamlit UI is gone: the browser talks to these JSON endpoints and
renders the arena itself with WebGL.
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from server.config import APP_NAME, APP_VERSION, WEB_DIR
from server.state import STATE


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=APP_VERSION, docs_url="/api/docs",
                  redoc_url=None, openapi_url="/api/openapi.json")

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # ------------------------------------------------------------------ base
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return STATE.health()

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return STATE.meta()

    # ----------------------------------------------------------------- arena
    @app.post("/api/arena/config")
    def arena_config(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.arena_configure(payload or {})

    @app.get("/api/arena/state")
    def arena_state() -> dict[str, Any]:
        return STATE.arena_state()

    @app.post("/api/arena/run")
    def arena_run(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        body = payload or {}
        return STATE.arena_set_running(bool(body.get("running", True)))

    @app.post("/api/arena/step")
    def arena_step(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        body = payload or {}
        return STATE.arena_step(int(body.get("steps", 1)))

    @app.post("/api/arena/reset")
    def arena_reset() -> dict[str, Any]:
        return STATE.arena_reset()

    # ------------------------------------------------------------ playground
    @app.post("/api/playground/config")
    def playground_config(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.playground_configure(payload or {})

    @app.post("/api/playground/action")
    def playground_action(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.playground_action(payload or {})

    @app.post("/api/playground/reset")
    def playground_reset() -> dict[str, Any]:
        return STATE.playground_reset()

    @app.post("/api/playground/recording")
    def playground_recording(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.playground_recording(payload or {})

    @app.get("/api/playground/demos.csv")
    def playground_demos() -> PlainTextResponse:
        return PlainTextResponse(
            STATE.demos_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="neural_arena_demos.csv"'},
        )

    @app.post("/api/playground/demos/save")
    def playground_save_demos() -> dict[str, Any]:
        return STATE.save_demos()

    # -------------------------------------------------------------- minigames
    @app.post("/api/aim/start")
    def aim_start(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        body = payload or {}
        return STATE.aim_start(float(body.get("duration_seconds", 30.0)))

    @app.post("/api/aim/tap")
    def aim_tap(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if "cell" not in payload:
            raise HTTPException(status_code=422, detail="cell is required")
        return STATE.aim_tap(int(payload["cell"]))

    @app.get("/api/aim/state")
    def aim_state() -> dict[str, Any]:
        return STATE.aim_state()

    @app.get("/api/aim/scene")
    def aim_scene() -> dict[str, Any]:
        from server import scene as scene_module
        from server.minigames import public_aim_game

        return {
            "static": scene_module.aim_index_scene("aim-lab"),
            "game": public_aim_game(STATE.aim_game),
            "best": STATE.aim_best,
        }

    @app.post("/api/dodge/start")
    def dodge_start() -> dict[str, Any]:
        return STATE.dodge_start()

    @app.post("/api/dodge/step")
    def dodge_step(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        body = payload or {}
        return STATE.dodge_step(int(body.get("dx", 0)), int(body.get("dy", 0)))

    @app.get("/api/dodge/state")
    def dodge_state() -> dict[str, Any]:
        return STATE.dodge_state()

    @app.get("/api/dodge/scene")
    def dodge_scene() -> dict[str, Any]:
        from server import scene as scene_module

        return {
            "static": scene_module.dodge_arena_scene("dodge-grid"),
            "game": STATE.dodge_game,
            "best": STATE.dodge_best,
        }

    # --------------------------------------------------------------- training
    @app.get("/api/training/status")
    def training_status() -> dict[str, Any]:
        return STATE.training_snapshot()

    @app.post("/api/training/start")
    def training_start(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.training_start(payload or {})

    @app.post("/api/training/{command}")
    def training_command(command: str) -> dict[str, Any]:
        if command not in {"pause", "resume", "stop", "save"}:
            raise HTTPException(status_code=404, detail=f"Unknown training command {command!r}")
        return STATE.training_command(command)

    # ------------------------------------------------------------------ stats
    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return STATE.stats()

    @app.post("/api/heatmap")
    def heatmap(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.heatmap(payload or {})

    # -------------------------------------------------------------- benchmark
    @app.get("/api/benchmark")
    def benchmark() -> dict[str, Any]:
        return STATE.benchmark_snapshot()

    @app.post("/api/benchmark/start")
    def benchmark_start(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.benchmark_start(payload or {})

    @app.post("/api/benchmark/stop")
    def benchmark_stop() -> dict[str, Any]:
        return STATE.benchmark_stop()

    # -------------------------------------------------------------------- ttk
    @app.get("/api/ttk")
    def ttk_defaults() -> dict[str, Any]:
        return STATE.ttk_defaults()

    @app.post("/api/ttk/simulate")
    def ttk_simulate(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        return STATE.ttk_simulate(payload or {})

    # ------------------------------------------------------------------- maps
    @app.get("/api/maps")
    def maps_index() -> dict[str, Any]:
        from env.maps import MAP_NAMES

        return {
            "maps": list(MAP_NAMES),
            "notices": STATE.pop_notices(),
            "detail_presets": STATE.meta()["detail_presets"],
        }

    @app.get("/api/maps/{name}/scene")
    def map_scene(name: str, detail: int = 160, spawns: bool = True) -> dict[str, Any]:
        _require_map(name)
        return STATE.map_scene(name, int(detail), bool(spawns))

    @app.post("/api/maps/{name}/randomize")
    def map_randomize(name: str) -> dict[str, Any]:
        _require_map(name)
        return STATE.randomize_map(name)

    @app.post("/api/maps/{name}/reset")
    def map_reset(name: str) -> dict[str, Any]:
        _require_map(name)
        return STATE.reset_map(name)

    @app.post("/api/maps/custom/objects")
    def map_add_object(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return STATE.add_custom_object(payload)

    @app.delete("/api/maps/custom/objects/{index}")
    def map_delete_object(index: int) -> dict[str, Any]:
        return STATE.delete_custom_object(index)

    @app.get("/api/maps/custom/export")
    def map_export() -> Response:
        return Response(
            content=STATE.export_custom_map(),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="custom_map.json"'},
        )

    @app.post("/api/maps/custom/import")
    async def map_import(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=422, detail="Empty upload body.")
        return STATE.import_custom_map(raw)

    # --------------------------------------------------------------- frontend
    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def _require_map(name: str) -> None:
    from env.maps import MAP_NAMES

    if not any(candidate.lower() == str(name).lower() for candidate in MAP_NAMES):
        raise HTTPException(status_code=404, detail=f"Unknown map {name!r}")


app = create_app()


def main() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()

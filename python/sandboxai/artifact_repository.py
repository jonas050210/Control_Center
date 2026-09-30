"""Single read-only gateway for persisted runs, checkpoints and evaluations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .run_inspection import discover_run_directories, inspect_run, inspect_runs


class ArtifactRepository:
    """Indexes run artifacts without owning training or mutating files."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).expanduser().resolve()

    def run_directories(self) -> list[Path]:
        return discover_run_directories(self.output_root)

    def list_runs(self, limit: int = 100) -> dict[str, Any]:
        return inspect_runs(self.output_root, limit=limit, event_limit=0)

    def inspect_run(self, run: str | Path, event_limit: int = 50) -> dict[str, Any]:
        return inspect_run(run, event_limit=event_limit)

    def dashboard(self, active_processes: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
        run_dirs = self.run_directories()
        latest = inspect_run(run_dirs[-1], event_limit=5) if run_dirs else None
        return {
            "output_root": str(self.output_root),
            "run_count": len(run_dirs),
            "latest_run": latest,
            "active_processes": active_processes(),
        }

    def checkpoints(self, limit: int = 300) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_dir in self.run_directories():
            report = inspect_run(run_dir)
            checkpoints = report.get("checkpoints", {})
            run_id = report.get("run_id") or run_dir.name
            directory = Path(checkpoints.get("directory", run_dir / "checkpoints"))
            for item in checkpoints.get("entries", []):
                name = item.get("name", "")
                kind = {"latest.zip": "latest", "best_eval.zip": "best"}.get(name, "checkpoint")
                entries.append({
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "kind": kind,
                    "path": str(directory / name),
                    "bytes": item.get("bytes"),
                    "modified_utc": item.get("modified_utc"),
                })
            final_path = run_dir / "final.zip"
            if final_path.is_file():
                modified = final_path.stat().st_mtime
                import time
                entries.append({
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "kind": "final",
                    "path": str(final_path),
                    "bytes": final_path.stat().st_size,
                    "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified)),
                })
        entries.sort(key=lambda item: item.get("modified_utc") or "", reverse=True)
        return entries[:limit]

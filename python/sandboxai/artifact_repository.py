"""Single read-only gateway for persisted runs, checkpoints and evaluations."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .control_center_schema import DashboardSnapshot, ProcessSnapshot
from .run_inspection import (
    discover_run_directories,
    inspect_run,
    inspect_run_checkpoints,
    inspect_runs,
)


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    keep_newest_runs: int = 25
    keep_checkpoints_per_run: int = 10

    def __post_init__(self) -> None:
        if self.keep_newest_runs < 1 or self.keep_checkpoints_per_run < 1:
            raise ValueError("retention counts must be positive")


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

    def dashboard(self, active_processes: Callable[[], list[ProcessSnapshot]]) -> DashboardSnapshot:
        run_dirs = self.run_directories()
        latest = inspect_run(run_dirs[-1], event_limit=5) if run_dirs else None
        return {
            "output_root": str(self.output_root),
            "run_count": len(run_dirs),
            "latest_run": latest,
            "active_processes": active_processes(),
        }

    def resolve_training_log(self, run: str | Path | None = None) -> Path | None:
        if run is not None:
            candidate = Path(run).expanduser()
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                return candidate / "logs" / "training.jsonl"
            return None
        runs = self.run_directories()
        return runs[-1] / "logs" / "training.jsonl" if runs else None

    def evaluation_roots(self, run: str | Path | None = None) -> list[Path]:
        if run is not None:
            path = Path(run).expanduser()
            return [path / "evaluations" if (path / "evaluations").is_dir() else path]
        roots = [path / "evaluations" for path in self.run_directories()]
        shared = self.output_root / "evaluations"
        if shared.is_dir():
            roots.append(shared)
        return roots

    def checkpoints(self, limit: int = 300) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_dir in self.run_directories():
            report = inspect_run_checkpoints(run_dir)
            checkpoints = report.get("checkpoints", {})
            run_id = report.get("run_id") or run_dir.name
            directory = Path(checkpoints.get("directory", run_dir / "checkpoints"))
            for item in checkpoints.get("entries", []):
                name = item.get("name", "")
                kind = {
                    "latest.zip": "latest",
                    "best_eval.zip": "best",
                    "best.zip": "best",
                }.get(name, "checkpoint")
                entries.append(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "kind": kind,
                        "path": str(directory / name),
                        "bytes": item.get("bytes"),
                        "modified_utc": item.get("modified_utc"),
                    }
                )
            final_path = run_dir / "final.zip"
            if final_path.is_file():
                modified = final_path.stat().st_mtime
                import time

                entries.append(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "kind": "final",
                        "path": str(final_path),
                        "bytes": final_path.stat().st_size,
                        "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified)),
                    }
                )
        entries.sort(key=lambda item: item.get("modified_utc") or "", reverse=True)
        return entries[:limit]

    def cleanup_plan(self, policy: RetentionPolicy) -> dict[str, list[str]]:
        """Return safe deletion candidates without changing the filesystem."""
        runs = self.run_directories()
        removable_runs: list[str] = []
        for path in runs[: -policy.keep_newest_runs]:
            report = inspect_run(path)
            state = str((report.get("status") or {}).get("state", ""))
            if state in {"finished", "stopped", "error"} and not path.is_symlink():
                removable_runs.append(str(path.resolve()))

        removable_checkpoints: list[str] = []
        for run in runs[-policy.keep_newest_runs :]:
            directory = run / "checkpoints"
            candidates = (
                [
                    path
                    for path in directory.glob("*.zip")
                    if path.name not in {"latest.zip", "best_eval.zip", "final.zip"}
                    and not path.is_symlink()
                ]
                if directory.is_dir()
                else []
            )
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            removable_checkpoints.extend(
                str(path.resolve()) for path in candidates[policy.keep_checkpoints_per_run :]
            )
        return {"runs": removable_runs, "checkpoints": removable_checkpoints}

    def cleanup(self, policy: RetentionPolicy, *, confirm: bool = False) -> dict[str, list[str]]:
        """Apply a reviewed cleanup plan; defaults to a non-destructive preview."""
        plan = self.cleanup_plan(policy)
        if not confirm:
            return plan
        root = self.output_root.resolve()
        for raw in plan["checkpoints"]:
            path = Path(raw).resolve()
            if root in path.parents:
                path.unlink(missing_ok=True)
        for raw in plan["runs"]:
            path = Path(raw).resolve()
            if root in path.parents and path != root:
                shutil.rmtree(path)
        return plan

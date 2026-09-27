"""Persistent, dependency-free experiment and metric tracking.

Every training/evaluation run owns a directory with an atomic manifest and an
append-only JSONL metric stream.  The format is deliberately simple so it can
be consumed by the CLI dashboard, a future UI, or ordinary analysis scripts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values into strict JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentTracker:
    """Own the durable record for one pipeline operation."""

    def __init__(
        self,
        kind: str,
        params: Optional[Dict[str, Any]] = None,
        root: Union[str, Path] = "logs/experiments",
        run_id: Optional[str] = None,
    ) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{stamp}_{kind}_{uuid.uuid4().hex[:8]}"
        self.kind = kind
        self.root = Path(root)
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.run_dir / "manifest.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.manifest: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "kind": kind,
            "status": "RUNNING",
            "started_at": utc_now(),
            "finished_at": None,
            "params": _json_safe(params or {}),
            "summary": {},
            "artifacts": [],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": self._git_commit(),
            },
        }
        self._save_manifest()

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    def _save_manifest(self) -> None:
        _atomic_json(self.manifest_path, self.manifest)

    def log_metrics(self, step: int, metrics: Dict[str, Any], phase: str = "train") -> None:
        event = {
            "timestamp": utc_now(),
            "step": int(step),
            "phase": phase,
            "metrics": _json_safe(metrics),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(event), separators=(",", ":"), allow_nan=False) + "\n"
            )
            handle.flush()

    def add_artifact(self, path: Union[str, Path], role: str) -> Dict[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Experiment artifact does not exist: {artifact_path}")
        artifact = {
            "role": role,
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": file_sha256(artifact_path),
        }
        self.manifest["artifacts"].append(artifact)
        self._save_manifest()
        return artifact

    def finish(self, summary: Optional[Dict[str, Any]] = None, status: str = "COMPLETED") -> None:
        self.manifest["status"] = status
        self.manifest["finished_at"] = utc_now()
        self.manifest["summary"] = _json_safe(summary or {})
        self._save_manifest()

    def fail(self, error: BaseException) -> None:
        self.finish(
            {"error_type": type(error).__name__, "error": str(error)}, status="FAILED"
        )

    @staticmethod
    def list_runs(root: Union[str, Path] = "logs/experiments") -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        root_path = Path(root)
        if not root_path.exists():
            return manifests
        for path in sorted(root_path.glob("*/manifest.json"), reverse=True):
            try:
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return manifests

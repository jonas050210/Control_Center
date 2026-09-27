"""Discover, validate, and summarize a repository of recording sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from data_pipeline.stats import inspect_dataset
from data_pipeline.validate import validate_dataset


def build_catalog(
    data_root: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    max_frame_checks: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(data_root)
    sessions: List[Dict[str, Any]] = []
    if (root / "metadata.json").is_file() and (root / "samples.jsonl").is_file():
        discovered = [root]
    elif root.exists():
        discovered = sorted(
            meta.parent
            for meta in root.rglob("metadata.json")
            if (meta.parent / "samples.jsonl").is_file()
        )
    else:
        discovered = []
    for session_path in discovered:
        validation = validate_dataset(session_path, max_frame_checks=max_frame_checks)
        entry: Dict[str, Any] = {
            "path": str(session_path),
            "valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "validation_stats": validation.stats,
        }
        if validation.is_valid:
            entry["statistics"] = inspect_dataset(session_path)
        sessions.append(entry)
    report = {
        "data_root": str(root),
        "sessions": sessions,
        "summary": {
            "session_count": len(sessions),
            "valid_sessions": sum(entry["valid"] for entry in sessions),
            "invalid_sessions": sum(not entry["valid"] for entry in sessions),
            "total_steps": sum(entry["validation_stats"].get("total_steps", 0) for entry in sessions),
        },
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalog SandboxAI datasets")
    parser.add_argument("--data", "-d", default="datasets")
    parser.add_argument("--output", "-o", default="logs/dataset_catalog.json")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()
    report = build_catalog(args.data, args.output, args.max_frames)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["session_count"] or report["summary"]["invalid_sessions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Dataset Validation Tool for SandboxAI (M1).

Checks metadata schema conformance, action bounds, frame integrity,
strictly monotonic timestamps, and synchronization consistency.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from data_pipeline.actions import MouseBinner
from data_pipeline.schema import (
    ACTION_SPACE_SPEC,
    DatasetMetadata,
    DatasetSample,
    SCHEMA_VERSION,
)

logger = logging.getLogger("SandboxAI.Validator")


@dataclasses.dataclass
class ValidationReport:
    """Detailed results of dataset validation."""

    is_valid: bool
    dataset_path: str
    schema_version: str
    total_steps: int
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    stats: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def summary(self) -> str:
        status_str = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Dataset Validation: {status_str} ===",
            f"Path: {self.dataset_path}",
            f"Schema Version: {self.schema_version}",
            f"Total Steps: {self.total_steps}",
        ]
        if self.stats:
            lines.append("Stats:")
            for k, v in self.stats.items():
                lines.append(f"  - {k}: {v}")

        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors[:20]:  # print first 20 errors
                lines.append(f"  [ERROR] {err}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more errors")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"  [WARN]  {w}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more warnings")

        return "\n".join(lines)


def validate_dataset(
    dataset_path: Union[str, Path],
    max_frame_checks: Optional[int] = None,
) -> ValidationReport:
    """Validates a dataset recording directory against the M1 schema and constraints."""
    path = Path(dataset_path)
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if not path.exists() or not path.is_dir():
        return ValidationReport(
            is_valid=False,
            dataset_path=str(path),
            schema_version="unknown",
            total_steps=0,
            errors=[f"Dataset directory does not exist or is not a directory: {path}"],
        )

    # 1. Check metadata.json
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return ValidationReport(
            is_valid=False,
            dataset_path=str(path),
            schema_version="missing",
            total_steps=0,
            errors=["Missing metadata.json file"],
        )

    try:
        metadata = DatasetMetadata.load(meta_path)
    except Exception as e:
        return ValidationReport(
            is_valid=False,
            dataset_path=str(path),
            schema_version="corrupted",
            total_steps=0,
            errors=[f"Failed to parse metadata.json: {e}"],
        )

    schema_version = metadata.schema_version
    if not schema_version:
        errors.append("metadata.json missing 'schema_version'")
    elif schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        errors.append(
            f"Unsupported schema major version {schema_version}; validator supports {SCHEMA_VERSION}"
        )

    recording_status = metadata.summary_stats.get("status", "complete")
    if recording_status != "complete":
        errors.append(f"Recording is not complete (status={recording_status})")
    if (path / ".recording").exists():
        errors.append("Recording lock marker is present; session may still be active or was interrupted")

    if not metadata.session_id:
        errors.append("metadata.json missing 'session_id'")

    target_w = metadata.capture_config.frame_width
    target_h = metadata.capture_config.frame_height
    target_fps = metadata.capture_config.target_fps
    num_bins_x = metadata.mouse_config.num_bins_x
    num_bins_y = metadata.mouse_config.num_bins_y

    for axis, num_bins, edges in (
        ("x", num_bins_x, metadata.mouse_config.bin_edges_x),
        ("y", num_bins_y, metadata.mouse_config.bin_edges_y),
    ):
        try:
            MouseBinner(
                num_bins=int(num_bins),
                strategy=metadata.mouse_config.binning_strategy,
                custom_edges=edges or None,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"Invalid mouse bin metadata for axis {axis}: {exc}")
    if not math.isfinite(metadata.mouse_config.sensitivity_scale):
        errors.append("Mouse sensitivity_scale must be finite")

    if target_w <= 0 or target_h <= 0:
        errors.append(f"Invalid frame dimensions in metadata: {target_w}x{target_h}")

    if target_fps <= 0:
        errors.append(f"Invalid target_fps in metadata: {target_fps}")

    # 2. Check samples.jsonl
    samples_path = path / "samples.jsonl"
    if not samples_path.exists():
        return ValidationReport(
            is_valid=False,
            dataset_path=str(path),
            schema_version=schema_version,
            total_steps=0,
            errors=["Missing samples.jsonl file"],
        )

    total_steps = 0
    t_prev: Optional[float] = None
    expected_step_idx = 0
    dt_list: List[float] = []

    frame_checks_count = 0
    check_all_frames = max_frame_checks is None
    referenced_frames: set[str] = set()
    resolved_root = path.resolve()

    with open(samples_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                sample_dict = json.loads(line_str)
                sample = DatasetSample.from_dict(sample_dict)
            except Exception as e:
                errors.append(f"Line {line_num}: Malformed sample JSON: {e}")
                continue

            # Sequential step index check
            if sample.step_idx != expected_step_idx:
                errors.append(
                    f"Line {line_num}: Step index out of order (expected {expected_step_idx}, got {sample.step_idx})"
                )
            expected_step_idx += 1
            total_steps += 1

            if not sample.is_valid:
                warnings.append(f"Line {line_num}: Sample is explicitly marked invalid")
            if not math.isfinite(sample.timestamp) or not math.isfinite(sample.dt):
                errors.append(f"Line {line_num}: timestamp/dt must be finite")

            # Timestamp monotonicity check
            if t_prev is not None:
                if sample.timestamp <= t_prev:
                    errors.append(
                        f"Line {line_num}: Non-monotonic timestamp (t_prev={t_prev:.6f}, t_curr={sample.timestamp:.6f})"
                    )
                calculated_dt = sample.timestamp - t_prev
                if abs(sample.dt - calculated_dt) > 1e-4:
                    warnings.append(
                        f"Line {line_num}: Stored dt ({sample.dt:.6f}) differs from timestamp delta ({calculated_dt:.6f})"
                    )
                dt_list.append(sample.dt)

                # Check for severe lag spikes / frame drops
                target_interval = 1.0 / max(1.0, target_fps)
                if sample.dt > target_interval * 3.0:
                    warnings.append(
                        f"Line {line_num}: Large frame delay (dt={sample.dt:.3f}s vs target {target_interval:.3f}s)"
                    )
            t_prev = sample.timestamp

            # Action bounds check
            act = sample.actions
            if act.move_x not in (-1, 0, 1):
                errors.append(f"Line {line_num}: Invalid move_x value {act.move_x}")
            if act.move_y not in (-1, 0, 1):
                errors.append(f"Line {line_num}: Invalid move_y value {act.move_y}")
            for flag_name, flag_val in [
                ("jump", act.jump),
                ("crouch", act.crouch),
                ("sprint", act.sprint),
                ("reload", act.reload),
                ("fire", act.fire),
                ("ads", act.ads),
            ]:
                if flag_val not in (0, 1):
                    errors.append(f"Line {line_num}: Invalid binary flag {flag_name}={flag_val}")

            # Mouse bin and finite continuous delta checks
            if not math.isfinite(act.mouse_dx) or not math.isfinite(act.mouse_dy):
                errors.append(f"Line {line_num}: Mouse deltas must be finite")
            if not (0 <= act.mouse_dx_bin < num_bins_x):
                errors.append(
                    f"Line {line_num}: mouse_dx_bin {act.mouse_dx_bin} out of range [0, {num_bins_x - 1}]"
                )
            if not (0 <= act.mouse_dy_bin < num_bins_y):
                errors.append(
                    f"Line {line_num}: mouse_dy_bin {act.mouse_dy_bin} out of range [0, {num_bins_y - 1}]"
                )

            # Frame image check
            if sample.frame_file in referenced_frames:
                errors.append(f"Line {line_num}: Duplicate frame reference: {sample.frame_file}")
            referenced_frames.add(sample.frame_file)
            frame_path = (path / sample.frame_file).resolve()
            try:
                frame_path.relative_to(resolved_root)
            except ValueError:
                errors.append(f"Line {line_num}: Frame path escapes dataset directory: {sample.frame_file}")
                continue
            if check_all_frames or frame_checks_count < max_frame_checks:
                frame_checks_count += 1
                if not frame_path.exists():
                    errors.append(f"Line {line_num}: Referenced frame not found on disk: {sample.frame_file}")
                else:
                    try:
                        with Image.open(frame_path) as img:
                            if img.size != (target_w, target_h):
                                errors.append(
                                    f"Line {line_num}: Frame dimensions {img.size} mismatch metadata {target_w}x{target_h}"
                                )
                            if img.mode != "RGB":
                                warnings.append(
                                    f"Line {line_num}: Frame color mode {img.mode} is not RGB"
                                )
                    except Exception as e:
                        errors.append(f"Line {line_num}: Corrupted frame image {sample.frame_file}: {e}")

    if total_steps == 0:
        errors.append("Dataset contains 0 valid steps in samples.jsonl")
    stored_steps = metadata.summary_stats.get("total_steps")
    if stored_steps not in (None, 0) and int(stored_steps) != total_steps:
        errors.append(
            f"metadata total_steps={stored_steps} does not match samples.jsonl count={total_steps}"
        )
    frames_dir = path / "frames"
    if frames_dir.is_dir():
        disk_frames = {
            str(frame.relative_to(path)).replace("\\", "/")
            for frame in frames_dir.rglob("*")
            if frame.is_file() and ".tmp." not in frame.name
        }
        orphan_count = len(disk_frames - referenced_frames)
        if orphan_count:
            warnings.append(f"Dataset contains {orphan_count} unreferenced frame file(s)")
        stats["orphan_frames"] = orphan_count

    # Summary statistics calculation
    if dt_list:
        mean_dt = sum(dt_list) / len(dt_list)
        effective_fps = 1.0 / mean_dt if mean_dt > 0 else 0.0
        stats["mean_dt_sec"] = round(mean_dt, 4)
        stats["effective_fps"] = round(effective_fps, 2)
        stats["min_dt_sec"] = round(min(dt_list), 4)
        stats["max_dt_sec"] = round(max(dt_list), 4)

    stats["total_steps"] = total_steps
    stats["frame_checks_performed"] = frame_checks_count

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid=is_valid,
        dataset_path=str(path),
        schema_version=schema_version,
        total_steps=total_steps,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SandboxAI M1 dataset")
    parser.add_argument("path", nargs="?", help="Path to the dataset directory")
    parser.add_argument("--dataset", "-d", dest="dataset_opt", help="Path to dataset directory")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max number of frame images to verify (default: all)",
    )
    args = parser.parse_args()

    dataset_path = args.path or args.dataset_opt
    if not dataset_path:
        parser.print_help()
        sys.exit(1)

    report = validate_dataset(dataset_path, max_frame_checks=args.max_frames)
    print(report.summary())
    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()

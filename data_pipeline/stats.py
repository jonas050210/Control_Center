"""Dataset Statistics and Inspection Tool for SandboxAI (M1).

Computes comprehensive dataset metrics: action distributions, mouse histograms,
effective FPS, timing jitter, and storage footprint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data_pipeline.schema import DatasetMetadata, DatasetSample


def compute_ascii_histogram(
    bins: Sequence[int],
    num_bins: int,
    bar_width: int = 30,
) -> List[str]:
    """Generates an ASCII bar chart for discrete bin counts."""
    counts = np.bincount(bins, minlength=num_bins)
    max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
    lines = []
    for b_idx, count in enumerate(counts):
        bar_len = int((count / max_count) * bar_width)
        bar_str = "█" * bar_len
        pct = (count / len(bins) * 100.0) if len(bins) > 0 else 0.0
        lines.append(f"  Bin {b_idx:02d}: {bar_str:<{bar_width}} {count:6d} ({pct:5.1f}%)")
    return lines


def inspect_dataset(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """Calculates full dataset inspection metrics."""
    path = Path(dataset_path)
    meta_path = path / "metadata.json"
    samples_path = path / "samples.jsonl"

    if not meta_path.exists() or not samples_path.exists():
        raise FileNotFoundError(f"Missing metadata.json or samples.jsonl in {path}")

    metadata = DatasetMetadata.load(meta_path)

    samples: List[DatasetSample] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                samples.append(DatasetSample.from_dict(json.loads(line_str)))

    if not samples:
        return {"error": "Dataset has 0 samples"}

    total_steps = len(samples)
    dts = [s.dt for s in samples[1:]] if len(samples) > 1 else [0.0]
    total_duration = sum(dts)
    mean_dt = float(np.mean(dts)) if dts else 0.0
    std_dt = float(np.std(dts)) if dts else 0.0
    min_dt = float(np.min(dts)) if dts else 0.0
    max_dt = float(np.max(dts)) if dts else 0.0
    effective_fps = 1.0 / mean_dt if mean_dt > 0 else 0.0

    # Actions accumulation
    move_x_vals = [s.actions.move_x for s in samples]
    move_y_vals = [s.actions.move_y for s in samples]
    jumps = [s.actions.jump for s in samples]
    crouches = [s.actions.crouch for s in samples]
    sprints = [s.actions.sprint for s in samples]
    reloads = [s.actions.reload for s in samples]
    fires = [s.actions.fire for s in samples]
    adses = [s.actions.ads for s in samples]

    mouse_dxs = [s.actions.mouse_dx for s in samples]
    mouse_dys = [s.actions.mouse_dy for s in samples]
    mouse_dx_bins = [s.actions.mouse_dx_bin for s in samples]
    mouse_dy_bins = [s.actions.mouse_dy_bin for s in samples]

    # Action frequencies (%)
    action_stats = {
        "forward_pct": round(sum(1 for y in move_y_vals if y == 1) / total_steps * 100.0, 2),
        "backward_pct": round(sum(1 for y in move_y_vals if y == -1) / total_steps * 100.0, 2),
        "strafe_left_pct": round(sum(1 for x in move_x_vals if x == -1) / total_steps * 100.0, 2),
        "strafe_right_pct": round(sum(1 for x in move_x_vals if x == 1) / total_steps * 100.0, 2),
        "idle_move_pct": round(
            sum(1 for x, y in zip(move_x_vals, move_y_vals) if x == 0 and y == 0) / total_steps * 100.0,
            2,
        ),
        "fire_lmb_pct": round(sum(fires) / total_steps * 100.0, 2),
        "ads_rmb_pct": round(sum(adses) / total_steps * 100.0, 2),
        "jump_pct": round(sum(jumps) / total_steps * 100.0, 2),
        "crouch_pct": round(sum(crouches) / total_steps * 100.0, 2),
        "sprint_pct": round(sum(sprints) / total_steps * 100.0, 2),
        "reload_pct": round(sum(reloads) / total_steps * 100.0, 2),
    }

    # Mouse statistics
    def calc_dist_metrics(arr: List[float]) -> Dict[str, float]:
        np_arr = np.asarray(arr, dtype=np.float64)
        return {
            "mean": round(float(np.mean(np_arr)), 3),
            "std": round(float(np.std(np_arr)), 3),
            "min": round(float(np.min(np_arr)), 3),
            "max": round(float(np.max(np_arr)), 3),
            "p5": round(float(np.percentile(np_arr, 5)), 3),
            "p25": round(float(np.percentile(np_arr, 25)), 3),
            "median": round(float(np.median(np_arr)), 3),
            "p75": round(float(np.percentile(np_arr, 75)), 3),
            "p95": round(float(np.percentile(np_arr, 95)), 3),
        }

    mouse_dx_metrics = calc_dist_metrics(mouse_dxs)
    mouse_dy_metrics = calc_dist_metrics(mouse_dys)

    # Disk usage
    total_bytes = 0
    frame_bytes = 0
    num_frame_files = 0
    for p in path.rglob("*"):
        if p.is_file():
            size = p.stat().st_size
            total_bytes += size
            if "frames" in p.parts:
                frame_bytes += size
                num_frame_files += 1

    avg_frame_kb = (frame_bytes / num_frame_files / 1024.0) if num_frame_files > 0 else 0.0

    return {
        "session_id": metadata.session_id,
        "schema_version": metadata.schema_version,
        "source": metadata.source,
        "target_fps": metadata.capture_config.target_fps,
        "resolution": f"{metadata.capture_config.frame_width}x{metadata.capture_config.frame_height}",
        "total_steps": total_steps,
        "duration_seconds": round(total_duration, 3),
        "effective_fps": round(effective_fps, 2),
        "dt_metrics": {
            "mean_sec": round(mean_dt, 4),
            "std_sec": round(std_dt, 4),
            "min_sec": round(min_dt, 4),
            "max_sec": round(max_dt, 4),
        },
        "action_stats": action_stats,
        "mouse_dx_metrics": mouse_dx_metrics,
        "mouse_dy_metrics": mouse_dy_metrics,
        "mouse_dx_bins": mouse_dx_bins,
        "mouse_dy_bins": mouse_dy_bins,
        "num_bins_x": metadata.mouse_config.num_bins_x,
        "num_bins_y": metadata.mouse_config.num_bins_y,
        "storage": {
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "frame_files_count": num_frame_files,
            "avg_frame_kb": round(avg_frame_kb, 2),
        },
    }


def format_inspection_report(metrics: Dict[str, Any]) -> str:
    """Formats inspection metrics into a clear human-readable console report."""
    lines = [
        "============================================================",
        f"  SandboxAI Dataset Inspection Report",
        "============================================================",
        f"Session ID       : {metrics['session_id']}",
        f"Schema Version   : {metrics['schema_version']}",
        f"Source           : {metrics['source']}",
        f"Resolution       : {metrics['resolution']}",
        f"Total Steps      : {metrics['total_steps']}",
        f"Duration         : {metrics['duration_seconds']} s",
        f"Effective FPS    : {metrics['effective_fps']} (target {metrics['target_fps']})",
        f"Frame Interval dt: {metrics['dt_metrics']['mean_sec'] * 1000:.1f} ms ± {metrics['dt_metrics']['std_sec'] * 1000:.1f} ms",
        f"Storage Footprint: {metrics['storage']['total_mb']} MB (avg frame {metrics['storage']['avg_frame_kb']} KB)",
        "",
        "--- Action Frequencies ---",
        f"  Forward (W)     : {metrics['action_stats']['forward_pct']:5.1f}%",
        f"  Backward (S)    : {metrics['action_stats']['backward_pct']:5.1f}%",
        f"  Strafe Left (A) : {metrics['action_stats']['strafe_left_pct']:5.1f}%",
        f"  Strafe Right (D): {metrics['action_stats']['strafe_right_pct']:5.1f}%",
        f"  Idle (No Move)  : {metrics['action_stats']['idle_move_pct']:5.1f}%",
        f"  Fire (LMB)      : {metrics['action_stats']['fire_lmb_pct']:5.1f}%",
        f"  ADS (RMB)       : {metrics['action_stats']['ads_rmb_pct']:5.1f}%",
        f"  Jump (Space)    : {metrics['action_stats']['jump_pct']:5.1f}%",
        f"  Crouch (Ctrl/C) : {metrics['action_stats']['crouch_pct']:5.1f}%",
        f"  Sprint (Shift)  : {metrics['action_stats']['sprint_pct']:5.1f}%",
        f"  Reload (R)      : {metrics['action_stats']['reload_pct']:5.1f}%",
        "",
        "--- Mouse Continuous Deltas (pixels / step) ---",
        f"  Mouse dX: mean={metrics['mouse_dx_metrics']['mean']:+.2f}, std={metrics['mouse_dx_metrics']['std']:.2f}, "
        f"p5={metrics['mouse_dx_metrics']['p5']:+.1f}, median={metrics['mouse_dx_metrics']['median']:+.1f}, p95={metrics['mouse_dx_metrics']['p95']:+.1f}",
        f"  Mouse dY: mean={metrics['mouse_dy_metrics']['mean']:+.2f}, std={metrics['mouse_dy_metrics']['std']:.2f}, "
        f"p5={metrics['mouse_dy_metrics']['p5']:+.1f}, median={metrics['mouse_dy_metrics']['median']:+.1f}, p95={metrics['mouse_dy_metrics']['p95']:+.1f}",
        "",
        "--- Mouse dX Bin Distribution Histogram ---",
    ]
    lines.extend(
        compute_ascii_histogram(
            metrics["mouse_dx_bins"],
            metrics["num_bins_x"],
        )
    )
    lines.append("")
    lines.append("--- Mouse dY Bin Distribution Histogram ---")
    lines.extend(
        compute_ascii_histogram(
            metrics["mouse_dy_bins"],
            metrics["num_bins_y"],
        )
    )
    lines.append("============================================================")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a SandboxAI M1 dataset")
    parser.add_argument("path", nargs="?", help="Path to the dataset directory")
    parser.add_argument("--dataset", "-d", dest="dataset_opt", help="Path to dataset directory")
    args = parser.parse_args()

    dataset_path = args.path or args.dataset_opt
    if not dataset_path:
        parser.print_help()
        sys.exit(1)

    try:
        metrics = inspect_dataset(dataset_path)
        print(format_inspection_report(metrics))
    except Exception as e:
        print(f"[ERROR] Inspection failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

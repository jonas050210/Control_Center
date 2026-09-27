"""Offline behavior-cloning evaluation on held-out recorded sessions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from bc.dataset import GameplayDataset
from bc.policy import BCPolicy
from data_pipeline.schema import DatasetSample
from monitoring.experiments import ExperimentTracker

CATEGORICAL_ACTIONS = (
    "move_x",
    "move_y",
    "jump",
    "crouch",
    "sprint",
    "reload",
    "fire",
    "ads",
    "mouse_dx_bin",
    "mouse_dy_bin",
)


def _macro_f1(truth: List[int], prediction: List[int]) -> float:
    labels = sorted(set(truth) | set(prediction))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_imitation(
    checkpoint_path: Union[str, Path],
    data_path: Union[str, Path],
    device: str = "cpu",
    max_samples: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    context_start_step: Optional[int] = None,
    output_report: Optional[Union[str, Path]] = None,
    experiment_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Measure per-head accuracy/F1 and joint action agreement."""
    policy = BCPolicy.load_from_checkpoint(checkpoint_path, device=device)
    sessions = GameplayDataset.discover_sessions(data_path)
    if not sessions:
        raise ValueError(f"No dataset sessions found under {data_path}")
    tracker = (
        ExperimentTracker(
            "imitation_eval",
            {
                "checkpoint": str(checkpoint_path),
                "data": str(data_path),
                "start_step": start_step,
                "end_step": end_step,
                "context_start_step": context_start_step,
            },
            experiment_root,
        )
        if experiment_root
        else None
    )
    truths: DefaultDict[str, List[int]] = defaultdict(list)
    predictions: DefaultDict[str, List[int]] = defaultdict(list)
    mouse_error = {"x_pixels": [], "y_pixels": [], "x_bins": [], "y_bins": []}
    joint_correct = 0
    count = 0
    confidence: DefaultDict[str, List[float]] = defaultdict(list)

    try:
        for session in sessions:
            policy.reset()
            with (session / "samples.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if max_samples is not None and count >= max_samples:
                        break
                    if not line.strip():
                        continue
                    sample = DatasetSample.from_dict(json.loads(line))
                    if context_start_step is not None and sample.step_idx < context_start_step:
                        continue
                    if end_step is not None and sample.step_idx >= end_step:
                        continue
                    if not sample.is_valid:
                        continue
                    frame = session / sample.frame_file
                    if not frame.is_file():
                        continue
                    with Image.open(frame) as image:
                        predicted, raw = policy.predict(image, deterministic=True)
                    # Context frames initialize the recurrent state but are not
                    # validation targets, matching GameplayDataset windows.
                    if start_step is not None and sample.step_idx < start_step:
                        continue
                    all_correct = True
                    for action_name in CATEGORICAL_ACTIONS:
                        truth = int(getattr(sample.actions, action_name))
                        pred = int(getattr(predicted, action_name))
                        truths[action_name].append(truth)
                        predictions[action_name].append(pred)
                        all_correct = all_correct and truth == pred
                        if action_name in raw.get("confidence", {}):
                            confidence[action_name].append(raw["confidence"][action_name])
                    joint_correct += int(all_correct)
                    mouse_error["x_pixels"].append(abs(predicted.mouse_dx - sample.actions.mouse_dx))
                    mouse_error["y_pixels"].append(abs(predicted.mouse_dy - sample.actions.mouse_dy))
                    mouse_error["x_bins"].append(
                        abs(predicted.mouse_dx_bin - sample.actions.mouse_dx_bin)
                    )
                    mouse_error["y_bins"].append(
                        abs(predicted.mouse_dy_bin - sample.actions.mouse_dy_bin)
                    )
                    count += 1
                if max_samples is not None and count >= max_samples:
                    break
        if count == 0:
            raise ValueError("No valid samples were available for imitation evaluation")

        action_metrics = {}
        for name in CATEGORICAL_ACTIONS:
            truth, pred = truths[name], predictions[name]
            action_metrics[name] = {
                "accuracy_pct": round(
                    100.0 * sum(t == p for t, p in zip(truth, pred)) / count, 3
                ),
                "macro_f1": round(_macro_f1(truth, pred), 5),
                "mean_confidence": round(float(np.mean(confidence[name])), 5)
                if confidence[name]
                else None,
                "support": count,
            }
        report: Dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "data_path": str(data_path),
            "sessions": len(sessions),
            "step_range": [start_step, end_step],
            "context_start_step": context_start_step,
            "samples": count,
            "joint_action_accuracy_pct": round(joint_correct / count * 100.0, 3),
            "actions": action_metrics,
            "mouse_error": {
                key: {"mae": round(float(np.mean(values)), 4), "p95": round(float(np.percentile(values, 95)), 4)}
                for key, values in mouse_error.items()
            },
        }
        if output_report:
            report_path = Path(output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if tracker:
            tracker.log_metrics(0, report, phase="evaluation")
            if output_report:
                tracker.add_artifact(output_report, "imitation_report")
            tracker.finish(report)
            report["run_id"] = tracker.run_id
        return report
    except Exception as exc:
        if tracker:
            tracker.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC on recorded gameplay")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--data", "-d", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--end_step", type=int, default=None)
    parser.add_argument("--context_start_step", type=int, default=None)
    parser.add_argument("--output", default="logs/imitation_evaluation.json")
    parser.add_argument("--experiment_root", default="logs/experiments")
    args = parser.parse_args()
    report = evaluate_imitation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        device=args.device,
        max_samples=args.max_samples,
        start_step=args.start_step,
        end_step=args.end_step,
        context_start_step=args.context_start_step,
        output_report=args.output,
        experiment_root=args.experiment_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

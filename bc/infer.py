"""CLI and Python utility for running inference with a trained Behavioral Cloning checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from bc.policy import BCPolicy
from data_pipeline.schema import DatasetSample


def run_inference_on_image(
    checkpoint_path: Union[str, Path],
    image: Union[Image.Image, np.ndarray, str, Path],
    device: str = "cpu",
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Runs BC policy inference on a single image input."""
    policy = BCPolicy.load_from_checkpoint(checkpoint_path, device=device)
    if isinstance(image, (str, Path)):
        img_obj = Image.open(image)
    else:
        img_obj = image
    action_state, raw_dict = policy.predict(img_obj, deterministic=deterministic)
    return {
        "action_state": action_state.to_dict(),
        "raw_dict": raw_dict,
    }


def run_inference_on_dataset(
    checkpoint_path: Union[str, Path],
    dataset_path: Union[str, Path],
    max_samples: Optional[int] = 50,
    device: str = "cpu",
    deterministic: bool = True,
) -> List[Dict[str, Any]]:
    """Runs BC policy inference across recorded samples in a dataset session."""
    policy = BCPolicy.load_from_checkpoint(checkpoint_path, device=device)
    ds_path = Path(dataset_path)
    samples_file = ds_path / "samples.jsonl"
    if not samples_file.exists():
        raise FileNotFoundError(f"Missing samples.jsonl in {ds_path}")

    results: List[Dict[str, Any]] = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line_str = line.strip()
            if not line_str:
                continue
            sample = DatasetSample.from_dict(json.loads(line_str))
            frame_p = ds_path / sample.frame_file
            if not frame_p.exists():
                continue
            with Image.open(frame_p) as img:
                act_state, raw_dict = policy.predict(img, deterministic=deterministic)

            results.append({
                "step_idx": sample.step_idx,
                "timestamp": sample.timestamp,
                "predicted_move_x": act_state.move_x,
                "predicted_move_y": act_state.move_y,
                "predicted_fire": act_state.fire,
                "predicted_mouse_dx_bin": act_state.mouse_dx_bin,
                "predicted_mouse_dy_bin": act_state.mouse_dy_bin,
                "ground_truth_move_x": sample.actions.move_x,
                "ground_truth_move_y": sample.actions.move_y,
                "ground_truth_fire": sample.actions.fire,
                "ground_truth_mouse_dx_bin": sample.actions.mouse_dx_bin,
                "ground_truth_mouse_dy_bin": sample.actions.mouse_dy_bin,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using a trained BC policy checkpoint")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to bc_best.pt / bc_latest.pt")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to frame image (optional)")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Path to dataset session directory (optional)")
    parser.add_argument("--max_samples", "-n", type=int, default=10, help="Max dataset samples to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (cpu/cuda)")
    args = parser.parse_args()

    if args.dataset:
        print(f"[INFO] Running inference over dataset: {args.dataset}")
        results = run_inference_on_dataset(args.checkpoint, args.dataset, max_samples=args.max_samples, device=args.device)
        print(f"[INFO] Evaluated {len(results)} samples.")
        print(json.dumps(results[:3], indent=2))
        return

    policy = BCPolicy.load_from_checkpoint(args.checkpoint, device=args.device)

    if args.image and Path(args.image).exists():
        img = Image.open(args.image)
        print(f"[INFO] Loaded image from: {args.image} ({img.size})")
    else:
        # Dummy frame
        img = Image.new("RGB", (policy.target_w, policy.target_h), color=(50, 60, 70))
        print(f"[INFO] Generated dummy {policy.target_w}x{policy.target_h} RGB frame for inference test")

    action_state, raw_dict = policy.predict(img, deterministic=True)

    print("\n=== Predicted Action State ===")
    print(json.dumps(action_state.to_dict(), indent=2))


if __name__ == "__main__":
    main()

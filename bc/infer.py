"""CLI Tool for running inference with a trained Behavioral Cloning checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from bc.policy import BCPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using a trained BC policy checkpoint")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to bc_best.pt / bc_latest.pt")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to frame image (optional, generates dummy if None)")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (cpu/cuda)")
    args = parser.parse_args()

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

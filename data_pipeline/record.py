"""CLI Entry point for recording SandboxAI gameplay datasets (M1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_pipeline.recorder import SessionRecorder
from data_pipeline.schema import MouseConfig
from data_pipeline.stats import format_inspection_report, inspect_dataset
from data_pipeline.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record gameplay data for SandboxAI Behavioral Cloning (M1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="datasets",
        help="Root directory where recorded sessions will be saved",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default=None,
        help="Custom session identifier (default: auto timestamped UUID)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target capture rate in frames per second (Hz)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Target observation frame width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Target observation frame height in pixels",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Frame image compression format",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG compression quality (1-100)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in synthetic mock mode (generates simulated frames/inputs without Roblox/display)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="ttk_testing",
        choices=["ttk_testing", "godot_sandbox", "desktop"],
        help="Data source label",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Window title pattern to auto-detect and crop to (e.g. 'Roblox', 'Godot')",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        default=None,
        help="Explicit capture Region of Interest (left top width height)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=21,
        help="Number of mouse discretization bins (must be an odd integer >= 3)",
    )
    parser.add_argument(
        "--bin_strategy",
        type=str,
        default="symmetric_log",
        choices=["symmetric_log", "uniform", "quantile"],
        help="Mouse delta discretization binning strategy",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip automatic validation upon session completion",
    )

    args = parser.parse_args()

    mouse_config = MouseConfig(
        sensitivity_scale=1.0,
        binning_strategy=args.bin_strategy,
        num_bins_x=args.bins,
        num_bins_y=args.bins,
    )

    print("============================================================")
    print("  SandboxAI Data Recorder (M1)")
    print("============================================================")
    print(f"Mode          : {'SYNTHETIC MOCK' if args.mock else 'LIVE CAPTURE'}")
    print(f"Source        : {args.source if not args.mock else 'mock_synthetic'}")
    print(f"Target FPS    : {args.fps} Hz")
    print(f"Frame Res     : {args.width}x{args.height} ({args.format.upper()})")
    print(f"Mouse Bins    : {args.bins} ({args.bin_strategy})")
    print(f"Output Root   : {Path(args.output).resolve()}")
    if args.duration:
        print(f"Duration Limit: {args.duration} s")
    else:
        print("Duration Limit: None (Press Ctrl+C to stop recording)")
    print("============================================================")

    recorder = SessionRecorder(
        output_dir=args.output,
        session_id=args.session_id,
        source_name=args.source,
        target_fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        image_format=args.format,
        jpeg_quality=args.quality,
        mouse_config=mouse_config,
        window_title=args.window or ("Roblox" if args.source == "ttk_testing" else None),
        roi=args.roi,
        is_mock=args.mock,
    )

    print(f"[INFO] Recording session: {recorder.session_id}")
    print("[INFO] Starting capture loop... Press Ctrl+C at any time to finish.")

    try:
        metadata = recorder.record(max_duration=args.duration)
    except KeyboardInterrupt:
        print("\n[INFO] Caught Ctrl+C, finalizing dataset...")
        recorder.request_stop()
        metadata = recorder.record()

    session_dir = recorder.session_dir
    print(f"\n[INFO] Session saved to: {session_dir}")
    print(f"[INFO] Total steps recorded: {metadata.summary_stats.get('total_steps', 0)}")
    print(f"[INFO] Duration: {metadata.summary_stats.get('duration_seconds', 0.0)} s")
    print(f"[INFO] Effective FPS: {metadata.summary_stats.get('effective_fps', 0.0)} Hz")

    if not args.no_validate:
        print("\n[INFO] Running automatic dataset validation...")
        report = validate_dataset(session_dir)
        print(report.summary())

        if report.is_valid:
            print("\n[INFO] Inspection summary:")
            metrics = inspect_dataset(session_dir)
            print(format_inspection_report(metrics))
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()

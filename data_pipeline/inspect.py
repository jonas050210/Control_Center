"""Generate a visual, self-contained HTML inspection report for a session."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import List, Union

from PIL import Image

from data_pipeline.schema import DatasetSample
from data_pipeline.stats import inspect_dataset
from data_pipeline.validate import validate_dataset


def generate_inspection_report(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path],
    max_frames: int = 24,
) -> Path:
    dataset = Path(dataset_path)
    validation = validate_dataset(dataset)
    statistics = inspect_dataset(dataset) if validation.total_steps else {"error": "empty"}
    samples: List[DatasetSample] = []
    with (dataset / "samples.jsonl").open("r", encoding="utf-8") as handle:
        samples = [DatasetSample.from_dict(json.loads(line)) for line in handle if line.strip()]
    if max_frames > 0 and len(samples) > max_frames:
        if max_frames == 1:
            samples = [samples[len(samples) // 2]]
        else:
            indices = sorted(
                {round(i * (len(samples) - 1) / (max_frames - 1)) for i in range(max_frames)}
            )
            samples = [samples[index] for index in indices]

    cards = []
    for sample in samples:
        with Image.open(dataset / sample.frame_file) as frame:
            frame = frame.convert("RGB")
            frame.thumbnail((320, 240), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=82)
        action = sample.actions
        image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        cards.append(
            f"<article><img src='{image_url}' alt='frame {sample.step_idx}'>"
            f"<b>step {sample.step_idx}</b><code>move=({action.move_x},{action.move_y}) "
            f"look=({action.mouse_dx_bin},{action.mouse_dy_bin}) fire={action.fire} "
            f"ads={action.ads} reload={action.reload}</code></article>"
        )
    validation_class = "ok" if validation.is_valid else "bad"
    payload = html.escape(json.dumps(statistics, indent=2))
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Dataset inspection</title>
<style>body{{font:14px system-ui;background:#0d121a;color:#edf3fa;margin:30px}}h1{{margin-bottom:4px}}.ok{{color:#5ce0a6}}.bad{{color:#ff756f}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}article{{background:#161f2c;padding:10px;border-radius:8px;border:1px solid #29364a}}img{{width:100%;image-rendering:auto}}code{{display:block;white-space:normal;color:#b7c5d9;margin-top:6px}}pre{{background:#111925;padding:14px;overflow:auto}}</style></head>
<body><h1>SandboxAI Dataset</h1><p>{html.escape(str(dataset))}</p><h2 class='{validation_class}'>Validation: {'PASSED' if validation.is_valid else 'FAILED'}</h2>
<pre>{html.escape(validation.summary())}</pre><h2>Timeline sample</h2><div class='grid'>{''.join(cards)}</div><h2>Statistics</h2><pre>{payload}</pre></body></html>"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual dataset report")
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--output", "-o", default="logs/dataset_inspection.html")
    parser.add_argument("--max_frames", type=int, default=24)
    args = parser.parse_args()
    print(generate_inspection_report(args.dataset, args.output, args.max_frames))


if __name__ == "__main__":
    main()

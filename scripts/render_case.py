from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image, ImageDraw


def _load_record(path: Path, sample_id: str | None) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        if sample_id is None or record.get("sample_id") == sample_id:
            return record
    raise ValueError(f"No matching record found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a causal selected-frame manifest.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    args = parser.parse_args()
    record = _load_record(args.input, args.sample_id)
    query_time_s = float(record["query_time_s"])
    frames = record.get("selected_frames", [])
    if any(float(frame["timestamp_s"]) > query_time_s + 1e-6 for frame in frames):
        raise ValueError("refusing to render a record containing future frames")
    tile_width, tile_height = 280, 120
    canvas = Image.new("RGB", (tile_width * max(1, len(frames)), tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    if not frames:
        draw.text((12, 12), "No selected frames", fill="black")
    for index, frame in enumerate(frames):
        x = index * tile_width
        timestamp_s = float(frame["timestamp_s"])
        draw.rectangle((x + 4, 4, x + tile_width - 4, tile_height - 4), fill=(225, 232, 240), outline="black")
        draw.text((x + 14, 18), f"frame_id={frame['frame_id']}", fill="black")
        draw.text((x + 14, 46), f"absolute={timestamp_s:.2f}s", fill="black")
        draw.text((x + 14, 74), f"relative={timestamp_s - query_time_s:.2f}s", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote causal case visualization to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

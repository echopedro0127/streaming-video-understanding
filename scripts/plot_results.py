from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from streaming_video_understanding.analysis.summaries import build_comparison_rows


CSV_FIELDS = (
    "policy_id",
    "policy_label",
    "question_accuracy",
    "p50_latency_s",
    "p95_latency_s",
    "peak_gpu_memory_mb",
    "average_input_frames",
    "retained_frames",
    "compression_ratio",
    "throughput_samples_per_s",
    "num_samples",
)

POLICY_COLORS = ("#3366cc", "#dc3912", "#109618", "#990099")


def _annotation_offsets(count: int) -> list[tuple[int, int]]:
    """Return vertically separated label offsets for collocated points."""
    center = (count - 1) / 2.0
    return [(6, int(round((index - center) * 22))) for index in range(count)]


def _save_figure(fig: Any, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, format="png")
    plt.close(fig)


def _scatter_axes(rows: list[dict[str, Any]]) -> tuple[list[str], list[float], list[float]]:
    labels = [str(row["policy_label"]) for row in rows]
    accuracies = [float(row["question_accuracy"]) * 100.0 for row in rows]
    return labels, accuracies, list(POLICY_COLORS[: len(rows)])


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, accuracies, colors = _scatter_axes(rows)
    outputs = [
        output_dir / "accuracy-vs-latency.png",
        output_dir / "accuracy-vs-gpu-memory.png",
        output_dir / "accuracy-vs-context.png",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = [float(row["p50_latency_s"]) for row in rows]
    upper_errors = [max(0.0, float(row["p95_latency_s"]) - float(row["p50_latency_s"])) for row in rows]
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.errorbar(
            x[index],
            accuracies[index],
            xerr=[[0.0], [upper_errors[index]]],
            fmt="o",
            color=color,
            capsize=4,
            label=label,
        )
    ax.set_xlabel("P50 latency (s); whisker ends at P95")
    ax.set_ylabel("Question accuracy (%)")
    ax.set_title("Streaming policy accuracy vs latency")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(fig, outputs[0])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = [float(row["peak_gpu_memory_mb"]) for row in rows]
    ax.scatter(x, accuracies, c=colors, s=75)
    for x_value, y_value, label, offset in zip(x, accuracies, labels, _annotation_offsets(len(rows))):
        ax.annotate(label, (x_value, y_value), xytext=offset, textcoords="offset points", fontsize=8)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.set_xlim(min(x) - 10.0, max(x) + 10.0)
    ax.set_xlabel("Peak GPU memory (MB)")
    ax.set_ylabel("Question accuracy (%)")
    ax.set_title("Streaming policy accuracy vs GPU memory")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    _save_figure(fig, outputs[1])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = [float(row["average_input_frames"]) for row in rows]
    ax.scatter(x, accuracies, c=colors, s=75)
    for x_value, y_value, label, offset in zip(x, accuracies, labels, _annotation_offsets(len(rows))):
        ax.annotate(label, (x_value, y_value), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set_xlim(min(x) - 0.25, max(x) + 0.25)
    ax.set_xlabel("Average input frames")
    ax.set_ylabel("Question accuracy (%)")
    ax.set_title("Streaming policy accuracy vs visual context")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    _save_figure(fig, outputs[2])
    return outputs


def write_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot fixed-budget streaming policy summaries.")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = build_comparison_rows(args.summary)
    csv_path = args.output_dir / "policy-comparison.csv"
    write_comparison_csv(rows, csv_path)
    outputs = write_plots(rows, args.output_dir)
    print(f"wrote comparison CSV to {csv_path}")
    for path in outputs:
        print(f"wrote figure to {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

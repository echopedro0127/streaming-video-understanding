"""Fixed-frame MCQ versus choice-free counting pilot with atomic paired resume."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_eval import _load_samples
from streaming_video_understanding.evaluation.count_readout import PairedCountBackend, summarize_count_pairs
from streaming_video_understanding.evaluation.engine import evaluate_sample
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.evaluation.splits import load_split_sample_ids
from streaming_video_understanding.models.counting import build_count_prompt, numeric_options
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.config import configuration_id, load_config
from streaming_video_understanding.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists() and not args.resume:
        parser.error("output exists; use --resume")
    annotation_path = (args.annotations if args.annotations.is_absolute() else args.data_root / args.annotations).resolve()
    config = load_config(args.config)
    if config.experiment.visual_budget != 8:
        parser.error("count-readout-v1 requires visual_budget=8")
    if config.model.backend != "smolvlm":
        parser.error("this real pilot requires the smolvlm backend")
    if config.dataset.context_horizon_s != 60.0 or config.runtime.sample_fps != 1.0 or config.model.max_new_tokens != 8:
        parser.error("count-readout-v1 requires 60s horizon, 1 FPS and 8 generated tokens")
    ids = load_split_sample_ids(args.split)
    if len(ids) != 20:
        parser.error("count-readout-v1 requires the frozen 20-QA split; use --limit for smoke")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    excluded = set(split.get("excluded_video_ids", ()))
    samples, reader = _load_samples(args.data_root, annotation_path=annotation_path,
                                    sample_fps=config.runtime.sample_fps)
    samples = sorted((sample for sample in samples if sample.sample_id in ids), key=lambda sample: sample.sample_id)
    if len(samples) != len(ids) or {sample.sample_id for sample in samples} != ids:
        parser.error("split coverage mismatch or duplicate annotation IDs")
    if len({sample.video_id for sample in samples}) != len(samples):
        parser.error("pilot requires exactly one QA per video")
    if any(sample.video_id in excluded for sample in samples):
        parser.error("pilot overlaps excluded diagnostic videos")
    for sample in samples:
        numeric_options(sample.options)
    protocol = {
        "version": "choice-free-count-v1",
        "source_config_id": configuration_id(config, policy_id="recent-8", backend="smolvlm"),
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "order": "count-first at odd index in sorted frozen IDs",
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"choice-free-count-v1-{digest}"
    store = ResumeStore(args.output)
    if store.completed_keys - {(sample.sample_id, run_id) for sample in samples}:
        parser.error("output contains unexpected sample/config IDs")
    selected = samples[:args.limit] if args.limit is not None else samples
    pending = [(index, sample) for index, sample in enumerate(selected)
               if not store.is_complete(sample.sample_id, run_id)]
    # Skip model loading completely if the requested subset is already complete.
    backend = None
    if pending:
        set_seed(config.experiment.seed)
        backend = SmolVLMBackend(model_id=config.model.model_id, max_new_tokens=config.model.max_new_tokens)
    failures = 0
    for index, sample in pending:
        paired = PairedCountBackend(backend, count_first=bool(index % 2))
        started = time.perf_counter()
        try:
            baseline = evaluate_sample(sample, config=config, frame_reader=reader, backend=paired,
                                       output_store=None, policy_id="recent-8", config_id=run_id)
            counts = numeric_options(sample.options)
            mcq, count = paired.responses["mcq"], paired.responses["count"]
            mcq_count = counts[ord(mcq.answer) - 65] if mcq.parse_valid else None
            baseline.pop("latency_ms")  # old field excludes the second generation
            baseline.pop("model_latency_ms")
            record = dict(baseline, protocol=protocol, record_type="paired-count-readout-v1",
                          option_counts=list(counts), ground_truth_count=counts[ord(sample.answer) - 65],
                          count_prompt=build_count_prompt(sample.question),
                          generation_order=["count", "mcq"] if paired.count_first else ["mcq", "count"],
                          shared_image_objects=True, inference_calls=2,
                          paired_pipeline_latency_ms=(time.perf_counter() - started) * 1000.0,
                          methods={
                              "mcq": dict(count=mcq_count, answer=mcq.answer, raw_text=mcq.raw_text,
                                          parse_valid=mcq.parse_valid, model_latency_ms=mcq.latency_ms,
                                          processor_ms=mcq.processor_latency_ms, generation_ms=mcq.generation_latency_ms),
                              "count": dict(count=count.predicted_count, answer=count.answer, raw_text=count.raw_text,
                                            parse_valid=count.parse_valid, model_latency_ms=count.latency_ms,
                                            processor_ms=count.processor_latency_ms, generation_ms=count.generation_latency_ms),
                          })
            # Peak is for the shared two-generation query, not a per-mode estimate.
            record["shared_peak_gpu_memory_mb"] = record.pop("peak_gpu_memory_mb")
            record.pop("correct")  # pair summary computes both numeric accuracies
            store.append(record)
            print(f"completed={len(store.completed_keys)}/{len(samples)} sample={sample.sample_id}", flush=True)
        except Exception as exc:
            failures += 1
            error_path = args.output.with_name(args.output.stem + "-errors.jsonl")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(sample_id=sample.sample_id, config_id=run_id,
                                             error_type=type(exc).__name__, error=str(exc))) + "\n")
            print(f"failed {sample.sample_id}: {exc}", file=sys.stderr, flush=True)
    print(f"processed={len(pending)-failures} skipped={len(selected)-len(pending)} failures={failures}")
    if len(store.completed_keys) == len(samples):
        rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != len(samples) or len({(row["sample_id"], row["config_id"]) for row in rows}) != len(rows):
            raise ValueError("duplicate result rows; refusing final summary")
        summary = summarize_count_pairs(rows)
        summary["protocol"] = protocol
        summary["config_id"] = run_id
        summary["mean_shared_decode_ms"] = sum(row["decode_latency_ms"] for row in rows) / len(rows)
        summary["peak_shared_gpu_memory_mb"] = max(row["shared_peak_gpu_memory_mb"] for row in rows)
        summary["inference_calls"] = sum(row["inference_calls"] for row in rows)
        args.output.with_name(args.output.stem + "-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print("Full pilot complete; paired summary saved", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

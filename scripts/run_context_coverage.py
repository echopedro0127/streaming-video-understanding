"""Matched B=8 temporal-context probe, with bounded causal image retention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_eval import _load_samples
from streaming_video_understanding.evaluation.context_probe import choose_review_cases, evaluate_context_sample, summarize_context_probe
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.evaluation.splits import load_split_sample_ids
from streaming_video_understanding.models.counting import numeric_options
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.config import configuration_id, load_config
from streaming_video_understanding.utils.seed import set_seed
from streaming_video_understanding.streaming.change_memory import ChangeMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--change-probe", action="store_true", help="opt-in Recent8/prefix8/change4+4 diagnostic")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists() and not args.resume:
        parser.error("output exists; use --resume")
    annotation = (args.annotations if args.annotations.is_absolute() else args.data_root / args.annotations).resolve()
    config = load_config(args.config)
    if (config.model.backend != "smolvlm" or config.experiment.visual_budget != 8
        or config.runtime.sample_fps != 1.0 or config.model.max_new_tokens != 8 or config.dataset.context_horizon_s != 60):
        parser.error("v1 requires SmolVLM, B=8, 1 FPS, 8 generation tokens and base horizon60")
    ids = load_split_sample_ids(args.split)
    if len(ids) != 20:
        parser.error("use frozen20-QA pilot split, --limit for smoke")
    samples, reader = _load_samples(args.data_root, annotation_path=annotation, sample_fps=1.0)
    samples = sorted((sample for sample in samples if sample.sample_id in ids), key=lambda sample: sample.sample_id)
    if len(samples) != 20 or {sample.sample_id for sample in samples} != ids or len({sample.video_id for sample in samples}) != 20:
        parser.error("pilot coverage/duplicate/video mismatch")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if any(sample.video_id in split.get("excluded_video_ids", []) for sample in samples):
        parser.error("pilot overlaps excluded diagnostic videos")
    for sample in samples:
        numeric_options(sample.options)
    reviews = choose_review_cases(samples)
    protocol = dict(version="context-coverage-v1", budget=8, window_s=60,
                    prefix_scope="all sampled frames at or before query, known query time",
                    bin_rule="within-bin closest to centre; earlier timestamp/id ties",
                    source_config_id=configuration_id(config, policy_id="recent-8", backend="smolvlm"),
                    annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                    split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
                    review_sample_ids=reviews, order="rotate recent/window/prefix by sorted sample index")
    if args.change_probe:
        protocol.update(version='change-memory-v1', change_threshold=ChangeMemory.threshold,
                        change_cooldown_s=ChangeMemory.cooldown_s, thumbnail='RGB32x32 bilinear; adjacent MAD/255',
                        history_rule='top4 evicted score; noncandidate0; earlier timestamp/id ties',
                        order='rotate recent/prefix/change by sorted sample index',
                        review_material='prior context-coverage preselected3 cases; human verification pending')
    run_id = protocol['version'] + '-' + hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    store = ResumeStore(args.output)
    if store.completed_keys - {(sample.sample_id, run_id) for sample in samples}:
        parser.error("output contains unexpected sample/config keys")
    selected = samples[:args.limit] if args.limit is not None else samples
    pending = [(index, sample) for index, sample in enumerate(selected) if not store.is_complete(sample.sample_id, run_id)]
    backend = None
    if pending:
        set_seed(config.experiment.seed)
        backend = SmolVLMBackend(model_id=config.model.model_id, max_new_tokens=8)
    failed = 0
    for index, sample in pending:
        material = None
        if sample.sample_id in reviews.values() and not args.change_probe:
            # Hashed folder names cannot escape the output tree via annotation IDs.
            material = args.output.parent / (args.output.stem + "-materials") / hashlib.sha256(sample.sample_id.encode()).hexdigest()[:12]
        try:
            record = evaluate_context_sample(sample, reader=reader, backend=backend,
                                             order_offset=index, material_dir=material, change_probe=args.change_probe)
            record.update(config_id=run_id, protocol=protocol, record_type=protocol['version'])
            if material:
                record["material_folder"] = material.name
            store.append(record)
            print(f"completed={len(store.completed_keys)}/20 sample={sample.sample_id}", flush=True)
        except Exception as exc:
            failed += 1
            error_path = args.output.with_name(args.output.stem + "-errors.jsonl")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(sample_id=sample.sample_id, config_id=run_id,
                                             error_type=type(exc).__name__, error=str(exc))) + "\n")
            print(f"failed {sample.sample_id}: {exc}", file=sys.stderr, flush=True)
    print(f"processed={len(pending)-failed} skipped={len(selected)-len(pending)} failed={failed}", flush=True)
    if len(store.completed_keys) == 20:
        rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 20 or len({(row["sample_id"], row["config_id"]) for row in rows}) != 20:
            raise ValueError("duplicate rows; refusing completed summary")
        summary = summarize_context_probe(rows)
        summary.update(config_id=run_id, protocol=protocol,
                       inference_calls=sum(row["inference_calls"] for row in rows),
                       peak_retained_images=max(row["peak_retained_images"] for row in rows),
                       mean_ingest_and_optional_render_ms=sum(row["ingest_and_optional_render_ms"] for row in rows)/20,
                       peak_shared_gpu_memory_mb=max(row["shared_peak_gpu_memory_mb"] for row in rows))
        if args.change_probe:
            summary.update(change_memory_peak_images=max(row['change_memory_peak_images'] for row in rows),
                           combined_retained_image_bound=32,
                           mean_change_candidates=sum(row['change_candidates'] for row in rows)/20)
        args.output.with_name(args.output.stem + "-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print("Full context probe complete", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

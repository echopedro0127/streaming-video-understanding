"""Causal temporal-coverage diagnostic and blinded sampled contact sheets."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import math
from pathlib import Path
import time

from PIL import Image, ImageDraw

from ..analysis.allocation import allocation_diagnosis
from ..models.base import ModelRequest
from ..models.counting import numeric_options
from ..streaming.context_bank import CONTEXT_MODES, ContextFrameBank
from ..streaming.change_memory import ChangeMemory
from .engine import _peak_gpu_memory_mb, _reset_gpu_peak_memory
from .metrics import _percentile


def choose_review_cases(samples) -> dict[str, str]:
    groups = {}
    for sample in samples:
        stratum = "short" if sample.query_time_s <= 60 else "medium" if sample.query_time_s <= 180 else "long"
        groups.setdefault(stratum, []).append(sample.sample_id)
    return {group: min(ids, key=lambda sample_id: hashlib.sha256(
                f"coverage-review-v1:{group}:{sample_id}".encode()).hexdigest())
            for group, ids in sorted(groups.items())}


class CausalTimeline:
    """One bounded thumbnail canvas at a time; never receives labels/options."""

    def __init__(self, output_dir: Path, *, sample_id: str, query_time_s: float, page_size: int = 64):
        if isinstance(query_time_s, bool) or not math.isfinite(query_time_s) or query_time_s < 0:
            raise ValueError("timeline query time must be finite and non-negative")
        if type(page_size) is not int or page_size <= 0 or page_size > 64:
            raise ValueError("timeline page_size must be in1..64")
        self.output_dir, self.sample_id = Path(output_dir), sample_id
        self.query_time_s, self.page_size = query_time_s, page_size
        self.canvas = None
        self.pages = []
        self.count = 0

    def add(self, image, frame):
        if not math.isfinite(frame.timestamp_s) or frame.timestamp_s < 0:
            raise ValueError("invalid timeline timestamp")
        if frame.timestamp_s > self.query_time_s + 1e-6:
            raise ValueError("future timeline frame")
        columns = min(8, self.page_size)
        if self.canvas is None:
            self.canvas = Image.new("RGB", (columns * 160, math.ceil(self.page_size / columns) * 110 + 40), "white")
            draw = ImageDraw.Draw(self.canvas)
            draw.text((8, 8), f"{self.sample_id} | sampled causal prefix <= {self.query_time_s:.2f}s | page {len(self.pages)+1}", fill="black")
        index = self.count % self.page_size
        x, y = (index % columns) * 160, (index // columns) * 110 + 40
        thumb = image.copy()
        try:
            thumb.thumbnail((156, 86))
            self.canvas.paste(thumb, (x + 2, y + 2))
        finally:
            thumb.close()
        ImageDraw.Draw(self.canvas).text((x + 2, y + 90), f"id={frame.frame_id} t={frame.timestamp_s:.2f}s", fill="black")
        self.count += 1
        if self.count % self.page_size == 0:
            self._flush()

    def _flush(self):
        if self.canvas is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"prefix-{len(self.pages)+1:03}.jpg"
            self.canvas.save(path, quality=88)
            self.pages.append(path)
            self.canvas.close()
            self.canvas = None

    def finish(self):
        self._flush()
        return tuple(self.pages)

    def abort(self):
        if self.canvas is not None:
            self.canvas.close()
            self.canvas = None


def _render_selection(bank, path: Path, sample_id: str):
    canvas = Image.new("RGB", (1440, 400), "white")
    try:
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 8), f"{sample_id} | selected causal frames; no answer labels", fill="black")
        for row_index, mode in enumerate(CONTEXT_MODES):
            y = 40 + row_index * 120
            draw.text((4, y + 30), mode, fill="black")
            for index, (image, frame) in enumerate(zip(bank.images(mode), bank.frames(mode))):
                x = 160 + index * 160
                thumb = image.copy()
                try:
                    thumb.thumbnail((156, 86))
                    canvas.paste(thumb, (x + 2, y))
                finally:
                    thumb.close()
                draw.text((x + 2, y + 90), f"id={frame.frame_id} t={frame.timestamp_s:.2f}s", fill="black")
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=88)
    finally:
        canvas.close()


def evaluate_context_sample(sample, *, reader, backend, order_offset=0, material_dir: Path | None = None, change_probe=False):
    counts = numeric_options(sample.options)
    started = time.perf_counter()
    bank = ContextFrameBank(query_time_s=sample.query_time_s)
    change = ChangeMemory(query_time_s=sample.query_time_s) if change_probe else None
    timeline = CausalTimeline(material_dir, sample_id=sample.sample_id, query_time_s=sample.query_time_s) if material_dir else None
    modes = ('recent-8', 'time-bin-prefix-8', 'change-4+4') if change_probe else CONTEXT_MODES
    order = modes[order_offset % 3:] + modes[:order_offset % 3]
    def observe(image, frame):
        if change is not None:
            change.update(image.copy(), frame)
        if timeline is not None:
            timeline.add(image, frame)
    def close_banks():
        errors = []
        for owned_bank in (bank, change):
            if owned_bank is not None:
                try:
                    owned_bank.close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
    try:
        ingest_started = time.perf_counter()
        for image, frame in reader.read_until(sample.video_path, end_time_s=sample.query_time_s):
            bank.update(image, frame, on_frame=observe)
        pages = timeline.finish() if timeline else ()
        if material_dir:
            _render_selection(bank, Path(material_dir) / "selection.jpg", sample.sample_id)
        ingest_ms = (time.perf_counter() - ingest_started) * 1000
        _reset_gpu_peak_memory()
        conditions = {}
        for mode in order:
            frames = change.frames() if mode == 'change-4+4' else bank.frames(mode)
            if len(frames) != 8:
                raise ValueError(f"{sample.sample_id}: {mode} has {len(frames)} frames, expected8")
            timestamps = tuple(frame.timestamp_s for frame in frames)
            response = backend.infer(ModelRequest(
                images=change.images() if mode == 'change-4+4' else bank.images(mode), timestamps_s=timestamps, question=sample.question,
                options=sample.options, query_time_s=sample.query_time_s,
                include_timestamps=False, allow_invalid_answer=True,
            ))
            count = counts[ord(response.answer) - 65] if response.parse_valid else None
            conditions[mode] = dict(
                answer=response.answer, raw_text=response.raw_text, count=count,
                parse_valid=response.parse_valid, model_latency_ms=response.latency_ms,
                processor_ms=response.processor_latency_ms, generation_ms=response.generation_latency_ms,
                selected_frames=[asdict(frame) for frame in frames], timestamps_s=list(timestamps),
                temporal_span_s=timestamps[-1]-timestamps[0], oldest_frame_age_s=sample.query_time_s-timestamps[0],
                latest_frame_age_s=sample.query_time_s-timestamps[-1], input_frames=len(frames),
            )
        record = dict(sample_id=sample.sample_id, video_id=sample.video_id,
                      question=sample.question, options=list(sample.options), query_time_s=sample.query_time_s,
                      ground_truth=sample.answer, ground_truth_count=counts[ord(sample.answer)-65],
                      mode_order=list(order), conditions=conditions, inference_calls=3,
                      sampled_prefix_frames=bank.processed_frames, peak_retained_images=bank.peak_retained_images,
                      ingest_and_optional_render_ms=ingest_ms, shared_peak_gpu_memory_mb=_peak_gpu_memory_mb(),
                      rendered_prefix_pages=len(pages), rendered_material=bool(material_dir))
        if change is not None:
            record.update(change_memory_peak_images=change.peak_retained_images,
                          combined_retained_image_bound=32, change_candidates=change.change_candidates,
                          change_history_scores=change.history_scores())
    except BaseException as primary:
        try:
            close_banks()
        except BaseException as cleanup:
            primary.add_note(f"context bank cleanup failed: {cleanup!r}")
        if timeline:
            timeline.abort()
        raise
    close_banks()
    gc.collect()
    record["pipeline_latency_ms"] = (time.perf_counter() - started) * 1000
    return record


def summarize_context_probe(rows, *, n_bootstrap=2000):
    if not rows:
        raise ValueError("context probe has no complete samples")
    summaries, paired = {}, {}
    modes = tuple(sorted(rows[0]['conditions']))
    if any(set(row['conditions']) != set(modes) for row in rows):
        raise ValueError('mixed context conditions')
    for mode in modes:
        results = [row["conditions"][mode] for row in rows]
        summaries[mode] = dict(
            correct=sum(result["count"] == row["ground_truth_count"] for row, result in zip(rows, results)),
            accuracy=sum(result["count"] == row["ground_truth_count"] for row, result in zip(rows, results))/len(rows),
            parse_valid_count=sum(result["parse_valid"] for result in results),
            mean_temporal_span_s=sum(result["temporal_span_s"] for result in results)/len(rows),
            mean_oldest_frame_age_s=sum(result["oldest_frame_age_s"] for result in results)/len(rows),
            mean_model_latency_ms=sum(result["model_latency_ms"] for result in results)/len(rows),
            p95_model_latency_ms=_percentile([result["model_latency_ms"] for result in results], 95),
        )
        if mode != "recent-8":
            pairs = []
            for row in rows:
                left = row["conditions"]["recent-8"]["count"] == row["ground_truth_count"]
                right = row["conditions"][mode]["count"] == row["ground_truth_count"]
                pairs.append(dict(sample_id=row["sample_id"], video_id=row["video_id"], question=row["question"],
                                  recent_correct=left, option_correct=right,
                                  group="both_correct" if left and right else "memory_hurt" if left
                                  else "memory_helped" if right else "both_wrong"))
            result = allocation_diagnosis(pairs, n_bootstrap=n_bootstrap)
            paired[mode+"_vs_recent-8"] = {key: result[key] for key in ("accuracy_delta", "helped", "hurt", "bootstrap")}
    return dict(num_samples=len(rows), num_videos=len({row["video_id"] for row in rows}),
                conditions=summaries, paired=paired,
                scope="Query-aware causal coverage diagnostic, not evidence recall or persistent streaming memory")

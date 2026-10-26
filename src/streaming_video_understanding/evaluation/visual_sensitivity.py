"""Fixed-selection counterfactuals, not normal streaming-policy evaluations."""
from __future__ import annotations

from dataclasses import asdict
import gc
import math
import time

from PIL import Image

from ..analysis.allocation import allocation_diagnosis
from ..models.base import ModelRequest
from ..models.counting import numeric_options
from ..models.prompt import build_prompt
from .engine import _peak_gpu_memory_mb, _reset_gpu_peak_memory
from .metrics import _percentile


MODES = ('original', 'blank', 'reverse', 'option-cycle')
OPTION_CYCLE = (1, 2, 3, 0)  # new option position -> original position


def validate_source(sample, source):
    expected = dict(sample_id=sample.sample_id, video_id=sample.video_id,
                    question=sample.question, options=list(sample.options),
                    ground_truth=sample.answer, query_time_s=sample.query_time_s)
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError('source sample/question/options/truth/time mismatch')
    numeric_options(sample.options)
    if sample.answer not in ('A', 'B', 'C', 'D'):
        raise ValueError('source ground truth must be A-D')
    if not math.isfinite(sample.query_time_s) or sample.query_time_s < 0:
        raise ValueError('source query time must be finite and non-negative')
    manifest = source.get('conditions', {}).get('change-4+4', {}).get('selected_frames')
    if not isinstance(manifest, list) or len(manifest) != 8:
        raise ValueError('source requires eight Change-4+4 frames')
    previous_id, previous_t = -1, -math.inf
    for frame in manifest:
        i, t = frame.get('frame_id'), frame.get('timestamp_s')
        if type(i) is not int or i <= previous_id:
            raise ValueError('source frame IDs must be increasing nonnegative integers')
        if isinstance(t, bool) or not isinstance(t, (float,int)) or not math.isfinite(t) or t < 0 or t < previous_t or t > sample.query_time_s + 1e-6:
            raise ValueError('source timestamps must be causal and chronological')
        previous_id, previous_t = i, t
    return manifest


def evaluate_visual_sensitivity(sample, source, *, reader, backend, order_offset=0):
    manifest = validate_source(sample, source)
    expected = {frame['frame_id']: frame['timestamp_s'] for frame in manifest}
    retained, blanks = [], []
    incoming = None
    started = time.perf_counter()
    def close_images():
        errors = []
        for image in [*retained, *blanks, *([incoming] if incoming is not None else [])]:
            try:
                image.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
    try:
        found = []
        previous_id, previous_t = -1, -math.inf
        for image, frame in reader.read_until(sample.video_path, end_time_s=sample.query_time_s):
            incoming = image
            i, t = frame.frame_id, frame.timestamp_s
            if type(i) is not int or i <= previous_id:
                raise ValueError('decoded frame IDs must increase')
            if isinstance(t, bool) or not math.isfinite(t) or t < 0 or t < previous_t:
                raise ValueError('invalid decoded timestamp')
            if t > sample.query_time_s + 1e-6:
                raise ValueError('future decoded frame rejected')
            previous_id, previous_t = i, t
            if i in expected:
                if abs(t - expected[i]) > 1e-6:
                    raise ValueError('selected timestamp differs from source')
                retained.append(image)
                found.append(asdict(frame))
            else:
                image.close()
            incoming = None
        if found != manifest:
            raise ValueError('missing selected source frames')
        decode_ms = (time.perf_counter() - started) * 1000
        for image in retained:
            blanks.append(Image.new('RGB', image.size, (127,127,127)))
        _reset_gpu_peak_memory()
        order = MODES[order_offset % 4:] + MODES[:order_offset % 4]
        conditions = {}
        for mode in order:
            permutation = OPTION_CYCLE if mode == 'option-cycle' else (0,1,2,3)
            options = tuple(sample.options[index] for index in permutation)
            images = tuple(blanks) if mode == 'blank' else tuple(reversed(retained)) if mode == 'reverse' else tuple(retained)
            actual_frames = [] if mode == 'blank' else list(reversed(manifest)) if mode == 'reverse' else list(manifest)
            # No timestamp text in any condition. Actual provenance, including
            # reversed order and synthetic blanks, is recorded separately below.
            response = backend.infer(ModelRequest(images=images, timestamps_s=(),
                question=sample.question, options=options, query_time_s=sample.query_time_s,
                include_timestamps=False, allow_invalid_answer=True))
            if response.parse_valid and response.answer not in ('A','B','C','D'):
                raise ValueError('backend marks non-A-D answer as parse-valid')
            mapped = chr(65 + permutation[ord(response.answer)-65]) if response.parse_valid else None
            count = numeric_options(sample.options)[ord(mapped)-65] if mapped is not None else None
            conditions[mode] = dict(answer=response.answer, raw_text=response.raw_text,
                parse_valid=response.parse_valid, mapped_answer=mapped, count=count,
                options=list(options), option_permutation=list(permutation),
                input_source_frames=actual_frames, synthetic_images=8 if mode=='blank' else 0,
                input_frames=8, prompt=build_prompt(question=sample.question, options=options,
                    timestamps_s=(), query_time_s=sample.query_time_s, include_timestamps=False),
                model_latency_ms=response.latency_ms, processor_ms=response.processor_latency_ms,
                generation_ms=response.generation_latency_ms)
        row = dict(sample_id=sample.sample_id, video_id=sample.video_id, question=sample.question,
                   options=list(sample.options), ground_truth=sample.answer,
                   ground_truth_count=numeric_options(sample.options)[ord(sample.answer)-65],
                   query_time_s=sample.query_time_s, source_selected_frames=manifest,
                   mode_order=list(order), conditions=conditions, inference_calls=4,
                   decode_latency_ms=decode_ms, retained_source_images=len(retained),
                   generated_blank_images=len(blanks), shared_peak_gpu_memory_mb=_peak_gpu_memory_mb())
    except BaseException as primary:
        try:
            close_images()
        except BaseException as cleanup:
            primary.add_note(f'visual-sensitivity cleanup failed: {cleanup!r}')
        raise
    close_images()
    gc.collect()
    row['pipeline_latency_ms'] = (time.perf_counter()-started)*1000
    return row


def summarize_visual_sensitivity(rows, *, n_bootstrap=2000):
    if not rows or any(set(row['conditions']) != set(MODES) for row in rows):
        raise ValueError('require complete four-condition rows')
    conditions, comparisons = {}, {}
    for mode in MODES:
        results = [row['conditions'][mode] for row in rows]
        latencies = [x['model_latency_ms'] for x in results]
        conditions[mode] = dict(correct=sum(x['mapped_answer']==row['ground_truth'] for row,x in zip(rows,results)),
            parse_valid_count=sum(x['parse_valid'] for x in results),
            mean_model_latency_ms=sum(latencies)/len(rows), p95_model_latency_ms=_percentile(latencies,95))
        conditions[mode]['accuracy'] = conditions[mode]['correct']/len(rows)
        if mode == 'original':
            continue
        valid, consistent, sticky, pairs = 0,0,0,[]
        for row in rows:
            base, candidate = row['conditions']['original'], row['conditions'][mode]
            if base['parse_valid'] and candidate['parse_valid']:
                valid += 1
                consistent += base['mapped_answer'] == candidate['mapped_answer']
                sticky += base['answer'] == candidate['answer']
            left, right = base['mapped_answer']==row['ground_truth'], candidate['mapped_answer']==row['ground_truth']
            pairs.append(dict(sample_id=row['sample_id'],video_id=row['video_id'],question=row['question'],
                recent_correct=left,option_correct=right,group='both_correct' if left and right else 'memory_hurt' if left else 'memory_helped' if right else 'both_wrong'))
        diagnosis = allocation_diagnosis(pairs,n_bootstrap=n_bootstrap)
        comparisons[mode] = dict(valid_pairs=valid, content_consistent=consistent,
            content_consistency=consistent/valid if valid else None,
            label_sticky=sticky if mode=='option-cycle' else None,
            paired={k:diagnosis[k] for k in ('accuracy_delta','helped','hurt','bootstrap')})
    return dict(num_samples=len(rows),num_videos=len({row['video_id'] for row in rows}),
        inference_calls=sum(row['inference_calls'] for row in rows),conditions=conditions,comparisons=comparisons,
        scope='Counterfactual diagnostics on inspected20QA; not streaming benchmark gains or proof of visual grounding')

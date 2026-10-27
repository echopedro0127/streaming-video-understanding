"""Source-locked posthoc exact-black artifact ablation (no new model calls)."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from PIL import Image
from streaming_video_understanding.data.perception_review import sha, verify_bundle, verify_frame
from streaming_video_understanding.evaluation.black_frame_control import is_exact_black, replay_gated
from streaming_video_understanding.evaluation.event_state import (
    config_digest, selection_for, replay_prefix, summarize_event_run, score_event_run,
)
from streaming_video_understanding.evaluation.perception_probe import QUESTION, DESCRIPTIONS

# Already inspected sources; these locks prevent accidental substitution, not
# hindsight bias. The control is explicitly posthoc on this cohort.
SOURCES = {
    '79': ('85e51ae6bba974b3da22bd4a12ec71b84987450975b84b7f100b13e47e36e8b5',
           '474b086c1d64727e851037fd50a27fa82ef74fc63c317c85497c5b37d479fbfe',
           'ea931d2c2726b118b224ab3f480d5bb38dd48d00069053d84ff0ecfaffea5b07'),
    '75': ('1e8fd26ed74f746b571ff27d63926e644885f32b1f69983dfd86f5f9afe4869b',
           'cc32eee904005b3f704fa0a1ac6d23dad3c58444f39ac0e0a663c1109805ae23',
           'a0f7c5e3d6d94b5f6213556d03585570d72bee4a9476b0ed4b4d5cf4b8950238'),
    '76': ('40d0242cf1bc82191a14bf0cd55aa52aa4560b6a721a0ccb999ca04ed5ad3c6f',
           '541b6f88b5e07fba07a505ede1e7992033bf7976ea88df6b591c7f8f26537021',
           'e691bec1d099a887afdf63e4e8351c519fefe7fbb2de0ce34df93fb43e3ae08d'),
}


def evaluate_case(case):
    base = ROOT / 'results/event-state/validation-3' / f'sample{case}'
    config = json.loads((ROOT / f'configs/event-state-validation-v1/sample{case}.json').read_text(encoding='utf-8'))
    paths = [base / name for name in ('manifest.json', 'predictions.jsonl', 'event-labels-submitted-v1.json')]
    before = [sha(path) for path in paths]
    if tuple(before) != SOURCES[case]:
        raise ValueError(f'sample{case}: source hashes differ from locked artifacts')
    manifest = verify_bundle(paths[0])
    if manifest.get('selection_config') != selection_for(config):
        raise ValueError('source selection differs from frozen config')
    rows = [json.loads(line) for line in paths[1].read_text(encoding='utf-8').splitlines() if line.strip()]
    protocol = dict(version='event-state-v1', config_sha256=config_digest(config), manifest_sha256=before[0],
                    model_id=config['model_id'], max_new_tokens=8, seed=config['seed'], input_frames=1,
                    question=QUESTION, descriptions=list(DESCRIPTIONS), option_order='original',
                    timestamp_text=False, unknown_policy=config['unknown_policy'])
    replay_prefix(rows, manifest, config, run_id='event-state-v1-' + config_digest(protocol)[:12],
                  protocol=protocol, manifest_sha256=before[0])
    if len(rows) != len(manifest['frames']):
        raise ValueError('complete source prefix required')
    flags = []
    for frame in manifest['frames']:
        with Image.open(verify_frame(base, frame)) as image:
            flags.append(is_exact_black(image))
    # Work exclusively on copies. Original raw/debounced traces remain intact.
    gated = deepcopy(rows)
    for row, trace in zip(gated, replay_gated(rows, flags)):
        row['policy_traces']['black_gate_raw'] = trace
    combined_config = deepcopy(config)
    combined_config['policies']['black_gate_raw'] = dict(enter_frames=1, exit_frames=1)
    labels = json.loads(paths[2].read_text(encoding='utf-8'))
    summary = summarize_event_run(gated, combined_config)
    scores = score_event_run(summary, labels, manifest_sha256=before[0], config=combined_config)
    dense = {}
    for policy in combined_config['policies']:
        mismatches = []
        for row in gated:
            t = row['timestamp_s']
            target = sum(event['start_s'] <= t for event in labels['intervals'])
            prediction = row['policy_traces'][policy]['count']
            if prediction != target:
                mismatches.append(dict(timestamp_s=t, count=prediction, target=target, absolute_error=abs(prediction-target)))
        dense[policy] = dict(num_queries=len(rows), correct=len(rows)-len(mismatches),
                             absolute_error_sum=sum(m['absolute_error'] for m in mismatches), mismatches=mismatches)
    if before != [sha(path) for path in paths]:
        raise ValueError('source artifacts changed during replay')
    return dict(video_id=f'sample_{case}', num_frames=len(rows), manifest_sha256=before[0],
                predictions_sha256=before[1], labels_sha256=before[2], scores=scores, dense=dense,
                gated_frames=[dict(frame_id=r['frame_id'], timestamp_s=r['timestamp_s'], image_sha256=r['image_sha256'],
                                   original_category=r['category']) for r, flag in zip(rows, flags) if flag])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new artifact path')
    cases = [evaluate_case(case) for case in SOURCES]
    aggregate = {}
    for policy in ('raw', 'debounced', 'black_gate_raw'):
        metrics = [case['scores']['policies'][policy] for case in cases]
        tp = sum(m['event_true_positives'] for m in metrics)
        predicted = sum(m['predicted_events'] for m in metrics)
        truth = sum(case['scores']['true_events'] for case in cases)
        nq = sum(len(case['scores']['query_times_s']) for case in cases)
        dense_n = sum(case['num_frames'] for case in cases)
        aggregate[policy] = dict(event_tp=tp, predicted_events=predicted, human_events=truth,
            micro_precision=tp/predicted if predicted else None, micro_recall=tp/truth if truth else None,
            micro_f1=2*tp/(predicted+truth) if predicted+truth else None,
            primary_queries=nq,
            primary_count_mae=sum(m['count_mae']*len(c['scores']['query_times_s']) for c,m in zip(cases,metrics))/nq,
            primary_correct=sum(round(m['query_accuracy']*len(c['scores']['query_times_s'])) for c,m in zip(cases,metrics)),
            mean_signed_media_confirmation_delay_s=sum(m['mean_confirmation_delay_s']*m['event_true_positives'] for m in metrics)/tp if tp else None,
            dense_queries=dense_n, dense_correct=sum(c['dense'][policy]['correct'] for c in cases),
            dense_count_mae=sum(c['dense'][policy]['absolute_error_sum'] for c in cases)/dense_n)
    report = dict(version='exact-black-posthoc-v1', scope='Posthoc artifact replay on previously inspected three-video cohort; no held-out claim',
                  gate_rule='Every decoded RGB channel value equals zero; any timestamp; supply unknown to raw1/1 counter',
                  unknown_policy='hold active state; reset pending confirmation streaks', new_model_calls=0,
                  measured_runtime_savings=None, measured_gpu_savings=None,
                  delay_definition='Signed media-time delay on IoU>=0.5 matched events; not live wall-clock latency',
                  source_hashes_unchanged=True, aggregate=aggregate, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(aggregate, indent=2))
    print(f'Posthoc report: {args.output}')


if __name__ == '__main__':
    main()

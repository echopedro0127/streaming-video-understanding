"""Prepare causal prefix review, run resumable state traces, score human events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))

from scripts.run_eval import _load_samples
from streaming_video_understanding.data.perception_review import prepare_bundle,verify_bundle,sha
from streaming_video_understanding.data.event_review import prepare_review_video,write_event_review,verify_clip
from streaming_video_understanding.evaluation.event_state import validate_config,config_digest,selection_for,classify_frame,apply_states,replay_prefix,summarize_event_run,score_event_run
from streaming_video_understanding.evaluation.perception_probe import QUESTION,DESCRIPTIONS
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.seed import set_seed


def rows_at(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare')
    for name in ('config','data-root','annotations','bundle'):
        prep.add_argument('--'+name,type=Path,required=True)
    run=subs.add_parser('run');score=subs.add_parser('score')
    for command in (run,score):
        for name in ('config','manifest','output'):
            command.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--resume',action='store_true');run.add_argument('--limit',type=int)
    score.add_argument('--predictions',type=Path,required=True);score.add_argument('--labels',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'));validate_config(config)
    selection=selection_for(config)
    if args.command=='prepare':
        annotation=(args.annotations if args.annotations.is_absolute() else args.data_root/args.annotations).resolve()
        samples,reader=_load_samples(args.data_root,annotation_path=annotation,sample_fps=1.0)
        chosen=[s for s in samples if s.sample_id==config['sample_id']]
        if len(chosen)!=1:
            raise ValueError('expected one source sample')
        manifest=prepare_bundle(selection,chosen,reader=reader,output_dir=args.bundle,
            annotation_sha256=sha(annotation),embed_selection=True,review_page=False)
        prepare_review_video(chosen[0].video_path,manifest,end_time_s=config['end_time_s'])
        print(f'Blinded review page: {write_event_review(manifest,end_time_s=config["end_time_s"])}')
        return 0
    manifest=verify_bundle(args.manifest)
    if manifest.get('selection_config')!=selection or manifest.get('selection_config_sha256')!=config_digest(selection):
        parser.error('bundle differs from fixed event selection/config')
    if 'review_video' in manifest:
        verify_clip(args.manifest,manifest)
        if manifest['review_video']['cutoff_s']!=config['end_time_s']:
            parser.error('review cutoff differs from event protocol')
    if args.output.resolve() in (args.manifest.resolve(),args.config.resolve()):
        parser.error('output must not replace manifest/config')
    digest=sha(args.manifest)
    protocol=dict(version='event-state-v1',config_sha256=config_digest(config),manifest_sha256=digest,
                  model_id=config['model_id'],max_new_tokens=8,seed=config['seed'],input_frames=1,
                  question=QUESTION,descriptions=list(DESCRIPTIONS),option_order='original',
                  timestamp_text=False,unknown_policy=config['unknown_policy'])
    run_id='event-state-v1-'+config_digest(protocol)[:12]
    if args.command=='score':
        if args.output.resolve() in (args.labels.resolve(),args.predictions.resolve()):
            parser.error('score output must not replace labels/predictions')
        if 'review_video' not in manifest:
            parser.error('human scoring requires the bound prefix review clip')
        rows=rows_at(args.predictions)
        replay_prefix(rows,manifest,config,run_id=run_id,protocol=protocol,manifest_sha256=digest)
        if len(rows)!=len(manifest['frames']):
            parser.error('score requires complete continuous prefix')
        result=score_event_run(summarize_event_run(rows,config),json.loads(args.labels.read_text(encoding='utf-8')),
                               manifest_sha256=digest,config=config)
        result.update(manifest_sha256=digest,predictions_sha256=sha(args.predictions),labels_sha256=sha(args.labels))
        write_json(args.output,result)
        print(f'Human event scores: {args.output}')
        return 0
    if args.output.exists() and not args.resume:
        parser.error('output exists; use --resume')
    if args.limit is not None and args.limit<=0:
        parser.error('--limit must be positive')
    rows=rows_at(args.output) if args.output.exists() else []
    counters=replay_prefix(rows,manifest,config,run_id=run_id,protocol=protocol,manifest_sha256=digest)
    store=ResumeStore(args.output)
    if len(store.completed_keys)!=len(rows):
        parser.error('duplicate completed prefix keys')
    frames=manifest['frames'];selected=frames[:args.limit] if args.limit is not None else frames
    pending=selected[len(rows):];backend=None
    if pending:
        set_seed(config['seed'])
        backend=SmolVLMBackend(model_id=config['model_id'],max_new_tokens=8)
    processed=0
    for frame in pending:
        started=time.perf_counter()
        try:
            row=classify_frame(frame,root=args.manifest.parent,backend=backend)
            row.update(config_id=run_id,protocol=protocol,manifest_sha256=digest,
                       policy_traces=apply_states(row,counters))
            row['pipeline_latency_ms']=(time.perf_counter()-started)*1000
            store.append(row);rows.append(row);processed+=1
            print(f'completed={len(rows)}/{len(frames)} t={frame["timestamp_s"]:.3f}s',flush=True)
        except Exception as exc:
            error=args.output.with_name(args.output.stem+'-errors.jsonl')
            error.parent.mkdir(parents=True,exist_ok=True)
            with error.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(dict(frame_key=frame['frame_key'],config_id=run_id,error=str(exc),error_type=type(exc).__name__))+'\n')
            print(f'Stopped at failed causal frame {frame["frame_key"]}: {exc}',file=sys.stderr,flush=True)
            return 1
    print(f'processed={processed} skipped={min(len(rows)-processed,len(selected))} failed=0',flush=True)
    if len(rows)==len(frames):
        replay_prefix(rows,manifest,config,run_id=run_id,protocol=protocol,manifest_sha256=digest)
        summary=summarize_event_run(rows,config);summary.update(config_id=run_id,protocol=protocol)
        write_json(args.output.with_name(args.output.stem+'-summary.json'),summary)
        print('Continuous causal prefix complete; event metrics pending human intervals',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

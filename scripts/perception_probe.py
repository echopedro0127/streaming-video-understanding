"""Prepare blinded images, run paired perception, or score separate human labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from scripts.run_eval import _load_samples
from streaming_video_understanding.data.perception_review import prepare_bundle,verify_bundle,sha
from streaming_video_understanding.evaluation.perception_probe import infer_perception,summarize_perception,score_perception,validate_predictions,CATEGORIES,DESCRIPTIONS,QUESTION,CYCLE
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.config import load_config
from streaming_video_understanding.utils.seed import set_seed

FROZEN_SELECTION_SHA256=hashlib.sha256(json.dumps(json.loads((ROOT/'configs'/'perception-probe-v1.json').read_text(encoding='utf-8')),sort_keys=True).encode()).hexdigest()
MODEL_ID='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prepare=subs.add_parser('prepare')
    for name in ('selection','data-root','annotations','bundle'):
        prepare.add_argument('--'+name,type=Path,required=True)
    run=subs.add_parser('run')
    for name in ('manifest','config','output'):
        run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--resume',action='store_true')
    run.add_argument('--limit',type=int)
    score=subs.add_parser('score')
    for name in ('manifest','predictions','labels','output'):
        score.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        annotation=(args.annotations if args.annotations.is_absolute() else args.data_root/args.annotations).resolve()
        samples,reader=_load_samples(args.data_root,annotation_path=annotation,sample_fps=1.0)
        manifest=prepare_bundle(json.loads(args.selection.read_text(encoding='utf-8')),samples,
                                reader=reader,output_dir=args.bundle,annotation_sha256=sha(annotation))
        print(f'Prepared {manifest}; review page: {manifest.parent / "review.html"}')
        return 0
    manifest=verify_bundle(args.manifest)
    if manifest.get('selection_config_sha256')!=FROZEN_SELECTION_SHA256:
        parser.error('manifest must match the fixed perception selection')
    digest=sha(args.manifest)
    if args.output.resolve()==args.manifest.resolve():
        parser.error('output must not replace manifest')
    if args.command=='score':
        if args.output.resolve() in (args.predictions.resolve(),args.labels.resolve()):
            parser.error('score output must not replace predictions/labels')
        rows=read_rows(args.predictions)
        validate_predictions(manifest,rows,manifest_sha256=digest)
        if len(rows)!=len(manifest['frames']) or len({x['config_id'] for x in rows})!=1:
            parser.error('score requires complete single-run predictions')
        labels=json.loads(args.labels.read_text(encoding='utf-8'))
        result=score_perception(rows,labels,manifest_sha256=digest)
        result.update(manifest_sha256=digest,predictions_sha256=sha(args.predictions),labels_sha256=sha(args.labels))
        write_json(args.output,result)
        print(f'Human-labelled scores: {args.output}')
        return 0
    if args.output.exists() and not args.resume:
        parser.error('output exists; use --resume')
    if args.limit is not None and args.limit<=0:
        parser.error('--limit must be positive')
    config=load_config(args.config)
    if config.model.backend!='smolvlm' or config.model.max_new_tokens!=8 or config.model.model_id!=MODEL_ID:
        parser.error('perception-v1 requires SmolVLM2-500M-Video-Instruct and8generation tokens')
    protocol=dict(version='perception-probe-v1',manifest_sha256=digest,model_id=config.model.model_id,
        max_new_tokens=8,seed=config.experiment.seed,input_frames=1,question=QUESTION,
        categories=list(CATEGORIES),descriptions=list(DESCRIPTIONS),option_cycle=list(CYCLE),
        order='alternate original/cycle-first by manifest frame index',timestamp_text=False)
    run_id='perception-probe-v1-'+hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
    store=ResumeStore(args.output)
    frames=manifest['frames']
    if store.completed_keys-{(x['frame_key'],run_id) for x in frames}:
        parser.error('output contains unexpected frame/config keys')
    if args.output.exists():
        old=read_rows(args.output)
        if len(old)!=len(store.completed_keys) or any(x.get('protocol')!=protocol for x in old):
            parser.error('duplicate/mixed output rows')
        validate_predictions(manifest,old,manifest_sha256=digest)
    selected=frames[:args.limit] if args.limit is not None else frames
    pending=[(i,x) for i,x in enumerate(selected) if not store.is_complete(x['frame_key'],run_id)]
    backend=None
    if pending:
        set_seed(config.experiment.seed)
        backend=SmolVLMBackend(model_id=config.model.model_id,max_new_tokens=8)
    failed=0
    for index,frame in pending:
        try:
            row=infer_perception(frame,root=args.manifest.parent,backend=backend,cycle_first=bool(index%2))
            row.update(sample_id=frame['frame_key'],source_sample_id=frame['sample_id'],
                       config_id=run_id,protocol=protocol,manifest_sha256=digest)
            # ResumeStore uses sample_id as its opaque primary key; keep source
            # QA identity separately and validate against the manifest on reads.
            store.append(row)
            print(f'completed={len(store.completed_keys)}/{len(frames)} frame={frame["frame_key"]}',flush=True)
        except Exception as exc:
            failed+=1
            error=args.output.with_name(args.output.stem+'-errors.jsonl')
            error.parent.mkdir(parents=True,exist_ok=True)
            with error.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(dict(frame_key=frame['frame_key'],config_id=run_id,error_type=type(exc).__name__,error=str(exc)))+'\n')
            print(f'failed {frame["frame_key"]}: {exc}',file=sys.stderr,flush=True)
    print(f'processed={len(pending)-failed} skipped={len(selected)-len(pending)} failed={failed}',flush=True)
    if len(store.completed_keys)==len(frames):
        completed=read_rows(args.output)
        validate_predictions(manifest,completed,manifest_sha256=digest)
        summary=summarize_perception(completed)
        summary.update(config_id=run_id,protocol=protocol)
        write_json(args.output.with_name(args.output.stem+'-summary.json'),summary)
        print('Paired perception complete; accuracy pending human review',flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())

"""Four matched counterfactual conditions on frozen Change-4+4 selections."""
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
from streaming_video_understanding.evaluation.visual_sensitivity import evaluate_visual_sensitivity, summarize_visual_sensitivity, validate_source
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.evaluation.splits import load_split_sample_ids
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.config import load_config,configuration_id
from streaming_video_understanding.utils.seed import set_seed

FROZEN_SOURCE_ID='change-memory-v1-0fbce55d18ea'
FROZEN_SOURCE_SHA256='39be41e1322e91eb489f52881601a47e66aae298840db6a376de38db08bfa4e4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data-root','annotations','split','source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.output.resolve()==args.source.resolve():
        parser.error('output must not replace source')
    if args.limit is not None and args.limit<=0:
        parser.error('--limit must be positive')
    if args.output.exists() and not args.resume:
        parser.error('output exists; use --resume')
    annotation=(args.annotations if args.annotations.is_absolute() else args.data_root/args.annotations).resolve()
    if sha(args.source)!=FROZEN_SOURCE_SHA256:
        parser.error('source file is not the audited frozen Change-memory artifact')
    config=load_config(args.config)
    if (config.model.backend!='smolvlm' or config.experiment.visual_budget!=8 or config.runtime.sample_fps!=1.0
        or config.model.max_new_tokens!=8 or config.dataset.context_horizon_s!=60):
        parser.error('requires SmolVLM/B8/1FPS/8generationtokens/basehorizon60')
    ids=load_split_sample_ids(args.split)
    rows=read_rows(args.source)
    if len(ids)!=20 or len(rows)!=20 or {x['sample_id'] for x in rows}!=ids or len({x['video_id'] for x in rows})!=20:
        parser.error('source/split must cover same20uniqueQA/20videos')
    source_protocol=rows[0]['protocol']
    source_id='change-memory-v1-'+hashlib.sha256(json.dumps(source_protocol,sort_keys=True).encode()).hexdigest()[:12]
    if (source_id!=FROZEN_SOURCE_ID or source_protocol.get('version')!='change-memory-v1'
        or source_protocol.get('annotation_sha256')!=sha(annotation)
        or source_protocol.get('split_sha256')!=sha(args.split)
        or source_protocol.get('source_config_id')!=configuration_id(config,policy_id='recent-8',backend='smolvlm')
        or any(x.get('record_type')!='change-memory-v1' or x.get('config_id')!=source_id or x['protocol']!=source_protocol for x in rows)):
        parser.error('source protocol/config/annotation/split mismatch')
    samples,reader=_load_samples(args.data_root,annotation_path=annotation,sample_fps=1.0)
    samples=sorted((x for x in samples if x.sample_id in ids),key=lambda x:x.sample_id)
    if len(samples)!=20 or {x.sample_id for x in samples}!=ids or len({x.video_id for x in samples})!=20:
        parser.error('annotations do not match frozen20cohort')
    sources={x['sample_id']:x for x in rows}
    for sample in samples:
        validate_source(sample,sources[sample.sample_id])
    protocol=dict(version='visual-sensitivity-v1',source_sha256=sha(args.source),source_config_id=source_id,
                  annotation_sha256=sha(annotation),split_sha256=sha(args.split),
                  base_config_id=configuration_id(config,policy_id='recent-8',backend='smolvlm'),
                  blank='eight same-size RGB127 images',image_order='reverse exactly',option_cycle=[1,2,3,0],
                  order='rotate original/blank/reverse/option-cycle by sorted frozen index',
                  timestamp_metadata='empty request timestamps, no timestamp text; actual provenance recorded separately')
    run_id='visual-sensitivity-v1-'+hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
    store=ResumeStore(args.output)
    expected={(x.sample_id,run_id) for x in samples}
    if store.completed_keys-expected:
        parser.error('output contains unexpected keys/source hash')
    if args.output.exists() and len(read_rows(args.output))!=len(store.completed_keys):
        parser.error('duplicate or missing output keys')
    selected=samples[:args.limit] if args.limit is not None else samples
    pending=[(i,x) for i,x in enumerate(selected) if not store.is_complete(x.sample_id,run_id)]
    backend=None
    if pending:
        set_seed(config.experiment.seed)
        backend=SmolVLMBackend(model_id=config.model.model_id,max_new_tokens=8)
    failed=0
    for index,sample in pending:
        try:
            record=evaluate_visual_sensitivity(sample,sources[sample.sample_id],reader=reader,backend=backend,order_offset=index)
            record.update(config_id=run_id,protocol=protocol,record_type='visual-sensitivity-v1')
            store.append(record)
            print(f'completed={len(store.completed_keys)}/20 sample={sample.sample_id}',flush=True)
        except Exception as exc:
            failed+=1
            error_path=args.output.with_name(args.output.stem+'-errors.jsonl')
            error_path.parent.mkdir(parents=True,exist_ok=True)
            with error_path.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(dict(sample_id=sample.sample_id,config_id=run_id,error_type=type(exc).__name__,error=str(exc)))+'\n')
            print(f'failed {sample.sample_id}: {exc}',file=sys.stderr,flush=True)
    print(f'processed={len(pending)-failed} skipped={len(selected)-len(pending)} failed={failed}',flush=True)
    if len(store.completed_keys)==20:
        completed=read_rows(args.output)
        if len(completed)!=20:
            raise ValueError('duplicate rows; refusing summary')
        summary=summarize_visual_sensitivity(completed)
        summary.update(config_id=run_id,protocol=protocol,
                       mean_decode_latency_ms=sum(x['decode_latency_ms'] for x in completed)/20,
                       shared_peak_gpu_memory_mb=max(x['shared_peak_gpu_memory_mb'] for x in completed))
        args.output.with_name(args.output.stem+'-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
        print('Full visual sensitivity diagnostic complete',flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())

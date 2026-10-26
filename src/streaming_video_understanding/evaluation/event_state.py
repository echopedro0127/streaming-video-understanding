"""Continuous causal classification/state replay and human interval scoring."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from PIL import Image

from ..data.perception_review import verify_frame,owned_image
from ..models.base import ModelRequest
from ..models.prompt import build_prompt
from ..streaming.event_state import EventCounter
from .perception_probe import CATEGORIES,DESCRIPTIONS,QUESTION
from .engine import _peak_gpu_memory_mb,_reset_gpu_peak_memory
from .metrics import _percentile

MODEL_ID='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'


def config_digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def validate_config(config):
    if (config.get('version')!='event-state-v1' or config.get('model_id')!=MODEL_ID
        or config.get('sample_fps')!=1.0 or config.get('max_new_tokens')!=8
        or config.get('max_image_side')!=640 or config.get('jpeg_quality')!=95
        or config.get('unknown_policy')!='hold active state; reset pending confirmation streaks'
        or config.get('event_iou_threshold')!=0.5 or type(config.get('seed')) is not int or config['seed']<0
        or config.get('policies')!={'raw':{'enter_frames':1,'exit_frames':1},'debounced':{'enter_frames':2,'exit_frames':2}}):
        raise ValueError('unsupported fixed event-state protocol')
    end=config['end_time_s'];stop=config['frame_ids_stop'];queries=config['query_times_s']
    if isinstance(end,bool) or not math.isfinite(end) or end<=0 or type(stop) is not int or stop<1:
        raise ValueError('invalid event-state horizon/frame count')
    if not queries or any(isinstance(q,bool) or not math.isfinite(q) or q<0 or q>end for q in queries) or queries!=sorted(set(queries)):
        raise ValueError('query timestamps must be unique, ordered and causal')


def selection_for(config):
    validate_config(config)
    return dict(version='perception-probe-v1',sample_fps=1.0,max_image_side=640,jpeg_quality=95,
        selection_scope='Continuous known-video diagnostic; no certified event labels',event_config_sha256=config_digest(config),
        cases=[dict(sample_id=config['sample_id'],query_time_s=config['end_time_s'],frame_ids=list(range(config['frame_ids_stop'])))])


def make_counters(config):
    return {name:EventCounter(**kwargs) for name,kwargs in config['policies'].items()}


def classify_frame(frame,*,root,backend):
    started=time.perf_counter()
    path=verify_frame(root,frame)
    _reset_gpu_peak_memory()
    with owned_image(Image.open(path)) as stored:
        with owned_image(stored.convert('RGB')) as image:
            response=backend.infer(ModelRequest(images=(image,),timestamps_s=(),question=QUESTION,
                options=DESCRIPTIONS,query_time_s=frame['timestamp_s'],include_timestamps=False,allow_invalid_answer=True))
    if response.parse_valid and response.answer not in ('A','B','C','D'):
        raise ValueError('parse-valid answer must be A-D')
    category=CATEGORIES[ord(response.answer)-65] if response.parse_valid else None
    return dict(sample_id=frame['frame_key'],frame_key=frame['frame_key'],source_sample_id=frame['sample_id'],
        video_id=frame['video_id'],frame_id=frame['frame_id'],timestamp_s=frame['timestamp_s'],
        observed_at_s=frame['timestamp_s'],source_cutoff_s=frame['query_time_s'],image_sha256=frame['image_sha256'],
        answer=response.answer,raw_text=response.raw_text,parse_valid=response.parse_valid,category=category,
        input_frames=1,inference_calls=1,request_timestamps_s=[],timestamp_text=False,
        prompt=build_prompt(question=QUESTION,options=DESCRIPTIONS,timestamps_s=(),query_time_s=frame['timestamp_s'],include_timestamps=False),
        model_latency_ms=response.latency_ms,processor_ms=response.processor_latency_ms,generation_ms=response.generation_latency_ms,
        peak_gpu_memory_mb=_peak_gpu_memory_mb(),pipeline_latency_ms=(time.perf_counter()-started)*1000)


def apply_states(row,counters):
    return {name:dict(counter.update(row['category'],row['timestamp_s']),state=counter.snapshot()) for name,counter in counters.items()}


def replay_prefix(rows,manifest,config,*,run_id,protocol,manifest_sha256):
    counters=make_counters(config)
    if len(rows)>len(manifest['frames']):
        raise ValueError('output is longer than source prefix')
    for row,frame in zip(rows,manifest['frames']):
        if row.get('frame_key')!=frame['frame_key'] or row.get('sample_id')!=frame['frame_key']:
            raise ValueError('output must be a contiguous source prefix')
        if (row.get('config_id')!=run_id or row.get('protocol')!=protocol or row.get('manifest_sha256')!=manifest_sha256
            or row.get('source_sample_id')!=frame['sample_id'] or row.get('video_id')!=frame['video_id']
            or row.get('frame_id')!=frame['frame_id'] or row.get('timestamp_s')!=frame['timestamp_s']
            or row.get('observed_at_s')!=frame['timestamp_s'] or row.get('source_cutoff_s')!=frame['query_time_s']
            or row.get('image_sha256')!=frame['image_sha256']):
            raise ValueError('output protocol/source/causal prefix mismatch')
        valid=row.get('parse_valid')
        if type(valid) is not bool or valid and row['answer'] not in ('A','B','C','D'):
            raise ValueError('invalid category parsing schema')
        category=CATEGORIES[ord(row['answer'])-65] if valid else None
        if (row.get('category')!=category or row.get('input_frames')!=1 or row.get('inference_calls')!=1
            or row.get('request_timestamps_s')!=[] or row.get('timestamp_text') is not False
            or row.get('prompt')!=build_prompt(question=QUESTION,options=DESCRIPTIONS,timestamps_s=(),
                       query_time_s=frame['timestamp_s'],include_timestamps=False)):
            raise ValueError('output category/input/prompt mismatch')
        if apply_states(row,counters)!=row.get('policy_traces'):
            raise ValueError('stored state trace differs from deterministic replay')
    return counters


def summarize_event_run(rows,config):
    if not rows:
        raise ValueError('no event-state frames')
    policies={}
    for name in config['policies']:
        events=[row['policy_traces'][name]['closed_event'] for row in rows if row['policy_traces'][name]['closed_event']]
        last=rows[-1]['policy_traces'][name]
        if last['active']:
            state=last['state']
            events.append(dict(start_s=state['active_start_s'],confirmed_at_s=state['active_confirmed_at_s'],
                               end_s=config['end_time_s'],confirmed_end_at_s=None,end_censored=True))
        if len(events)!=last['count']:
            raise ValueError('event intervals disagree with causal count')
        queries=[]
        for q in config['query_times_s']:
            past=[r for r in rows if r['timestamp_s']<=q+1e-6]
            queries.append(dict(timestamp_s=q,count=past[-1]['policy_traces'][name]['count'] if past else 0))
        policies[name]=dict(final_count=last['count'],intervals=events,query_counts=queries)
    latencies=[r['model_latency_ms'] for r in rows]
    total_pipeline=sum(r['pipeline_latency_ms'] for r in rows)
    return dict(num_frames=len(rows),inference_calls=sum(r['inference_calls'] for r in rows),
        input_frames_per_call=1,total_input_images=len(rows),policies=policies,
        mean_model_latency_ms=sum(latencies)/len(latencies),p50_model_latency_ms=_percentile(latencies,50),
        p95_model_latency_ms=_percentile(latencies,95),cumulative_model_latency_ms=sum(latencies),
        cumulative_pipeline_latency_ms=total_pipeline,
        processing_throughput_fps=1000*len(rows)/total_pipeline if total_pipeline else None,
        peak_gpu_memory_mb=max(r['peak_gpu_memory_mb'] for r in rows),event_metrics=None,
        human_label_status='pending',scope='Single known-video causal diagnostic;185calls differ from1call/8image baselines')


def interval_matches(predicted,truth,threshold):
    # Offline ordered dynamic programming: maximize one-to-one match count,
    # then total IoU. This metadata workspace is not online model state.
    dp=[[(0,0.0,()) for _ in range(len(truth)+1)] for _ in range(len(predicted)+1)]
    for i,p in enumerate(predicted,1):
        for j,t in enumerate(truth,1):
            intersection=max(0,min(p['end_s'],t['end_s'])-max(p['start_s'],t['start_s']))
            union=max(p['end_s'],t['end_s'])-min(p['start_s'],t['start_s'])
            iou=intersection/union if union else 0
            candidates=[dp[i-1][j],dp[i][j-1]]
            if iou>=threshold:
                old=dp[i-1][j-1]
                candidates.append((old[0]+1,old[1]+iou,old[2]+((i-1,j-1),)))
            dp[i][j]=max(candidates,key=lambda item:(item[0],item[1]))
    return dp[-1][-1][2]


def score_event_run(summary,labels,*,manifest_sha256,config):
    if (labels.get('annotation_source')!='human_self_reported' or labels.get('reviewed_full_prefix') is not True
        or not isinstance(labels.get('reviewer'),str) or not labels['reviewer'].strip()):
        raise ValueError('explicit complete-prefix human review required')
    if labels.get('manifest_sha256')!=manifest_sha256:
        raise ValueError('human label manifest hash mismatch')
    if labels.get('counting_convention') not in ('split_on_presenter_return','merge_brief_returns_within_same_insertion'):
        raise ValueError('explicit counting convention required')
    truth=labels.get('intervals')
    if not isinstance(truth,list):
        raise ValueError('human intervals must be an explicit list, including empty')
    if not truth and labels.get('no_events_confirmed') is not True:
        raise ValueError('empty human interval list needs explicit negative confirmation')
    previous_end=0.0
    for event in truth:
        start,end=event['start_s'],event['end_s']
        if (any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in (start,end))
            or start<previous_end or end<=start or end>config['end_time_s'] or type(event.get('end_censored')) is not bool
            or event['end_censored'] and end!=config['end_time_s']):
            raise ValueError('intervals must be ordered, nonoverlapping, positive and inside prefix')
        previous_end=end
    result=dict(annotation_source='human_self_reported',counting_convention=labels['counting_convention'],
                true_events=len(truth),query_times_s=config['query_times_s'],policies={})
    for name,policy in summary['policies'].items():
        predicted=policy['intervals'];pairs=interval_matches(predicted,truth,config['event_iou_threshold']);tp=len(pairs)
        queries=policy['query_counts']
        if [q['timestamp_s'] for q in queries]!=config['query_times_s']:
            raise ValueError('query points differ from frozen protocol')
        targets=[sum(event['start_s']<=q['timestamp_s'] for event in truth) for q in queries]
        errors=[abs(q['count']-target) for q,target in zip(queries,targets)]
        delays=[predicted[i]['confirmed_at_s']-truth[j]['start_s'] for i,j in pairs]
        result['policies'][name]=dict(event_true_positives=tp,predicted_events=len(predicted),
            event_precision=tp/len(predicted) if predicted else None,event_recall=tp/len(truth) if truth else None,
            event_f1=2*tp/(len(predicted)+len(truth)) if predicted or truth else None,
            count_mae=sum(errors)/len(errors),query_accuracy=sum(e==0 for e in errors)/len(errors),
            query_ground_truth=targets,mean_confirmation_delay_s=sum(delays)/len(delays) if delays else None,
            confirmation_delay_definition='signed media time, not wall-clock end-to-end latency')
    return result

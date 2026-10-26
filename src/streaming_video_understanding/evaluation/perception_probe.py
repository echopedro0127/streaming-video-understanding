"""Single-frame category probe; accuracy requires separate explicit human labels."""
from __future__ import annotations

from collections import Counter
import math

from PIL import Image

from ..data.perception_review import verify_frame,owned_image
from ..models.base import ModelRequest
from ..models.prompt import build_prompt
from .metrics import _percentile


CATEGORIES=('presenter','inserted_scene','transition','uncertain')
DESCRIPTIONS=('A teaching presenter beside a whiteboard or teaching board.',
              'An inserted movie or performance scene, not the teaching presenter.',
              'A blank screen or transition where neither scene is visible.',
              'The image is too ambiguous to decide among the other categories.')
QUESTION='Which description best matches the visible image?'
CYCLE=(1,2,3,0)


def infer_perception(frame,*,root,backend,cycle_first=False):
    path=verify_frame(root,frame)
    conditions={}
    order=('option-cycle','original') if cycle_first else ('original','option-cycle')
    with owned_image(Image.open(path)) as stored:
        with owned_image(stored.convert('RGB')) as image:
            for mode in order:
                permutation=CYCLE if mode=='option-cycle' else (0,1,2,3)
                options=tuple(DESCRIPTIONS[i] for i in permutation)
                response=backend.infer(ModelRequest(images=(image,),timestamps_s=(),question=QUESTION,
                    options=options,query_time_s=frame['query_time_s'],include_timestamps=False,allow_invalid_answer=True))
                if response.parse_valid and response.answer not in ('A','B','C','D'):
                    raise ValueError('parse-valid response must be A-D')
                category=CATEGORIES[permutation[ord(response.answer)-65]] if response.parse_valid else None
                conditions[mode]=dict(answer=response.answer,raw_text=response.raw_text,parse_valid=response.parse_valid,
                    category=category,options=list(options),option_permutation=list(permutation),
                    prompt=build_prompt(question=QUESTION,options=options,timestamps_s=(),
                                        query_time_s=frame['query_time_s'],include_timestamps=False),
                    model_latency_ms=response.latency_ms,input_frames=1)
    return dict(frame_key=frame['frame_key'],sample_id=frame['sample_id'],video_id=frame['video_id'],
                frame_id=frame['frame_id'],timestamp_s=frame['timestamp_s'],query_time_s=frame['query_time_s'],
                image_sha256=frame['image_sha256'],mode_order=list(order),conditions=conditions,inference_calls=2)


def summarize_perception(rows):
    if not rows or len({x['frame_key'] for x in rows})!=len(rows):
        raise ValueError('require nonempty unique perception rows')
    if any(set(x['conditions'])!={'original','option-cycle'} for x in rows):
        raise ValueError('incomplete paired perception row')
    valid=[x for x in rows if all(c['parse_valid'] for c in x['conditions'].values())]
    result=dict(num_frames=len(rows),num_videos=len({x['video_id'] for x in rows}),
                inference_calls=sum(x['inference_calls'] for x in rows),valid_pairs=len(valid),
                semantic_consistent=sum(x['conditions']['original']['category']==x['conditions']['option-cycle']['category'] for x in valid),
                label_sticky=sum(x['conditions']['original']['answer']==x['conditions']['option-cycle']['answer'] for x in valid),
                accuracy=None,label_status='pending_human_review',conditions={})
    for mode in ('original','option-cycle'):
        items=[x['conditions'][mode] for x in rows]
        latencies=[x['model_latency_ms'] for x in items]
        result['conditions'][mode]=dict(parse_valid_count=sum(x['parse_valid'] for x in items),
            category_distribution=dict(Counter(x['category'] if x['category'] is not None else 'invalid' for x in items)),
            letter_distribution=dict(Counter(x['answer'] for x in items)),
            mean_model_latency_ms=sum(latencies)/len(latencies),p95_model_latency_ms=_percentile(latencies,95))
    return result


def validate_predictions(manifest,rows,*,manifest_sha256):
    lookup={x['frame_key']:x for x in manifest['frames']}
    if len({x['frame_key'] for x in rows})!=len(rows):
        raise ValueError('duplicate prediction frames')
    for row in rows:
        frame=lookup.get(row['frame_key'])
        if frame is None or row.get('manifest_sha256')!=manifest_sha256:
            raise ValueError('prediction manifest mismatch')
        if row.get('source_sample_id',row.get('sample_id'))!=frame['sample_id']:
            raise ValueError('prediction QA source mismatch')
        if 'source_sample_id' in row and row.get('sample_id')!=row['frame_key']:
            raise ValueError('prediction primary key mismatch')
        if any(row.get(key)!=frame[key] for key in ('video_id','frame_id','timestamp_s','query_time_s','image_sha256')):
            raise ValueError('prediction image/source mismatch')
        if row.get('inference_calls')!=2 or row.get('mode_order') not in (['original','option-cycle'],['option-cycle','original']):
            raise ValueError('prediction pair/order mismatch')
        if set(row['conditions'])!={'original','option-cycle'}:
            raise ValueError('incomplete prediction pair')
        for mode,item in row['conditions'].items():
            permutation=CYCLE if mode=='option-cycle' else (0,1,2,3)
            options=tuple(DESCRIPTIONS[i] for i in permutation)
            if type(item['parse_valid']) is not bool or item['parse_valid'] and item['answer'] not in ('A','B','C','D'):
                raise ValueError('invalid answer validity schema')
            category=CATEGORIES[permutation[ord(item['answer'])-65]] if item['parse_valid'] else None
            if item['category']!=category or item['option_permutation']!=list(permutation):
                raise ValueError('prediction category mapping mismatch')
            if (item['options']!=list(options) or item['input_frames']!=1 or item['prompt']!=build_prompt(
                question=QUESTION,options=options,timestamps_s=(),query_time_s=frame['query_time_s'],include_timestamps=False)):
                raise ValueError('prediction input/prompt mismatch')


def score_perception(rows,labels,*,manifest_sha256):
    summarize_perception(rows)
    if any(row.get('manifest_sha256')!=manifest_sha256 for row in rows):
        raise ValueError('prediction manifest hash mismatch')
    if (labels.get('annotation_source')!='human_self_reported' or labels.get('reviewed') is not True
        or not isinstance(labels.get('reviewer'),str) or not labels['reviewer'].strip()):
        raise ValueError('explicit human self-reported review metadata required')
    if labels.get('manifest_sha256')!=manifest_sha256:
        raise ValueError('label manifest hash mismatch')
    entries=labels.get('labels',[])
    lookup={x['frame_key']:x['category'] for x in entries}
    if len(entries)!=len(lookup) or set(lookup)!={x['frame_key'] for x in rows} or any(x not in CATEGORIES for x in lookup.values()):
        raise ValueError('human labels must cover every unique frame with a known category')
    resolved=[x for x in rows if lookup[x['frame_key']]!='uncertain']
    counts=Counter(lookup[x['frame_key']] for x in resolved)
    result=dict(annotation_source='human_self_reported',num_frames=len(rows),resolved_frames=len(resolved),
                resolved_coverage=len(resolved)/len(rows),human_category_counts=dict(Counter(lookup.values())),
                conditions={},binary_balanced_recall=None)
    for mode in ('original','option-cycle'):
        recalls={category:sum(x['conditions'][mode]['category']==category for x in resolved if lookup[x['frame_key']]==category)/count
                 for category,count in counts.items()}
        result['conditions'][mode]=dict(accuracy=sum(x['conditions'][mode]['category']==lookup[x['frame_key']] for x in resolved)/len(resolved) if resolved else None,
                                       per_class_recall=recalls,binary_balanced_recall=(recalls['presenter']+recalls['inserted_scene'])/2 if 'presenter' in recalls and 'inserted_scene' in recalls else None)
    result['binary_balanced_recall']=result['conditions']['original']['binary_balanced_recall']
    return result

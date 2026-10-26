"""Portable causal image bundle and a blinded local reviewer page."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def owned_image(image):
    try:
        yield image
    except BaseException as primary:
        try:
            image.close()
        except BaseException as cleanup:
            primary.add_note(f'image cleanup failed: {cleanup!r}')
        raise
    else:
        image.close()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_path(root, frame):
    root=Path(root).resolve()
    path=(root/frame['image']).resolve()
    if not path.is_relative_to(root) or Path(frame['image']).is_absolute():
        raise ValueError('image path must stay inside bundle')
    return path


def verify_frame(root,frame):
    t,q=frame['timestamp_s'],frame['query_time_s']
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in (t,q)) or t>q+1e-6:
        raise ValueError('noncausal/invalid frame timestamp')
    if type(frame['frame_id']) is not int or frame['frame_id']<0:
        raise ValueError('invalid frame ID')
    path=image_path(root,frame)
    if sha(path)!=frame['image_sha256']:
        raise ValueError('image hash mismatch')
    return path


def verify_bundle(manifest_path):
    path=Path(manifest_path)
    manifest=json.loads(path.read_text(encoding='utf-8'))
    frames=manifest['frames']
    if manifest.get('version')!='perception-probe-v1' or not frames or len({x['frame_key'] for x in frames})!=len(frames):
        raise ValueError('invalid/duplicate perception manifest')
    if len({(x['sample_id'],x['frame_id']) for x in frames})!=len(frames):
        raise ValueError('duplicate source frame')
    selection=manifest.get('selection_config')
    if selection is None:
        selection=json.loads((Path(__file__).resolve().parents[3]/'configs'/'perception-probe-v1.json').read_text(encoding='utf-8'))
    config_sha=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
    expected=[(case['sample_id'],i,case['query_time_s']) for case in selection['cases'] for i in case['frame_ids']]
    actual=[(frame['sample_id'],frame['frame_id'],frame['query_time_s']) for frame in frames]
    if manifest.get('selection_config_sha256')!=config_sha or expected!=actual:
        raise ValueError('manifest does not match complete frozen selection')
    for frame in frames:
        verify_frame(path.parent,frame)
    return manifest


def prepare_bundle(config,samples,*,reader,output_dir,annotation_sha256,embed_selection=False,review_page=True):
    output_dir=Path(output_dir)
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    path=output_dir/'manifest.json'
    if path.exists():
        old=verify_bundle(path)
        if old.get('selection_config_sha256')!=config_sha or old.get('annotation_sha256')!=annotation_sha256:
            raise ValueError('existing bundle config/annotation mismatch')
        if review_page:
            write_review_page(path)
        return path
    if config.get('version')!='perception-probe-v1' or config.get('sample_fps')!=1.0 or config.get('max_image_side')!=640 or config.get('jpeg_quality')!=95:
        raise ValueError('unsupported perception export configuration')
    lookup={x.sample_id:x for x in samples}
    frames=[]
    for case in config['cases']:
        sample=lookup[case['sample_id']]
        if sample.query_time_s!=case['query_time_s']:
            raise ValueError('query time differs from frozen selection')
        ids=case['frame_ids']
        if not ids or any(type(i) is not int or i<0 for i in ids) or ids!=sorted(set(ids)):
            raise ValueError('selected IDs must be increasing unique integers')
        wanted=set(ids);found=set();last_id=-1;last_t=-math.inf
        for image,frame in reader.read_until(sample.video_path,end_time_s=sample.query_time_s):
            with owned_image(image):
                t=frame.timestamp_s
                if (type(frame.frame_id) is not int or frame.frame_id<=last_id or isinstance(t,bool)
                    or not math.isfinite(t) or t<0 or t<last_t or t>sample.query_time_s+1e-6):
                    raise ValueError('invalid/future frame before export')
                last_id,last_t=frame.frame_id,t
                if frame.frame_id not in wanted:
                    continue
                key=hashlib.sha256(f'{sample.sample_id}:{frame.frame_id}'.encode()).hexdigest()[:12]
                target=output_dir/'images'/f'{key}.jpg'
                target.parent.mkdir(parents=True,exist_ok=True)
                with owned_image(image.copy()) as thumb:
                    thumb.thumbnail((640,640))
                    thumb.save(target,quality=95)
                found.add(frame.frame_id)
                frames.append(dict(frame_key=key,sample_id=sample.sample_id,video_id=sample.video_id,
                    frame_id=frame.frame_id,timestamp_s=t,query_time_s=sample.query_time_s,
                    image=f'images/{key}.jpg',image_sha256=sha(target)))
        if found!=wanted:
            raise ValueError('selected frames missing during export')
    if len({x['frame_key'] for x in frames})!=len(frames):
        raise ValueError('duplicate export cases')
    manifest=dict(version='perception-probe-v1',selection_config_sha256=config_sha,
                  annotation_sha256=annotation_sha256,selection_scope=config['selection_scope'],
                  max_image_side=640,jpeg_quality=95,label_status='pending_human_review',frames=frames)
    if embed_selection:
        manifest['selection_config']=config
    path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    verify_bundle(path)
    if review_page:
        write_review_page(path)
    return path


def write_review_page(manifest_path):
    manifest_path=Path(manifest_path)
    manifest=verify_bundle(manifest_path)
    payload=dict(manifest_sha256=sha(manifest_path),frames=manifest['frames'])
    template=Path(__file__).resolve().parents[3]/'assets'/'perception-review.html'
    text=template.read_text(encoding='utf-8').replace('__MANIFEST_DATA__',json.dumps(payload,ensure_ascii=True).replace('<','\\u003c'))
    page=manifest_path.parent/'review.html'
    page.write_text(text,encoding='utf-8')
    return page

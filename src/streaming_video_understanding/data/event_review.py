"""Local bounded-prefix video and blinded human interval review page."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

from .perception_review import verify_bundle,sha


def verify_clip(manifest_path,manifest):
    root=Path(manifest_path).parent.resolve()
    meta=manifest['review_video'];path=(root/meta['path']).resolve()
    if not path.is_relative_to(root) or Path(meta['path']).is_absolute():
        raise ValueError('review video path escapes bundle')
    if sha(path)!=meta['sha256']:
        raise ValueError('review video hash mismatch')
    if not math.isfinite(meta['duration_s']) or meta['duration_s']<=0 or abs(meta['duration_s']-meta['cutoff_s'])>0.11:
        raise ValueError('review video must cover exactly the complete prefix within encoding tolerance')


def prepare_review_video(source_video,manifest_path,*,end_time_s):
    manifest_path=Path(manifest_path)
    manifest=verify_bundle(manifest_path)
    source_sha=sha(source_video)
    if 'review_video' in manifest:
        verify_clip(manifest_path,manifest)
        if manifest.get('source_video_sha256')!=source_sha or manifest['review_video']['cutoff_s']!=end_time_s:
            raise ValueError('existing video/source boundary mismatch')
        return
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe are required for human prefix review')
    target=manifest_path.parent/'review-prefix.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(source_video),'-map','0:v:0',
                    '-an','-vf','fps=10,scale=640:-2','-t',str(end_time_s),'-c:v','libx264','-preset','veryfast',
                    '-crf','26','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True,capture_output=True)
    probe=subprocess.run([ffprobe,'-v','error','-show_entries','format=duration','-of','json',str(target)],
                         check=True,capture_output=True,text=True)
    duration=float(json.loads(probe.stdout)['format']['duration'])
    manifest.update(source_video_sha256=source_sha,review_video=dict(path=target.name,sha256=sha(target),
                    duration_s=duration,cutoff_s=end_time_s,fps=10,silent=True))
    verify_clip(manifest_path,manifest)
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def write_event_review(manifest_path,*,end_time_s):
    manifest_path=Path(manifest_path)
    manifest=verify_bundle(manifest_path)
    verify_clip(manifest_path,manifest)
    payload=dict(manifest_sha256=sha(manifest_path),end_time_s=end_time_s,
                 video=manifest['review_video']['path'],frames=manifest['frames'])
    template=Path(__file__).resolve().parents[3]/'assets'/'event-review.html'
    text=template.read_text(encoding='utf-8').replace('__EVENT_DATA__',json.dumps(payload).replace('<','\\u003c'))
    target=manifest_path.parent/'event-review.html'
    target.write_text(text,encoding='utf-8')
    return target

import pytest

from streaming_video_understanding.streaming.event_state import EventCounter


def test_count_available_only_on_confirmation_and_two_events():
    state=EventCounter(enter_frames=2,exit_frames=2)
    assert state.update('presenter',0)['count']==0
    assert state.update('inserted_scene',1)['count']==0
    opened=state.update('inserted_scene',2)
    assert opened['count']==1
    assert opened['opened_event']=={'start_s':1.0,'confirmed_at_s':2.0}
    assert state.update('presenter',3)['active']
    closed=state.update('presenter',4)
    assert not closed['active'] and closed['closed_event']['end_s']==3
    assert state.update('inserted_scene',5)['count']==1
    assert state.update('inserted_scene',6)['count']==2


def test_unknown_holds_state_but_resets_streaks():
    state=EventCounter(enter_frames=2,exit_frames=2)
    for t,label in enumerate(['inserted_scene','uncertain','inserted_scene',None,'transition']):
        assert state.update(label,t)['count']==0
    state.update('inserted_scene',5)
    assert state.update('inserted_scene',6)['active']
    assert state.update('presenter',7)['active']
    assert state.update('uncertain',8)['active']
    assert state.update('presenter',9)['active']
    assert not state.update('presenter',10)['active']


def test_prefix_invariance_and_flicker():
    sequence=['presenter','inserted_scene','presenter','inserted_scene','inserted_scene','transition','presenter']
    full=EventCounter(enter_frames=2,exit_frames=2)
    trace=[full.update(label,i) for i,label in enumerate(sequence)]
    for n in range(1,len(sequence)+1):
        state=EventCounter(enter_frames=2,exit_frames=2)
        assert [state.update(label,i) for i,label in enumerate(sequence[:n])]==trace[:n]
    raw=EventCounter(enter_frames=1,exit_frames=1)
    assert [raw.update(label,i)['count'] for i,label in enumerate(sequence)][-1]==2
    assert trace[-1]['count']==1


@pytest.mark.parametrize('timestamp',[-1,float('nan'),True,0])
def test_invalid_times_do_not_mutate_state(timestamp):
    state=EventCounter(enter_frames=2,exit_frames=2)
    state.update('presenter',0)
    before=state.snapshot()
    with pytest.raises(ValueError):
        state.update('inserted_scene',timestamp)
    assert state.snapshot()==before


def tiny_run(tmp_path):
    import json,hashlib
    from PIL import Image
    config=json.loads(open('configs/event-state-v1.json',encoding='utf-8').read())
    config.update(sample_id='s',end_time_s=7.0,frame_ids_stop=7,query_times_s=[0.0,2.0,4.0,7.0])
    frames=[]
    for i in range(7):
        p=tmp_path/f'{i}.jpg'
        with Image.new('RGB',(16,12)) as im:
            im.save(p)
        frames.append(dict(frame_key=f'f{i}',sample_id='s',video_id='v',frame_id=i,timestamp_s=float(i),query_time_s=7.0,
                           image=p.name,image_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return config,dict(frames=frames)


def test_single_current_image_resume_replays_exact_prefix(tmp_path):
    from streaming_video_understanding.evaluation.event_state import make_counters,classify_frame,apply_states,replay_prefix,summarize_event_run
    from streaming_video_understanding.models.base import ModelResponse
    config,manifest=tiny_run(tmp_path);counters=make_counters(config);calls=[]
    letters=iter(['A','B','B','A','A','B','B'])
    class Backend:
        def infer(self,request):
            calls.append(request)
            return ModelResponse(raw_text='label',answer=next(letters),latency_ms=1)
    rows=[]
    for i,frame in enumerate(manifest['frames']):
        row=classify_frame(frame,root=tmp_path,backend=Backend())
        row.update(config_id='run',protocol={'p':1},manifest_sha256='m',policy_traces=apply_states(row,counters))
        rows.append(row)
        assert calls[-1].query_time_s==float(i)
        assert calls[-1].timestamps_s==() and len(calls[-1].images)==1
    restored=replay_prefix(rows[:3],manifest,config,run_id='run',protocol={'p':1},manifest_sha256='m')
    assert restored['debounced'].count==1
    assert [apply_states(row,restored) for row in rows[3:]]==[row['policy_traces'] for row in rows[3:]]
    with pytest.raises(ValueError,match='prefix'):
        replay_prefix([rows[0],rows[2]],manifest,config,run_id='run',protocol={'p':1},manifest_sha256='m')
    rows[0]['policy_traces']['raw']['count']=99
    with pytest.raises(ValueError,match='state'):
        replay_prefix(rows,manifest,config,run_id='run',protocol={'p':1},manifest_sha256='m')


def test_human_interval_metrics_use_confirmation_counts_and_unique_matching():
    from streaming_video_understanding.evaluation.event_state import score_event_run
    config=dict(end_time_s=10.0,query_times_s=[1.0,2.0,5.0,10.0],event_iou_threshold=0.5)
    labels=dict(manifest_sha256='m',annotation_source='human_self_reported',reviewer='r',reviewed_full_prefix=True,
                counting_convention='split_on_presenter_return',intervals=[dict(start_s=1.0,end_s=4.0,end_censored=False)])
    summary=dict(policies={'p':dict(intervals=[dict(start_s=1.0,end_s=4.0,confirmed_at_s=2.0,end_censored=False)],
                 query_counts=[dict(timestamp_s=t,count=c) for t,c in zip(config['query_times_s'],[0,1,1,1])])})
    result=score_event_run(summary,labels,manifest_sha256='m',config=config)
    assert result['policies']['p']['event_precision']==1
    assert result['policies']['p']['count_mae']==0.25
    assert result['policies']['p']['query_accuracy']==0.75
    assert result['policies']['p']['mean_confirmation_delay_s']==1.0
    labels['reviewed_full_prefix']=False
    with pytest.raises(ValueError,match='human'):
        score_event_run(summary,labels,manifest_sha256='m',config=config)


def test_optional_embedded_selection_export_preserves_default_contract(tmp_path):
    from pathlib import Path
    from PIL import Image
    from streaming_video_understanding.data.schema import StreamingSample,FrameRecord
    from streaming_video_understanding.data.perception_review import prepare_bundle,verify_bundle
    s=StreamingSample('s','v',Path('v'),'q',('a','b','c','d'),'A',2.0,'','')
    class Reader:
        def read_until(self,*args,**kwargs):
            for i in range(2):
                yield Image.new('RGB',(8,8)),FrameRecord(i,float(i))
    selection=dict(version='perception-probe-v1',sample_fps=1.0,max_image_side=640,jpeg_quality=95,
                   selection_scope='continuous-test',cases=[dict(sample_id='s',query_time_s=2.0,frame_ids=[0,1])])
    path=prepare_bundle(selection,[s],reader=Reader(),output_dir=tmp_path,annotation_sha256='a',embed_selection=True,review_page=False)
    assert verify_bundle(path)['selection_config']==selection
    assert not (tmp_path/'review.html').exists()


def test_event_cli_resume_and_gap_guard(monkeypatch,tmp_path):
    import json,hashlib,sys
    import scripts.event_state_probe as cli
    from streaming_video_understanding.models.base import ModelResponse
    from streaming_video_understanding.evaluation.event_state import selection_for,config_digest
    config,manifest=tiny_run(tmp_path)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    selection=selection_for(config)
    manifest.update(version='perception-probe-v1',selection_config=selection,selection_config_sha256=config_digest(selection))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));output=tmp_path/'predictions.jsonl'
    loaded=[]
    class Backend:
        def infer(self,request):
            return ModelResponse(answer='A',raw_text='A',latency_ms=1)
    def factory(**kwargs):
        loaded.append(True);return Backend()
    monkeypatch.setattr(cli,'SmolVLMBackend',factory)
    monkeypatch.setattr(sys,'argv',['event_state_probe.py','run','--config',str(config_path),
        '--manifest',str(path),'--output',str(output),'--limit','3','--resume'])
    assert cli.main()==0
    before=output.read_bytes()
    assert cli.main()==0 and output.read_bytes()==before and len(loaded)==1
    rows=output.read_text().splitlines();output.write_text(rows[0]+'\n'+rows[2]+'\n')
    with pytest.raises(ValueError,match='prefix'):
        cli.main()
    assert len(loaded)==1


def test_event_review_export_requires_full_review_and_valid_intervals(tmp_path):
    import shutil,subprocess
    if not shutil.which('node'):
        pytest.skip('Node unavailable')
    template=open('assets/event-review.html',encoding='utf-8').read()
    import json
    payload=dict(manifest_sha256='m',end_time_s=10.0,video='review-prefix.mp4',frames=[])
    script=template.replace('__EVENT_DATA__',json.dumps(payload)).split('<script>')[1].split('</script>')[0]
    stub="""const elements={};const el=()=>({value:'',checked:false,currentTime:0,appendChild(){},replaceChildren(){}});
const document={getElementById(id){return elements[id]||(elements[id]=el());},createElement:el};
const localStorage={getItem(){return null;},setItem(){}};const window={};
"""
    checks="""const assert=require('node:assert/strict');assert.throws(()=>buildExport());
document.getElementById('reviewer').value='r';document.getElementById('reviewed').checked=true;
document.getElementById('convention').value='split_on_presenter_return';assert.throws(()=>buildExport());
intervals=[{start_s:1,end_s:4,end_censored:false}];assert.equal(buildExport().intervals.length,1);
intervals=[{start_s:1,end_s:11,end_censored:false}];assert.throws(()=>buildExport());
intervals=[{start_s:1,end_s:5,end_censored:false},{start_s:4,end_s:6,end_censored:false}];assert.throws(()=>buildExport());
"""
    result=subprocess.run(['node'],input=stub+script+checks,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('field,value',[('unknown_policy','increment on unknown'),('event_iou_threshold',0.1),('seed',True)])
def test_frozen_config_rejects_unimplemented_semantics(field,value):
    import json
    from streaming_video_understanding.evaluation.event_state import validate_config
    config=json.loads(open('configs/event-state-v1.json',encoding='utf-8').read())
    config[field]=value
    with pytest.raises(ValueError):
        validate_config(config)


def test_empty_human_events_require_explicit_negative_confirmation():
    from streaming_video_understanding.evaluation.event_state import score_event_run
    labels=dict(manifest_sha256='m',annotation_source='human_self_reported',reviewer='r',reviewed_full_prefix=True,
                counting_convention='split_on_presenter_return',intervals=[])
    config=dict(end_time_s=10.0,query_times_s=[10.0],event_iou_threshold=0.5)
    with pytest.raises(ValueError,match='empty'):
        score_event_run(dict(policies={}),labels,manifest_sha256='m',config=config)


def test_short_review_clip_cannot_support_full_prefix_review(tmp_path):
    import hashlib
    from streaming_video_understanding.data.event_review import verify_clip
    clip=tmp_path/'clip.mp4';clip.write_bytes(b'clip')
    meta=dict(path=clip.name,sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),duration_s=1.0,cutoff_s=10.0)
    with pytest.raises(ValueError,match='prefix'):
        verify_clip(tmp_path/'manifest.json',dict(review_video=meta))

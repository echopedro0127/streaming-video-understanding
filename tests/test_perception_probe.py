from pathlib import Path
import hashlib
import json

from PIL import Image
import pytest

from streaming_video_understanding.evaluation.perception_probe import infer_perception, summarize_perception, score_perception, CATEGORIES
from streaming_video_understanding.data.perception_review import verify_bundle, write_review_page
from streaming_video_understanding.models.base import ModelResponse


def bundle(tmp_path):
    image=tmp_path/'frame.jpg'
    with Image.new('RGB',(16,12),'red') as im:
        im.save(image)
    row=dict(frame_key='f1',sample_id='s',video_id='v',frame_id=2,timestamp_s=2.0,query_time_s=3.0,
             image='frame.jpg',image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    path=tmp_path/'manifest.json'
    selection=dict(version='perception-probe-v1',cases=[dict(sample_id='s',query_time_s=3.0,frame_ids=[2])])
    path.write_text(json.dumps(dict(version='perception-probe-v1',frames=[row],selection_config=selection,
        selection_config_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest())),encoding='utf-8')
    return path,row


class Backend:
    def __init__(self,fail=False):
        self.calls=[];self.fail=fail
    def infer(self,request):
        self.calls.append(request)
        request.images[0].getpixel((0,0))
        if self.fail:
            raise RuntimeError('model failure')
        return ModelResponse(answer='C',raw_text='C',latency_ms=1)


def test_single_frame_mapping_and_closed_image(tmp_path):
    path,frame=bundle(tmp_path);backend=Backend()
    row=infer_perception(frame,root=tmp_path,backend=backend)
    assert len(backend.calls)==2
    assert row['conditions']['original']['category']=='transition'
    assert row['conditions']['option-cycle']['category']=='uncertain'
    assert all(len(x.images)==1 and x.timestamps_s==() and not x.include_timestamps for x in backend.calls)
    assert backend.calls[0].images[0] is backend.calls[1].images[0]
    with pytest.raises(ValueError):
        backend.calls[0].images[0].getpixel((0,0))
    summary=summarize_perception([row])
    assert summary['accuracy'] is None
    assert summary['semantic_consistent']==0 and summary['label_sticky']==1


def test_model_failure_closes_image(tmp_path):
    path,frame=bundle(tmp_path);backend=Backend(fail=True)
    with pytest.raises(RuntimeError,match='model failure'):
        infer_perception(frame,root=tmp_path,backend=backend)
    with pytest.raises(ValueError):
        backend.calls[0].images[0].getpixel((0,0))


def test_image_hash_and_path_guards(tmp_path):
    path,frame=bundle(tmp_path)
    assert verify_bundle(path)['frames']==[frame]
    bad=dict(frame,image='../outside.jpg')
    data=json.loads(path.read_text());data['frames']=[bad]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='path'):
        verify_bundle(path)
    frame['image_sha256']='wrong'
    data['frames']=[frame];path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):
        verify_bundle(path)


def test_human_scoring_requires_matching_explicit_provenance(tmp_path):
    path,frame=bundle(tmp_path)
    row=infer_perception(frame,root=tmp_path,backend=Backend())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    row['manifest_sha256']=digest
    labels=dict(manifest_sha256=digest,annotation_source='human_self_reported',reviewer='Reviewer',
                reviewed=True,labels=[dict(frame_key='f1',category='transition')])
    score=score_perception([row],labels,manifest_sha256=digest)
    assert score['resolved_frames']==1 and score['conditions']['original']['accuracy']==1
    assert score['binary_balanced_recall'] is None
    labels['labels'][0]['category']='uncertain'
    score=score_perception([row],labels,manifest_sha256=digest)
    assert score['resolved_frames']==0 and score['conditions']['original']['accuracy'] is None
    labels['annotation_source']='assistant_guess'
    with pytest.raises(ValueError,match='human'):
        score_perception([row],labels,manifest_sha256=digest)


def test_review_page_has_no_predictions_or_default_labels(tmp_path):
    path,frame=bundle(tmp_path)
    page=write_review_page(path)
    text=page.read_text(encoding='utf-8')
    assert 'frame.jpg' in text and 'human_self_reported' in text
    assert 'prediction' not in text.lower()
    assert 'checked="checked"' not in text


def test_prediction_manifest_and_mapping_validation(tmp_path):
    from streaming_video_understanding.evaluation.perception_probe import validate_predictions
    path,frame=bundle(tmp_path)
    manifest=verify_bundle(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    row=infer_perception(frame,root=tmp_path,backend=Backend())
    row['manifest_sha256']=digest
    validate_predictions(manifest,[row],manifest_sha256=digest)
    row['conditions']['original']['category']='presenter'
    with pytest.raises(ValueError,match='mapping'):
        validate_predictions(manifest,[row],manifest_sha256=digest)


def test_causal_export_closes_rejected_future_frame(tmp_path):
    from streaming_video_understanding.data.perception_review import prepare_bundle
    from streaming_video_understanding.data.schema import FrameRecord,StreamingSample
    s=StreamingSample('s','v',Path('v.mp4'),'q',('a','b','c','d'),'A',3.0,'','')
    images=[]
    class Reader:
        def read_until(self,*args,**kwargs):
            for i in range(5):
                image=Image.new('RGB',(16,12));images.append(image)
                yield image,FrameRecord(i,float(i))
    config=dict(version='perception-probe-v1',sample_fps=1.0,max_image_side=640,jpeg_quality=95,
                selection_scope='test',cases=[dict(sample_id='s',query_time_s=3.0,frame_ids=[2])])
    with pytest.raises(ValueError,match='future'):
        prepare_bundle(config,[s],reader=Reader(),output_dir=tmp_path/'out',annotation_sha256='a')
    assert not (tmp_path/'out'/'manifest.json').exists()
    for image in images:
        with pytest.raises(ValueError):
            image.getpixel((0,0))


def test_browser_export_logic_requires_explicit_complete_review(tmp_path):
    import shutil,subprocess
    if not shutil.which('node'):
        pytest.skip('Node unavailable for offline reviewer script test')
    path,frame=bundle(tmp_path)
    html=write_review_page(path).read_text(encoding='utf-8')
    script=html.split('<script>')[1].split('</script>')[0]
    stub="""const elements={};const el=()=>({value:'',checked:false,appendChild(){}});
const document={getElementById(id){return elements[id]||(elements[id]=el());},createElement:el};
const localStorage={getItem(){return null;},setItem(){}};const window={};
"""
    assertions="""const assert=require('node:assert/strict');
assert.throws(()=>buildExport());document.getElementById('reviewer').value='human';
document.getElementById('confirmed').checked=true;assert.throws(()=>buildExport());
labels.f1='presenter';const exported=buildExport();assert.equal(exported.labels[0].category,'presenter');
assert.equal(exported.annotation_source,'human_self_reported');assert.equal(exported.reviewed,true);
labels.f1='assistant';assert.throws(()=>buildExport());console.log('review_export_logic_verified');
"""
    result=subprocess.run(['node'],input=stub+script+assertions,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_cli_resume_and_separate_label_scoring(monkeypatch,tmp_path):
    import sys
    import scripts.perception_probe as cli
    path,frame=bundle(tmp_path);output=tmp_path/'pairs.jsonl';loaded=[]
    def factory(**kwargs):
        loaded.append(True);return Backend()
    monkeypatch.setattr(cli,'SmolVLMBackend',factory)
    monkeypatch.setattr(cli,'FROZEN_SELECTION_SHA256',json.loads(path.read_text())['selection_config_sha256'],raising=False)
    monkeypatch.setattr(sys,'argv',['perception_probe.py','run','--manifest',str(path),
        '--config','configs/adaptive-semantic-1fps.yaml','--output',str(output),'--resume'])
    assert cli.main()==0
    before=output.read_bytes()
    assert cli.main()==0 and output.read_bytes()==before and len(loaded)==1
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    labels=tmp_path/'human.json';score=tmp_path/'score.json'
    labels.write_text(json.dumps(dict(manifest_sha256=digest,annotation_source='human_self_reported',
        reviewer='human',reviewed=True,labels=[dict(frame_key='f1',category='transition')])))
    monkeypatch.setattr(sys,'argv',['perception_probe.py','score','--manifest',str(path),
        '--predictions',str(output),'--labels',str(labels),'--output',str(score)])
    assert cli.main()==0
    assert json.loads(score.read_text())['conditions']['original']['accuracy']==1


def test_removed_frame_is_not_accepted_as_complete_bundle(tmp_path):
    path,frame=bundle(tmp_path)
    data=json.loads(path.read_text())
    extra=dict(frame,frame_key='f2',frame_id=1)
    data['frames']=[extra,frame]
    data['selection_config']['cases'][0]['frame_ids']=[1,2]
    data['selection_config_sha256']=hashlib.sha256(json.dumps(data['selection_config'],sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(data))
    verify_bundle(path)
    data['frames'].pop();path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='selection'):
        verify_bundle(path)


def test_inference_cleanup_does_not_mask_model_error(tmp_path):
    path,frame=bundle(tmp_path)
    class Failing:
        def infer(self,request):
            def fail_close():
                raise OSError('close failed')
            request.images[0].close=fail_close
            raise RuntimeError('primary model error')
    with pytest.raises(RuntimeError,match='primary model error') as caught:
        infer_perception(frame,root=tmp_path,backend=Failing())
    assert any('close failed' in note for note in caught.value.__notes__)


def test_export_cleanup_does_not_mask_causal_rejection(tmp_path):
    from streaming_video_understanding.data.perception_review import prepare_bundle
    from streaming_video_understanding.data.schema import FrameRecord,StreamingSample
    s=StreamingSample('s','v',Path('v.mp4'),'q',('a','b','c','d'),'A',3.0,'','')
    class Broken:
        def close(self):
            raise OSError('close failed')
    class Reader:
        def read_until(self,*args,**kwargs):
            yield Broken(),FrameRecord(0,4.0)
    config=dict(version='perception-probe-v1',sample_fps=1.0,max_image_side=640,jpeg_quality=95,
                selection_scope='test',cases=[dict(sample_id='s',query_time_s=3.0,frame_ids=[2])])
    with pytest.raises(ValueError,match='future') as caught:
        prepare_bundle(config,[s],reader=Reader(),output_dir=tmp_path/'out',annotation_sha256='a')
    assert any('close failed' in note for note in caught.value.__notes__)


def test_cli_rejects_other_model_before_loading(monkeypatch,tmp_path):
    import sys
    from dataclasses import replace
    import scripts.perception_probe as cli
    from streaming_video_understanding.utils.config import load_config
    path,frame=bundle(tmp_path)
    config=load_config('configs/adaptive-semantic-1fps.yaml')
    monkeypatch.setattr(cli,'FROZEN_SELECTION_SHA256',json.loads(path.read_text())['selection_config_sha256'],raising=False)
    monkeypatch.setattr(cli,'load_config',lambda *args:replace(config,model=replace(config.model,model_id='other/model')))
    monkeypatch.setattr(cli,'SmolVLMBackend',lambda **kwargs:pytest.fail('unexpected model load'))
    monkeypatch.setattr(sys,'argv',['perception_probe.py','run','--manifest',str(path),'--config','x',
                                  '--output',str(tmp_path/'out.jsonl'),'--resume'])
    with pytest.raises(SystemExit):
        cli.main()

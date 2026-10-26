from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
from streaming_video_understanding.models.base import ModelResponse
from streaming_video_understanding.evaluation.visual_sensitivity import evaluate_visual_sensitivity, summarize_visual_sensitivity


def sample():
    return StreamingSample('s', 'v', Path('v.mp4'), 'How many clips?', ('1.', '2.', '3.', '4.'), 'D', 11.0, 'Counting', 'Concurrent')


def source():
    s = sample()
    return dict(sample_id=s.sample_id, video_id=s.video_id, question=s.question,
                options=list(s.options), ground_truth=s.answer, query_time_s=s.query_time_s,
                conditions={'change-4+4': dict(answer='C', selected_frames=[
                    dict(frame_id=i, timestamp_s=float(i)) for i in (0, 1, 2, 3, 8, 9, 10, 11)])})


class Reader:
    def __init__(self, future=False):
        self.images = []
        self.future = future
    def read_until(self, path, end_time_s):
        for i in range(13 if self.future else 12):
            image = Image.new('RGB', (16, 12), (i, i, i))
            self.images.append(image)
            yield image, FrameRecord(i, float(i))


class Backend:
    def __init__(self, fail=False, invalid=False):
        self.calls = []
        self.fail, self.invalid = fail, invalid
    def infer(self, request):
        pixels = [im.getpixel((0, 0))[0] for im in request.images]
        self.calls.append((request, pixels))
        if self.fail:
            raise RuntimeError('model failure')
        return ModelResponse(raw_text='?', answer='INVALID_MCQ', latency_ms=1, parse_valid=False) if self.invalid else ModelResponse(raw_text='C', answer='C', latency_ms=1)


def test_matched_inputs_mapping_and_cleanup():
    reader, backend = Reader(), Backend()
    row = evaluate_visual_sensitivity(sample(), source(), reader=reader, backend=backend)
    assert row['mode_order'] == ['original', 'blank', 'reverse', 'option-cycle']
    assert [pixels for _, pixels in backend.calls] == [[0,1,2,3,8,9,10,11], [127]*8, [11,10,9,8,3,2,1,0], [0,1,2,3,8,9,10,11]]
    for request, _ in backend.calls:
        assert request.timestamps_s == () and not request.include_timestamps
        assert request.allow_invalid_answer
    assert backend.calls[-1][0].options == ('2.', '3.', '4.', '1.')
    assert row['conditions']['option-cycle']['mapped_answer'] == 'D'
    assert row['conditions']['option-cycle']['count'] == 4
    assert row['conditions']['blank']['input_source_frames'] == []
    assert row['conditions']['blank']['synthetic_images'] == 8
    assert row['conditions']['reverse']['input_source_frames'][0]['frame_id'] == 11
    for image in [*reader.images, *(im for request, _ in backend.calls for im in request.images)]:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    summary = summarize_visual_sensitivity([row], n_bootstrap=20)
    assert summary['conditions']['option-cycle']['correct'] == 1
    assert summary['comparisons']['option-cycle']['content_consistent'] == 0
    assert summary['comparisons']['option-cycle']['label_sticky'] == 1


@pytest.mark.parametrize('fail,invalid', [(True,False), (False,True)])
def test_error_and_invalid_output_handling(fail, invalid):
    reader, backend = Reader(), Backend(fail=fail, invalid=invalid)
    if fail:
        with pytest.raises(RuntimeError, match='model failure'):
            evaluate_visual_sensitivity(sample(), source(), reader=reader, backend=backend)
    else:
        row = evaluate_visual_sensitivity(sample(), source(), reader=reader, backend=backend)
        summary = summarize_visual_sensitivity([row], n_bootstrap=20)
        assert all(x['correct']==0 and x['parse_valid_count']==0 for x in summary['conditions'].values())
    for image in [*reader.images, *(im for request, _ in backend.calls for im in request.images)]:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_future_unselected_frame_is_rejected_and_closed():
    reader, backend = Reader(future=True), Backend()
    with pytest.raises(ValueError, match='future'):
        evaluate_visual_sensitivity(sample(), source(), reader=reader, backend=backend)
    assert not backend.calls
    for image in reader.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


@pytest.mark.parametrize('field,value', [('question','changed'), ('options',['4.','3.','2.','1.']), ('sample_id','other'), ('query_time_s',12)])
def test_source_mismatch_fails_before_decoding(field,value):
    saved=source(); saved[field]=value
    reader=Reader()
    with pytest.raises(ValueError, match='source'):
        evaluate_visual_sensitivity(sample(), saved, reader=reader, backend=Backend())
    assert not reader.images


def test_missing_selected_frame_releases_images():
    saved=source(); saved['conditions']['change-4+4']['selected_frames'][-1]['timestamp_s']=10.5
    reader=Reader()
    with pytest.raises(ValueError, match='timestamp'):
        evaluate_visual_sensitivity(sample(), saved, reader=reader, backend=Backend())
    for image in reader.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_cli_four_way_resume_and_changed_source_rejection(monkeypatch, tmp_path):
    import hashlib,json,sys
    import scripts.run_visual_sensitivity as cli
    from streaming_video_understanding.utils.config import load_config,configuration_id
    samples=[replace(sample(),sample_id=f's{i:02}',video_id=f'v{i:02}') for i in range(20)]
    annotation,split,sources,output=[tmp_path/name for name in ('ann.jsonl','split.json','source.jsonl','out.jsonl')]
    annotation.write_text('frozen',encoding='utf-8')
    split.write_text(json.dumps({'sample_ids':[s.sample_id for s in samples]}),encoding='utf-8')
    proto=dict(version='change-memory-v1',source_config_id=configuration_id(load_config('configs/adaptive-semantic-1fps.yaml'),policy_id='recent-8',backend='smolvlm'),
               annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),split_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
    source_id='change-memory-v1-'+hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest()[:12]
    saved=[]
    for s in samples:
        row=source();row.update(sample_id=s.sample_id,video_id=s.video_id,protocol=proto,config_id=source_id,record_type='change-memory-v1');saved.append(row)
    sources.write_text('\n'.join(json.dumps(x) for x in saved)+'\n',encoding='utf-8')
    monkeypatch.setattr(cli,'FROZEN_SOURCE_ID',source_id,raising=False)
    monkeypatch.setattr(cli,'FROZEN_SOURCE_SHA256',hashlib.sha256(sources.read_bytes()).hexdigest(),raising=False)
    monkeypatch.setattr(cli,'_load_samples',lambda *args,**kwargs:(samples,Reader()))
    loaded=[]
    def factory(**kwargs):
        loaded.append(True);return Backend()
    monkeypatch.setattr(cli,'SmolVLMBackend',factory)
    monkeypatch.setattr(sys,'argv',['run_visual_sensitivity.py','--config','configs/adaptive-semantic-1fps.yaml',
        '--data-root',str(tmp_path),'--annotations',str(annotation),'--split',str(split),
        '--source',str(sources),'--output',str(output),'--limit','4','--resume'])
    assert cli.main()==0
    before=output.read_bytes()
    records=list(map(json.loads,before.decode().splitlines()))
    assert len(records)==4 and len({x['mode_order'][0] for x in records})==4
    assert cli.main()==0 and output.read_bytes()==before and len(loaded)==1
    alternate=tmp_path/'alternate.jsonl'
    new_proto=dict(proto,change_threshold=0.09)
    new_id='change-memory-v1-'+hashlib.sha256(json.dumps(new_proto,sort_keys=True).encode()).hexdigest()[:12]
    alternate.write_text('\n'.join(json.dumps(dict(x,protocol=new_proto,config_id=new_id)) for x in saved)+'\n',encoding='utf-8')
    old_argv=list(sys.argv)
    sys.argv[sys.argv.index('--source')+1]=str(alternate)
    sys.argv[sys.argv.index('--output')+1]=str(tmp_path/'fresh.jsonl')
    with pytest.raises(SystemExit):
        cli.main()
    assert len(loaded)==1
    sys.argv[:]=old_argv
    sources.write_text(sources.read_text()+'\n',encoding='utf-8')
    with pytest.raises(SystemExit):
        cli.main()
    assert len(loaded)==1

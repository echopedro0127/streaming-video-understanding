from copy import deepcopy

import pytest
from PIL import Image

from streaming_video_understanding.evaluation.black_frame_control import is_exact_black, replay_gated


def test_exact_zero_does_not_suppress_dark_visible_pixel():
    with Image.new('RGB', (4, 4)) as image:
        assert is_exact_black(image)
        image.putpixel((3, 3), (0, 0, 1))
        assert not is_exact_black(image)


def test_black_inside_event_holds_active_and_preserves_inputs():
    rows = [dict(timestamp_s=float(i), category=c) for i, c in enumerate(
        ['inserted_scene', 'presenter', 'inserted_scene', 'presenter', 'presenter'])]
    before = deepcopy(rows)
    black = [True, False, False, True, False]
    traces = replay_gated(rows, black)
    assert [r['count'] for r in traces] == [0, 0, 1, 1, 1]
    assert traces[3]['active']
    assert traces[4]['closed_event']['end_s'] == 4.0
    assert rows == before
    for n in range(1, len(rows) + 1):
        assert replay_gated(rows[:n], black[:n]) == traces[:n]


def test_gate_rejects_missing_flags_and_noncausal_rows():
    with pytest.raises(ValueError, match='one flag'):
        replay_gated([dict(timestamp_s=0, category='presenter')], [])
    with pytest.raises(ValueError, match='strictly increasing'):
        replay_gated([dict(timestamp_s=1, category='presenter'),
                      dict(timestamp_s=0, category='inserted_scene')], [False, False])


def test_replay_rejects_replaced_source_before_scoring(tmp_path, monkeypatch):
    from scripts import black_frame_control as cli
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    configs = tmp_path / 'configs/event-state-validation-v1'
    configs.mkdir(parents=True)
    (configs / 'sample79.json').write_text('{}', encoding='utf-8')
    base = tmp_path / 'results/event-state/validation-3/sample79'
    base.mkdir(parents=True)
    for name in ('manifest.json', 'predictions.jsonl', 'event-labels-submitted-v1.json'):
        (base / name).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hashes differ'):
        cli.evaluate_case('79')


def test_cli_refuses_existing_output_without_touching_it(tmp_path, monkeypatch):
    from scripts import black_frame_control as cli
    output = tmp_path / 'existing.json'
    output.write_text('keep original', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['black_frame_control.py', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert output.read_text(encoding='utf-8') == 'keep original'

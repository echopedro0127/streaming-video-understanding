from pathlib import Path

import pytest
from PIL import Image

from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
from streaming_video_understanding.evaluation.context_probe import (
    CausalTimeline, choose_review_cases, evaluate_context_sample, summarize_context_probe,
)
from streaming_video_understanding.models.base import ModelResponse


def sample(sample_id="s1", video_id="v1", time=80):
    return StreamingSample(sample_id=sample_id, video_id=video_id, video_path=Path("video.mp4"),
                           question="How many times?", options=("0", "1", "2", "3"),
                           answer="A", query_time_s=time, task_type="unused", temporal_clue_type="unused")


def test_timeline_is_causal_and_retains_only_one_canvas(tmp_path):
    timeline = CausalTimeline(tmp_path, sample_id="s1", query_time_s=4, page_size=4)
    for index in range(5):
        image = Image.new("RGB", (16, 16))
        timeline.add(image, FrameRecord(index, float(index)))
        image.close()
    paths = timeline.finish()
    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
    assert timeline.canvas is None
    bad = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match="future"):
        timeline.add(bad, FrameRecord(5, 5.0))
    bad.close()


@pytest.mark.parametrize("time", [float("nan"), -1.0, True])
def test_timeline_rejects_invalid_query_time(tmp_path, time):
    with pytest.raises(ValueError):
        CausalTimeline(tmp_path, sample_id="s", query_time_s=time)


def test_review_sampling_is_time_stratified_and_answer_independent():
    from dataclasses import replace
    rows = [sample("a", time=16), sample("b", time=50), sample("c", time=86),
            sample("d", time=178), sample("e", time=810)]
    chosen = choose_review_cases(rows)
    assert set(chosen) == {"short", "medium", "long"}
    assert chosen == choose_review_cases([replace(row, answer="D") for row in reversed(rows)])


class Reader:
    def __init__(self):
        self.images = []
    def read_until(self, video_path, end_time_s):
        for index in range(int(end_time_s) + 1):
            image = Image.new("RGB", (4, 4))
            self.images.append(image)
            yield image, FrameRecord(index, float(index))


class Backend:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail
    def infer(self, request):
        assert request.readout_mode == "mcq"
        assert request.allow_invalid_answer
        request.images[0].getpixel((0, 0))
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("model failed")
        return ModelResponse(raw_text="A", answer="A", latency_ms=1.0)


def test_evaluator_uses_three_eight_frame_conditions_and_releases_all_images():
    reader, backend = Reader(), Backend()
    row = evaluate_context_sample(sample(), reader=reader, backend=backend, order_offset=1)
    assert row["mode_order"] == ["time-bin-60s-8", "time-bin-prefix-8", "recent-8"]
    assert len(backend.requests) == 3
    assert row["inference_calls"] == 3
    assert row["peak_retained_images"] <= 24
    assert all(len(condition["selected_frames"]) == 8 for condition in row["conditions"].values())
    assert [frame["frame_id"] for frame in row["conditions"]["recent-8"]["selected_frames"]] == list(range(73, 81))
    assert row["conditions"]["time-bin-prefix-8"]["temporal_span_s"] == 70.0
    for image in reader.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_model_error_releases_bank_images_and_preserves_exception():
    reader = Reader()
    with pytest.raises(RuntimeError, match="model failed"):
        evaluate_context_sample(sample(), reader=reader, backend=Backend(fail=True))
    for image in reader.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_summary_uses_same_qa_denominator_for_each_context():
    row = evaluate_context_sample(sample(), reader=Reader(), backend=Backend())
    result = summarize_context_probe([row], n_bootstrap=20)
    assert result["num_samples"] == 1
    assert result["conditions"]["recent-8"]["accuracy"] == 1.0
    assert result["paired"]["time-bin-prefix-8_vs_recent-8"]["accuracy_delta"] == 0.0


def test_runner_smoke_resume_preserves_three_mode_order_and_does_not_load_model(monkeypatch, tmp_path):
    import json
    import sys
    import scripts.run_context_coverage as cli
    rows = [sample(f"s{index:02}", f"v{index:02}") for index in range(20)]
    constructed = []
    def factory(**kwargs):
        constructed.append(True)
        return Backend()
    monkeypatch.setattr(cli, "SmolVLMBackend", factory)
    monkeypatch.setattr(cli, "_load_samples", lambda *args, **kwargs: (rows, Reader()))
    split, annotation, output = tmp_path / "split.json", tmp_path / "ann.jsonl", tmp_path / "out.jsonl"
    split.write_text(json.dumps({"sample_ids": [row.sample_id for row in rows]}), encoding="utf-8")
    annotation.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_context_coverage.py", "--config", "configs/adaptive-semantic-1fps.yaml",
                                     "--data-root", str(tmp_path), "--annotations", str(annotation),
                                     "--split", str(split), "--output", str(output), "--limit", "3", "--resume"])
    assert cli.main() == 0
    before = output.read_bytes()
    saved = [json.loads(line) for line in before.decode().splitlines()]
    assert len(saved) == 3
    assert len({row["mode_order"][0] for row in saved}) == 3
    assert all(row["inference_calls"] == 3 and row["peak_retained_images"] <= 24 for row in saved)
    assert cli.main() == 0
    assert len(constructed) == 1
    assert output.read_bytes() == before

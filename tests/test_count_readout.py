from dataclasses import replace
from pathlib import Path

import pytest

from streaming_video_understanding.models.counting import build_count_prompt, parse_count, numeric_options
from streaming_video_understanding.evaluation.count_readout import PairedCountBackend, select_counting_cases, summarize_count_pairs
from streaming_video_understanding.models.base import ModelRequest, ModelResponse


@pytest.mark.parametrize("text,value", [("0", 0), (" 12. ", 12), ("3 times!", 3),
                                       ("1 time", 1), ("2 or 3", None), ("C", None),
                                       ("-1", None), ("2.5", None), ("There are 3", None)])
def test_strict_frozen_count_parser(text, value):
    assert parse_count(text) == value


def test_numeric_options_require_unique_nonnegative_integer_values():
    assert numeric_options(("2.", "0.", "4.", "1.")) == (2, 0, 4, 1)
    for options in (("2", "02", "3", "4"), ("cat", "0", "1", "2"), ("0", "1")):
        with pytest.raises(ValueError):
            numeric_options(options)


def test_count_prompt_contains_question_but_no_options_or_letter_instruction():
    prompt = build_count_prompt("How many times has she waved?")
    assert "How many times has she waved?" in prompt
    assert "Options:" not in prompt
    assert "A, B, C" not in prompt


def request():
    from PIL import Image
    return ModelRequest(images=(Image.new("RGB", (4, 4)),), timestamps_s=(1.0,),
                        question="How many times?", options=("0.", "1.", "2.", "3."),
                        query_time_s=2.0, include_timestamps=False)


@pytest.mark.parametrize("count_first", [False, True])
def test_pair_reuses_exact_images_and_has_no_ground_truth_argument(count_first):
    class Fake:
        def __init__(self):
            self.calls = []
        def infer(self, item):
            self.calls.append(item)
            if item.readout_mode == "count":
                return ModelResponse(raw_text="9", answer="OUT_OF_OPTIONS", latency_ms=2.0,
                                     predicted_count=9, parse_valid=True)
            return ModelResponse(raw_text="B", answer="B", latency_ms=3.0)
    fake = Fake()
    pair = PairedCountBackend(fake, count_first=count_first)
    original = request()
    response = pair.infer(original)
    assert response.answer == "B"
    assert len(fake.calls) == 2
    assert all(item.images is original.images for item in fake.calls)
    assert all(item.timestamps_s is original.timestamps_s for item in fake.calls)
    assert [item.readout_mode for item in fake.calls] == (["count", "mcq"] if count_first else ["mcq", "count"])
    assert pair.responses["count"].predicted_count == 9


def sample(sample_id, video_id, answer="A"):
    from streaming_video_understanding.data.schema import StreamingSample
    return StreamingSample(sample_id=sample_id, video_id=video_id, video_path=Path(video_id + "/video.mp4"),
                           question="How many times has she waved so far?", options=("0", "1", "2", "3"),
                           answer=answer, query_time_s=5, task_type="unused", temporal_clue_type="unused")


def test_sampling_is_video_disjoint_answer_independent_and_order_independent():
    rows = [sample("a", "v1"), sample("b", "v1"), sample("c", "v2"), sample("d", "v3")]
    chosen = select_counting_cases(rows, excluded_video_ids={"v2"}, seed="20261026")
    second = select_counting_cases([replace(row, answer="D") for row in reversed(rows)],
                                   excluded_video_ids={"v2"}, seed="20261026")
    assert chosen == second
    assert len(chosen) == 2
    assert "c" not in chosen
    assert len(set(chosen) & {"a", "b"}) == 1


def test_summary_counts_invalid_and_outside_options_without_nearest_choice_repair():
    rows = [dict(sample_id="s1", video_id="v1", ground_truth_count=2, option_counts=[0, 1, 2, 3],
                 methods={"mcq": dict(count=2, model_latency_ms=10.0), "count": dict(count=9, model_latency_ms=2.0)}),
            dict(sample_id="s2", video_id="v2", ground_truth_count=1, option_counts=[0, 1, 2, 3],
                 methods={"mcq": dict(count=0, model_latency_ms=30.0), "count": dict(count=None, model_latency_ms=8.0)})]
    report = summarize_count_pairs(rows, n_bootstrap=50)
    assert report["methods"]["mcq"]["accuracy"] == 0.5
    assert report["methods"]["count"]["accuracy"] == 0.0
    assert report["methods"]["count"]["parse_valid_count"] == 1
    assert report["methods"]["count"]["outside_option_count"] == 1
    assert report["methods"]["count"]["valid_count_mae"] == 7.0
    assert report["methods"]["count"]["valid_count_mae_denominator"] == 1
    assert report["paired"]["accuracy_delta"] == -0.5
    assert report["methods"]["mcq"]["p50_model_latency_ms"] == 20.0
    assert report["methods"]["count"]["p95_model_latency_ms"] == pytest.approx(7.7)


@pytest.mark.parametrize("mode,text,answer,count,valid", [
    ("mcq", "C", "C", None, True),
    ("count", "2 times", "C", 2, True),
    ("count", "9", "OUT_OF_OPTIONS", 9, True),
    ("count", "two or three", "INVALID_COUNT", None, False),
    ("mcq", "I do not know", "INVALID_MCQ", None, False),
])
def test_real_backend_branch_uses_choice_free_prompt_and_frozen_parser(mode, text, answer, count, valid):
    backend = backend_for_text(text)
    response = backend.infer(replace(request(), readout_mode=mode, allow_invalid_answer=(mode == "mcq" and not valid)))
    assert response.answer == answer
    assert response.predicted_count == count
    assert response.parse_valid == valid
    assert ("Options:" in backend.processor.prompt) == (mode == "mcq")
    assert ("A. 0." in backend.processor.prompt) == (mode == "mcq")


def backend_for_text(text):
    from types import SimpleNamespace
    from streaming_video_understanding.models.smolvlm import SmolVLMBackend

    class Inputs(dict):
        def to(self, device):
            return self
    class Tokens:
        def __getitem__(self, key):
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.prompt = messages[0]["content"][-1]["text"]
            return Inputs(input_ids=SimpleNamespace(shape=(1, 1)))
        def batch_decode(self, ids, **kwargs):
            return [text if isinstance(text, str) else text["mcq" if "Options:" in self.prompt else "count"]]
    class FakeOOM(Exception):
        pass
    backend = object.__new__(SmolVLMBackend)
    backend._torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=FakeOOM))
    backend.model = SimpleNamespace(device="cpu", generate=lambda **kwargs: Tokens())
    backend.processor = Processor()
    backend.max_new_tokens = 8
    return backend


def test_default_mcq_parser_still_raises_on_invalid_output():
    with pytest.raises(ValueError):
        backend_for_text("I do not know").infer(request())


@pytest.mark.parametrize("count_first", [False, True])
def test_pilot_retains_invalid_mcq_and_completes_both_generations(count_first):
    pair = PairedCountBackend(backend_for_text({"mcq": "I do not know", "count": "3"}), count_first=count_first)
    response = pair.infer(request())
    assert response.answer == "INVALID_MCQ"
    assert response.parse_valid is False
    assert pair.responses["count"].predicted_count == 3
    assert len(pair.responses) == 2


def test_runner_resumes_invalid_mcq_as_completed_and_does_not_reload_model(monkeypatch, tmp_path):
    import json
    import sys
    from PIL import Image
    import scripts.run_count_readout as cli
    from streaming_video_understanding.data.schema import FrameRecord

    rows = [sample(f"s{index:02}", f"v{index:02}") for index in range(20)]
    images = []
    class Reader:
        def read_until(self, video_path, end_time_s):
            image = Image.new("RGB", (4, 4))
            images.append(image)
            yield image, FrameRecord(0, 0.0)
    constructed = []
    def factory(**kwargs):
        constructed.append(True)
        return backend_for_text({"mcq": "I do not know", "count": "3"})
    monkeypatch.setattr(cli, "SmolVLMBackend", factory)
    monkeypatch.setattr(cli, "_load_samples", lambda *args, **kwargs: (rows, Reader()))
    split, annotation, output = tmp_path / "split.json", tmp_path / "annotations.jsonl", tmp_path / "out.jsonl"
    split.write_text(json.dumps({"sample_ids": [row.sample_id for row in rows]}), encoding="utf-8")
    annotation.write_text("frozen annotation fingerprint", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_count_readout.py", "--config", "configs/adaptive-semantic-1fps.yaml",
                                     "--data-root", str(tmp_path), "--annotations", str(annotation),
                                     "--split", str(split), "--output", str(output), "--limit", "2", "--resume"])
    assert cli.main() == 0
    before = output.read_bytes()
    results = [json.loads(line) for line in before.decode().splitlines()]
    assert len(results) == 2
    assert all(row["methods"]["mcq"]["count"] is None for row in results)
    assert all(not row["methods"]["mcq"]["parse_valid"] for row in results)
    assert results[0]["generation_order"] == ["mcq", "count"]
    assert results[1]["generation_order"] == ["count", "mcq"]
    for image in images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    assert cli.main() == 0
    assert output.read_bytes() == before
    assert len(constructed) == 1

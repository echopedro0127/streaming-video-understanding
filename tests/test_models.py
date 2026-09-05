from pathlib import Path

import pytest
from PIL import Image

from streaming_video_understanding.models.base import ModelRequest
from streaming_video_understanding.models.mock import MockVLMBackend
from streaming_video_understanding.models.prompt import build_prompt, parse_answer_letter


def test_timestamp_prompt_contains_absolute_and_relative_time():
    prompt = build_prompt(
        question="What is the person holding?",
        options=("a cup", "a ball", "a book", "a phone"),
        timestamps_s=(12.5, 18.0),
        query_time_s=20.0,
        include_timestamps=True,
    )
    assert "absolute_time=12.50s" in prompt
    assert "relative_to_query=-2.00s" in prompt
    assert "answer with only one letter" in prompt.lower()


def test_parse_answer_accepts_only_unambiguous_letter():
    assert parse_answer_letter("B") == "B"
    assert parse_answer_letter("The answer is C.") == "C"
    with pytest.raises(ValueError, match="ambiguous"):
        parse_answer_letter("A or B")


def test_mock_backend_returns_configured_answer_and_records_request():
    backend = MockVLMBackend(answer="D")
    request = ModelRequest(
        images=(Image.new("RGB", (8, 8)),),
        timestamps_s=(3.0,),
        question="What is visible?",
        options=("cat", "dog", "ball", "car"),
        query_time_s=4.0,
        include_timestamps=False,
    )

    response = backend.infer(request)

    assert response.answer == "D"
    assert backend.call_count == 1
    assert backend.last_request == request

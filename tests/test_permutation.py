from pathlib import Path

from streaming_video_understanding.data.schema import StreamingSample
from streaming_video_understanding.evaluation.permutation import permute_sample


def _sample() -> StreamingSample:
    return StreamingSample(
        sample_id="sample-1",
        video_id="video-1",
        video_path=Path("video-1.mp4"),
        question="Which object is visible?",
        options=("a cup", "a ball", "a book", "a phone"),
        answer="C",
        query_time_s=10.0,
        task_type="Object Recognition",
        temporal_clue_type="Prior",
        frames_required="single",
    )


def test_option_permutation_is_deterministic_and_remaps_ground_truth():
    original = _sample()

    first = permute_sample(original, seed=42)
    second = permute_sample(original, seed=42)

    assert first.sample.options == second.sample.options
    assert first.sample.answer == second.sample.answer
    assert first.sample.options != original.options
    original_correct_option = original.options[ord(original.answer) - ord("A")]
    new_correct_index = first.sample.options.index(original_correct_option)
    assert first.sample.answer == chr(ord("A") + new_correct_index)
    assert first.original_answer == original.answer
    assert first.permutation != (0, 1, 2, 3)


def test_option_permutation_does_not_mutate_original_sample():
    original = _sample()

    permuted = permute_sample(original, seed=7)

    assert original.options == ("a cup", "a ball", "a book", "a phone")
    assert original.answer == "C"
    assert permuted.sample.sample_id == original.sample_id

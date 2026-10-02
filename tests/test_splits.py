from pathlib import Path

from streaming_video_understanding.data.schema import StreamingSample
from streaming_video_understanding.evaluation.splits import (
    build_deterministic_split,
    load_split_sample_ids,
    write_split_files,
)


TASK_TYPES = (
    "Counting",
    "Action Recognition",
    "Object Recognition",
    "Attribute Recognition",
    "Event Understanding",
    "Text-Rich Understanding",
    "Causal Reasoning",
    "Spatial Understanding",
)


def _samples() -> list[StreamingSample]:
    samples: list[StreamingSample] = []
    for index in range(500):
        samples.append(
            StreamingSample(
                sample_id=f"sample-{index:03d}",
                video_id=f"video-{index // 5:03d}",
                video_path=Path(f"video-{index // 5:03d}.mp4"),
                question="Which option is visible?",
                options=("one", "two", "three", "four"),
                answer="ABCD"[index % 4],
                query_time_s=float(index % 90),
                task_type=TASK_TYPES[index % len(TASK_TYPES)],
                temporal_clue_type="Prior" if index % 3 else "Concurrent",
                frames_required="multiple" if index % 2 else "single",
            )
        )
    return samples


def test_deterministic_split_is_reproducible_and_covers_required_dimensions():
    samples = _samples()

    first = build_deterministic_split(samples, validation_size=100, seed=42)
    second = build_deterministic_split(samples, validation_size=100, seed=42)

    assert [sample.sample_id for sample in first.validation] == [
        sample.sample_id for sample in second.validation
    ]
    assert len(first.validation) == 100
    assert len(first.final_test) == 400
    assert {
        sample.sample_id for sample in first.validation
    }.isdisjoint(sample.sample_id for sample in first.final_test)
    assert {
        sample.sample_id for sample in first.validation
    } | {sample.sample_id for sample in first.final_test} == {
        sample.sample_id for sample in samples
    }

    validation = first.validation
    assert {sample.task_type for sample in validation} == set(TASK_TYPES)
    assert sum(sample.frames_required == "multiple" for sample in validation) >= 30
    assert sum(sample.frames_required == "single" for sample in validation) >= 20
    assert sum(sample.temporal_clue_type == "Prior" for sample in validation) >= 10
    assert sum(sample.temporal_clue_type == "Concurrent" for sample in validation) >= 20
    assert {sample.answer for sample in validation} == {"A", "B", "C", "D"}


def test_split_writer_emits_public_metadata_without_video_paths(tmp_path: Path):
    split = build_deterministic_split(_samples(), validation_size=100, seed=42)

    paths = write_split_files(split, tmp_path)

    assert paths["validation"].is_file()
    assert paths["final_test"].is_file()
    assert paths["stats"].is_file()
    assert "video_path" not in paths["validation"].read_text(encoding="utf-8")


def test_split_loader_reads_sample_ids(tmp_path: Path):
    path = tmp_path / "validation.json"
    path.write_text('{"sample_ids": ["s1", "s2"]}', encoding="utf-8")

    assert load_split_sample_ids(path) == {"s1", "s2"}

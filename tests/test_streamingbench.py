from pathlib import Path

import pytest

from streaming_video_understanding.data.streamingbench import load_streamingbench_csv


def test_loader_preserves_options_and_query_time(tmp_path: Path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "question_id,task_type,question,time_stamp,answer,options,frames_required,temporal_clue_type,video\n"
        "v1_1,Object Perception,What is visible?,00:00:04,B,\"['A. cat', 'B. dog', 'C. ball', 'D. car']\",single,Prior,data/v1.mp4\n",
        encoding="utf-8",
    )

    rows = load_streamingbench_csv(csv_path)

    assert rows[0].sample_id == "v1_1"
    assert rows[0].query_time_s == 4.0
    assert rows[0].options == ("cat", "dog", "ball", "car")
    assert rows[0].answer == "B"
    assert rows[0].frames_required == "single"


def test_loader_uses_sample_directory_as_video_id(tmp_path: Path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "question_id,task_type,question,time_stamp,answer,options,frames_required,temporal_clue_type,video\n"
        "v1_1,Object Perception,What is visible?,00:00:04,B,\"['A. cat', 'B. dog']\",single,Prior,sample_7/video.mp4\n",
        encoding="utf-8",
    )

    rows = load_streamingbench_csv(csv_path)

    assert rows[0].video_id == "sample_7"


def test_json_loader_accepts_annotation_records(tmp_path: Path):
    path = tmp_path / "rows.json"
    path.write_text(
        '[{"question_id": "v1_1", "task_type": "Object Perception", '
        '"question": "What is visible?", "time_stamp": "4", "answer": "B", '
        '"options": ["A. cat", "B. dog", "C. ball", "D. car"], '
        '"video": "data/v1.mp4"}]',
        encoding="utf-8",
    )

    from streaming_video_understanding.data.streamingbench import load_streamingbench_file

    rows = load_streamingbench_file(path)

    assert rows[0].answer == "B"
    assert rows[0].options == ("cat", "dog", "ball", "car")


def test_loader_reports_missing_required_column(tmp_path: Path):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        "question_id,question,time_stamp,answer,options,video\n"
        "v1_1,What is visible?,00:00:04,B,\"['A. cat']\",data/v1.mp4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task_type"):
        load_streamingbench_csv(csv_path)

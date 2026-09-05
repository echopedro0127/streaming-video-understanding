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


def test_loader_reports_missing_required_column(tmp_path: Path):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        "question_id,question,time_stamp,answer,options,video\n"
        "v1_1,What is visible?,00:00:04,B,\"['A. cat']\",data/v1.mp4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task_type"):
        load_streamingbench_csv(csv_path)

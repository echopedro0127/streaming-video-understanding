import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_annotations.py"
SPEC = importlib.util.spec_from_file_location("download_annotations", SCRIPT_PATH)
DOWNLOAD_ANNOTATIONS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DOWNLOAD_ANNOTATIONS)


def test_api_row_is_mapped_to_local_video_path():
    row = {
        "question_id": "Real-Time Visual Understanding_sample_7_3",
        "task_type": "Object Perception",
        "question": "What is visible?",
        "time_stamp": "00:02:18",
        "answer": "C",
        "options": "['A. one', 'B. two', 'C. three', 'D. four']",
        "frames_required": "single",
        "temporal_clue_type": "Prior",
    }

    mapped = DOWNLOAD_ANNOTATIONS._map_row(row, max_sample=50)

    assert mapped["video"] == "sample_7/video.mp4"
    assert mapped["question_id"] == row["question_id"]


def test_rows_outside_bounded_subset_are_skipped():
    row = {
        "question_id": "Real-Time Visual Understanding_sample_51_1",
        "task_type": "Object Perception",
        "question": "What is visible?",
        "time_stamp": "00:02:18",
        "answer": "C",
        "options": "['A. one', 'B. two', 'C. three', 'D. four']",
    }

    assert DOWNLOAD_ANNOTATIONS._map_row(row, max_sample=50) is None

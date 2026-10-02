import json
from pathlib import Path

from streaming_video_understanding.evaluation.metrics import summarize_records
from streaming_video_understanding.evaluation.profiling import RuntimeProfileStore


def test_runtime_profile_store_writes_stage_timing_record(tmp_path: Path):
    path = tmp_path / "profile.jsonl"
    store = RuntimeProfileStore(path)

    store.append(
        {
            "sample_id": "sample-1",
            "config_id": "recent-8",
            "decode_ms": 1.0,
            "sampling_ms": 2.0,
            "memory_ms": 3.0,
            "processor_ms": 4.0,
            "generation_ms": 5.0,
            "write_ms": 6.0,
            "total_ms": 21.0,
        }
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["sample_id"] == "sample-1"
    assert record["decode_ms"] == 1.0
    assert record["total_ms"] == 21.0


def test_runtime_profile_store_rejects_missing_identity(tmp_path: Path):
    store = RuntimeProfileStore(tmp_path / "profile.jsonl")

    try:
        store.append({"decode_ms": 1.0})
    except ValueError as exc:
        assert "sample_id" in str(exc)
    else:
        raise AssertionError("missing profile identity should fail")


def test_metrics_summarize_stage_timings():
    summary = summarize_records(
        [
            {
                "decode_ms": 1.0,
                "sampling_ms": 2.0,
                "memory_ms": 3.0,
                "processor_ms": 4.0,
                "generation_ms": 5.0,
                "write_ms": 6.0,
                "total_ms": 21.0,
            },
            {
                "decode_ms": 3.0,
                "sampling_ms": 4.0,
                "memory_ms": 5.0,
                "processor_ms": 6.0,
                "generation_ms": 7.0,
                "write_ms": 8.0,
                "total_ms": 33.0,
            },
        ]
    )

    assert summary["stage_timing"]["decode_ms"]["mean_ms"] == 2.0
    assert summary["stage_timing"]["total_ms"]["p50_ms"] == 27.0

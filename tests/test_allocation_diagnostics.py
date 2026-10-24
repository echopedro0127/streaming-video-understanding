import pytest

from streaming_video_understanding.analysis.allocation import allocation_diagnosis, evidence_review_queue


def _row(sample, video, left, right, question="What is visible now?"):
    group = ("both_correct" if left and right else "memory_hurt" if left
             else "memory_helped" if right else "both_wrong")
    return dict(sample_id=sample, video_id=video, question=question,
                recent_correct=left, option_correct=right, group=group,
                query_time_s=20.0, options=["cat", "dog", "ball", "car"])


def test_cluster_bootstrap_is_paired_video_based_and_order_independent():
    rows = [_row(f"s{i}", "sample_1", False, True) for i in range(5)]
    rows += [_row("s5", "sample_2", True, False, "What happened before?")]
    report = allocation_diagnosis(rows, seed=42, n_bootstrap=400)
    assert report == allocation_diagnosis(list(reversed(rows)), seed=42, n_bootstrap=400)
    assert report["n_videos"] == 2
    assert report["n_pairs"] == 6
    assert report["accuracy_delta"] == pytest.approx(4 / 6)
    assert report["video_macro_accuracy_delta"] == 0.0
    assert report["bootstrap"]["unit"] == "video"
    assert report["bootstrap"]["ci95"] == [-1.0, 1.0]
    assert report["route_proxy_groups"]["current"]["helped"] == 5
    assert report["route_proxy_groups"]["history"]["hurt"] == 1
    assert "not evidence labels" in report["grouping_note"]


@pytest.mark.parametrize("rows,n", [([], 20), ([_row("s", "v", True, True)], 0),
                                   ([_row("s", "v", True, True)] * 2, 20)])
def test_diagnosis_rejects_invalid_inputs(rows, n):
    with pytest.raises(ValueError):
        allocation_diagnosis(rows, n_bootstrap=n)


def test_constant_pairs_have_zero_delta_and_interval():
    report = allocation_diagnosis([_row("s1", "v1", True, True),
                                   _row("s2", "v2", False, False)], n_bootstrap=50)
    assert report["accuracy_delta"] == 0.0
    assert report["bootstrap"]["ci95"] == [0.0, 0.0]


def test_review_queue_is_blinded_balanced_deterministic_and_unannotated():
    rows = [_row(f"s{i}", f"v{i}", i % 2 == 0, i % 3 == 0) for i in range(30)]
    queue = evidence_review_queue(rows, limit=12, seed=7)
    assert queue == evidence_review_queue(list(reversed(rows)), limit=12, seed=7)
    assert len(queue) == len({row["sample_id"] for row in queue}) == 12
    outcome_by_id = {row["sample_id"]: row["group"] for row in rows}
    outcome_order = [outcome_by_id[row["sample_id"]] for row in queue]
    # Round-robin positions must not reveal the outcome of otherwise blinded rows.
    assert outcome_order != sorted(set(outcome_by_id.values())) * 3
    for row in queue:
        assert row["annotation_status"] == "pending"
        assert row["evidence_intervals"] == []
        assert row["evidence_type"] is None
        assert row["evidence_available"] is None
        assert "group" not in row
        assert "ground_truth" not in row
        assert "prediction" not in row


def test_invalid_review_limit_does_not_produce_silent_empty_queue():
    with pytest.raises(ValueError):
        evidence_review_queue([_row("s", "v", True, True)], limit=0)


def test_diagnosis_cli_writes_blinded_queue_and_refuses_overwrite(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    base = dict(sample_id="s1", video_id="sample_1", question="What is visible now?",
                query_time_s=2.0, ground_truth="A", prediction="A", correct=True,
                options=["cat", "dog", "ball", "car"], selected_frames=[])
    left, right = tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"
    left.write_text(json.dumps(dict(base, policy_id="recent-8")) + "\n", encoding="utf-8")
    right.write_text(json.dumps(dict(base, policy_id="adaptive-semantic-v1")) + "\n", encoding="utf-8")
    output = tmp_path / "report"
    command = [sys.executable, str(Path(__file__).parents[1] / "scripts/diagnose_allocation.py"),
               "--baseline-input", str(left), "--baseline-policy", "recent-8",
               "--candidate-input", str(right), "--candidate-policy", "adaptive-semantic-v1",
               "--output-dir", str(output), "--bootstrap", "20"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "diagnosis.json").read_text())["n_pairs"] == 1
    original = (output / "evidence-review.jsonl").read_bytes()
    assert json.loads(original)["annotation_status"] == "pending"
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "already exist" in repeated.stderr
    assert (output / "evidence-review.jsonl").read_bytes() == original

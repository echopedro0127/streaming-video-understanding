from copy import deepcopy

import pytest

from streaming_video_understanding.analysis.option_stability import analyze_option_stability


def original():
    return dict(sample_id="s1", video_id="sample_73", policy_id="recent-8",
                question="How many times?", query_time_s=10.0, ground_truth="A",
                prediction="A", correct=True, options=["zero", "one", "two", "three"],
                selected_frames=[dict(frame_id=1, timestamp_s=1.0)], sample_fps=1.0,
                timestamps_s=[1.0], config_id="recent-8-testhash",
                timestamp_prompt="Answer with one letter.\nQuestion: How many times?\nOptions:\nA. zero\nB. one\nC. two\nD. three")


def permuted(prediction="B"):
    row = original()
    row.update(option_permutation=[1, 0, 3, 2], option_permutation_seed=17,
               original_options=row["options"], original_ground_truth="A",
               permuted_options=["one", "zero", "three", "two"],
               options=["one", "zero", "three", "two"],
               ground_truth="B", prediction=prediction, correct=prediction == "B",
               config_id="recent-8-testhash-option-permutation-17",
               timestamp_prompt="Answer with one letter.\nQuestion: How many times?\nOptions:\nA. one\nB. zero\nC. three\nD. two")
    return row


def analyze(left=None, right=None):
    return analyze_option_stability([original() if left is None else left],
                                   [permuted() if right is None else right],
                                   sample_ids=["s1"], policies=["recent-8"], seeds=[17])


def test_changed_letter_can_preserve_same_answer_content():
    result = analyze()
    row = result["rows"][0]
    assert row["semantic_consistent"] is True
    assert row["original_prediction"] == "A"
    assert row["permuted_prediction"] == "B"
    assert row["mapped_prediction"] == "A"
    assert row["predicted_content"] == "zero"
    assert row["label_sticky"] is False
    assert result["summary"]["conditions"][0]["semantic_consistency"] == 1.0


def test_same_letter_can_mean_different_content_and_wrong_answer():
    result = analyze(right=permuted("A"))
    row = result["rows"][0]
    assert row["semantic_consistent"] is False
    assert row["mapped_prediction"] == "B"
    assert row["label_sticky"] is True
    condition = result["summary"]["conditions"][0]
    assert condition["correct_to_wrong"] == 1
    assert condition["label_sticky_opportunities"] == 1


def test_fixed_prediction_position_is_not_stickiness_opportunity():
    row = permuted("A")
    row.update(option_permutation=[0, 2, 1, 3],
               options=["zero", "two", "one", "three"],
               permuted_options=["zero", "two", "one", "three"],
               timestamp_prompt="Answer with one letter.\nQuestion: How many times?\nOptions:\nA. zero\nB. two\nC. one\nD. three",
               ground_truth="A", correct=True)
    result = analyze(right=row)
    assert result["summary"]["conditions"][0]["label_sticky_rate"] is None


@pytest.mark.parametrize("field,value", [
    ("option_permutation", [0, 0, 2, 3]),
    ("option_permutation", [True, 0, 3, 2]),
    ("ground_truth", "A"),
    ("question", "different"),
    ("query_time_s", 12.0),
    ("prediction", "invalid"),
    ("correct", False),
    ("original_ground_truth", "C"),
    ("options", ["wrong", "zero", "three", "two"]),
    ("selected_frames", [dict(frame_id=2, timestamp_s=2.0)]),
    ("option_permutation_seed", True),
])
def test_rejects_uncontrolled_or_invalid_comparisons(field, value):
    row = permuted()
    row[field] = value
    with pytest.raises(ValueError):
        analyze(right=row)


def test_rejects_missing_and_duplicate_expected_matrix_records():
    kwargs = dict(sample_ids=["s1"], policies=["recent-8"], seeds=[17])
    with pytest.raises(ValueError, match="missing"):
        analyze_option_stability([original()], [], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_option_stability([original()], [permuted(), permuted()], **kwargs)


def test_rejects_ambiguous_original_option_text():
    row = original()
    row["options"] = ["zero", " ZERO ", "two", "three"]
    with pytest.raises(ValueError, match="ambiguous"):
        analyze(left=row)


def test_deterministic_order_and_unrelated_original_policies_are_ignored():
    unrelated = deepcopy(original())
    unrelated["policy_id"] = "uniform-8"
    assert analyze_option_stability([unrelated, original()], [permuted()],
             sample_ids=["s1"], policies=["recent-8"], seeds=[17]) == analyze()


def test_same_frames_in_changed_image_order_are_a_confound():
    left, right = original(), permuted()
    left["selected_frames"] = [dict(frame_id=1, timestamp_s=1.0), dict(frame_id=2, timestamp_s=2.0)]
    right["selected_frames"] = list(reversed(left["selected_frames"]))
    left["timestamps_s"] = [1.0, 2.0]
    right["timestamps_s"] = [2.0, 1.0]
    with pytest.raises(ValueError, match="selected frames changed"):
        analyze(left, right)


@pytest.mark.parametrize("change", ["config", "stale_prompt", "instructions", "timestamps"])
def test_rejects_hidden_model_input_or_configuration_confounds(change):
    row = permuted()
    if change == "config":
        row["config_id"] = "other-model-option-permutation-17"
    elif change == "stale_prompt":
        row["timestamp_prompt"] = original()["timestamp_prompt"]
    elif change == "instructions":
        row["timestamp_prompt"] = row["timestamp_prompt"].replace("Answer", "Guess")
    else:
        row["timestamps_s"] = [2.0]
    with pytest.raises(ValueError):
        analyze(right=row)


def test_cli_produces_summary_and_refuses_to_overwrite(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    left, right, split = tmp_path / "original.jsonl", tmp_path / "permuted.jsonl", tmp_path / "split.json"
    left.write_text(json.dumps(original()) + "\n", encoding="utf-8")
    right.write_text(json.dumps(permuted()) + "\n", encoding="utf-8")
    split.write_text(json.dumps({"sample_ids": ["s1"]}), encoding="utf-8")
    output = tmp_path / "report"
    command = [sys.executable, str(Path(__file__).parents[1] / "scripts/analyze_option_stability.py"),
               "--original", str(left), "--permuted", str(right), "--split", str(split),
               "--policies", "recent-8", "--seeds", "17", "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_comparisons"] == 1
    before = (output / "comparisons.jsonl").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr
    assert (output / "comparisons.jsonl").read_bytes() == before

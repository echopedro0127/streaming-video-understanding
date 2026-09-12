# Results Analysis and Failure Review Design

## Goal

Turn the completed four-policy StreamingBench run into public, reproducible
evidence without changing the evaluator or rerunning the 1,000 real-model
queries.

## Scope

This phase covers two small post-processing tools:

1. A summary/plot tool that reads the four checked-in summary JSON files and
   produces a comparison CSV plus three PNG figures.
2. A failure-review tool that reads per-policy failure JSONL files, selects a
   deterministic, deduplicated set of 30 cases, and writes a Markdown/CSV
   review manifest. Existing `render_case.py` remains responsible for turning
   selected records into timestamp-labeled contact sheets.

The phase does not change model inference, frame selection, scoring, resume
behavior, or the raw local JSONL outputs. It does not infer a failure category
automatically; category labels remain an explicit human-review field.

## Inputs and outputs

Inputs are resolved relative to the repository root or supplied explicitly:

- `results/full-recent-8-summary.json`
- `results/full-uniform-8-summary.json`
- `results/full-recent-4-uniform-memory-4-summary.json`
- `results/full-recent-4-timestamp-aware-memory-4-summary.json`
- The corresponding `*-failures.jsonl` files

The plot tool writes ignored local artifacts under `results/analysis/`:

- `policy-comparison.csv`
- `accuracy-vs-latency.png`
- `accuracy-vs-gpu-memory.png`
- `accuracy-vs-context.png`

The failure tool writes ignored local review artifacts under
`results/failure_review/`:

- `selected_failures.jsonl`, containing the original evidence plus a stable
  `review_id`
- `failure_review.csv`, with blank `category` and `notes` fields for manual
  review
- `failure_review.md`, containing a compact table and the allowed category
  vocabulary

The tools must fail with a clear message when an input is missing, malformed,
or contains records from mixed policies.

## Plot design

`plot_results.py` accepts repeated `--summary` paths and an `--output-dir`.
Each summary is normalized into one row with a human-readable policy label,
accuracy, P50/P95 latency, peak GPU memory, average input frames, retained
frames, compression ratio, throughput, and sample count.

The figures use the same policy order and colors in every plot:

1. Accuracy vs P50 latency, with P95 latency represented by a horizontal error
   bar where the axes remain readable in seconds.
2. Accuracy vs peak GPU memory.
3. Accuracy vs average input frames, with point labels showing the policy.

The script uses a non-interactive Matplotlib backend, creates parent
directories, and saves deterministic PNGs. It does not include raw per-sample
latency outliers in the plot inputs; the summary's P50/P95 values are used and
the README records the suspend/interruption caveat.

## Failure-review design

`select_failures.py` accepts repeated failure JSONL paths, `--limit` (default
30), a seed, and an output directory. It validates every record's causal
manifest before selection, rejects any selected frame after `query_time_s`,
deduplicates by `(sample_id, policy_id)`, and uses a stable seeded ordering.
The selection is stratified as evenly as possible across policies and then
truncated to the requested limit. If fewer than the requested number of
unique failures exist, it reports the available count instead of fabricating
rows.

The CSV/Markdown review fields are:

- `review_id`
- `sample_id`
- `policy_id`
- `video_id`
- `question`
- `query_time_s`
- `prediction`
- `ground_truth`
- `category`
- `notes`

Allowed categories are `forgotten_early_evidence`, `temporal_ordering_error`,
`distractor_frames`, `insufficient_recent_context`,
`memory_compression_loss`, `semantic_retrieval_failure`,
`timestamp_confusion`, and `other`. The script leaves category and notes
empty so the report cannot claim automated manual analysis.

## Testing and verification

Add unit tests before implementation for:

- summary normalization and stable policy ordering;
- missing/malformed summary validation;
- causal validation of failure manifests;
- deterministic, policy-balanced, deduplicated failure selection;
- CSV/Markdown output fields.

Run the targeted tests first, then the full suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py
.\.venv\Scripts\python.exe -m pytest -q
```

After the code tests pass, run both tools against the four real summaries and
failure files, verify four 250-row input JSONL files are unchanged, inspect
the generated figures, render five representative cases with
`scripts/render_case.py`, and update README/PROJECT_PLAN with only measured
facts and explicitly labeled human-review status.

## Non-goals

- No new VLM or embedding model.
- No additional large dataset or model download.
- No accuracy claim based on a manually chosen subset.
- No deletion or rewriting of raw experiment outputs.

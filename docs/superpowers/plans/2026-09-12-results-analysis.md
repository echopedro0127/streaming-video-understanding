# Results Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic post-processing tools that turn the completed four-policy run into public comparison figures and a human-reviewable failure-analysis set.

**Architecture:** Keep model inference and raw JSONL untouched. Put reusable validation, normalization, and selection logic in small `analysis` modules; keep `scripts/plot_results.py` and `scripts/select_failures.py` as thin CLI adapters. Local generated artifacts stay under ignored `results/analysis/` and `results/failure_review/`.

**Tech Stack:** Python 3.13, standard library JSON/CSV/pathlib, Matplotlib with the non-interactive `Agg` backend, pytest, existing Pillow/FrameReader renderer.

---

### Task 1: Add analysis dependencies and summary primitives

**Files:**
- Modify: `requirements.txt`
- Create: `src/streaming_video_understanding/analysis/__init__.py`
- Create: `src/streaming_video_understanding/analysis/summaries.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing tests**

Add tests requiring `normalize_summary` to map a summary object to
`policy_id`, `policy_label`, `question_accuracy`, `p50_latency_s`,
`p95_latency_s`, `peak_gpu_memory_mb`, `average_input_frames`,
`retained_frames`, `compression_ratio`, `throughput_samples_per_s`, and
`num_samples`; require missing `question_accuracy` to raise `ValueError`; and
require `ordered_summary_rows` to return the four fixed-policy order.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py
```

Expected: import or attribute failures because the new functions do not exist.

- [ ] **Step 3: Implement the minimal summaries module**

Implement policy label/order constants and pure functions with explicit numeric
validation. Convert milliseconds to seconds only in the normalized row and
raise messages naming missing or invalid fields.

- [ ] **Step 4: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py
```

Expected: the summary tests pass.

- [ ] **Step 5: Add and install Matplotlib in the E-drive environment**

Add `matplotlib>=3.9` to `requirements.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install "matplotlib>=3.9" --cache-dir .cache\pip --timeout 120 --retries 5
```

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt src/streaming_video_understanding/analysis tests/test_analysis.py
git commit -m "feat: add results analysis primitives"
```

### Task 2: Implement deterministic failure selection and review output

**Files:**
- Modify: `tests/test_analysis.py`
- Modify: `src/streaming_video_understanding/analysis/failures.py`
- Create: `scripts/select_failures.py`

- [ ] **Step 1: Write failing failure-selection tests**

Add tests requiring `select_failures(records, limit=4, seed=7)` to reject a
future-frame manifest, deduplicate `(sample_id, policy_id)`, return the same
ordering for the same seed, and select at least one case from each available
policy before filling remaining slots. Add output tests requiring CSV headers
`review_id`, `sample_id`, `policy_id`, `video_id`, `question`, `query_time_s`,
`prediction`, `ground_truth`, `category`, and `notes`.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py -k failure
```

Expected: failures because the selector and review writers are not implemented.

- [ ] **Step 3: Implement the selector and writers**

Implement `validate_causal_manifest`, seeded policy-balanced selection, stable
`review_id` values, and JSONL/CSV/Markdown writers. Markdown must list the
allowed manual categories and state that `category` and `notes` are blank for
human review. Do not infer a failure category in code.

- [ ] **Step 4: Implement the CLI**

`scripts/select_failures.py` accepts repeated `--input`, `--limit`, `--seed`,
and `--output-dir`; parses JSONL with line-numbered errors; validates causal
manifests; creates the output directory; writes `selected_failures.jsonl`,
`failure_review.csv`, and `failure_review.md`; and prints counts and paths.

- [ ] **Step 5: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py
```

Expected: all analysis tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/streaming_video_understanding/analysis/failures.py scripts/select_failures.py tests/test_analysis.py
git commit -m "feat: add deterministic failure review selection"
```

### Task 3: Implement comparison summaries and trade-off plots

**Files:**
- Modify: `tests/test_analysis.py`
- Create: `scripts/plot_results.py`

- [ ] **Step 1: Add failing plot-input tests**

Add tests requiring `build_comparison_rows` to load four summary paths in any
order, return the fixed policy order, convert latency to seconds, and raise a
clear `ValueError` when `num_samples` is not 250.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py -k summary
```

Expected: the comparison-row test fails because the helper is not implemented.

- [ ] **Step 3: Implement the plotting CLI**

Implement `build_comparison_rows` and a CLI accepting repeated `--summary`
paths and `--output-dir`. Write `policy-comparison.csv` and three deterministic
PNG files using Matplotlib `Agg`: accuracy vs P50 latency with P95 as an
error-bar endpoint, accuracy vs peak GPU memory, and accuracy vs average input
frames. Validate exactly one summary per fixed policy and 250 samples per
summary. Use consistent colors, labels, axes, grid, legend, and `tight_layout`.

- [ ] **Step 4: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py
```

Expected: all analysis tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/plot_results.py tests/test_analysis.py
git commit -m "feat: add streaming policy trade-off plots"
```

### Task 4: Run real artifacts, render cases, and finish documentation

**Files:**
- Modify: `PROJECT_PLAN.md`
- Modify: `README.md`
- Local ignored outputs: `results/analysis/`, `results/failure_review/`, and `results/failure_review/renders/`

- [ ] **Step 1: Run the complete test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all existing and analysis tests pass.

- [ ] **Step 2: Generate comparison artifacts**

```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py --summary results/full-recent-8-summary.json --summary results/full-uniform-8-summary.json --summary results/full-recent-4-uniform-memory-4-summary.json --summary results/full-recent-4-timestamp-aware-memory-4-summary.json --output-dir results/analysis
```

Expected: one CSV and three non-empty PNG files under `results/analysis/`.

- [ ] **Step 3: Select the 30-case review set**

```powershell
.\.venv\Scripts\python.exe scripts/select_failures.py --input results/full-recent-8-failures.jsonl --input results/full-uniform-8-failures.jsonl --input results/full-recent-4-uniform-memory-4-failures.jsonl --input results/full-recent-4-timestamp-aware-memory-4-failures.jsonl --limit 30 --seed 20260912 --output-dir results/failure_review
```

Expected: deterministic JSONL, CSV, and Markdown files with up to 30 unique
cases and blank manual category fields.

- [ ] **Step 4: Render five representative cases**

Read the selected review JSONL, choose five cases covering at least three
policies, and run `scripts/render_case.py` once per case with its `sample_id`
and source policy JSONL. Verify each PNG is non-empty and all displayed frame
timestamps are causal.

- [ ] **Step 5: Complete manual review fields**

Inspect all 30 selected records and contact sheets. Fill category and notes
with evidence-based labels without changing prediction or ground truth. Record
the reviewed count and three to five representative examples in README.

- [ ] **Step 6: Update status and verify repository hygiene**

Update `PROJECT_PLAN.md` with measured review status, update README links to
public figures if they are added to Git, run `git diff --check`, run the full
tests again, and verify `git status` contains no datasets, model weights, raw
JSONL, or private paths.

- [ ] **Step 7: Commit the public report changes**

```powershell
git add README.md PROJECT_PLAN.md assets/figures
git commit -m "docs: add streaming benchmark analysis"
```

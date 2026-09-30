# Scale Streaming Evaluation to 100 Videos Design

## Goal

Extend the existing StreamingBench real-time evaluation from the validated
1--50 subset (50 videos / 250 QA rows) to a reproducible 1--100 subset (100
videos / 500 QA rows) without changing the existing 1--50 data or result
artifacts.

The scale-up should produce evidence about data ingestion, resumable
evaluation, long-tail runtime, and per-video aggregation. It must not create
claims merely by increasing the sample count.

## Scope and non-goals

In scope:

- Reuse the already verified `Real-Time%20Visual%20Understanding_51-100.zip`.
- Add support for the public `real-time-51-100` archive while retaining
  `real-time-1-50` compatibility.
- Generate one unified annotation file for samples 1--100 with explicit
  archive-relative video paths.
- Validate archive extraction and annotation/video consistency before inference.
- Run two policies on all 500 QA rows: `recent-8` and
  `recent-4+timestamp-aware-memory-4`.
- Preserve per-sample JSONL, resume behavior, causal audit, runtime summaries,
  and failure records.

Out of scope:

- Downloading another model or changing the SmolVLM backend.
- Running the full 203GB StreamingBench release.
- Running the 16-frame policy that already exceeded the local GPU/context
  budget.
- Re-running the existing 1--50 four-policy benchmark unless a regression is
  detected.

## Data layout

The existing subset remains untouched. The new archive is extracted beside it:

```text
data/streamingbench/
├── real-time-1-50/
│   ├── sample_1/
│   └── ...
├── real-time-51-100/
│   ├── sample_51/
│   └── ...
└── annotations-1-100.jsonl
```

The unified annotation rows use paths such as:

```text
real-time-1-50/sample_7/video.mp4
real-time-51-100/sample_73/video.mp4
```

This allows `run_eval.py` to use `data/streamingbench` as the data root while
keeping the original subset annotation file and paths reproducible.

The already downloaded archive currently has a URL-encoded filename in the
project root. The downloader must accept an explicit archive path so it can
validate and extract that file without downloading a second copy.

## Components

### 1. Multi-subset archive downloader

Modify `scripts/download_data.py` to use a checked-in subset table containing
the URL, expected archive size, archive filename, and extraction directory for
`real-time-1-50` and `real-time-51-100`.

Add an optional `--archive-path` argument. When supplied, the script validates
the existing archive and skips network download; otherwise it uses the
existing resumable downloader. Extraction remains path-safe and idempotent.

The command for the verified local archive will be equivalent to:

```powershell
python scripts/download_data.py `
  --subset real-time-51-100 `
  --destination data/streamingbench `
  --archive-path "E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\Real-Time%20Visual%20Understanding_51-100.zip" `
  --confirm-large-download
```

### 2. Unified annotation downloader

Modify `scripts/download_annotations.py` to support `--max-sample 100` and an
output path outside an individual subset directory. Rows for samples 1--50
receive the `real-time-1-50/` prefix; rows for samples 51--100 receive the
`real-time-51-100/` prefix.

The downloader must reject duplicate question IDs and report the exact row
count. The expected output is 500 unique rows.

### 3. Dataset integrity validator

Add `scripts/validate_dataset.py` with checks for:

- expected sample range 1--100;
- exactly one real `sample_N/video.mp4` per sample directory;
- no missing annotation video paths;
- no duplicate question IDs;
- five QA rows per video for the selected real-time subset;
- query timestamps that are non-negative and finite;
- optional archive CRC validation before extraction.

The validator should ignore `__MACOSX` metadata files when counting real
videos, because the downloaded archive contains macOS metadata entries.

### 4. Scale evaluation configuration

Add `configs/scale_100.yaml` with the existing model, seed, 60-second causal
horizon, 1 FPS sampling, batch size 1, and only these policies:

```yaml
policy_ids:
  - recent-8
  - recent-4+timestamp-aware-memory-4
```

The output files must be separate from the existing 250-QA results, for
example:

```text
results/scale-100-recent-8.jsonl
results/scale-100-recent-4-timestamp-aware-memory-4.jsonl
```

Evaluation will first run a mock/configuration smoke test, then a small real
model smoke test, and only then the full 500-QA runs. Full runs must use
`--resume`; an interruption must not require completed samples to be repeated.

### 5. Scale-specific analysis

Add a compact scale summary containing:

- question and video-macro accuracy;
- P50/P95 latency and mean latency with outlier count;
- peak GPU memory;
- input and retained frame counts;
- throughput;
- per-video sample count and accuracy;
- future-frame violation count;
- failed/resumed/skipped sample counts.

The report must compare scale-100 results against the existing 250-QA main
results without implying that the two runs are identical experimental splits.

## Failure modes to measure honestly

The scale-up should record real failures rather than inventing them. The
engineering report should check whether the larger run exposes:

1. archive/path mismatches across two extraction roots;
2. duplicate or missing sample IDs;
3. annotation rows with missing videos or invalid timestamps;
4. long-tail decode latency on longer videos;
5. interrupted evaluation and resume correctness;
6. per-video performance variance hidden by overall accuracy;
7. growth in failure-case selection and rendering cost.

Any issue becomes a resume-worthy engineering claim only after a reproducible
failure, a code or protocol change, and a before/after verification artifact.

## Acceptance criteria

The scale-up is complete when:

- the 51--100 archive is validated and extracted without redownloading;
- 100 sample directories and 500 unique QA rows pass dataset validation;
- the old 1--50 evaluator and results remain unchanged;
- both scale policies complete 500/500 QA rows or have a documented resumable
  partial state;
- the causal audit reports zero future-frame violations;
- resume is demonstrated by a controlled interrupted run or an equivalent
  completed-sample skip test;
- runtime and per-video summaries are saved;
- the README and PROJECT_PLAN describe only measured scale-up results;
- tests and `pip check` pass.

## Resource budget

- No new model download.
- The verified 51--100 archive is approximately 8.71GB.
- Expected additional extracted data is approximately 15--20GB based on the
  existing subset.
- Full scale evaluation is 1,000 model calls (500 QA rows × 2 policies) and
  should be run on AC power with sleep disabled.
- All data, model cache, pip cache, and generated results remain on E drive.

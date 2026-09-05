# Streaming Video Understanding with Bounded Temporal Memory

## Status

Approved design for the first public-ready MVP. This document defines the scope that will be implemented before optional model or memory extensions.

## Goal

Build a reproducible streaming video QA evaluator that compares recent-only, uniform-history, bounded-memory, and timestamp-aware bounded-memory policies under a fixed visual input budget on a small public StreamingBench split.

## Research question

Under a fixed visual input budget, can bounded temporal memory improve streaming video understanding compared with recent-only or uniform-history sampling?

The project is an online/streaming complement to TAMS: TAMS studies offline long-video temporal grounding under compressed representations; this project studies causal online QA with bounded historical memory and local recent context.

## Scope and non-goals

### MVP scope

- Primary benchmark: StreamingBench, `Real-Time Visual Understanding_1-50.zip`.
- Approximately 50 videos and approximately 250 timestamped multiple-choice QA items.
- Main context horizon: the 60 seconds preceding each dataset query timestamp, matching the official StreamingBench main setting.
- Primary local model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
- Frame-based multi-image inference with explicit timestamp text in the prompt.
- Four policies under a fixed budget of 8 frames:
  - Recent-8.
  - Uniform-History-8.
  - Recent-4 + Uniform-Memory-4.
  - Recent-4 + Timestamp-aware-Memory-4.
- Budget sensitivity for B=4 and B=16 on Recent-B and Timestamp-aware-Memory-B.
- Deterministic seeds, JSONL per-sample results, resumable runs, leakage audits, latency, throughput, and GPU memory measurements.
- Automatic failure-case export.

### Explicit non-goals

- No full 203GB StreamingBench download.
- No training or LoRA in the MVP.
- No semantic embedding model or learned memory module in the MVP.
- No local Cosmos3 deployment or remote API dependency in the MVP.
- No claim of state-of-the-art performance.
- No redistribution of benchmark videos, model weights, or API credentials.

## Data protocol

Each StreamingBench row supplies a video identifier/path, question, options, answer letter, and a `time_stamp`. The dataset timestamp is the only source of the primary protocol's query time.

For a query at `t`:

```text
visible context = [max(0, t - 60 seconds), t]
future context  = (t, video_end]
```

The evaluator processes frames in chronological order. A query-specific `StreamingMemory` is updated only as frames/chunks become visible; it is frozen at `t` before sampling for inference. Cached decoded frames may be reused for performance only when their source timestamps are at or before the current query time.

Every selected frame must satisfy:

```python
frame.timestamp <= query_time_s + timestamp_tolerance_s
```

The selected frame list, visible range, query time, and leakage verdict are persisted in the result. A violation raises a dedicated `FutureFrameLeakageError` and the sample is not scored.

The primary evaluator will support `Prior` and `Concurrent` temporal clue metadata without changing the causal boundary. `Concurrent` may select a frame at the query boundary; all other selected frames must still be no later than that boundary.

NExT-GQA is a secondary diagnostic only. Its ground-truth evidence spans may be used to compute temporal-grounding metrics or a separately named evidence-complete protocol, but they must not define the primary StreamingBench query time.

## Methods

All main comparisons use the same `visual_budget` and the same question set.

### Recent Window

Select the latest B visible frames within the active 60-second context horizon.

### Uniform History

Select B frames with deterministic temporal stratification over the active visible context. The implementation must be stream-compatible: bins are derived from the currently visible interval, and no future frame is inspected.

### Recent + Uniform Memory

Split B into a recent local window and historical memory. For B=8:

```text
recent_capacity = 4
memory_capacity = 4
```

The historical memory covers older visible content using deterministic temporal bins. It remains bounded as the stream grows.

### Timestamp-aware Bounded Memory

Use the same selected frames as the bounded-memory comparison, but add stable metadata to the prompt for every frame:

```text
[frame 1 | absolute_time=12.50s | relative_to_query=-47.50s]
```

This isolates the effect of explicit temporal information from the effect of changing the visual selection policy.

Semantic retrieval is a future extension, not part of the first implementation.

## Model interface

The model backend is a small interface accepting:

- ordered frame images;
- ordered frame timestamps;
- question text;
- answer options;
- whether timestamp annotations are enabled.

The backend returns raw text, normalized answer letter, token counts when available, and runtime metadata. `MockVLMBackend` is used for unit tests and smoke tests. `SmolVLMBackend` is optional at import time and raises a clear installation/runtime error if Transformers or Torch is unavailable.

The prompt requests only one answer letter from A-D. The parser accepts a standalone letter and rejects ambiguous multi-letter responses rather than silently guessing.

## Metrics and artifacts

Per-sample JSONL records include:

- sample and configuration identifiers;
- video ID, question, options, query timestamp;
- prediction, ground truth, correctness;
- selected recent/memory frame IDs and timestamps;
- visible range and leakage audit;
- input frame count and retained frame count;
- end-to-end latency and model latency;
- peak allocated/reserved CUDA memory when available;
- error category and exception text when the sample fails.

Aggregates include:

- question-level accuracy;
- video-macro accuracy;
- mean, P50, and P95 latency;
- throughput;
- peak GPU memory;
- visual budget and compression ratio.

The evaluator resumes from completed `(sample_id, config_id)` keys. Existing valid records are not recomputed unless `--overwrite` is explicit.

## Failure analysis

All incorrect predictions are exported to a separate failure JSONL file. A renderer will create a timestamp-labeled contact sheet for selected frames. Manual analysis is deferred until the core run has at least 20 failures and will use the categories:

- forgotten early evidence;
- temporal ordering error;
- insufficient recent context;
- distractor frames;
- memory selection failure;
- temporal coverage failure;
- timestamp confusion.

The README will only include measured examples and numbers after the corresponding run artifacts exist.

## Runtime and reproducibility

- Native Windows is the first execution target because CUDA and FFmpeg are already visible there; WSL is a fallback, not a dependency.
- Use Python 3.12 in a project-local virtual environment unless an installation probe proves 3.13 support for all required wheels.
- Use `uv` when available for environment creation and dependency installation.
- Pin direct dependencies in `requirements.txt` with an environment note for the PyTorch CUDA wheel.
- Seed Python, NumPy, Torch, and sampler tie-breaking.
- Use one sample per inference call in the MVP to reduce OOM risk.
- Synchronize CUDA before latency measurements.
- Keep data, model weights, caches, credentials, and large results out of Git.

## Repository boundaries

```text
README.md                         public project overview
PROJECT_PLAN.md                   phase/status tracker
requirements.txt                  direct dependencies
configs/                          reproducible experiment configurations
scripts/                          user-facing commands
src/data/                         StreamingBench schema and loading
src/models/                       mock and Transformers backends
src/streaming/                    causal buffer and sampling policies
src/evaluation/                   evaluator, resume, metrics, failures
src/utils/                        config, seed, logging, runtime helpers
tests/                            unit and integration tests
experiments/                      small checked-in manifests only
results/                          ignored runtime output
assets/                           small checked-in figures/contact sheets only
```

## Risk controls

1. If the StreamingBench archive cannot be downloaded or its data terms cannot be confirmed, stop before bulk download and use NExT-GQA as the primary fallback.
2. If SmolVLM2-500M exceeds the 6GB GPU budget, switch to CPU/mock smoke tests and then the 256M fallback; do not silently move to a larger model.
3. If real inference is too slow for the week, complete the evaluator and controlled sampler study with the real model on a smaller subset, then report the reduced sample count honestly.
4. Cosmos3 is never a dependency for MVP completion.

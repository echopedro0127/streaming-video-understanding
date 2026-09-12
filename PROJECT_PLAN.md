# Streaming Video Understanding MVP Plan

## Objective

Create a public-ready, reproducible streaming video QA evaluator that measures whether bounded temporal memory improves accuracy and efficiency under a fixed visual input budget.

## Current status

| Phase | Status | Evidence |
|---|---|---|
| Machine audit | Complete | RTX 3060 Laptop GPU, 6GB VRAM, FFmpeg available, 395GB free disk reported by user |
| Design | Complete | `docs/superpowers/specs/2026-09-05-streaming-video-understanding-design.md` |
| Git baseline | Complete | `a0aacc1 chore: initialize repository hygiene` |
| Repository skeleton | Complete | `README.md`, configs, scripts, package layout, and public-data `.gitignore` rules |
| MVP evaluator | Complete | 47 unit tests pass; causal evaluator, four policy families, resume store, metrics, failure export, and CLI are implemented |
| Real-model smoke test | Complete | SmolVLM2 real-video inference succeeded on RTX 3060; peak GPU memory was approximately 2.58GB |
| Core experiment | Complete | Four fixed-budget policies completed on 250 QA rows each; causal audit found 0 future-frame violations and summaries are saved under `results/` |
| Failure analysis | Complete | 30 failure cases manually reviewed in `docs/FAILURE_ANALYSIS.md`; five representative contact sheets rendered; trade-off plots generated under `assets/figures/` |
| Budget sensitivity | Complete | Four safe policies completed on the first 50 QA rows; 200/200 records passed the causal audit; 16-frame probe excluded after reaching the 6GB/context limit |

## Research scope

Primary benchmark: StreamingBench `Real-Time Visual Understanding_1-50.zip` (approximately 50 videos and 250 timestamped QA items). The complete 203GB release is intentionally not downloaded.

Primary model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, loaded only when the optional runtime dependencies are installed. The evaluator and all policy tests must work with the mock backend without a GPU.

Main causal context: the 60 seconds preceding the dataset query timestamp. The primary query timestamp comes from the benchmark row, never from ground-truth evidence.

## MVP experiment matrix

| Group | Configurations | Purpose |
|---|---|---|
| Main fixed budget | Recent-8, Uniform-8, Recent-4 + Memory-4, Recent-4 + Timestamp-aware-Memory-4 | Controlled comparison with exactly 8 input frames |
| Budget sensitivity | B=4: Recent-4 vs Recent-2 + Timestamp-aware-Memory-2; B=8: Recent-8 vs Recent-4 + Timestamp-aware-Memory-4 | Safe context-size trade-off on a 50-QA split |
| Runtime | One sample per call, CUDA synchronization, resume enabled | Latency and OOM-safe profiling |

## Deliverables

- `README.md` with motivation, protocol, method, installation, data preparation, metrics, limitations, and links to upstream data/model terms.
- `requirements.txt` with direct dependencies and a note for installing the correct PyTorch CUDA wheel.
- `configs/` YAML files for main and sensitivity runs.
- `scripts/download_data.py` that downloads only the selected StreamingBench archive after an explicit confirmation flag.
- `scripts/download_annotations.py` that retrieves the bounded 250-row QA annotation subset and maps rows to local video paths.
- `scripts/run_eval.py` for deterministic, resumable evaluation.
- `scripts/benchmark_runtime.py` for aggregate latency/memory summaries.
- `scripts/render_case.py` for timestamp-labeled selected-frame contact sheets.
- Typed data records, causal streaming memory, deterministic samplers, model backends, evaluator, resume store, metrics, and failure export under `src/`.
- Unit tests for timestamp parsing, causal bounds, memory capacity, deterministic selection, answer parsing, resume behavior, and metric aggregation.

## Execution phases

### MVP

1. Build the package and test harness.
2. Implement benchmark-row parsing and strict causal timestamp validation.
3. Implement Recent, Uniform, Bounded Memory, and Timestamp-aware policies.
4. Implement mock backend and optional SmolVLM backend.
5. Implement resumable evaluator and metrics.
6. Run synthetic smoke tests and a real-model 3–5 sample smoke test.

### MVP verification evidence

- `python -m pytest -q` passes with **47 tests**.
- `python scripts/run_eval.py --config configs/main.yaml --backend mock --limit 3 --output results/mock-smoke.jsonl --overwrite` writes 3 records.
- Re-running with `--resume` reports `processed=0 skipped=3`, confirming sample/config resumability.
- `python scripts/benchmark_runtime.py --input results/mock-smoke.jsonl --output results/mock-smoke-summary.json` writes the expected summary keys.
- `python scripts/render_case.py --input results/mock-smoke.jsonl --output results/mock-smoke-case.png` writes a causal timestamp contact sheet.
- `python scripts/run_eval.py --config configs/main.yaml --backend mock --all-policies --limit 1 ...` writes 4 policy records, confirming YAML-driven matrix execution.
- Resume IDs include a hash of the dataset/runtime/model/backend/policy configuration; changing those settings cannot silently reuse old records.
- Real-data annotations support CSV, JSON, JSONL, and optional Parquet; video decoding uses PyAV with an FFmpeg/ffprobe fallback.
- These smoke artifacts are intentionally ignored by Git; they are local verification evidence, not benchmark results.
- Real-model smoke completed on one real video; the cached model ran without OOM and recorded latency / GPU-memory fields.
- A 10-sample pilot completed all 40 policy/sample evaluations after leading-letter answer parsing was made tolerant of explanatory model outputs. It is a pipeline check, not a final benchmark result.

### Core Experiment

1. Download only `Real-Time Visual Understanding_1-50.zip` after checking disk and terms.
2. Run the fixed-budget matrix on all available rows or a documented subset.
3. Save per-sample JSONL, summary CSV/JSON, leakage audit, and failure JSONL.
4. Generate accuracy/latency/GPU-memory trade-off plots.
5. Manually categorize 20–30 measured failure cases.

### Budget Sensitivity

- Four policies completed on 50 QA rows from `sample_1`–`sample_10`, with 200 total records and 0 future-frame violations.
- The compact comparison is recorded in `assets/figures/sensitivity-comparison.csv`; raw JSONL remains local and ignored by Git.
- A 16-frame SmolVLM probe was stopped after the first diagnostic record because the 6GB GPU was nearly full and the processor reported a sequence longer than the 8192-token limit; it is a documented hardware boundary, not a benchmark result.

### Optional Extension

- NExT-GQA temporal-grounding diagnostic.
- Semantic memory policy using a separately pinned lightweight embedding model.
- Qwen2.5-VL-3B-AWQ comparison.
- Cosmos3-Nano hosted cross-model check on 30–50 identical selected-frame cases.

## Stop conditions

- Do not download the full StreamingBench release.
- Do not add Cosmos3 as a required dependency.
- Do not replace the 6GB-safe primary model with a larger model without a new measurement.
- Do not write accuracy improvements or resume-ready numerical claims before measured artifacts exist.
- If the real model cannot run, complete and verify the evaluator with the mock backend, report the blocker, and do not fabricate model results.

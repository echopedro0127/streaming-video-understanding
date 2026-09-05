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
| MVP evaluator | Complete | 26 unit tests pass; causal evaluator, four policy families, resume store, metrics, failure export, and CLI are implemented |
| Real-model smoke test | Not started |  |
| Core experiment | Not started |  |
| Failure analysis | Not started |  |

## Research scope

Primary benchmark: StreamingBench `Real-Time Visual Understanding_1-50.zip` (approximately 50 videos and 250 timestamped QA items). The complete 203GB release is intentionally not downloaded.

Primary model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, loaded only when the optional runtime dependencies are installed. The evaluator and all policy tests must work with the mock backend without a GPU.

Main causal context: the 60 seconds preceding the dataset query timestamp. The primary query timestamp comes from the benchmark row, never from ground-truth evidence.

## MVP experiment matrix

| Group | Configurations | Purpose |
|---|---|---|
| Main fixed budget | Recent-8, Uniform-8, Recent-4 + Memory-4, Recent-4 + Timestamp-aware-Memory-4 | Controlled comparison with exactly 8 input frames |
| Budget sensitivity | Recent-4 vs Timestamp-aware-Memory-4; Recent-16 vs Timestamp-aware-Memory-16 | Context-size trade-off |
| Runtime | One sample per call, CUDA synchronization, resume enabled | Latency and OOM-safe profiling |

## Deliverables

- `README.md` with motivation, protocol, method, installation, data preparation, metrics, limitations, and links to upstream data/model terms.
- `requirements.txt` with direct dependencies and a note for installing the correct PyTorch CUDA wheel.
- `configs/` YAML files for main and sensitivity runs.
- `scripts/download_data.py` that downloads only the selected StreamingBench archive after an explicit confirmation flag.
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

- `python -m pytest -q` passes with **26 tests**.
- `python scripts/run_eval.py --config configs/main.yaml --backend mock --limit 3 --output results/mock-smoke.jsonl --overwrite` writes 3 records.
- Re-running with `--resume` reports `processed=0 skipped=3`, confirming sample/config resumability.
- `python scripts/benchmark_runtime.py --input results/mock-smoke.jsonl --output results/mock-smoke-summary.json` writes the expected summary keys.
- `python scripts/render_case.py --input results/mock-smoke.jsonl --output results/mock-smoke-case.png` writes a causal timestamp contact sheet.
- These smoke artifacts are intentionally ignored by Git; they are local verification evidence, not benchmark results.
- Real-model smoke is still pending because PyTorch / Transformers / PyAV and model weights have not been installed or downloaded.

### Core Experiment

1. Download only `Real-Time Visual Understanding_1-50.zip` after checking disk and terms.
2. Run the fixed-budget matrix on all available rows or a documented subset.
3. Save per-sample JSONL, summary CSV/JSON, leakage audit, and failure JSONL.
4. Generate accuracy/latency/GPU-memory trade-off plots.
5. Manually categorize 20–30 measured failure cases.

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

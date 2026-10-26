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
| Task 1-7 implementation | Complete | Semantic policy IDs, CLIP scoring/cache, evaluator integration, profiling, and tests are implemented at current HEAD |
| Option-aware retrieval validation | Complete for 100 videos; full strategy matrix pending | 2 policies × 500 QA = 1,000 records across `sample_1`–`sample_100`; combined Option-Semantic accuracy is 50.8% and Option-Semantic-Diverse is 50.6%; causal, budget, option-schema, and runtime violations are all 0 |
| Task 8 smoke and documentation | Complete after this update | One-sample mock and real SmolVLM semantic-diverse smoke evidence is recorded below |
| Full semantic accuracy validation | Complete for the 50-video subset; partial at 100-video scale | Five-policy validation is complete on 250 QA from `real-time-1-50`; at 100-video scale, Recent-8, Recent-4, Timestamp-aware-Memory-4, and both Option-aware policies have measured results, while Uniform-8 and Q-only semantic policies still need a full run |

## Research scope

### Human-labelled perception scoring (2026-10-26)

- User submitted24labels, all image-manifest bindings valid:19presenter,
  4inserted_scene,1transition; coverage24/24. No new model calls or training.
- Submitted-v1: original23/24(95.8%), cycled22/24(91.7%). Presenter19/19 both;
  inserted_scene4/4 versus3/4; transition0/1 both. Binary balanced recall excludes
  transition and is not overall accuracy. Small purposeful3-video probe only.
- One dark presenter/board frame submitted as transition awaits user adjudication;
  preserve original labels/scores and report any later revision separately.
- Pre-label snapshots/raw predictions unchanged. Next: continuous-event evidence,
  then perception-to-cumulative-state validation; not a500QA accuracy gain.
- Artifacts: `experiments/2026-10-26/perception-probe/human-review-v1/`.

### Human-reviewable perception probe (2026-10-26)

- Implemented prepare/run/score CLI and blinded local category-review page.
 24purposeful frames from3preselected cases, same JPEGs for human and model;
 portable timing-free selection/image-hash manifest, images remain local.
- Completed24pairs/48generations with unchanged SmolVLM2-500M. Parse48/48,
 semantic category consistency23/24, displayed-letter stickiness0/24.
 Pre-label summary retains null accuracy; submitted human scores are separate.
- Source/image/selection/causal/prompt/mapping/protocol/runtime violations0;
 counterbalanced12/12; resume skipped24 unchanged.526tests passed; code and
 independent artifact reviews passed. Browser export logic tested with Node;
 browser layout/save picker not manually verified.
- Review page: `results/perception-probe/review.html`; user exports matching
 `human-labels.json` there. Actual human annotation and continuous-event labels
 cannot be replaced by assistant guesses or semantic-consistency measurements.
- Artifacts: `experiments/2026-10-26/perception-probe/README.md`.

### Fixed-selection visual sensitivity (2026-10-26)

- Complete:20QA/20videos,80generations; fixed Change-4+4 source selections.
  Original / same-size blank / reversed images / option-cycle[1,2,3,0].
  No model, prompt template, baseline or selection-threshold changes.
- Original/blank/reverse each4/20(20%); option-cycle8/20(40%). Blank/reverse
  content consistency17/20 and18/20. Option-cycle content consistency1/20,
  displayed-letter stickiness18/20; +20pp CI95[-15,+55]pp is not a video gain.
- Constant displayed C matches each condition's accuracy. Findings suggest
  option-position dependence, not verified grounding or proof of ignoring images.
  Reverse invariance alone is expected for some total-count tasks.
- All audit/runtime violations0; original manifests/responses reproduce20/20;
  order5/5/5/5, smoke/full resumes skip4/20 unchanged,513tests pass.
- Exact frozen local source JSONL is required; hash includes timing fields.
  Public summaries alone do not enable turnkey replay of this diagnostic.
- Artifacts: `experiments/2026-10-26/visual-sensitivity/README.md`.
  Next: human-verified perception/event boundaries before cumulative-event state.

### Frame-change memory probe (2026-10-26)

- Complete: same20QA/20videos, Recent-8 / Prefix-8 / Change-4+4,60generations.
  Causal adjacent32x32RGB pixel difference, fixed0.08 threshold/2s cooldown,
  four recent and four top-scored historical frames; no new dependencies.
- Change4/20(20%) versus controls3/20(15%); helped1/hurt0, CI95[0,+15]pp.
  Posthoc constant-C gets the same4/20 and identical correctness vector;
  no verified memory gain or semantic event-counting claim.
- Audit/runtime violations0;40historical control predictions/manifests match.
  Change bank peak8images; paired diagnostic capacity bound32+incoming,
  excluding decoder/framework buffers. Smoke/full resume skip3/20 unchanged.
- Full suite503passed. Review fixed exceptional cleanup paths before inference.
  Human event annotations remain pending. Next priority: visual sensitivity and
  option-order controls, then verified event evidence before expanding/tuning.
- Artifacts: `experiments/2026-10-26/change-memory/README.md`.

### Causal temporal coverage probe (2026-10-26)

- Complete: frozen20QA/20videos, threeMCQconditions,60generations. SameB8/model;
  Recent-8,60s time bins and whole-prefix time bins all score3/20(15%).
- Mean selected spans7.007/45.696/200.350s; wider span alone yields no gain.
  Both paired helped/hurt0/0; degenerate bootstrap[0,0] is not population equivalence.
- Audit/runtime violations0; historical Recent-8 frames/predictions match20/20;
  five short-prefix controls agree. Retained-image peak23 (bound24+incoming).
  Smoke/full resume skip3/20 unchanged. Full suite492passed.
- Three preselected cases reviewed from1FPS contact sheets; observations are
  assistant-generated and pending human verification, not evidence labels.
- Report: `experiments/2026-10-26/context-coverage/README.md`.
  Next: verify event intervals before event-change/cumulative-state diagnostics.

### Choice-free counting pilot (2026-10-26)

- Completed 20 paired QA / 40 generations on 20 videos excluded from the earlier
  11-case diagnostic. Frozen answer-independent selection, one QA per video;
  videos appeared in earlier benchmark runs, so not pristine unseen data.
- Same Recent-8 images; MCQ/count order 10/10. Accuracy 3/20 vs 2/20, delta -5 pp,
  CI95 [-25,+15] pp; no gain established. Both parse 20/20; count emits 1 in 17/20,
  outside choices in 15/20. No nearest-choice repair.
- Audit/runtime errors 0, MCQ predictions/frames reproduce 20/20. Resume skipped 20
  unchanged. Tests: 474 passed. Report: `experiments/2026-10-26/choice-free-count/README.md`.
- Next priority: evidence answerability and event history, not parser/routing tuning.

### Counting-case diagnosis (2026-10-26)

- Completed 66 inferences on the 11 previously helped/hurt QA: 22 same-config
  replay controls plus 44 option permutations (seeds 17/29, two policies).
- Controls reproduced all historical predictions and selected-frame manifests.
  All 44 permutation comparisons passed config/prompt/timestamp/frame controls;
  66 profiles present; missing/duplicate/protocol/runtime violations 0.
- Answer-content consistency: Recent-8 6/22 (27.3%), adaptive-v1 7/22 (31.8%).
  These are selected-case stability measurements, not benchmark accuracy.
- Full suite: 450 passed. Public methods, results, audit and next decision:
  `experiments/2026-10-26/README.md`. No inference/routing behavior changed.

### Selective-memory follow-up (2026-10-24)

- Implemented `adaptive-semantic-v1`: frozen question-only 8+0 / 6+2 / 4+4
  allocation with auditable rules; reuses Q-only retrieval and the existing model.
- Implemented video-cluster paired bootstrap and a blinded 60-QA evidence review
  queue. All evidence annotations are pending human review.
- Existing tail-split diagnosis: Recent-8 95/250 versus Option-4+4 90/250;
  helped/hurt 5/10; delta -2.0 pp, video-cluster CI95 [-5.2, +1.2] pp.
- Option-aware 6+2 completed 250/250 on videos 51–100: 93 correct (37.2%),
  audit violations/runtime failures 0. Against Recent-8: helped/hurt 4/6,
  -0.8 pp (CI95 [-3.2, +1.6]); against Option-4+4: 4/1, +1.2 pp
  (CI95 [-0.4, +2.8]). Published small summaries: `experiments/2026-10-25/`.
- Adaptive current/unspecified/counting real smoke completed 3/3; real resume
  skipped all 3 with unchanged output checksum. Full tail evaluation completed
  250/250: 92 correct (36.8%), audit/runtime violations 0; against Recent-8,
  helped/hurt 4/7, -1.2 pp, video-cluster CI95 [-4.0, +1.6] pp.
- All 98 current-route predictions and frame manifests match Recent-8 exactly;
  the counting group is 31/150 versus 34/150. No overall improvement established.
- Current candidate horizon is 60 seconds; this is causal prefix evaluation,
  not yet a persistent incremental ingestion engine.
- Full method, commands and limitations: `docs/SELECTIVE_MEMORY_STAGE1.md`.

Primary benchmark: StreamingBench `Real-Time Visual Understanding_1-50.zip` (approximately 50 videos and 250 timestamped QA items). The complete 203GB release is intentionally not downloaded.

Primary model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, loaded only when the optional runtime dependencies are installed. The evaluator and all policy tests must work with the mock backend without a GPU.

Main causal context: the 60 seconds preceding the dataset query timestamp. The primary query timestamp comes from the benchmark row, never from ground-truth evidence.

## MVP experiment matrix

| Group | Configurations | Purpose |
|---|---|---|
| Main fixed budget | Recent-8, Uniform-8, Recent-4 + Memory-4, Recent-4 + Timestamp-aware-Memory-4 | Controlled comparison with exactly 8 input frames |
| Option-aware ablation | Recent-4 + Option-Semantic-Memory-4, Recent-4 + Option-Semantic-Diverse-Memory-4 | Four candidate-answer hypotheses per historical frame under the same 4+4 budget |
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

## Task 8: Semantic stage status

Task 1-7 implementation is complete. Task 8 smoke and documentation are
complete after this update. Semantic retrieval remains an experiment with no
proven accuracy gain.

| Check | Current evidence |
|---|---|
| Full pytest after Task 7 | **249 passed** |
| CLIP cache | `E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\hf_cache\hub\models--openai--clip-vit-base-patch32`; approximately **1157.8 MiB** (about **1.16 GB**) measured after download, including cache artifacts |
| Mock semantic-diverse smoke | `processed=1 skipped=0`; selected/input frames `5`; `query_time=4.0`; timestamps `0..4`; trace count `1`; `retrieval_latency_ms=990.394 ms`; mock pipeline evidence only |
| Real SmolVLM semantic-diverse smoke | `processed=1 skipped=0`; no OOM; `input_frames=5`; timestamps `0..4`; trace count `1`; `retrieval_ms=493.4114`; `model_latency_ms=3433.9946`; record `latency_ms=4201.0253`; `peak_gpu_memory_mb=2877.4521`; profile `total_ms=4702.2538` |
| Full semantic accuracy | **Complete for the 50-video subset**; 5 policies × 250 QA = 1,250 records. The 100-video scale currently has measured Recent-8, Recent-4, Timestamp-aware, and Option-aware results, while Uniform-8 and Q-only semantic full runs remain open |

### Available 250-QA semantic validation

The completed run used one process with a shared feature cache. All policies had
`runtime_failed=0`, `causal_violations=0`, `budget_violations=0`, average input
frames `7.944`, and peak GPU memory `3834.389648 MB`.

| Policy | Correct / 250 | Accuracy | P50 latency (ms) | P95 latency (ms) | Retrieval mean (ms) |
|---|---:|---:|---:|---:|---:|
| recent-8 | 163 | 65.2% | 22380.957 | 59667.738 | 0 |
| uniform-8 | 157 | 62.8% | 21910.201 | 59004.514 | 0 |
| recent-4+uniform-memory-4 | 160 | 64.0% | 21861.579 | 58798.954 | 0 |
| recent-4+semantic-memory-4 | 159 | 63.6% | 21922.348 | 59710.063 | 774.724 |
| recent-4+semantic-diverse-memory-4 | 157 | 62.8% | 21928.612 | 58401.495 | 47.460 |

`semantic_selection_trace` was non-empty for 248/250 records for each semantic
policy; the two missing traces had no historical frames at the query point.
All selected timestamps were causal. Recent-8 remains best at 65.2%:
semantic-memory is -1.6 percentage points and semantic-diverse is -2.4 points
relative to Recent-8. No accuracy improvement is claimed.

Retrieval latency is cache- and execution-order-sensitive, not an isolated
cold-start comparison: semantic-memory ran before semantic-diverse and includes
cache population, while semantic-diverse mostly read the warm shared cache.
Generated summary files under `results/semantic-validation-real-time-1-50-summary.json`
and the per-policy summary JSON files are ignored local artifacts.

The five visible frames in the one-sample smoke are due to 1fps sampling at
`query_time=4.0`, not a full eight-frame input. Record latency and profile
total are different accounting layers. These smokes validate runtime and the
causal pipeline only.

### Option-aware retrieval validation

The option-aware run used the same available `real-time-1-50` subset and added
two policies under the same 4+4 visual budget. Each historical frame was scored
against the question plus each of four candidate answers; the ground-truth
answer was never passed to retrieval.

| Policy | Correct / 250 | Accuracy | P50 latency (ms) | P95 latency (ms) | Retrieval mean (ms) |
|---|---:|---:|---:|---:|---:|
| recent-4+option-semantic-memory-4 | 164 | 65.6% | 22576.610 | 68135.936 | 78.465 |
| recent-4+option-semantic-diverse-memory-4 | 164 | 65.6% | 22679.149 | 69329.830 | 55.419 |

The run completed `500` records (`250 QA × 2 policies`) with
`runtime_failed=0`, `causal_violations=0`, `budget_violations=0`, and
`option_schema_violations=0`. Average input frames were `7.944` and peak GPU
memory was `3834.390 MB`. Against Recent-8, the paired helped/hurt/unchanged
counts were `10/9/231` for option-semantic and `12/11/227` for
option-semantic-diverse; both measured `+0.4` percentage points. Against
Q-only semantic memory, the counts were `8/3/239` and `9/4/237`, respectively,
or `+2.0` percentage points. These are subset results, not a generalization
claim.

Retrieval latency is affected by shared image-feature-cache warm-up and policy
execution order. The first option-aware policy includes more cache work, while
the second mostly reads warm features; the timing values should be interpreted
with this caveat.

## Scale-100 Option-aware expansion (2026-10-05)

The 51-100 archive was extracted from the previously verified local archive and
evaluated with the two Option-aware policies. The run used the main
`data/streamingbench` root after an initial worktree path mismatch was
diagnosed; existing 1-50 results were preserved.

| Scope | Records | Result |
|---|---:|---|
| 1-50, Option-Semantic-Memory-4 | 250 | 65.6% |
| 1-50, Option-Semantic-Diverse-Memory-4 | 250 | 65.6% |
| 51-100, Option-Semantic-Memory-4 | 250 | 36.0% |
| 51-100, Option-Semantic-Diverse-Memory-4 | 250 | 35.6% |
| 1-100, Option-Semantic-Memory-4 | 500 | 50.8% |
| 1-100, Option-Semantic-Diverse-Memory-4 | 500 | 50.6% |

The combined 1-100 run produced 1,000 records with zero runtime errors,
future-frame violations, frame-budget violations, or option-schema violations.
Peak GPU memory was approximately 3.83 GB. The large gap between the first and
second 50-video blocks is recorded as a long-tail / split-difficulty finding,
not as an accuracy improvement claim.

Still pending at 100-video scale:

- Uniform-8;
- Q-only semantic-memory and semantic-diverse-memory;
- Recent-4 + Uniform-Memory-4;
- full M=8/M=16 and K=16 ablations;
- calibrated retrieval confidence, option permutation stability, and
  answer/clarification routing;
- 100-video failure-case re-sampling and manual analysis.

The detailed handoff for further discussion is
`PROJECT_STATUS_FOR_GPT_UPDATED.md`.

## Stop conditions

- Do not download the full StreamingBench release.
- Do not add Cosmos3 as a required dependency.
- Do not replace the 6GB-safe primary model with a larger model without a new measurement.
- Do not write accuracy improvements or resume-ready numerical claims before measured artifacts exist.
- If the real model cannot run, complete and verify the evaluator with the mock backend, report the blocker, and do not fabricate model results.

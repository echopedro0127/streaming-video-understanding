# Choice-free counting readout pilot

Goal: compare original MCQ with direct integer counting using identical Recent-8
visual inputs on a video-disjoint, deterministically sampled 20-QA pilot.

Design: one counting QA per eligible video, excluding every video represented in
the earlier 11-case diagnostic. Within each video choose the minimum SHA256 of
`20261026:video_id:sample_id`, without consulting answers or correctness. Require
four distinct numeric options for post-generation evaluation. These videos were
in earlier benchmark runs, so this is video-disjoint from the diagnostic, not a
pristine held-out test. Freeze IDs before inference.

The same decoded image objects and timestamps feed both modes. Alternate which
mode runs first by index in sorted frozen sample IDs. Use existing model/config,
1 FPS, 60s horizon and eight-frame budget. Decode once per QA, generate twice;
this does not reuse VLM visual embeddings between generations.

MCQ prompt and answer parsing remain unchanged. Count mode omits every answer
option and asks for one non-negative integer. Freeze parser before smoke:
ASCII digits, optional `time`/`times`, optional terminal `.`/`!`; any other output
is invalid (incorrect, not a runtime failure). Numeric outputs outside the four
choices stay unmodified: no nearest-choice mapping or answer-based repair.

Primary metrics: exact count accuracy over all 20 QA (invalid outputs incorrect),
paired accuracy difference with video bootstrap CI, parse-valid rate, outside-
option count, and valid-output MAE with its explicit denominator. Report generation
latencies separately from one shared decode and full pipeline/cleanup time.

Implementation:

- [x] Test and add `models/counting.py`: strict parse, numeric option validation,
  question-only prompt; tests reject ambiguous counts and duplicate numeric options.
- [x] Add optional count mode to ModelRequest/Response and SmolVLM backend. Existing
  config hashing and default MCQ prompts/inference semantics must not change.
- [x] Test and add `evaluation/count_readout.py`: paired backend reuses exact
  images, counterbalanced order, no ground-truth access; integrate with existing
  evaluator image cleanup. Add deterministic video-disjoint sampling and summary.
- [x] Add `scripts/run_count_readout.py` with durable per-QA paired-result resume,
  distinct versioned config identity including split/annotation fingerprints,
  strict split coverage, runtime error sidecar and incremental flushed output.
- [x] Freeze `configs/splits/counting-readout-video-disjoint-20.json`; run tests,
  independent code review and 2-QA real smoke before extending the same output to20.
- [x] Audit all 20 paired rows/40 generations and publish measured results. No rule
  changes, parser tuning, new model, training or dependency downloads after outcomes.

Scope: this compares complete readout formats (instructions and availability of
choices), not an isolated estimate of a specific positional-bias mechanism. It
does not establish temporal evidence sufficiency or whole-history counting.

Completed: 20 pairs / 40 generations; no audit/runtime errors. MCQ 3/20 versus
count 2/20, delta -5 pp, CI95 [-25,+15]. Count emits 1 in 17/20; all outputs parse.
Smoke resume skipped 2 unchanged; full resume skipped 20 unchanged without model
loading. Full tests: 474 passed. Review resolved invalid-MCQ asymmetry before any
inference: pilot retains invalid outputs in both denominators, while default
baseline behavior is unchanged. Report: `experiments/2026-10-26/choice-free-count/README.md`.

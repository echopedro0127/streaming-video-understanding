# Selective temporal memory: implementation and experiment plan

> Execute in the existing optimization worktree. Preserve prior configurations,
> baseline behavior and results. Use test-first implementation and review checkpoints.

**Goal:** diagnose fixed eight-frame allocation, then add a question-only rule router
with auditable 8+0 / 6+2 / 4+4 decisions using the existing semantic selector.

**Architecture:** a pure router accepts only question text and budget, returns an
allocation and explicit rule/version. The evaluator resolves that decision into
an existing recent/semantic policy. Stored policy identity remains the adaptive
policy; resolved policy and route are additional metadata. Unknown questions use
6+2. Counting uses 4+4 with temporal diversity, explicitly a proxy for event coverage.
No answer, task label, evidence annotation, or future frame is visible to the router.

**Tech stack:** existing Python, pytest, PyYAML, SmolVLM2 and CLIP. No new dependencies.

## Work packages and verification

- [ ] 1. Resume `results/option-semantic-6plus2-real-time-51-100.jsonl` using the
  unchanged config/derived annotation. Verify unique sample/config keys, causal
  boundary, 6/2/8 limits, runtime failures and 250 completed rows before summary.
  Run `scripts/benchmark_runtime.py` and both `scripts/analyze_policy_pairs.py`
  comparisons only on complete matching sample sets. Keep the old artifacts.
- [x] 2. Add `tests/test_routing.py`: word-boundary matching, counting priority,
  explicit history/current cues, unknown fallback, empty question and budget != 8.
  Add evaluator tests: route is independent of answer/options/task labels;
  current route needs no scorer; historical route requires scorer; future frame
  guard and existing cleanup/resume behavior still hold. Observe RED first.
- [x] 3. Add `streaming/routing.py`, integrate in `evaluation/engine.py` and the
  scorer initializer in `scripts/run_eval.py`; add
  `configs/adaptive-semantic-1fps.yaml`. New ID `adaptive-semantic-v1` fixes B=8;
  revision requires a new version/ID to avoid invalid resume reuse. Preserve
  existing latency field semantics; publish full end-to-end profile separately.
- [ ] 4. Verify focused and full tests; run synthetic/mock routing and real-model
  smoke with distinct outputs and resume. Run GPU jobs sequentially.
- [x] 5. Add reusable diagnostic report with video-cluster paired bootstrap,
  per-route groups explicitly labeled rule proxies, plus an evidence review queue
  (empty human annotation fields, no invented evidence). Separate routing from
  evidence labels; do not tune rules on reported test outcomes.
- [x] 6. Update PROJECT_PLAN.md and a dated experiment report with exact counts,
  commands and limitations. Commit only source/config/tests/public reports.

Checkpoint: full suite 427 passed; independent review verified old baseline
selection and old configuration hashes, and prompted a review-order blinding fix.
Synthetic adaptive CLI smoke and resume passed. Real adaptive smoke waits for
the existing 6+2 job; do not mark work packages 1 or 4 complete prematurely.

## Research boundaries and subsequent packages

Current context horizon is 60 seconds. The evaluator re-reads visible prefixes
and reconstructs state at each QA. This package does not claim true incremental
ingestion, bounded total host memory, or zero repeated VLM encoding.

Evidence annotation (50–80 QA), event-bank writing, incremental ingestion and
OVO cross-benchmark evaluation are subsequent packages. Evidence labels need
human review; event/ingestion packages each need their own tested implementation.
Prior 500 QA have been inspected repeatedly: improvements there are exploratory,
not untouched-test generalization. Parameter selection needs video-disjoint data.

## Exact commands (from worktree, activated shared venv)

```powershell
python -m pytest -q tests/test_routing.py tests/test_evaluation.py
python -m pytest -q
python scripts/run_eval.py --config configs/adaptive-semantic-1fps.yaml --backend mock --policy adaptive-semantic-v1 --output results/smoke/adaptive-semantic-v1.jsonl --resume
python scripts/benchmark_runtime.py --input results/option-semantic-6plus2-real-time-51-100.jsonl --output results/option-semantic-6plus2-real-time-51-100-summary.json
```

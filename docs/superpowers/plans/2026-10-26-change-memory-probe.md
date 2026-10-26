# Frame-change memory probe implementation plan

Goal: test whether inexpensive visual-change retention helps the frozen counting
cohort, after wider temporal span alone produced no observed gain.

Design: compare Recent-8, whole-prefix eight-bin sampling and Change-4+4 on
the same20QA, model, prompt and B8 budget. Change-4+4 owns four recent images and
four evicted historical images. A causal adjacent-frame RGB32x32 mean absolute
difference /255 >=0.08 makes a frame a change candidate, subject to2s causal
cooldown. Candidate scores are frozen at arrival; historical top4 by descending
score, then earlier timestamp/ID. Noncandidate scores0 provide deterministic
fallback when fewer than4candidates exist. No labels/options/query semantics
enter selection, and no claim of semantic event counting is made.

Reuse the existing context bank and evaluator through an opt-in flag. The
diagnostic shares decode but still separately encodes each condition. Context
bank<=24 images and independent change bank<=8 images, plus incoming images
and one32x32thumbnail. Decoder/framework buffers excluded. Whole-prefix selectors
know query time, so this remains a query-aware causal diagnostic, not a deployed
persistent streaming system. Do not modify existing baseline default behavior.

Human verification remains pending: publish the preselected3case review queue
and boundary candidates, with no certified counts or evidence-recall claims.
Do not inject these observations, GT or textual event summaries into inference.

- [x] Add failing tests for change-bank bounded ownership, cooldown, top-score
  eviction, deterministic fallback, timestamps/future errors and cleanup.
- [x] Implement `streaming/change_memory.py`; run focused tests.
- [x] Add evaluator and CLI opt-in tests, invalid-output/model-error cleanup;
  implement optional mode in `evaluation/context_probe.py` and
  `scripts/run_context_coverage.py`, with new protocol version/hash/output.
- [x] Run full tests and independent review. Freeze thresholds before inference.
- [x] Run3QA smoke, verify unchanged resume, then complete20QA/60generations.
- [x] Audit old Recent-8 and prefix controls, nofuture/B8/capacity/order/errors,
  paired accuracy and latency. Document negative as well as positive outcomes.
- [x] Publish small summaries, audit and review queue; update status/plan and
  sync existing feature branch. Keep all original results unchanged.

Pre-inference verification:503 tests passed. Independent review found one
exceptional cleanup gap; two regression tests reproduced it before the fix.
Rereview approved with no remaining P1/P2 issues. Threshold/cooldown frozen
before smoke and not tuned against observed answers.

Outcome: Change4/20 versus controls3/20;1helped0hurt, CI[0,+15]pp. Posthoc
constant-C has identical per-QA correctness, so no verified memory improvement.
All40 oldcontrol manifests/predictions reproduce; causal/budget/runtime audits0.
Smoke/full resumes skip3/20 unchanged. Public results and pending-human evidence
queue are in `experiments/2026-10-26/change-memory/`.

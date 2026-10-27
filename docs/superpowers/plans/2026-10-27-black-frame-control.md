# Strict black-frame control implementation plan

**Goal:** Attribute startup-artifact gains using a separately labelled posthoc replay; preserve all original baselines, predictions and labels.

**Design:** Compare raw1/1, confirmation2/2 and exact-black gate + raw1/1 on the existing three-video cohort. Gate only when every decoded RGB channel value equals zero, at any timestamp. Supply unknown to the existing counter (hold active, reset pending streaks). No brightness tuning, label-based selection, model calls, claimed runtime savings or replacement of the frozen validation. Keep sample80 development results separate.

**Alternatives:** A brightness threshold risks suppressing visible dark scenes and needs separate calibration. Two-frame confirmation already exists but adds media-time delay. Exact-zero gating isolates the observed artifact with no tuned threshold; it cannot reject non-black classifier noise or distinguish a legitimate black scene.

**Architecture:** Add isolated replay helpers in `evaluation/black_frame_control.py`, tests in `tests/test_black_frame_control.py`, and `scripts/black_frame_control.py` CLI. Reuse manifest/image verification, full source protocol/state replay, event summaries and human scoring. Require fixed source prediction hashes and complete prefixes; reject existing output files. Publish anonymous metrics only, keeping reviewer identity and local paths private.

## Execution

- [x] Write behavioral tests: black versus dark visible pixels, unknown holds active, prefix invariance, source immutability and missing-frame rejection; observed missing-module failure before implementation.
- [x] Implement exact gate and causal replay;22focused tests pass, full suite548passes.
- [x] Run source-locked posthoc comparison on 79/75/76; separately check development80 and its dark frame145.145 (not gated).
- [x] Audit micro event precision/recall/F1, primary/dense count errors, signed media-time delays and unchanged source hashes. Zero new inference and no measured efficiency gain. Existing per-case scores reproduced exactly.
- [x] Publish report and update project status; stage only reviewed files, excluding the unrelated private plan.

The rule is fixed before this replay, but after inspecting this cohort's errors: this is not preregistered held-out validation. Future challenge cases must include short events/returns, dark scenes and non-startup noise.

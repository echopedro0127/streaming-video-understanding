# Counting-case option-order diagnostic

Goal: distinguish sensitivity of the answerer to option positions from changes
in selected visual evidence, on the 11 helped/hurt cases from adaptive v1.

This is a post-hoc diagnostic, not a representative benchmark or a test set for
claiming general accuracy improvement. Cases and seeds are frozen before runs.

Protocol:

- Freeze the 4 helped and 7 hurt sample IDs from the completed 250-pair analysis.
- Reuse `configs/adaptive-semantic-1fps.yaml`, seed 42, 1 FPS, 60s horizon, B=8,
  model/prompt/retrieval settings unchanged. Use only Recent-8 and Adaptive-v1.
- Reuse existing `--option-permutation-seed` implementation with seeds 17 and 29.
  Run 2 policies x 11 QA x 2 permutations = 44 new inferences, sequential GPU jobs.
- Also run 2 policies x 11 QA = 22 same-config unpermuted replay controls, for
  **66 total new inferences**. Original stored outcomes select the diagnostic
  cases; replay controls are the direct reference for permutation comparisons.
- Compare each condition to same-config replay results for the same policy.
  Validate question, timestamp, source options, answer remapping and selected-frame
  manifests. If frames differ, do not attribute a prediction change to option order.
- Require exact config ID suffix mapping, actual prompt option/metadata agreement,
  unchanged prompt instructions and ordered timestamps matching frame manifests.
  Compare replay predictions to the old stored outcomes separately; changed replay
  predictions are a reproducibility finding, not an option-permutation effect.
- Decode the predicted permuted letter back to its ORIGINAL option index and text;
  raw letter agreement is not semantic consistency. Check duplicate/ambiguous
  option text, valid permutations and missing/duplicate records explicitly.
- Report original-choice consistency, correctness transitions and label stickiness
  per policy/seed. If the same label also refers to the same content, it cannot
  diagnose position following. Do not interpret these fractions as confidence.
- All inference is causal. Audit expected 22 rows per seed, exact selected IDs,
  per-route budgets, option schema and runtime failures before interpretation.

Implementation steps:

- [x] Freeze split `configs/splits/adaptive-changed-counting-11.json`.
- [x] Run existing permutation tests; implement a small tested post-processing
  analyzer in `analysis/option_stability.py` and `scripts/analyze_option_stability.py`.
  Tests must catch comparing letters instead of content, incorrect remapping,
  selected-frame changes, missing pairs and duplicate records before implementation.
- [x] Run two existing evaluator commands per seed, keeping output and profile
  sidecars distinct by seed; resume completed keys only.
- [x] Audit and summarize 44 permutation and 22 replay rows. Retain all originals.
- [x] Record observed mechanisms versus unverified hypotheses, update project
  status and publish small aggregate artifacts with config/seed provenance.

Completed: 66/66 rows and profiles, zero audit/runtime errors, unchanged replay
predictions/frames. Content consistency: Recent-8 6/22, adaptive-v1 7/22.
Full tests 450 passed; independent code review approved the strengthened controls.
Results and limitations: `experiments/2026-10-26/README.md`.

Later evidence work: frame contact sheets and full causal-prefix evidence
annotation can test coverage and horizon truncation. Permutation consistency
does not show that selected frames contain sufficient evidence or that the
numerical answer is correct for visual reasons. No event counter, model change,
training, new dependency or large data download is part of this diagnostic.

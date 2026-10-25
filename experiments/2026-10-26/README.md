# Counting-case option-order diagnostic

Follow-up: [choice-free count pilot on 20 video-disjoint QA](choice-free-count/README.md).
This is separate; the diagnostic results below remain unchanged.

This is a **post-hoc diagnostic on 11 outcome-selected QA**, not a representative
benchmark or an accuracy improvement claim. Cases are exactly the four helped
and seven hurt items from the completed Recent-8 versus adaptive-v1 comparison.
The main 250-QA scores remain Recent-8 38.0%, adaptive-v1 36.8%.

## Protocol and controls

- Frozen case IDs: `configs/splits/adaptive-changed-counting-11.json` (10 videos).
- Same `configs/adaptive-semantic-1fps.yaml`, SmolVLM2-500M, 1 FPS, 60s horizon,
  eight-frame budget, seed 42. No model/prompt/routing changes.
- Two policies, two option-permutation seeds (17, 29): 44 inference rows.
- Fresh unpermuted same-config controls: 22 more inference rows; 66 total.
- All 22 controls reproduced historical predictions and selected-frame manifests
  exactly. The original Recent-8 experiment has a different whole-config hash,
  so permutation comparisons use fresh controls, not a waived config check.
- All 44 comparisons passed exact config suffix matching, prompt-option metadata
  agreement, unchanged non-option prompt content, ordered timestamp/frame input
  matching, causal/horizon/budget checks. All 66 profiles are present; no runtime
  failures or duplicate keys. The two seeds yield distinct permutations for every
  sample/policy pair.

## Measured results

Answer-content consistency maps the permuted prediction back to its original
option index; it is not raw-letter agreement. Each policy has 11 QA x 2 seeds =
22 comparisons. The comparisons are correlated repeated perturbations of 11 QA,
not 22 independent test samples.

| Policy | Same answer content | Content changed | Same label despite different content at that position |
|---|---:|---:|---:|
| Recent-8 | 6/22 (27.3%) | 16/22 (72.7%) | 6/21 eligible (28.6%) |
| Adaptive-v1 | 7/22 (31.8%) | 15/22 (68.2%) | 14/20 eligible (70.0%) |

The last denominator excludes cases where the originally selected option stays
at the same position: those cannot distinguish label following from content
following. This diagnostic is not calibrated confidence.

| Policy | Unpermuted control correct / 11 | Seed 17 correct / 11 | Seed 29 correct / 11 |
|---|---:|---:|---:|
| Recent-8 | 7 | 2 | 3 |
| Adaptive-v1 | 4 | 4 | 4 |

Adaptive's unchanged aggregate accuracy masks changed answers: across two seeds,
four correct-to-wrong transitions are offset by four wrong-to-correct transitions.
Recent-8 has ten correct-to-wrong and one wrong-to-correct transitions.
The subset selection favors particular historical outcomes, so these selected-case
accuracies cannot rank policies or replace the full 250-QA evaluation.

## Interpretation and next decision

With the recorded visual inputs fixed and unpermuted controls reproducible, the
answerer is sensitive to option ordering on these cases. This happens for
Recent-8 as well as memory routing, so memory compression is not a sufficient
explanation for all observed answer changes. The adaptive same-label effect is
evidence of positional sensitivity here, not proof that the model ignores video.

Priority: validate a readout-stability intervention on a video-disjoint or newly
sampled counting set, keeping visual selection fixed. Do not add routing rules
based on these 11 outcomes. Event evidence coverage and the 60s horizon still
require independent visual annotation; this experiment does not measure evidence
recall, true counting, or calibrated answer/clarification routing.

## Reproduce and inspect

Run `scripts/run_eval.py` with the frozen split and each policy. For the two
permutations add `--option-permutation-seed 17` or `29`; for replay controls omit
that flag. Use distinct seed-specific output files and `--resume`.

```powershell
python scripts/analyze_option_stability.py --original results/counting-diagnosis/replay-controls.jsonl --permuted results/counting-diagnosis/options-seed-17.jsonl results/counting-diagnosis/options-seed-29.jsonl --split configs/splits/adaptive-changed-counting-11.json --output-dir results/analysis/counting-option-stability
python -m pytest -q
```

The analyzer refuses missing/duplicate matrix rows, uncontrolled input changes
and output overwrite. Full comparison traces remain local at
`results/analysis/2026-10-26-counting-option-stability/comparisons.jsonl`.
Small public evidence: `option-stability-summary.json` and `audit.json`.
All 450 repository tests passed; independent code review approved the control
checks. Raw videos, model files, private paths and large logs are not included.

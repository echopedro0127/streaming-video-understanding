# Three-video frozen event-state validation

Human review is complete. [Scores and failure attribution](human-review-v1/README.md)
report8human events: raw precision72.7% versus confirmation100%, both recall100%.
All gains are suppression of the same startup black-frame artifact; confirmation
adds1.001s media delay. This is not a general temporal-memory accuracy gain.
The original cost/audit JSONs remain pre-label snapshots; full prediction
snapshots and anonymous human intervals are now released under human-review-v1.

## Selection and scope

Freeze the existing event-state implementation, model, prompt and raw1/1 versus
confirmation2/2 rules; change only source sample, horizon, frame count and query
times. No new model, dependencies, implementation changes or threshold tuning.

Use the existing derived tail250annotations. Eligibility uses question text
(`movie clips`, `inserted`) and integer timestamps120..240s; exclude development
videos80,84,100. Within each eligible video select minimum SHA256 of
`event-validation-v1:question_id`, then select3videos by ascending SHA256 of
`event-validation-v1:video:video_id`. No answers, previous correctness, predicted
counts or human intervals enter selection.27videos eligible.

| Video | QA source | Prefix | Classifications |
|---|---|---:|---:|
| sample_79 | sample_79_2 | 185s | 185 |
| sample_75 | sample_75_2 | 203s | 203 |
| sample_76 | sample_76_2 | 203s | 203 |
| Total | 3videos | 591s | 591 |

[Frozen cohort and source hashes](../../../configs/event-state-validation-v1/cohort.json).
The three exact source-file hashes differ; source-level/near-duplicate independence
is not established. These videos participated in older QA benchmarks. They were
not used to debug these state rules, but are not pristine unseen data.

Freeze commit `88dd576`; startup-error analysis declared before inference in
`3df23cd`. All other rule/model values equal `configs/event-state-v1.json`.
Both counters share each frame's single classification output. Sequential GPU
jobs avoid concurrent model residency. Query points are fixed at0/20/40/60/80/100%
of each prefix. All sampled timestamps are a predeclared secondary count analysis,
not independent QA samples or a replacement for the six primary queries.

## Completed execution and cost

All591classifications completed;591/591parse-valid, source/causal/prompt/mapping/
protocol/replay/runtime violations0. Replayed complete traces reproduce stored
states and summaries; full resumes skip185/203/203 with unchanged file hashes.
Development-case predictions remain unchanged.17existing event-state tests pass;
the unchanged implementation's prior full suite passed543tests. No new algorithm
or code changes were introduced in this validation stage.

| Case | Frames | Mean model latency | P95 | GPU allocated peak |
|---|---:|---:|---:|---:|
| sample_79 | 185 | 879.452ms | 1288.031ms | 1298.91MiB |
| sample_75 | 203 | 764.502ms | 1217.101ms | 1298.91MiB |
| sample_76 | 203 | 646.774ms | 754.443ms | 1298.91MiB |
| Pooled | 591 | 760.047ms | 1219.457ms | 1298.91MiB |

Pooled P50=653.190ms; cumulative model449.188s, classification pipeline451.478s,
processing1.309FPS. Pooled quantiles use all per-frame observations, not averages
of per-video quantiles. Timing excludes video preparation/decoding/export, model
loading and log append/fsync; no end-to-end hard realtime or speedup claim. Same
algorithm with different content/runtime conditions does not establish an
optimization versus the development video's timings. Each counter keeps11fields;
evaluator metadata and disk review files remain additional O(N) artifacts.

[Performance and hashes](performance.json), [audit](audit.json). Final artifact
checks were performed by the main agent, not an additional independent reviewer.
Every pair shares one exact JPEG hash; this does not certify source independence.

## Human review

Open local `results/event-state/validation-3/review-index.html`. Each linked page
shows its bounded silent video and source-frame reference, no model results or
prefilled intervals. Review the full prefix, annotate nonoverlapping intervals
and specify the counting convention. Save separate `event-labels.json` files in:

- `results/event-state/validation-3/sample79/`
- `results/event-state/validation-3/sample75/`
- `results/event-state/validation-3/sample76/`

Files have the same name but must not overwrite another video's annotation.
Human labels are self-reported, not independently authenticated. When conventions
differ, results must be separated or explicitly qualified, not silently pooled.
No human label or interval truth is inferred from model predictions.

During annotation, public reports withheld per-case predicted counts and
intervals; raw traces stayed local. All3labels are now submitted and full model
snapshots are linked in the human-review report. This reduced accidental
contamination but is not an independently double-blind study. Model choices,
state rules and the selected cases are never changed according to these results.

## Reproduction

For each case79,75,76, run the existing CLI with its corresponding config:

```powershell
python scripts/event_state_probe.py prepare `
  --config configs/event-state-validation-v1/sample79.json `
  --data-root <path-to-real-time-51-100> `
  --annotations <absolute-path-to-annotations-tail-250.jsonl> `
  --bundle results/event-state/validation-3/sample79

python scripts/event_state_probe.py run `
  --config configs/event-state-validation-v1/sample79.json `
  --manifest results/event-state/validation-3/sample79/manifest.json `
  --output results/event-state/validation-3/sample79/predictions.jsonl --resume
```

After actual human export, use the same case paths with `score --predictions ...
--labels ... --output ...`; no inference is rerun for scoring. Complete-prefix,
image/video hashes and deterministic state replay are validated. Do not replace
any earlier development/500-QA results or mix query timestamps across configs.

## Planned interpretation

Report each video first, then event micro/macro metrics, count MAE on six frozen
points, dense-time/onset errors and confirmation delay. Three video cases are
not18independent primary-query tests or591independent sample-level tests. Separate
startup black-frame effects from later temporal classification errors; if benefits
are only startup-artifact suppression, do not claim general temporal memory gains.
Short insertions/short presenter returns may be lost or merged by two-frame
confirmation. Report those failures rather than retuning on this cohort.

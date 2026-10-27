# Exact-black artifact control (posthoc)

This separately labelled posthoc replay tests a simpler explanation for the
three-video confirmation gain. The cohort and its errors were already inspected;
this is not a new held-out or preregistered validation. All original predictions,
human annotations and baseline results are preserved. No model calls were added.

## Fixed rule and source verification

At every sampled timestamp, decode the existing source-verified JPEG to RGB.
Only if **every channel value is exactly zero**, supply unknown instead of the
stored category to a raw1/1 counter. Unknown holds active state and resets pending
streaks; it neither opens nor closes an event. No brightness threshold was tuned.
The gate is independent of questions, options, annotations and future frames.

The CLI locks all three manifest/prediction/label hashes, verifies ordered source
images and the original model/prompt/causal/state protocol, and requires complete
prefixes. It rejects existing output paths. The scorer and IoU>=0.5 matching are
unchanged. Human labels remain single-reviewer self-reported annotations.

## Results: 3 videos, 8 events, 591 sampled timestamps

| Metric | Raw1/1 | Confirmation2/2 | Exact-black + raw1/1 |
|---|---:|---:|---:|
| Predicted / matched events | 11 / 8 | 8 / 8 | 8 / 8 |
| Micro event precision | 72.7% | 100% | 100% |
| Micro event recall | 100% | 100% | 100% |
| Count MAE, 18 preset queries | 1 | 0 | 0 |
| Correct dense-time counts | 0/591 | 586/591 | 588/591 |
| Dense count MAE | 1.00508 | 0.00846 | 0.00508 |
| Mean signed media confirmation delay | 0.405s | 1.406s | 0.405s |

Exactly three frames are gated, one at t=0 in each video; all have the same JPEG
hash. The exact-black control matches confirmation's event and coarse count
metrics without its additional1.001s confirmation delay. The remaining three
dense-time errors are slight early detections relative to human onset labels;
these boundary differences are not future-frame access. Neither18queries nor
591timestamps are independent video-level test samples.

**Interpretation:** This cohort's measured precision benefit is explained by
startup-artifact suppression. It does not establish a general benefit of temporal
debouncing, semantic memory or retrieval. This control is posthoc and therefore
does not retrospectively replace the original frozen validation.

## Development check (excluded from aggregates)

The separate known sample80 prefix contains185frames and2human events. Only t=0
is gated. Gate+raw and confirmation both have precision/recall100% and six-query
MAE0; mean signed media delays are0.2875s and1.2885s respectively. The previously
adjudicated presenter frame145.145s is **not** gated: channel maxima255/252/241.
This checks the specific dark scene, not all low-light cases.

## Reproduce and artifacts

From the repository root, with the existing local bundles and original submitted
human labels (not redistributed):

```powershell
python scripts/black_frame_control.py --output results/event-state/black-frame-control-replay.json
python -m pytest tests/test_black_frame_control.py tests/test_event_state.py -q
```

- [Full anonymous report and per-case dense mismatches](report.json).
- [Separate development check and dark-frame pixel evidence](development-check.json).
- [Original frozen three-video scoring](../../2026-10-26/event-state-validation/human-review-v1/README.md).

This replay reuses classifications; it does not actually skip model calls or
measure online throughput, latency or GPU savings. Media-time confirmation delay
must not be presented as wall-clock inference latency. Local metadata/images are
O(N); only the counter itself has a constant number of fields.

Verification:22focused tests and548full-suite tests passed. Original raw and
confirmation per-case scores were reproduced exactly; all locked source hashes
remain unchanged. Artifact checks found no exported reviewer identities or local
absolute paths. Review was performed by the main agent, not an independent reviewer.

## Next validation

Freeze raw, exact-black+raw and confirmation rules before evaluating new challenge
videos. Include short insertions/returns, dark but meaningful content and nonzero
compression noise. All-zero gating may miss nearly black artifacts or suppress a
legitimate all-black scene; confirmation can miss brief events. Do not tune either
on the new evaluation labels. Report event recall, false positives, dense count
errors and media confirmation delay together.

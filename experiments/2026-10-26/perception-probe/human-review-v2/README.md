# User-adjudicated frame labels (v2)

The user explicitly finalized frame145 at145.145s (`302436c8409c`) as
`presenter` after error review. This supersedes both the submitted transition
label and an earlier conflicting confirmation in the conversation. No other
label or model prediction changed. This is **posthoc human adjudication**, not
an independent blinded second annotation.

The original [submitted-v1 scores](../human-review-v1/README.md) remain intact.
Local revision files are `human-labels-adjudicated-v2.json` and
`human-scores-adjudicated-v2.json`; the original `human-labels.json` export is
also retained unchanged. Use the explicit v2 filenames for subsequent scoring.

| Metric | Original options | Cycled options |
|---|---:|---:|
| v1 accuracy, retained for provenance | 23/24 (95.8%) | 22/24 (91.7%) |
| v2 accuracy | 24/24 (100%) | 23/24 (95.8%) |
| v2 presenter recall | 20/20 | 20/20 |
| v2 inserted-scene recall | 4/4 | 3/4 |
| v2 binary balanced recall | 100% | 87.5% |

All24labels are resolved:20presenter,4inserted_scene, no transition/uncertain
examples. Transition-class recognition is therefore **unmeasured**, not perfect.
The remaining error is frame85 at85.085s: cycle predicts presenter for the
user-labelled inserted dance scene; original predicts inserted_scene.

Only the label revision changes these scores. All48model generations, image
hashes and the23/24 semantic-consistency measurement remain unchanged. No new
inference or training occurred. Accuracy is on24purposeful frames from3videos
with4positive examples; it is not500-QA streaming understanding accuracy or
evidence that general event counting is solved. Do not advertise the posthoc
24/24 as a general perfect-recognition claim.

[Scores](scores.json), [revision provenance](audit.json).
Next evidence remains continuous-video event boundaries and counting convention,
followed by a causal perception-to-state experiment.

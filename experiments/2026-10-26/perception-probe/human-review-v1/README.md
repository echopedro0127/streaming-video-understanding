# Submitted human-label scores (v1)

Adjudication update: the user finalized the dark frame as presenter. See
[v2 revision and scores](../human-review-v2/README.md); v1 below is preserved as
submitted and is not silently overwritten.

The user exported24frame labels from the local review page after inference had
already completed. Source is a single reviewer's self-reported human annotation,
not independently authenticated or double-blind ground truth. Frame-level model
responses were not shown by the page. Original submitted labels and scores are
retained locally as `human-labels-submitted-v1.json` and
`human-scores-submitted-v1.json`; reviewer identity is not republished here.

All24labels match the frozen manifest, with no duplicates or omissions:
19presenter,4inserted_scene,1transition,0uncertain. Resolved coverage24/24.

| Metric | Original option order | Cycled option order |
|---|---:|---:|
| Correct | 23/24 | 22/24 |
| Resolved-frame accuracy | 95.8% | 91.7% |
| Presenter recall | 19/19 | 19/19 |
| Inserted-scene recall | 4/4 | 3/4 |
| Transition recall | 0/1 | 0/1 |
| Presenter/inserted balanced recall | 100% | 87.5% |

Balanced recall averages the presenter and inserted-scene recalls only; it
**excludes the transition class** and is not overall accuracy. Majority-presenter
on these submitted labels is19/24(79.2%). These are24purposeful frames from just
3videos, including only4movie/performance frames. No generalization, statistical
superiority or improved video-QA accuracy is established.

## Errors and label ambiguity

Two unique error frames under submitted labels:

- `sample_80_2`,85.085s, frame85: human inserted_scene. Original correctly selects
  inserted_scene; cycle selects presenter. The image shows two dancers on a
  performance set, illustrating residual option-order sensitivity. There is no
  measured internal causal explanation for the category change.
- `sample_80_2`,145.145s, frame145: human transition, both modes presenter. Visual
  inspection still reveals a presenter and teaching board under low light. The
  review instructions say lighting changes alone are not blank transitions.
  **User adjudication was requested; this submitted label is retained unchanged.**
  No assistant-generated correction is used in v1 scores.

Any later label change must be reported as a separate revision after error
review, retaining the submitted version. It must not silently turn these results
into a perfect-recognition claim. Label ambiguity cannot be resolved by treating
the model's answer as ground truth.

## Interpretation and next evidence

This small labelled probe supports checking a perception-to-state pipeline:
the model can distinguish these submitted presenter/inserted examples much more
successfully than it answered the earlier multi-frame counting questions.
However, task wording, one-frame versus eight-frame input, image resolution and
cohort all differ. These results do not isolate why counting failed, establish
robust event boundaries, or validate cumulative event counts.

Next verify continuous-video boundaries and the counting convention, then test
causal category smoothing and event-entry state against those annotations.
Changing every film shot, a brief presenter return, and a dark lighting effect
must not automatically become distinct counted insertions. Keep this24-frame
accuracy separate from the100-video/500-QA benchmark and resume claims.

[Scores](scores.json), [provenance audit](audit.json), [error metadata](errors.json).
All48model generations and their original hashes remain unchanged. No new
inference, training, model download or code changes were needed for scoring.

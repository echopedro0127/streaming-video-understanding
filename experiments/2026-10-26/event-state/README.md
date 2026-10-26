# Causal perception-to-event-state diagnostic

## Scope and frozen rules

One previously inspected video/query, `sample_80_2`, prefix0-185s. This is a
known-case architectural diagnostic, not an untouched test or the500-QA benchmark.
Configuration and implementation were frozen in `9b9ddd3` before classification.

Reuse SmolVLM2-500M and the four single-frame descriptions in the perception
probe, original option order, max8generated tokens, seed42. At1FPS classify
185current images (IDs0..184, actual times0..184.184s). Each model request has
only its current image and current timestamp boundary, no visible timestamps,
historical images, event labels, QA answers or cumulative count. Images are
max640-side JPEGs; classification and review use hash-bound material.

The exact same category sequence feeds two policies:

- **Raw1/1:** inserted_scene enters immediately; presenter exits immediately.
- **Debounced2/2:** require two consecutive inserted_scene frames to enter and
  two consecutive presenter frames to exit.

Transition, uncertain and invalid reset confirmation streaks but hold active
status. Count increments at confirmation, never backdated to estimated onset.
An event's estimated onset and its later confirmation timestamp are stored
separately. Active-at-cutoff intervals would be marked censored. No rules were
tuned after observing this run's predictions or endpoint answer.

## Observed predictions

| Measurement | Raw1/1 | Debounced2/2 |
|---|---:|---:|
| Final predicted event count | 3 | 2 |
| Shared classification calls | 185 | Same185, no extra calls |
| Query counts at20/60/90/110/130/185s | 1/1/2/3/3/3 | 0/0/1/2/2/2 |
| Human event precision/recall | Pending | Pending |
| Human interval-based count MAE | Pending | Pending |

Raw includes an initial candidate[0,1.001)s. The initial JPEG's RGB means are
[0,0,0], yet the model selects inserted_scene. Two-frame confirmation does not
admit this isolated prediction. This is an observed startup failure candidate,
not a substitute for complete-prefix human event annotation.

Both policies also predict estimated intervals[82.082,98.098) and
[100.100,124.124)s. Debounced entry confirmations occur at83.083 and101.101s,
1.001s after the first candidate frames; exit confirmations at99.099 and125.125s.
That is media-time confirmation delay, not end-to-end wall-clock answer latency.
No human event labels were generated or inferred from these intervals.

## Existing endpoint QA comparison: one case only

The existing dataset record at185s asks the total number of movie insertions;
its numeric ground truth is2. Reuse prior model outputs without rerunning them:

| Method | Predicted count | Absolute endpoint error | Calls / visual images |
|---|---:|---:|---|
| Recent-8 direct QA | 4 | 2 | 1 /8 |
| Change-4+4 direct QA | 4 | 2 | 1 /8 |
| Whole-prefix bins8 direct QA | 4 | 2 | 1 /8 |
| Raw state | 3 | 1 | Shared185 /185 |
| Debounced state | 2 | 0 | Shared185 /185 |

This single endpoint match does **not** validate event boundaries or a general
accuracy improvement. Human IoU matching and frozen six-query count metrics
remain null until the user supplies intervals. The old ground-truth count is
not used to classify frames, select rules or generate boundary annotations.

Comparisons are **not compute-matched**:185single-image calls versus one8-image
QA call; full-prefix processing versus60s candidate history for Recent/Change;
different task prompts and image resolutions. The prefix-bin control has full
history scope but still only8images. Only raw-versus-debounced is a controlled
state-rule comparison sharing exactly the same perception costs/predictions.

## Measured efficiency and state

- Model mean932.382ms/frame, P50835.942ms, P951329.697ms.
- Cumulative model172.491s; classified-frame pipeline173.272s.
- Processing throughput1.068FPS; allocated GPU peak1298.91MiB (~1.27GiB).
- 185 model calls and185input images total; no second pass for the other counter.
- Each counter has11scalar/optional fields; no historical image/token buffer.

Pipeline timing excludes model loading, prefix preparation/decoding, review-video
encoding and log append/fsync. It is not a measured live ingest-to-answer latency
or a hard realtime guarantee; P95 exceeds the1s sampling interval. GPU allocated
peak excludes reserved/context memory. The evaluator keeps manifest/results
metadata and disk JPEGs for audit, so the whole program is **not** claimed O(1)
space. Only online decision state has bounded field count; current-image/model
buffers remain additional resources. Generic video QA is not supported by an
event counter specialized to these categories.

## Audit

185complete ordered records,185/185parse-valid;144presenter and41inserted_scene.
No missing/duplicate frames, causal/input/prompt/mapping/protocol/state-replay or
runtime violations. All8overlap frames reproduce prior single-frame JPEG hashes,
prompts, answers and semantic categories. Contiguous-prefix replay reconstructs
both states and rejects gaps; smoke/full resumes skip3/185 unchanged.

543repository tests passed. Independent pre-inference code review approved after
fixing a short-prefix-video validation gap. Final artifact verification by the
main agent checked hashes, all source rows and deterministic replay; an additional
independent final reviewer was unavailable due to its usage limit. Do not describe
that final artifact audit as independently completed.

Review clip185.0s,10FPS/silent,3675513bytes;185JPEGs11101104bytes. Images/video,
raw predictions and future human labels stay under ignored results/. Public
[manifest](manifest.json), [summary](summary.json), [audit](audit.json) and
[endpoint comparison](qa-case-comparison.json) contain metadata only.

## Human annotation and commands

Open local `results/event-state/sample80-prefix/event-review.html`. Watch the
complete prefix and mark nonoverlapping insertion intervals[start,end). Internal
film shot changes do not automatically create new insertions. Explicitly choose
whether a brief presenter return splits an insertion. Mark active-at-cutoff
events censored; if no intervals exist, explicitly confirm no events. Export
`event-labels.json` into that same E-drive folder. No model predictions or guessed
intervals are prefilled. This is single-reviewer self-reported evidence, not an
independently authenticated or double-blind annotation.

```powershell
python scripts/event_state_probe.py prepare `
  --config configs/event-state-v1.json `
  --data-root <path-to-real-time-51-100> `
  --annotations <absolute-path-to-annotations-tail-250.jsonl> `
  --bundle results/event-state/sample80-prefix

python scripts/event_state_probe.py run `
  --config configs/event-state-v1.json `
  --manifest results/event-state/sample80-prefix/manifest.json `
  --output results/event-state/sample80-prefix/predictions.jsonl --resume

# After actual human annotation:
python scripts/event_state_probe.py score `
  --config configs/event-state-v1.json `
  --manifest results/event-state/sample80-prefix/manifest.json `
  --predictions results/event-state/sample80-prefix/predictions.jsonl `
  --labels results/event-state/sample80-prefix/event-labels.json `
  --output results/event-state/sample80-prefix/human-event-scores.json
```

Scoring validates complete-prefix provenance, then uses one-to-one IoU>=0.5
matching, precision/recall/F1 and count MAE/accuracy at the six frozen timestamps.
Those timestamps are correlated points on one video, not six independent test
videos. Very brief events can be missed by2-frame confirmation; long uncertain
stretches can merge events. Confirm these trade-offs on human intervals before
expansion, and use separate videos for future generalization checks.

Run `event-state-v1-e537544e096c`. Same cached model revision and6GB RTX3060Laptop
environment as prior probes; no new dependencies, model or dataset downloads.

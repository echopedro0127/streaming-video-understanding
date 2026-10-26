# Causal perception-to-event-state diagnostic

Goal: test a causal cumulative-event state on sample80_2,0-185s, using the
existing single-frame model and no new training/dependencies. Exploratory known
video, not a fresh test or proof of memory-policy gains.

Fixed protocol before predictions: sample IDs0..184 at1FPS,185causal frames;
JPEGmax640/Q95, same four descriptions/original option order/max8tokens/seed42.
One model call per frame, actual current timestamp as request boundary. No
future image, label, count or event interval enters classification or state.

Run two counters on exactly the same predictions: raw1/1 and debounced2/2 enter/
exit confirmations. Only inserted_scene enters; only presenter exits. Unknown,
transition and invalid reset streaks and hold active status. Count increments
at confirmation, never retroactively; interval estimates may use first streak
timestamp but store confirmation availability separately. Active-at-end events
are censored. State has constant field count, one current image; offline JPEGs,
manifest and diagnostic logs are not historical model context.

Resume must replay an exact contiguous prefix and validate stored state traces;
stop on errors rather than skip a failed timepoint. Include model/config/manifest
hashes. Record per-frame latency/P50/P95, throughput, GPU allocated peak, total
calls/images; distinguish185calls from earlier1call/8image baselines and scope/
resolution/task changes. No speedup or fair matched-compute claim.

Generate silent185s review clip + local blinded event interval page. Human marks
nonoverlapping[start,end) insertions and confirms complete prefix review; no
predictions/guessed intervals prefilled. Export explicit human_self_reported
labels bound to manifest. Score event IoU>=0.5 with one-to-one matching and
countMAE/query accuracy at frozen6timestamps only after actual human labels.
Separate official timestamped-QA counts from human boundary truth.

Files: streaming/event_state.py, evaluation/event_state.py,
scripts/event_state_probe.py, assets/event-review.html, tests/test_event_state.py,
configs/event-state-v1.json. Minimal optional embedding flag in existing
data/perception_review.py; default24frame probe behavior unchanged.

- [x] RED/GREEN causal state tests: confirmation availability, unknown holding,
  flicker,2separate events, prefix invariance, invalid chronology.
- [x] RED/GREEN interval scoring/human provenance/contiguous resume and export;
  implement CLI prepare/run/score and blinded UI.
- [x] Full tests and independent review; export185frames and review clip;
  run185model calls, verify resumed replay and old baselines remain unchanged.
- [x] Audit/publish predictions and costs, existing QA comparisons with explicit
  scope/cumulative-cost differences; no human event metrics until annotated.
- [x] Update project plan/status, sync branch, provide human review instructions.
- [x] Human complete-prefix intervals and final boundary/count scoring.

Pre-inference:543tests passed. Independent review fixed a short-clip validation
gap with a failing then passing regression. Encoding duration must match cutoff
within0.11s. Full review approved; actual clip185.0s,185JPEGs through184.184s,
11101104JPEG bytes +3675513video bytes. Manifest SHA
5a08c8a330eddb07e23a3616c561db7cfb7541d01872d6ff89da209c24fdb6a1.
UI export logic tested with Node; browser playback/file picker not manually
verified. No guessed human intervals or scores produced.

Run complete185calls: raw3/debounced2 at185s, endpoint QA truth2.185/185parsed,
old8overlap frames reproduce; all full-prefix source/hash/state checks passed.
Mean932.382ms, cumulative172.491smodel, allocated GPU peak1298.91MiB.
Smoke/full resumes skip3/185 unchanged. Independent code review approved;
main-agent final replay audit complete, additional final reviewer quota exhausted.
Human event intervals and corresponding metrics remain pending; published output
is under `experiments/2026-10-26/event-state/`.

Human event scoring now complete with2submitted intervals. Raw P/R=2/3,2/2;
confirmation P/R=2/2,2/2. Frozen-query countMAE1/0, media confirmation delay
.2875/1.2885s. Posthoc all185timestamps expose two lagged confirmation counts,
183/185correct; not a replacement for frozen6queries. No new model calls or
rule changes. Human scores and count figures published separately, original
pre-label snapshots retained. Single-reviewer/known-video limitations apply.

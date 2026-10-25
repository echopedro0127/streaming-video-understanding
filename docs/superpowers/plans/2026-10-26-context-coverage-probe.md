# Causal temporal coverage probe

Goal: distinguish recent-window selection from the 60s candidate cutoff under
the same eight-frame input budget, without altering the MCQ readout.

Frozen cohort: reuse counting-readout-video-disjoint-20.json. This is a controlled
follow-up on inspected pilot data, not an untouched generalization test.

Conditions (all MCQ, no timestamp prompt):

- recent-8: last eight sampled frames within [max(0,t-60),t].
- time-bin-60s-8: divide that interval into eight equal time bins; within each
  occupied bin keep the frame nearest its centre, ties to earlier timestamp/id.
- time-bin-prefix-8: same bin rule over [0,t].

These binned policies are new diagnostic conditions, not the existing endpoint-
index Uniform-8 baseline. Query t is known when constructing bins. No question,
options, answer or prediction influences frame selection. No future frame enters
the bank, inference or rendering. This is query-aware prefix evaluation, not a
persistent query-agnostic streaming-memory method or evidence oracle.

Implement a bounded image bank: three selectors retain at most24 distinct images
plus one incoming decoded image. Close evicted images immediately and all bank
images on success/error. Decoder/framework buffers are outside this bound. No
linear image archive or second decode is required.

Rotate three-condition generation order by index in sorted full split (7/7/6
first positions over20). Run60 generations, same model/config, source frames,
prompt/options, and invalid-MCQ-as-incorrect pilot rule. Store all three results
atomically per QA, with raw text, manifests, temporal spans/ages, shared decode
and peak memory. Resume IDs include protocol/split/annotation hashes.

Blinded qualitative material: before inference outcomes, choose one QA per query-
time stratum <=60 / (60,180] / >180 using minimum SHA256 of
`coverage-review-v1:stratum:sample_id`. Render causal 1-FPS prefix pages and selected
frame sheets with timestamps but no options, GT or predictions. Sampled inspection
cannot certify event counts; any assistant observations remain unverified until
human annotation, and no evidence-recall metric is fabricated.

- [x] Test and implement streaming/context_bank.py: bin ties, short prefixes,
  future/backward/invalid-frame rejection, <=24 retained frames, close on eviction.
- [x] Add bounded timeline rendering and tests; validate before rendering future data.
- [x] Add run_context_coverage.py: complete per-QA triples, rotated order, errors,
  resume and summary; preserve existing baselines/config hashes.
- [x] Full tests and code review; real three-QA smoke, unchanged resume, extend20.
- [x] Audit20 rows/60 calls, source Recent-8 reproduction, input equality for t<=60,
  temporal extent metrics and paired video bootstrap intervals.
- [x] Inspect preselected contact sheets as sampled assistant review; publish
  observations separately from verified evidence labels, plus measured results.

Completed: 20 triples / 60 generations, no runtime or audit violations; all three
methods 3/20 correct. Mean spans 7.007 / 45.696 / 200.350 seconds. Broader span
alone did not improve this pilot. Five short-prefix controls identical; old
Recent-8 answers/manifests reproduce20/20; resume skips20 with unchanged hash.
Tests492 passed; independent artifact review found no P1/P2 concerns. Three
assistant-reviewed cases remain pending human verification. Public artifacts:
`experiments/2026-10-26/context-coverage/`.

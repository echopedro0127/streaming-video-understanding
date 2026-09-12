# Failure Analysis

This is a manual review of the deterministic 30-record subset generated from
the four full-policy failure JSONL files. The subset contains 30 unique
`(sample_id, policy_id)` records: Recent-8 (8), Uniform-8 (8), Recent-4 +
Uniform-Memory-4 (7), and Recent-4 + Timestamp-aware-Memory-4 (7). All 30
selected manifests passed the causal audit; no selected timestamp is after its
query timestamp.

The labels below describe the most plausible failure mechanism visible in the
selected frames. They are not an automated diagnosis and should not be read as
a causal estimate for the full benchmark.

## Category counts

| Category | Cases |
|---|---:|
| forgotten_early_evidence | 3 |
| memory_compression_loss | 5 |
| distractor_frames | 2 |
| temporal_ordering_error | 6 |
| semantic_retrieval_failure | 14 |
| Total manually reviewed | 30 |

## Reviewed cases

| Case | Policy | Sample | Query (s) | Prediction → GT | Manual category | Evidence from selected causal frames |
|---:|---|---|---:|---|---|---|
| 01 | Recent-8 | sample_14_1 | 20.0 | C → D | forgotten_early_evidence | The question asks about the very beginning, while the window contains only 13–20s. |
| 02 | Uniform-8 | sample_2_4 | 211.0 | B → A | memory_compression_loss | The 151–211s history is represented by eight sparse frames and misses the decisive penalty/celebration transition. |
| 03 | Recent-4 + Uniform-Memory-4 | sample_35_2 | 360.0 | C → B | semantic_retrieval_failure | The person and kitchen tools are visible, but the fine-grained knife/towel combination is confused. |
| 04 | Recent-4 + Timestamp-aware-Memory-4 | sample_10_4 | 304.0 | D → C | distractor_frames | The final grill frames contain several similarly colored foods, making buns and pineapple easy to confuse. |
| 05 | Recent-8 | sample_39_2 | 105.0 | C → A | forgotten_early_evidence | A NOT-mentioned countertop item requires information outside the 98–105s local window. |
| 06 | Uniform-8 | sample_19_1 | 181.0 | A → B | semantic_retrieval_failure | The baking dish is present but small; the error is a color attribute mismatch rather than missing time coverage. |
| 07 | Recent-4 + Uniform-Memory-4 | sample_12_3 | 185.0 | A → D | semantic_retrieval_failure | The last cooking frames show multiple hand-held tools and the red tongs are not isolated clearly. |
| 08 | Recent-4 + Timestamp-aware-Memory-4 | sample_26_1 | 127.0 | B → A | distractor_frames | Seasoning and oil-related actions share the same kitchen composition in the 123–126s local frames. |
| 09 | Recent-8 | sample_22_3 | 183.0 | B → D | semantic_retrieval_failure | Nail color is a small visual attribute and is not legible at the sampled resolution. |
| 10 | Uniform-8 | sample_5_5 | 443.0 | B → D | memory_compression_loss | Sparse 383–443s samples do not preserve a clear goalkeeper-back close-up for the kit number. |
| 11 | Recent-4 + Uniform-Memory-4 | sample_36_1 | 202.0 | A → B | semantic_retrieval_failure | Cabinet color is visible but affected by scene lighting; the model selects the wrong nearby color option. |
| 12 | Recent-4 + Timestamp-aware-Memory-4 | sample_1_1 | 4.0 | A → D | semantic_retrieval_failure | The 3–4s frames visibly contain the Aeromexico/La Liga logo combination, but logo text is misread as FIFA. |
| 13 | Recent-8 | sample_16_5 | 659.0 | A → C | forgotten_early_evidence | The local 651–658s window shows cooking, while the question targets a broader video-level activity. |
| 14 | Uniform-8 | sample_3_5 | 171.0 | C → A | semantic_retrieval_failure | The final football frames show two people interacting, but the exact gesture is visually ambiguous. |
| 15 | Recent-4 + Uniform-Memory-4 | sample_26_5 | 386.0 | C → A | temporal_ordering_error | “Why” requires linking the earlier pour to the later pan-shaking action; sparse history weakens that causal link. |
| 16 | Recent-4 + Timestamp-aware-Memory-4 | sample_8_5 | 256.0 | A → D | temporal_ordering_error | The sampled penalty sequence ends before the outcome is fully unambiguous, so post-kick state is confused. |
| 17 | Recent-8 | sample_12_3 | 185.0 | A → D | semantic_retrieval_failure | The local cooking frames contain hand-held utensils, but the fine-grained red-tongs option is missed. |
| 18 | Uniform-8 | sample_41_4 | 120.0 | B → A | memory_compression_loss | The sparse 60–120s history contains several pouring scenes and does not preserve the decisive action clearly. |
| 19 | Recent-4 + Uniform-Memory-4 | sample_2_1 | 10.0 | C → A | semantic_retrieval_failure | The jersey back and number are visible in an early frame, but the player/number association is misread. |
| 20 | Recent-4 + Timestamp-aware-Memory-4 | sample_37_2 | 178.0 | A → B | temporal_ordering_error | The question asks what happens after the egg is held up; the transition to placing it is easy to reverse. |
| 21 | Recent-8 | sample_6_1 | 14.0 | B → D | temporal_ordering_error | The player’s action changes across 7–14s from standing/walking to kneeling, making the endpoint easy to miss. |
| 22 | Uniform-8 | sample_21_2 | 165.0 | A → C | semantic_retrieval_failure | Noodles and a wooden spoon appear in the same cooking scene; the object identity is confused. |
| 23 | Recent-4 + Uniform-Memory-4 | sample_21_2 | 165.0 | A → C | memory_compression_loss | Compared with the uniform policy, the four historical slots leave only sparse earlier evidence for the short noodle action. |
| 24 | Recent-4 + Timestamp-aware-Memory-4 | sample_21_3 | 185.0 | B → C | semantic_retrieval_failure | The utensil is present near the query, but whisk and tongs are visually similar at the sampled views. |
| 25 | Recent-8 | sample_4_4 | 131.0 | B → D | semantic_retrieval_failure | The scoreboard is small in the upper corner; the updated score requires OCR-like reading. |
| 26 | Uniform-8 | sample_9_5 | 75.0 | B → A | temporal_ordering_error | “Most likely do next” requires a future action that is intentionally unavailable under the causal protocol. |
| 27 | Recent-4 + Uniform-Memory-4 | sample_32_1 | 122.0 | A → B | semantic_retrieval_failure | The knife is visible while chopping the onion, but the option wording causes a fine-grained tool mismatch. |
| 28 | Recent-4 + Timestamp-aware-Memory-4 | sample_5_5 | 443.0 | B → D | memory_compression_loss | Only one historical close-up around 420s exposes the goalkeeper back; the remaining frames show distant field views. |
| 29 | Recent-8 | sample_5_3 | 147.0 | B → D | temporal_ordering_error | The kick trajectory and the player’s celebration are sampled around the transition, making the outcome easy to misorder. |
| 30 | Uniform-8 | sample_19_5 | 726.0 | D → C | semantic_retrieval_failure | The countertop material is visible but texture/color cues make marble and stainless steel easy to confuse. |

## Representative cases

- Case 01 is the clearest bounded-window limitation: a recent-only policy
  cannot answer a query about the beginning when the relevant evidence has
  left the local buffer.
- Case 12 shows that timestamps prevent leakage but do not solve fine-grained
  logo recognition; the relevant frame is present and the semantic read is
  still wrong.
- Cases 22–23 use the same query under two policies. Both select mostly
  near-query kitchen views, but the memory policy has fewer historical slots,
  illustrating how short actions can be lost under compression.
- Case 25 is an OCR-like failure: the scoreboard is present but too small in
  the fixed visual budget.
- Case 29 is a temporal outcome failure: answering “what happened” requires
  correctly ordering the kick, ball trajectory, and celebration.

The selected contact sheets are generated locally under
`results/failure_review/renders_all/`. Raw videos, per-sample JSONL, and local
review artifacts remain excluded from Git by design.

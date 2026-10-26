# Fixed-selection visual and option sensitivity

## Why this diagnostic

The prior Change-4+4 trial scored 4/20, but a constant-C rule had identical
correctness. This follow-up holds its selected frames fixed to test sensitivity
to visual content, image order and displayed option positions. No memory threshold
is tuned, no model/backend/prompt baseline is modified, and no new dependencies
or model downloads are introduced.

Same inspected 20 QA / 20 videos, previously seen in benchmark runs; not a pristine
test or the 500-QA benchmark. Source selections are pinned to audited run
`change-memory-v1-0fbce55d18ea`, SHA256 in `audit.json`. Decode one causal prefix
per QA, retaining only the eight specified frames. First condition rotates 5/5/5/5.
All methods use SmolVLM2-500M, eight images, original MCQ instructions and eight
generation tokens. Visual encoding runs separately for all four conditions.

| Condition | Image input | Displayed options |
|---|---|---|
| Original | Saved Change-4+4 selection | Original |
| Blank | Eight same-size RGB(127,127,127) images | Original |
| Reverse | Same eight source images in reverse order | Original |
| Option-cycle | Original images in original order | New positions contain old B,C,D,A |

Timestamp text is absent in every condition; requests have empty timestamp
metadata. Records separately store actual source timestamps and actual input
order. Blank records contain no claimed source evidence; reverse is an explicit
nonchronological counterfactual, not a normal causal policy. All real images
originate at or before the query timestamp. Predictions under option-cycle are
mapped back to original answer content before scoring; GT never enters inference.

## Measured results

| Metric | Original | Blank | Reverse | Option-cycle |
|---|---:|---:|---:|---:|
| Correct / 20 | 4 | 4 | 4 | 8 |
| Accuracy | 20% | 20% | 20% | 40% |
| Parse-valid / 20 | 20 | 20 | 20 | 20 |
| Same answer content as original | 20/20 | 17/20 | 18/20 | 1/20 |
| Same displayed letter as original | 20/20 | 17/20 | 18/20 | 18/20 |
| Mean model latency | 3.125s | 2.899s | 2.967s | 2.854s |
| P95 model latency | 4.463s | 3.238s | 3.527s | 3.321s |

Blank and reverse change 3 and 2 answer contents respectively, all wrong-to-wrong:
no accuracy difference, helped 0 / hurt 0. Degenerate empirical bootstrap [0,0] pp
does not establish population equivalence. Reverse invariance is also compatible
with correct total-count reasoning: such questions are often order-invariant.

Option-cycle changes 19/20 answer contents while preserving 18/20 displayed
letters (all 20 pairs valid; derangement moves every original option). It helps 8
and hurts 4: delta +20 pp, paired video-bootstrap CI95 [-15,+55] pp, 2,000 resamples,
seed 42. This is **not an accuracy improvement from a video method**.

## Position-bias control and interpretation

Displayed-letter distributions:

| Condition | A | B | C | D |
|---|---:|---:|---:|---:|
| Original | 0 | 0 | 19 | 1 |
| Blank | 2 | 0 | 18 | 0 |
| Reverse | 1 | 0 | 19 | 0 |
| Option-cycle | 1 | 0 | 19 | 0 |

Original GT positions are A 5 / B 3 / C 4 / D 8. Always choosing displayed C gets 4/20 in
the original layout and 8/20 after the frozen cycle, because new C contains old D.
These posthoc constant-position controls exactly match the measured accuracies;
they were not used for frame selection, cohort selection or tuning. Report the
content-consistency and letter-stickiness results together, not just 40% accuracy.

On this small counting cohort, replacing visual evidence with blanks leaves
17/20 answers unchanged, and option order strongly changes selected content.
These findings do not demonstrate reliable visual grounding of the prior single-
case gain. They also do not prove the network ignores all images: some predictions
change, blanks are out of distribution, only one permutation and one task family
are tested, and no internal mechanism or calibrated confidence is measured.

## Audit and runtime

20 complete quadruples / 80 generations; no missing/duplicate records, prompt,
provenance, mapping, protocol, future-frame, frame-budget or runtime violations.
Original manifests and answers/raw text/counts/validity reproduce the earlier
Change-4+4 run 20/20. First condition counts 5 each. Source bounds are validated before
retention/inference; unselected decoded images close immediately. Retain 8 source
images and generate 8 blank images, plus incoming decode; decoder/framework buffers
excluded. Shared GPU allocated peak 3537.18 MiB, shared decode 35.219s/QA. These are
not per-condition memory measurements or a demonstrated deployed speedup.

Four-QA smoke and full resumes skipped 4/20, preserving raw SHA256. Full suite:
513 tests passed. Pre-inference review fixed an over-permissive source guard;
a fresh-output regression first reproduced acceptance of an alternative source,
then passed after exact frozen-ID/hash checks. Existing baselines/results remain
unchanged. Independent full artifact review found no P1/P2 issues. Actual expanded
processor tokens/request flags are not logged per
response; prompt construction and empty timestamps are covered by code/tests.

- [Summary](summary.json), [audit and checksums](audit.json).
- Local ignored raw file: `results/visual-sensitivity/pilot-v1.jsonl`.
- Run ID: `visual-sensitivity-v1-95b23e88ccb2`.
- Implementation commit: `a6e5768`.
- Frozen cohort: `configs/splits/counting-readout-video-disjoint-20.json`.
- Annotation provenance remains derived prior baseline records, not a fresh
  official annotation download. Human event verification remains pending.

## Replay and reproducibility limit

```powershell
python scripts/run_visual_sensitivity.py `
  --config configs/adaptive-semantic-1fps.yaml `
  --data-root <path-to-real-time-51-100> `
  --annotations <absolute-path-to-annotations-tail-250.jsonl> `
  --split configs/splits/counting-readout-video-disjoint-20.json `
  --source results/change-memory/pilot-v1.jsonl `
  --output results/visual-sensitivity/pilot-v1.jsonl --resume
```

This frozen replay requires the exact archived local source JSONL. Its SHA256
includes timing fields, so regenerating the source experiment normally changes
its bytes and will be rejected. Raw files are Git-ignored; this command is **not
turnkey public reproduction** from the published small summaries alone. A portable
selection-only provenance artifact is future work. No claim that this diagnostic
is independently replayable without the frozen input artifact is made.

Same cached model revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`,
RTX 3060 Laptop 6GB, torch 2.14.0+cu126, transformers 5.16.1, Pillow 12.3.0.

## Next decision

Pause selection-threshold tuning on these 20 QA. Verify visual perception on a small
human-labelled clip-versus-presenter probe and establish event boundaries/counting
conventions before adding cumulative semantic-event state. The three-case evidence
review queue remains pending human verification. Do not turn the 40% option-cycle
diagnostic or the earlier +5 pp single-case result into a resume improvement claim.

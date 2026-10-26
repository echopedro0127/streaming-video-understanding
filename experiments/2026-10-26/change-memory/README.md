# Causal frame-change memory diagnostic

## Frozen method

Follow-up to the temporal coverage probe on the same frozen 20 QA / 20 videos.
This cohort appeared in earlier benchmark runs and is not an untouched test.
Same SmolVLM2-500M, MCQ prompt/options, B8, 1 FPS and eight generated tokens.
Each QA decodes once, then generates three times; condition order rotates 7/7/6.

- Recent-8: last eight frames in the 60s candidate horizon, unchanged control.
- Prefix-8: whole causal prefix divided into eight time bins, unchanged control.
- Change-4+4: four recent frames and four historical visual-change representatives.

For Change-4+4, resize successive incoming frames to RGB32x32 with bilinear
resampling. Score is mean absolute pixel difference across channels divided by 255.
A score >=0.08 and at least 2s since the last admitted candidate marks a visual
change candidate. Scores are computed at arrival using only the preceding frame;
cooldown applies to admitted candidates even if later discarded from memory.

After a frame leaves the four-frame recent buffer, keep the top four historical
scores; ties favor earlier timestamp then frame ID. Noncandidates score0 and fill
unused slots deterministically. A lighting change, motion or return to a presenter
can score highly: **visual-change candidates are not semantic events or counts**.
Question text, options, GT and the review queue do not enter frame ranking.

Parameters were fixed before inference, code-reviewed and saved in implementation
commit `610a579`; no threshold tuning from these answers. The module itself uses
no future information or final duration. Its paired evaluation reuses the existing
query-aware prefix sampler, so this is not a deployed persistent-memory system.

## Results

| Metric | Recent-8 | Prefix-8 | Change-4+4 |
|---|---:|---:|---:|
| Correct / 20 | 3 | 3 | 4 |
| Accuracy | 15% | 15% | 20% |
| Parse-valid / 20 | 20 | 20 | 20 |
| Mean selected temporal span | 7.007s | 200.350s | 219.019s |
| Mean model latency | 2.363s | 2.296s | 2.286s |
| P95 model latency | 2.854s | 2.536s | 2.533s |

Change versus Recent: +5 pp, helped 1 / hurt 0, paired video-bootstrap CI95
[0,+15] pp (2,000 resamples, seed 42). One changed outcome on 20 inspected videos
does not establish stable superiority. The prefix control has no correctness
differences; its degenerate[0,0] interval does not establish population equivalence.

Shared ingest 16.530s/QA, shared allocated GPU peak 3537.18 MiB. The change bank retains
at most 8 full-resolution images; observed peak 8. The context bank observed peak 23
(bound 24). Combined retention upper bound 32 images plus incoming images and one
32x32 thumbnail; decoder/framework buffers excluded. This is a conservative sum
of bank capacities, not a measured whole-process image or RAM peak. Model encoding
runs separately three times; small timing differences do not establish speedup.

## Important answer-distribution control

Change outputs C 19 times and D once; controls C 18 times and A twice. GT counts are
A 5 / B 3 / C 4 / D 8. A posthoc rule that always answers C also gets 4/20, with the
**same per-QA correctness vector** as Change-4+4. This does not prove the model
ignores images, but the observed gain cannot be presented as verified event
understanding or memory efficacy. The constant rule is a diagnostic, not a tuned
benchmark method, and was not used for frame selection.

Three answer letters change relative to Recent-8:

| Sample | GT | Recent | Change | Outcome |
|---|---|---|---|---|
| sample_76_4 | A | C | D | Both wrong |
| sample_79_5 | C | A | C | Helped |
| sample_92_2 | D | A | C | Both wrong |

The sole helped case asks how many movie clips have been inserted by333s, with
numeric options2/3/4/5. Historical representatives are at1.001,140.140,275.275,
290.290s. Ground-truth C corresponds to4; neither these four slots nor their
change scores certify four actual inserted events. No human evidence annotation
has yet linked these frames to the ground-truth count.

## Audit and artifacts

20 complete triples / 60 generations; no missing, duplicate, future-frame,
window-scope, chronology, B8, history-schema, protocol or runtime violations.
Both controls reproduce previous predictions and frame manifests on all 40
comparisons. Historical representatives are disjoint from the last four recent
frames; candidate/fallback scores retained in raw records. Smoke/full resumes
skip 3/20 with unchanged raw hash; 503 repository tests passed. Independent review
fixed two exceptional cleanup paths before inference, with failing regressions
then passing tests. Existing results and default protocol hashes are preserved.

Independent artifact review found no P1/P2 issues. Saved final history cannot
independently certify global top-four ranking without the entire candidate trace;
ranking is covered by code review and unit tests. Reported image peaks likewise
do not independently measure cleanup or whole-process RAM.

- [Machine summary](summary.json), [audit/checksums](audit.json).
- [Human review queue](evidence-review-queue.json): rough brackets only; pending
  verification, not official labels or evidence recall. No extra textual event
  memory was supplied to the model.
- Frozen split: `configs/splits/counting-readout-video-disjoint-20.json`.
- Annotation source: `results/real-run/annotations-tail-250.jsonl`, derived from
  older baseline records; provenance/hashes match the previous probe.
- Local ignored raw file: `results/change-memory/pilot-v1.jsonl`.
- Run ID: `change-memory-v1-0fbce55d18ea`.

Run from the repository root with the existing environment/cache:

```powershell
python scripts/run_context_coverage.py `
  --config configs/adaptive-semantic-1fps.yaml `
  --data-root <path-to-real-time-51-100> `
  --annotations <absolute-path-to-annotations-tail-250.jsonl> `
  --split configs/splits/counting-readout-video-disjoint-20.json `
  --output results/change-memory/pilot-v1.jsonl --change-probe --resume
```

RTX 3060 Laptop 6GB; torch 2.14.0+cu126, transformers 5.16.1, Pillow 12.3.0; cached model
revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. No new dependencies/downloads.

## Next decision

Keep this as an exploratory ablation. Before scaling or claiming accuracy gains,
verify event intervals/counting convention on the prepared cases and test visual
input sensitivity with matched blank/shuffled controls and option permutations.
Do not tune thresholds on this cohort or write a+5pp general improvement claim
into a resume. Cumulative semantic-event state and evidence recall remain pending.

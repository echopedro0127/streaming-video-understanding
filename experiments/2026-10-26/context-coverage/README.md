# Causal temporal coverage probe

## Question and protocol

Does spreading the same eight input frames across more history improve counting
on the frozen 20-QA/20-video follow-up cohort? All conditions use SmolVLM2-500M,
1 FPS, unchanged MCQ prompt/options, no timestamp prompt and eight generated tokens.
The cohort is video-disjoint from the earlier 11-case diagnostic but appeared in
earlier benchmark runs. It is not a pristine unseen test or the 500-QA benchmark.

| Condition | Candidate interval | Selection |
|---|---|---|
| recent-8 | [max(0,t-60),t] | Latest eight sampled frames |
| time-bin-60s-8 | [max(0,t-60),t] | Eight equal time bins |
| time-bin-prefix-8 | [0,t] | Eight equal time bins |

Within each bin choose its closest-to-centre frame; ties go to earlier timestamp,
then frame ID. These conditions differ from the existing endpoint-index Uniform-8.
Query timestamp is known for binning. Selection uses no question semantics,
choices, GT or predictions. Future frames are rejected before storage/rendering.
Invalid MCQ outputs are retained as incorrect, runtime failures remain errors.

Decode each QA prefix once; keep at most 24 distinct full-resolution images across
the three selectors, plus one incoming frame. Evicted images close immediately.
Decoder/framework buffers are outside that bound. Visual encoding still runs
separately for each condition. This is a query-aware causal-prefix diagnostic,
not query-agnostic persistent memory or a deployed efficiency comparison.

## Measured results

| Metric | Recent-8 | 60s bins | Prefix bins |
|---|---:|---:|---:|
| Correct / 20 | 3 | 3 | 3 |
| Accuracy | 15% | 15% | 15% |
| Parse-valid / 20 | 20 | 20 | 20 |
| Mean selected temporal span | 7.007s | 45.696s | 200.350s |
| Mean oldest-frame age | 7.779s | 48.970s | 214.586s |
| Mean model latency | 2.404s | 2.305s | 2.327s |
| P95 model latency | 2.886s | 2.571s | 2.555s |

Both binned methods versus Recent-8: delta 0 pp, helped 0 / hurt 0. Empirical paired
video-bootstrap CI95 is [0,0] pp (2,000 resamples, seed 42). This degenerates because
all observed correctness pairs agree; it **does not establish population
equivalence**, a tight generalization bound or adequate statistical power.

Prefix bins reproduce all 20 Recent-8 answer letters. Window bins change one
letter, wrong-to-wrong. Recent-8/prefix output C 18 times and A 2 times; window bins
C 17 times / A 3 times. These are observed output distributions, not calibrated
confidence or proof that the model ignores images.

Mean shared ingest, including optional rendering on 3 preselected cases, is 16.636s
per QA. Shared allocated GPU peak 3537.18 MiB; retained-image peak 23, within the 24
bound. GPU peak is shared, not a per-condition estimate. No overall speedup is
claimed from the small latency differences or shared decode.

## Audit and reproducibility

20 complete triples / 60 generations; first-condition counts 7/7/6. Missing,
duplicate, protocol, future-frame, window-scope, budget and runtime violations 0.
Historical Recent-8 answers and manifests reproduce 20/20. Five QA with t<=60s
have identical window/prefix manifests and predictions. Smoke3QA/9calls succeeded;
smoke and full resumes skipped 3 and 20 respectively with unchanged raw SHA256.
Full repository suite: 492 passed. Existing baselines, hashes and results unchanged.

Frozen IDs: `configs/splits/counting-readout-video-disjoint-20.json`.
Annotation provenance: `results/real-run/annotations-tail-250.jsonl` was derived
from prior baseline records, not newly downloaded official annotations. Recorded
annotation/split hashes are in `summary.json`; raw checksum is in `audit.json`.
Raw local records: `results/context-coverage/pilot-v1.jsonl`.
Run ID: `context-coverage-v1-8cff49fa1e3a`.

From the repository root, with the existing environment/cache and derived
annotations available (substitute your local data directory):

```powershell
python scripts/run_context_coverage.py `
  --config configs/adaptive-semantic-1fps.yaml `
  --data-root <path-to-real-time-51-100> `
  --annotations <absolute-path-to-annotations-tail-250.jsonl> `
  --split configs/splits/counting-readout-video-disjoint-20.json `
  --output results/context-coverage/pilot-v1.jsonl --resume
```

The real measurement used RTX 3060 Laptop 6GB, torch 2.14.0+cu126,
transformers 5.16.1, Pillow 12.3.0 and cached model revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. No model replacement or dependency
installation was required.

## Decision

Wider temporal span alone produced no counting gain here. This does not rule
out better frame selection or identify the internal cause of failure. Review
[sampled qualitative observations](qualitative-review.md) before the next test:
prefix bins can still miss an inserted scene between representatives. Prioritize
human-verified event intervals and a small event-change/cumulative-state
diagnostic. Do not enlarge the benchmark or tune routing thresholds solely from
this 20-QA result. Evidence recall and exact event counts remain unmeasured.

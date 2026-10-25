# Fixed-frame choice-free counting pilot

## Scope and selection

20 QA on 20 videos, one numeric counting question per video. Every video from
the earlier 11-case diagnostic is excluded. Within each eligible video choose
the minimum SHA256 of `20261026:video_id:sample_id`, without consulting answers
or correctness. IDs: `configs/splits/counting-readout-video-disjoint-20.json`.

These videos appeared in earlier benchmark runs: video-disjoint from the
diagnostic is **not a pristine unseen test**. This is a small readout pilot,
not the 250-QA benchmark or a memory-policy comparison.

## Matched conditions

Exact same decoded image objects and timestamps: Recent-8, 1 FPS, 60s candidate
horizon, SmolVLM2-500M, max eight generated tokens. Recent-8 exposes approximately
seven seconds of sampled context despite the larger candidate horizon. Decode
once per QA, generate twice; VLM visual encoding still runs separately. First
mode is counterbalanced 10/10 by alternating sorted frozen sample IDs.

- MCQ: original prompt and A-D parsing, four numeric choices.
- Choice-free: question and request for one non-negative integer; no choices or
  GT in the prompt. Values are not clipped/snapped to candidate counts. Parser
  frozen before smoke: ASCII digits, optional `time`/`times`, optional `.`/`!`.
- Pilot-specific invalid outputs from either method complete the QA as incorrect.
  Default baseline MCQ behavior is unchanged. Raw text/validity are retained;
  MAE is valid-output-only with its denominator. Runtime errors remain errors.

## Results

| Metric | MCQ | Choice-free count |
|---|---:|---:|
| Correct / 20 | 3 | 2 |
| Exact count accuracy | 15% | 10% |
| Parse-valid | 20/20 | 20/20 |
| Outside original choices | 0/20 | 15/20 |
| Count MAE (20 valid outputs each) | 1.85 | 3.20 |
| Mean model latency | 2.441 s | 2.269 s |
| P50 model latency | 2.346 s | 2.284 s |
| P95 model latency | 3.048 s | 2.455 s |

Count minus MCQ: **-5 pp**, helped 2 / hurt 3; video-cluster bootstrap CI95
**[-25,+15] pp**, 20 clusters, 2,000 resamples, seed 42. No gain established;
the wide interval also prevents a precise general inferiority claim.

Direct outputs: `1` in 17/20, `12`, `10`, `2` once each. Poor counting is not due
to failed numeric parsing here. Concentration at1 is an observed symptom, not
proof of a particular internal mechanism. Mean shared decode is 15.901 s/QA;
shared allocated GPU peak 3537.18 MiB, not a per-method memory estimate. This
paired 40-call experiment establishes no deployed end-to-end speedup.

## Audit and interpretation

20/20 paired records, 40 generations; missing/duplicate/protocol/runtime errors 0.
No overlap with the 10 diagnostic videos. Selection, parsing, prompt, GT mapping
and 10/10 order checks passed. MCQ predictions and frame manifests match historical
Recent-8 on 20/20 pilot QA.

Smoke: 2 QA; both smoke/full resumes preserved raw records and skipped completed
keys without loading a model. Full run added 18 / skipped 2; final resume skipped 20.
All 474 tests passed. Code review resolved invalid-MCQ asymmetry before inference.

Removing choices changes instructions and candidate priors: this is a composite
format intervention, not an isolated positional-bias estimate. It did not solve
counting. Next priority is evidence coverage/answerability and event history,
not tuning this parser or adding routing keywords. Recent-window evidence, the
60s candidate cutoff and reasoning limitations need separate causal controls.

## Reproduce

Config: `configs/adaptive-semantic-1fps.yaml`; fixed Recent-8.
Run ID: `choice-free-count-v1-840c22f32657`.
Cached SmolVLM revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
Measured environment: torch `2.14.0+cu126`, transformers `5.16.1`, Pillow `12.3.0`.

Annotation provenance: the local 250-row tail annotation was reconstructed from
prior baseline records, not freshly downloaded official annotations. Its SHA256
is recorded in `summary.json` and independently rechecked against that exact
source. The QA/options/timestamps/GT mappings for these 20 cases were audited;
this provenance remains a limitation, as in the earlier allocation experiments.

```powershell
python scripts/run_count_readout.py --config configs/adaptive-semantic-1fps.yaml --data-root $env:VIDEO_DATA_ROOT --annotations $env:QA_ANNOTATIONS --split configs/splits/counting-readout-video-disjoint-20.json --output results/count-readout/paired-recent8-v1.jsonl --resume
python -m pytest -q
```

Set variables to local public-data paths and HF cache to the data drive. Raw
records stay local; `summary.json`/`audit.json` are public small outputs. Earlier
benchmark scores are unchanged. Future prompt/parser changes require a new
protocol version and distinct result identity, not resume into this run.

# Eight-frame allocation: completed tail-split experiment

Data: StreamingBench videos 51–100, 250 timestamped QA.
Model: SmolVLM2-500M-Video-Instruct; CLIP ViT-B/32 retrieval; 1 FPS;
60-second candidate horizon; eight-frame input budget; seed 42 for 6+2.

| Policy | Correct | Accuracy |
|---|---:|---:|
| Recent-8 | 95/250 | 38.0% |
| Recent-4 + Option-Semantic-Memory-4 | 90/250 | 36.0% |
| Recent-6 + Option-Semantic-Memory-2 | 93/250 | 37.2% |
| Q-only Adaptive-Semantic-v1 | 92/250 | 36.8% |

The 6+2 run passed exact split coverage, unique sample/config key, causal frame,
6/2/8 capacity and runtime-failure audits. The unrounded summary and two paired
diagnoses in this directory are copied from the corresponding local generated
artifacts without modification. Both paired bootstrap intervals span zero.

Config: `configs/option-semantic-6plus2-1fps.yaml`.
Config ID: `recent-6+option-semantic-memory-2-0f9adb02ff39`.
Pairing: exact sample IDs and metadata, using `scripts/analyze_policy_pairs.py`.
Uncertainty: `scripts/diagnose_allocation.py`, paired video-cluster percentile
bootstrap, 50 clusters, 2,000 resamples, seed 42.

Local source SHA256:

- Raw 6+2 JSONL: `5077E979F83D41D5465389D46F51136596C76F072E556C825559883450633D8B`
- Derived tail annotation JSONL: `B9CA9584D4F5AFF717B64B0E1E606E188A76D12782045D55DE503CA046D0F888`

The annotation file was previously reconstructed from existing baseline records,
not freshly downloaded official annotations; video/question/timestamp/answer and
option alignment is checked by the paired analyzer. This provenance is retained
as a limitation. Data and raw traces stay in ignored local directories.

The run resumed 126 existing records and added 124 in a later session. Mixed
cache/session timings are descriptive, not a controlled speedup claim. Summary
latency is the legacy decode-plus-model field; broader profile total excludes
wrapper cleanup. GPU memory is allocated peak in MiB, not total process memory.

The fixed memory policies use option-aware retrieval; the adaptive policy uses
Q-only retrieval. Their comparison does not isolate allocation from retrieval.
The repository's question-rule groups and blank evidence-review queues are
diagnostic aids, not human evidence labels. No queue has been annotated.

## Completed adaptive evaluation

Config: `configs/adaptive-semantic-1fps.yaml`;
config ID: `adaptive-semantic-v1-e81bd6be6411`.
The final resume processed 201 and skipped 49 existing records. The 250 records
and 250 profiles passed split coverage, identity, causal/horizon/budget and
correctness checks. Runtime failures, duplicate keys and missing profiles: zero.

Against Recent-8: helped 4, hurt 7, delta -1.2 pp, video-cluster bootstrap
CI95 [-4.0,+1.6] pp. No overall improvement established.

- Current route: 59/98 versus baseline 59/98; all predictions and frame manifests
  match exactly (98/98).
- Counting route: 31/150 versus 34/150; helped 4, hurt 7.
- Unspecified: 2/2 versus 2/2; explicit-history: no real samples.

Latency mean/P50/P95: 21.101/18.024/48.922 s; allocated GPU peak 3834.31 MiB;
mean input 7.98 frames. Timings span sessions/cache states and do not establish
a controlled speedup. Full raw-output SHA256 is
`517C38544D8E5DEE0EB05372DE3803DB4FED72D13E95B251DD75265EF48B2CAA`.

Machine-readable evidence: `adaptive-semantic-v1-tail-summary.json`,
`recent8-vs-adaptive-v1-diagnosis.json`, `adaptive-semantic-v1-tail-audit.json`.
The next diagnostic step is reviewing the 11 helped/hurt cases and annotating
evidence; these outcomes alone do not establish the mechanism behind the loss.

# Eight-frame allocation: completed tail-split experiment

Data: StreamingBench videos 51–100, 250 timestamped QA.
Model: SmolVLM2-500M-Video-Instruct; CLIP ViT-B/32 retrieval; 1 FPS;
60-second candidate horizon; eight-frame input budget; seed 42 for 6+2.

| Policy | Correct | Accuracy |
|---|---:|---:|
| Recent-8 | 95/250 | 38.0% |
| Recent-4 + Option-Semantic-Memory-4 | 90/250 | 36.0% |
| Recent-6 + Option-Semantic-Memory-2 | 93/250 | 37.2% |

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

These results evaluate **option-aware fixed allocation**, not the new Q-only
adaptive router. The repository's question-rule groups and blank evidence-review
queues are diagnostic aids, not human evidence labels. The generated queues
have not been annotated.

# Selective temporal memory: stage 1

Updated: 2026-10-25 (Asia/Shanghai).

## Implemented

Verification: **427 tests passed**, including baseline compatibility, causal
guards, route independence, adaptive mock CLI resume, diagnostic CLI no-overwrite,
video-cluster bootstrap and review-order blinding regression tests.

`adaptive-semantic-v1` is a frozen question-only rule router with an eight-frame
readout budget. It reuses the existing Q-only CLIP retrieval path and model:

| Question cue | Recent / history | Resolved policy |
|---|---|---|
| Explicit present: now, currently, at the moment | 8 / 0 | Recent-8 |
| Explicit history: before, after, earlier, initially | 4 / 4 | Q-only Semantic-Memory-4 |
| Repeated-event counting: how many times, number of times | 4 / 4 | Q-only Semantic-Diverse-Memory-4 |
| No matching cue | 6 / 2 | Q-only Semantic-Memory-2 |

Priority is counting, then history, then current, then fallback. Thus “How many
cups are visible now?” uses recent frames; “How many times ...?” uses diverse
history. Word boundaries avoid substring matches such as “aftershave”. The router
does not accept answer options, ground truth, task labels or predictions.

Each adaptive result keeps its adaptive policy/config identity and adds a
`routing` object containing group, reason, capacities, resolved policy and rule
version. New evaluator records also include task type, temporal clue type and
context horizon for auditing. Existing config hashes remain unchanged; new rules
must use a new policy/version before resuming experiments.

Counting routing is a lexical heuristic and temporal diversity is not an event
counter. Retrieval scores are not calibrated answer confidence. This package
does not implement answer/clarification routing.

## Measured diagnosis on existing completed runs

Scope: videos 51–100, 250 exact paired QA, Recent-8 versus Option-Semantic-4+4.
This compares existing option-aware runs, **not** the new adaptive method.

| Measure | Value |
|---|---:|
| Recent-8 accuracy | 38.0% (95/250) |
| Option-Semantic-4+4 accuracy | 36.0% (90/250) |
| Helped / hurt | 5 / 10 |
| Accuracy difference | -2.0 percentage points |
| 95% paired video-cluster bootstrap interval | [-5.2, +1.2] percentage points |
| Bootstrap | 50 video clusters, 2,000 resamples, seed 42 |

The interval spans zero. These data do not establish a stable advantage for either
policy. All QA from a sampled video stay together during resampling.

Post-hoc question-rule groups:

| Lexical group | QA | Recent-8 correct | Option-4+4 correct | Helped / hurt |
|---|---:|---:|---:|---:|
| Repeated-event counting | 150 | 34 | 33 | 4 / 5 |
| Current | 98 | 59 | 55 | 1 / 5 |
| Unspecified | 2 | 2 | 2 | 0 / 0 |

These groups are proxies, not annotated evidence requirements. In particular,
current-scene questions may still require history. No rule was selected using
which model answer was correct. Since the benchmark has been examined repeatedly,
this is exploratory diagnosis; untouched-video validation is still needed.

## Evidence review

`scripts/diagnose_allocation.py` produces `diagnosis.json` and
`evidence-review.jsonl`. The 60-row review sample is outcome-balanced for diagnosis
and deterministically shuffled. It is not a representative accuracy sample.
Predictions, ground truth and outcome labels are omitted from the review file.
The earlier local `2026-10-24-option4-diagnosis` queue was generated before the
ordering fix and is superseded by `2026-10-24-option4-diagnosis-v2`.

Annotation procedure:

1. Watch each video only up to its query timestamp, with the question and options.
2. Mark sufficient evidence intervals as objects with `start_s` and `end_s`.
   All intervals must end at or before the query timestamp. Record every required
   event for ordering and repetition questions, not just a single matching frame.
3. Set `evidence_type` to `current`, `historical`, `ordering`, `counting` or
   `insufficient`; set `evidence_available` explicitly, add annotator and notes,
   and change `annotation_status` to `reviewed`.
4. Only then compare the selected-frame traces and classify selection versus
   reasoning failures. Report evidence older than the 60-second candidate horizon
   separately from within-horizon retrieval failures.

All 60 rows are currently **pending**. Empty intervals are unannotated, not a
claim that no evidence exists. Evidence hit/coverage/age metrics must wait for
validated annotations; question-rule groups cannot supply those labels.

## Reproduce

From the optimization checkout with the project environment activated:

```powershell
python -m pytest -q
python scripts/run_eval.py --config configs/adaptive-semantic-1fps.yaml --backend smolvlm --policy adaptive-semantic-v1 --data-root $env:VIDEO_DATA_ROOT --annotations $env:QA_ANNOTATIONS --output results/adaptive-semantic-v1.jsonl --profile-output results/adaptive-semantic-v1-profile.jsonl --resume
python scripts/diagnose_allocation.py --baseline-input $env:RECENT8_RESULTS --baseline-policy recent-8 --candidate-input $env:OPTION4_RESULTS --candidate-policy recent-4+option-semantic-memory-4 --video-min 51 --video-max 100 --output-dir results/analysis/option4-diagnosis
```

Set `VIDEO_DATA_ROOT`, `QA_ANNOTATIONS`, `RECENT8_RESULTS`, `OPTION4_RESULTS` to
your local public-data/results paths. Set `HF_HOME`/`HF_HUB_CACHE` on the data
drive before model loading. The diagnostic CLI refuses to overwrite reports or
human review files. No data or model download is required on the current machine.

The adaptive config also declares matched Q-only fixed allocations. Use
`--all-policies` for that controlled matrix; option-aware 6+2 is a separate
allocation ablation and must not be presented as the same retrieval algorithm.

## Running and remaining experiments

- Option-Semantic-6+2 is complete: 250/250 QA from videos 51–100, 93 correct
  (37.2%), zero missing IDs, duplicate keys, future-frame violations,
  recent/memory/readout budget violations, or runtime failures. All rows use the
  original config ID `recent-6+option-semantic-memory-2-0f9adb02ff39`.
- Relative to Recent-8 (95/250), 6+2 helped 4 and hurt 6: -0.8 pp,
  video-cluster bootstrap CI95 [-3.2, +1.6] pp.
- Relative to Option-4+4 (90/250), 6+2 helped 4 and hurt 1: +1.2 pp,
  CI95 [-0.4, +2.8] pp. Both intervals span zero; a stable gain has not
  been established. No rules are changed based on these outcomes.
- Legacy decode-plus-model latency: mean 25.948 s, P50 24.132 s, P95 54.483 s;
  peak allocated GPU memory 3834.31 MiB; average input 7.98 frames.
  The 250-row profile reports mean pipeline time 26.045 s, decode 23.199 s,
  retrieval 0.083 s, processor 0.786 s and generation 1.962 s. This resumed run
  spans sessions and cache states, so it does not establish a controlled speedup.
- An initial five-QA real adaptive smoke succeeded on sample_51; all five
  questions use the current route. A separate three-QA route-coverage split
  `configs/splits/adaptive-routing-smoke-tail.json` covers current, unspecified
  and counting routes: all three completed, actual allocations were 8+0 / 6+2 /
  4+4, all input/causal/route-budget checks passed, and no runtime failures were
  found. A real-model resume returned `processed=0 skipped=3`; the output SHA256
  remained `EC46B287ED22E9F99CB991E566D96A62EB7336A5724CD6DBE93D96AC83A2B71A`.
  No explicit-history question exists in the tail split
  under the frozen v1 rules; that branch has synthetic coverage only.
- Adaptive full tail-split evaluation completed 250/250 in
  `results/adaptive-semantic-v1-real-time-51-100.jsonl`, seeded with the three
  audited smoke records and matching profile/failure sidecars. It uses config ID
  `adaptive-semantic-v1-e81bd6be6411`. The final resume processed 201 and skipped
  49 existing records. Coverage, identity, causal/horizon/budget, correctness and
  profile audits passed: 250 QA and 250 profiles, zero missing/duplicate keys,
  protocol violations or runtime failures. The failure sidecar contains 158
  incorrect answers, not execution errors.
- During the first 14 newly resumed records, decoding accounted for 88.92% of
  legacy `latency_ms`. This is a partial-run observation, not a full benchmark
  result. Existing `latency_ms` is decode plus model time; use the profile's
  `total_ms` for broader pipeline time (still excluding wrapper cleanup).
- The candidate horizon is 60 seconds. The present evaluator re-reads prefixes
  and reconstructs memory for each QA. A fixed eight-frame model input does not
  establish bounded total RAM/disk or incremental encoding. Event writing and
  true incremental ingestion remain subsequent, separately controlled packages.

### Completed 6+2 artifacts

- `results/option-semantic-6plus2-real-time-51-100.jsonl` and its summary/profile;
- `results/analysis/51-100-recent8-vs-6plus2-*`;
- `results/analysis/51-100-option4-vs-6plus2-*`;
- cluster uncertainty: `results/analysis/2026-10-25-6plus2-diagnosis/diagnosis.json`
  and `results/analysis/2026-10-25-option4-vs-6plus2-diagnosis/diagnosis.json`.

These are local generated artifacts. The evidence-review queues remain pending;
the earlier 60-row v2 queue is the primary annotation set to avoid multiplying
manual work. H1 frame overlap and H3 keyword counts are proxies; H2 retrieval
agreement is not calibrated confidence; H4 both-wrong cases do not alone identify
the answerer as the cause.

Small completed summaries, paired uncertainty and source hashes are published
under [`experiments/2026-10-25`](../experiments/2026-10-25/README.md), without raw
videos, model weights or machine-specific paths.

The model selected option C in 162/250 cases (64.8%), while C is the correct
position in 54/250 (21.6%). This motivates later option-permutation checks;
it does not prove position bias independently of content and task difficulty.

### Completed adaptive evaluation

| Policy | Correct / 250 | Accuracy |
|---|---:|---:|
| Recent-8 | 95 | 38.0% |
| Option-Semantic-4+4 | 90 | 36.0% |
| Option-Semantic-6+2 | 93 | 37.2% |
| Q-only Adaptive-Semantic-v1 | 92 | 36.8% |

Adaptive versus Recent-8: helped 4, hurt 7, delta -1.2 pp; paired video-cluster
CI95 [-4.0, +1.6] pp, 2,000 resamples, seed 42. There is no established overall
gain. The option-aware policies differ in both retrieval and allocation; this
table is not an isolation of the router's effect against matched Q-only controls.

| Question-rule proxy | QA | Recent-8 correct | Adaptive correct | Helped / hurt |
|---|---:|---:|---:|---:|
| Current | 98 | 59 | 59 | 0 / 0 |
| Repeated-event counting | 150 | 34 | 31 | 4 / 7 |
| Unspecified | 2 | 2 | 2 | 0 / 0 |

All 98 current-route predictions and selected-frame manifests match Recent-8
exactly. Thus the current route preserves that baseline on this subset, while
the counting route accounts for the observed net loss of three correct answers.
This localizes the difference but does not identify whether missing evidence,
the 60-second horizon or the answerer caused it. Inspect the 11 helped/hurt cases
before changing rules. Explicit-history routing has no real samples in this split.

Adaptive legacy latency: mean 21.101 s, P50 18.024 s, P95 48.922 s; peak allocated
GPU memory 3834.31 MiB; mean input 7.98 frames. Mean profile total is 21.151 s,
decode 18.589 s, retrieval 0.035 s, processor 0.617 s and generation 1.895 s.
Multiple sessions/cache states preclude a controlled speedup claim. Option C is
predicted 168/250 times (67.2%); this remains a descriptive imbalance only.

Artifacts:

- `results/adaptive-semantic-v1-real-time-51-100.jsonl`, summary and profile;
- `results/analysis/51-100-recent8-vs-adaptive-v1-*` (4 helped, 7 hurt cases);
- `results/analysis/2026-10-25-adaptive-v1-diagnosis/diagnosis.json`;
- public summaries, audit and source hash under `experiments/2026-10-25/`.

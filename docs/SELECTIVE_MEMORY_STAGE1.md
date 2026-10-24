# Selective temporal memory: stage 1

Updated: 2026-10-24 (Asia/Shanghai).

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

- The existing Option-Semantic-6+2 evaluation was resumed from 126/250 completed
  rows with its original configuration and config ID. Full paired 6+2 summaries
  must wait for all 250 rows and causal/budget/key audits.
- Adaptive routing has synthetic/mock evaluator and CLI-resume tests. Its
  real-model smoke and accuracy run are pending the GPU slot. No adaptive accuracy
  improvement is claimed.
- During the first 14 newly resumed records, decoding accounted for 88.92% of
  legacy `latency_ms`. This is a partial-run observation, not a full benchmark
  result. Existing `latency_ms` is decode plus model time; use the profile's
  `total_ms` for broader pipeline time (still excluding wrapper cleanup).
- The candidate horizon is 60 seconds. The present evaluator re-reads prefixes
  and reconstructs memory for each QA. A fixed eight-frame model input does not
  establish bounded total RAM/disk or incremental encoding. Event writing and
  true incremental ingestion remain subsequent, separately controlled packages.

# Option-Aware Semantic Retrieval Ablation Design

## 1. Context and motivation

The current training-free semantic memory stage uses a Q-only CLIP query: the
question is encoded once and historical frames are ranked by question-frame
similarity. On the completed `real-time-1-50` validation run, Q-only semantic
memory achieved 63.6% accuracy and semantic-diverse memory achieved 62.8%,
compared with 65.2% for `recent-8`. The semantic policies changed only a
small number of paired answers, so the next controlled question is whether
the retrieval query is underspecified because it ignores the answer options.

This design adds an option-aware retrieval ablation. It is a validation-only
extension, not a replacement for the existing Q-only scorer or the original
four baselines.

## 2. Research question

Does conditioning frame retrieval on the question and all candidate answer
hypotheses improve bounded-memory streaming video QA compared with Q-only
retrieval under the same causal and visual-frame budget?

The experiment must not use the ground-truth answer during retrieval.

## 3. Scope and policy IDs

Add two new policy IDs in a separate experiment configuration:

- `recent-4+option-semantic-memory-4`
- `recent-4+option-semantic-diverse-memory-4`

Each policy keeps a maximum visual budget of eight frames: four recent frames
and four historical memory frames. Existing policy IDs, checked-in configs,
historical results, and Q-only semantic behavior remain unchanged.

## 4. Retrieval contract

The option-aware scorer receives only:

- the video path;
- causal historical frames and their images;
- the user question;
- the ordered candidate option strings.

It must not receive `StreamingSample.answer`, any ground-truth evidence, or a
correct-option indicator. The four options are treated as hypotheses that a
real user would see.

For each option, construct a deterministic text hypothesis:

```text
Question: {question}
Candidate answer: {option}
```

The scorer encodes all hypotheses as one text batch and reuses the existing
cached image features. For each historical frame it returns four cosine
similarities in option order. The first aggregation is deliberately simple:

```text
frame_score = max(score_A, score_B, score_C, score_D)
```

The aggregated frame score is passed to the existing semantic/diversity
selector. The scorer also exposes per-option scores for analysis; it does not
change answer generation or answer parsing.

## 5. Trace and metrics

Each selected option-aware memory frame records:

- `frame_id` and `timestamp_s`;
- `semantic_score` equal to the maximum option score;
- the four ordered `option_scores`;
- `best_option_index`;
- `option_margin`, defined as top-1 score minus top-2 score;
- the existing diversity `objective_score`.

The evaluator continues to record accuracy, latency, retrieval latency,
input-frame count, peak GPU memory, and causal manifests. Additional analysis
will compute:

- accuracy against `recent-8`, Q-only semantic memory, and uniform memory;
- paired helped / hurt / unchanged counts per QA item;
- option-score margin and answer-change rates;
- causal violations and visual-budget violations.

Option-aware retrieval latency must be reported separately from frame-selection
latency, as in the current semantic evaluator.

## 6. Causality, memory, and fairness

The option-aware path uses the same causal frame reader and history exclusion
as Q-only semantic retrieval. Recent frames are never scored as historical
memory. All selected timestamps must satisfy:

```text
timestamp_s <= query_time_s + timestamp_tolerance_s
```

The combined recent and memory capacities must not exceed
`config.experiment.visual_budget`. Both new methods use the same CLIP model,
cache format, candidate multiplier, cluster window, diversity weight, and
SmolVLM backend as the Q-only validation run.

The option-aware methods run in a separate configuration and output file so
that the existing 1,250-record result remains immutable. Paired comparisons
join records by `sample_id`; no new split or random sampling is introduced.

## 7. Implementation boundaries

Expected implementation changes are limited to:

- a separate option-aware scorer protocol and CLIP scorer method;
- evaluator policy parsing, selection, and trace serialization;
- a separate option-aware validation YAML config;
- CLI detection and lazy scorer construction;
- focused tests for batch text encoding, option ordering, no-GT retrieval,
  causal selection, budget limits, trace fields, and config loading.

No new model dependency, training stage, answer-reranking stage, confidence
router, or changes to the existing Q-only policy are included.

## 8. Failure handling

The scorer must reject empty questions or option lists, preserve option order,
and return one finite score per option for every candidate frame. Missing
images, malformed option scores, CLIP OOM, and inconsistent score dimensions
must raise clear errors before backend answer generation. Existing cache and
OOM handling remain in effect.

## 9. Acceptance criteria

The implementation is accepted only if:

1. Existing tests and all five existing validation policies remain unchanged.
2. New policies run without ground-truth access and load CLIP lazily.
3. New selected frames are causal and never exceed the eight-frame budget.
4. Traces contain finite four-way option scores and margins.
5. The option-aware validation run completes on all 250 available QA rows for
   both new policies, with resumability enabled.
6. The final report distinguishes measured accuracy from the hypothesis that
   option conditioning helps; no improvement is claimed unless the paired
   results support it.

## 10. Non-goals and follow-up

This ablation does not implement answer log-probabilities, calibrated
confidence, clarification questions, or chain-of-thought scoring. Those remain
separate follow-up experiments after retrieval quality is understood.

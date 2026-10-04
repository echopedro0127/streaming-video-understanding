# Option-Aware Semantic Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a validation-only option-aware semantic-memory ablation that scores each historical frame against four question-answer hypotheses while preserving every existing baseline and result.

**Architecture:** Reuse the current CLIP image-feature cache and causal semantic selector. Add a separate option-aware scorer protocol that returns four finite scores per frame; the evaluator aggregates each frame with the maximum option score, records option-level trace fields, and routes two new policy IDs through the same bounded-memory and causal checks. The CLI and configuration remain separate from the existing Q-only semantic experiment.

**Tech Stack:** Python 3.12+, PyTorch, Transformers CLIP, NumPy, PyYAML, pytest, existing FrameFeatureCache, select_semantic_history, and SmolVLM backend.

---

## Current implementation map

- retrieval/base.py: Q-only SemanticFrameScorer protocol.
- retrieval/clip.py: lazy CLIP loader, cached image features, one-text scoring.
- streaming/semantic.py: causal scalar-score memory selector.
- evaluation/engine.py: policy parsing, budget/causal checks, traces, timing.
- scripts/run_eval.py: one lazy scorer per process.
- analysis/summaries.py: extended display order with historical four-policy comparison intact.
- results/semantic-validation-real-time-1-50.jsonl: immutable 1,250-record Q-only result.

## Task 1: Define option-score aggregation and scorer protocol

Files:
- Create: src/streaming_video_understanding/retrieval/option.py
- Modify: src/streaming_video_understanding/retrieval/base.py
- Modify: src/streaming_video_understanding/retrieval/__init__.py
- Test: tests/test_option_retrieval.py

- [ ] Step 1: Write failing pure-helper tests.

Test max aggregation, deterministic argmax, top-1/top-2 margin, and rejection of empty options, wrong option counts, non-finite scores, non-integer frame IDs, and malformed inputs.

    def test_aggregate_option_scores_uses_max_and_margin():
        result = aggregate_option_scores(
            {7: (0.1, 0.8, 0.4, 0.2)}, expected_options=4
        )
        assert result[7].semantic_score == pytest.approx(0.8)
        assert result[7].best_option_index == 1
        assert result[7].option_margin == pytest.approx(0.4)

- [ ] Step 2: Confirm the tests fail.

    python -m pytest tests/test_option_retrieval.py -q

Expected: import/collection failure because retrieval.option does not exist.

- [ ] Step 3: Implement the pure helper and protocol.

Define OptionScoreItem with option_scores, semantic_score, best_option_index, and option_margin. Implement aggregate_option_scores(scores_by_frame, expected_options) with exact integer frame IDs, a positive exact integer option count, at least two options, exact score lengths, finite numeric scores, lowest-index tie breaking, and top-1 minus top-2 margin.

Add OptionAwareFrameScorer to retrieval/base.py. Its method is:

    score_option_frames(
        *,
        video_path: Path,
        frames: Sequence[FrameRecord],
        images_by_id: Mapping[int, Image.Image],
        question: str,
        options: Sequence[str],
    ) -> dict[int, tuple[float, ...]]

The protocol must not include a ground-truth answer parameter. Export the protocol, dataclass, and helper from retrieval/__init__.py.

- [ ] Step 4: Run focused tests and commit.

    python -m pytest tests/test_option_retrieval.py tests/test_retrieval.py -q
    git add src/streaming_video_understanding/retrieval/base.py src/streaming_video_understanding/retrieval/option.py src/streaming_video_understanding/retrieval/__init__.py tests/test_option_retrieval.py
    git commit -m "feat: define option-aware retrieval contract"

## Task 2: Extend CLIP to batch option hypotheses

Files:
- Modify: src/streaming_video_understanding/retrieval/clip.py
- Test: tests/test_retrieval.py

- [ ] Step 1: Add fake-model tests.

Call score_option_frames with four options. Assert that the processor receives the four Question/Candidate-answer strings in exact option order. Assert that the returned tuple has length four, image encoding happens once, option order is preserved, and empty option text, missing images, malformed text output, and CLIP OOM produce clear errors.

- [ ] Step 2: Refactor the shared encoding path minimally.

Keep public Q-only score_frames(..., query=...) behavior unchanged. Extract cached image-feature loading/encoding and add _encode_texts(queries) returning a finite [batch, dim] matrix. Keep _encode_text as a one-query wrapper. Implement score_option_frames by loading image features once, encoding four Question/Candidate-answer strings in one batch, computing cosine scores for each frame/option pair, and returning scores in option order.

- [ ] Step 3: Verify and commit.

    python -m pytest tests/test_retrieval.py tests/test_option_retrieval.py -q
    git add src/streaming_video_understanding/retrieval/clip.py tests/test_retrieval.py
    git commit -m "feat: score frames against option hypotheses"

## Task 3: Integrate option-aware policies into the causal evaluator

Files:
- Modify: src/streaming_video_understanding/evaluation/engine.py
- Test: tests/test_evaluation.py
- Test: tests/test_streaming.py only for shared regressions

- [ ] Step 1: Add failing evaluator tests.

Use a fake option-aware scorer and mock backend. Assert that the evaluator passes question/options but never the ground-truth answer, keeps at most eight frames, keeps every timestamp causal, and serializes option_scores with length four, best_option_index in range(4), and a finite option_margin. Also test that an over-budget policy is rejected before decoding/scoring and that a missing option-aware scorer prevents any backend call.

- [ ] Step 2: Extend strict policy parsing.

Recognize exactly recent-K+option-semantic-memory-M and recent-K+option-semantic-diverse-memory-M. Return option-semantic and option-semantic-diverse, reuse recent_capacity + memory_capacity <= visual_budget before decoding, and leave old policy IDs and kinds unchanged.

- [ ] Step 3: Add option-aware selection and trace timing.

Add _select_option_semantic_policy that calls score_option_frames with sample.question and sample.options, passes the max-aggregated scalar map to select_semantic_history, and uses diversity only for option-semantic-diverse. Merge option_scores, best_option_index, and option_margin into the existing selected-frame trace. Measure only scorer invocation as retrieval_latency_ms and keep selector work in sampling_ms.

- [ ] Step 4: Run tests and commit.

    python -m pytest tests/test_evaluation.py tests/test_streaming.py tests/test_retrieval.py -q
    python -m pytest -q
    git add src/streaming_video_understanding/evaluation/engine.py tests/test_evaluation.py tests/test_streaming.py
    git commit -m "feat: integrate option-aware memory policies"

## Task 4: Add isolated config and lazy CLI routing

Files:
- Create: configs/option-semantic-validation-1fps.yaml
- Modify: scripts/run_eval.py
- Modify: tests/test_checked_in_configs.py

- [ ] Step 1: Add config/helper tests.

Assert visual_budget == 8, policy_ids exactly (recent-4+option-semantic-memory-4, recent-4+option-semantic-diverse-memory-4), and _requires_semantic_scorer returns True for the option-aware policy. Add a monkeypatch test that the scorer is constructed once for an option-aware policy and not constructed for recent-8.

- [ ] Step 2: Create the config.

Copy dataset/model/runtime/retrieval values from configs/semantic-validation-1fps.yaml; set the two option-aware policy IDs and visual budget 8. Do not edit any existing config or result.

- [ ] Step 3: Update lazy predicate and route.

Make _requires_semantic_scorer true for both +semantic and +option-semantic. Instantiate one existing ClipFrameScorer per process and pass it to every evaluation. The evaluator chooses the Q-only or option-aware method from policy kind; non-semantic smoke still imports no CLIP.

- [ ] Step 4: Test and commit.

    python -m pytest tests/test_checked_in_configs.py tests/test_config.py tests/test_evaluation.py -q
    python scripts/run_eval.py --config configs/main.yaml --backend mock --limit 1 --policy recent-8 --output results/option-aware-nonsemantic-smoke.jsonl --overwrite
    git add configs/option-semantic-validation-1fps.yaml scripts/run_eval.py tests/test_checked_in_configs.py
    git commit -m "feat: configure option-aware semantic validation"

## Task 5: Extend reporting labels without breaking fixed-four comparison

Files:
- Modify: src/streaming_video_understanding/analysis/summaries.py
- Modify: tests/test_analysis.py

- [ ] Step 1: Add label/order regression tests.

Assert labels Recent-4 + Option-Semantic-Memory-4 and Recent-4 + Option-Semantic-Diverse-Memory-4. Assert the six existing policies remain before the two option-aware policies. Keep POLICY_ORDER, _policy_from_summary_path, and build_comparison_rows fixed-four behavior unchanged.

- [ ] Step 2: Implement constants and verify.

Add OPTION_SEMANTIC_POLICY_ORDER after the existing semantic order and use POLICY_ORDER + SEMANTIC_POLICY_ORDER + OPTION_SEMANTIC_POLICY_ORDER only for display ordering.

    python -m pytest tests/test_analysis.py -q
    git add src/streaming_video_understanding/analysis/summaries.py tests/test_analysis.py
    git commit -m "feat: report option-aware memory policies"

## Task 6: Run validation and document measured results

Files:
- Modify: README.md
- Modify: PROJECT_PLAN.md
- Generated/ignored: results/option-semantic-validation-real-time-1-50*.jsonl/json

- [ ] Step 1: Verify before long run.

    python -m pytest -q
    git diff --check

Expected: all tests pass and no unintended changes exist.

- [ ] Step 2: Run one option-aware mock smoke.

Set HF_HOME and HF_HUB_CACHE to the existing E project cache, then run:

    python scripts/run_eval.py --config configs/option-semantic-validation-1fps.yaml --data-root E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\data\streamingbench\real-time-1-50 --annotations annotations.jsonl --backend mock --limit 1 --policy recent-4+option-semantic-diverse-memory-4 --output results\option-aware-smoke.jsonl --overwrite

Expected: processed=1 skipped=0, four option scores per selected memory frame, causal timestamps, and at most eight input frames.

- [ ] Step 3: Run both policies on all available QA.

    python scripts/run_eval.py --config configs/option-semantic-validation-1fps.yaml --data-root E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\data\streamingbench\real-time-1-50 --annotations annotations.jsonl --backend smolvlm --all-policies --output results\option-semantic-validation-real-time-1-50.jsonl --profile-output results\option-semantic-validation-real-time-1-50-profile.jsonl --cache-decoded-frames --resume

Expected: 500 completed records, zero runtime failures, and unchanged existing Q-only results.

- [ ] Step 4: Generate paired summaries.

Use summarize_records on each new policy and validate every record with validate_causal_manifest. Report correct count, accuracy, P50/P95 latency, retrieval latency, peak GPU memory, average input frames, causal/budget violations, option-score validity, and helped/hurt/unchanged counts against Recent-8 and Q-only semantic memory.

- [ ] Step 5: Document measured values and commit.

Add the option-aware table, one trace example, paired comparison, and any cache-order latency caveat to README.md. Mark the ablation complete in PROJECT_PLAN.md only after all 500 records pass. If accuracy does not improve, document that negative result without changing the current strongest-baseline claim.

    python -m pytest -q
    git diff --check
    git status --short --branch
    git add README.md PROJECT_PLAN.md
    git commit -m "docs: record option-aware retrieval validation"

## Review checkpoints

- Use a fresh focused test and full-suite check at every task boundary.
- Preserve all existing Q-only policies, configs, result files, and summaries.
- Add no new model dependency; reuse the existing CLIP and SmolVLM caches.
- Do not claim an accuracy improvement until the paired 500-record run and causal audit are complete.


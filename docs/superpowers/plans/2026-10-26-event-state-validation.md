# Three-video frozen event-state validation

Goal: test the existing raw1/1 and confirmation2/2 rules on videos excluded from
state/perception debugging, without changing implementation, model or thresholds.

Selection before outputs: derived tail250annotations, question text contains
movie clips + inserted, integer timestamp120..240s; exclude80/84/100. Within each
eligible video choose minimum SHA256(event-validation-v1:question_id). Rank videos
by SHA256(event-validation-v1:video:video_id) and select first3. No GT/prediction/
correctness or human interval selection.27eligible; selected79_2@185s,
75_2@203s,76_2@203s. Distinct exact source-file hashes; near-duplicate/source
independence is not established. All appeared in earlier benchmark runs.

Only config fields sample_id/end_time/frame_ids_stop/query_times change from
development. Primary6queries at0/20/40/60/80/100%prefix. Secondary all actual
sample times and onset-neighborhood errors are declared before results. Fixed
model/prompts,1FPS,oneimage,max8tokens,seed42,640JPEG95,unknownhold/reset,
IoU.5,raw1/1,confirmation2/2.591calls total, GPU jobs sequential. No downloads.

Use existing prepare/run/score CLI. Separate config/bundle per video. Blinded
review index links three complete-prefix pages; each requires its own actual
human event-labels.json and counting convention. Do not generate truth or pool
different conventions silently. Human metrics stay null until submitted.

- [x] Verify answer-independent selection, annotation/source hashes, only allowed
  config differences, and focused existing event-state tests.
- [x] Freeze config/cohort commit before inference. Prepare three clipped videos,
  manifests and local blinded index; check exact duration and frame counts.
- [ ] Sequentially run591classifications; replay/source/prompt/mapping/state/cost
  audit and unchanged full resumes. No code/rule tuning against these results.
- [ ] Publish per-case predictions/costs, blank human metric fields, update status
  and sync branch. Preserve development video and all earlier benchmarks.
- [ ] User complete-prefix intervals, per-case and pooled/macro metrics; distinguish
  three correlated video cases from18primary queries or591sample timestamps.

Before inference:all3configs validate and only allowed fields differ; annotation
SHA matches; sources have distinct declared SHA256. Existing17event-state tests
pass. No implementation changes or additional dependencies in this stage.

Additional diagnostic declared before inference: distinguish initial black-frame
candidate differences from later classification flicker. If benefits are confined
to startup artifacts, do not generalize them to all temporal memory errors.
No black-frame filter or other policy is introduced into this frozen experiment.

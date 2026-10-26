# Human-reviewable single-frame perception probe

Goal: separate simple scene recognition from sparse multi-frame counting without
pretending assistant observations are verified human labels.

Design: freeze24purposeful frame IDs from the3already preselected review cases.
No correctness labels in config, manifest, inference or reviewer UI. RGB JPEGs
max640side, quality95 are the exact files used by both human and model. Export
source IDs/timestamps/query bounds/image hashes in a portable manifest (no timing
fields). Images and labels stay under ignored results/. No full-resolution archive.

Same SmolVLM2-500M, one image, max8tokens; original four semantic options and cycle
[1,2,3,0],24atomic pairs/48calls, alternating first position12/12. Categories:
presenter, inserted_scene, transition, uncertain. Fixed generic visual question,
no QA options, counts, GT, example labels or timestamps enter inference. Report
parse validity, semantic consistency and letter stickiness before human review;
accuracy is null until matching explicit human labels exist. This is a task/input
format change, not a direct comparison with the video QA accuracy or memory gain.

Local HTML displays only images/source timestamps and human category controls.
Save browser progress by manifest hash; export requires reviewer and explicit
self-review checkbox. Exported labels identify provenance as human_self_reported,
not independently authenticated. User chooses an E-drive save location.

Scoring checks exact manifest digest, unique complete frame IDs and category
values; excludes human-uncertain frames, reports resolved coverage, per-class
recall, resolved accuracy and presenter/inserted balanced recall only when both
classes have human labels. Never synthesize human review or certify event counts.

Files: data/perception_review.py, evaluation/perception_probe.py,
scripts/perception_probe.py, assets/perception-review.html,
tests/test_perception_probe.py, configs/perception-probe-v1.json.

- [x] Tests RED then GREEN: causal extraction, hash-locked image verification,
  semantic mapping/invalid output, paired cleanup, human-label provenance/coverage.
- [x] Add standalone HTML label export and CLI prepare/run/score; test UI export
  logic with Node if browser unavailable. Full tests and independent review.
- [ ] Export24images and blinded review page; run48generations with unchanged
  resume, audit hashes/order/mapping/latencies and publish unlabelled results.
- [ ] Update project status/plan, sync feature branch and give user the exact
  local review page/save location. Human annotation/scoring remains pending user.

Pre-inference:526tests passed. Independent review fixed selection completeness,
exact model enforcement and primary-exception preservation; four regressions
reproduced failures before fixes. Node verifies export logic; browser layout/file
picker behavior still needs manual confirmation. Real bundle24images/3videos,
1533124JPEG bytes, manifest SHA11623fac36c397bce2408ffa263536e0fa7fcde1ee972fc387b70e0df30184d9.
Rereview approved; user-facing manifest and progress key unchanged.

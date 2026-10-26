# Fixed-selection visual sensitivity plan

Goal: test whether the Change-4+4 pilot's strong C concentration persists when
visual content, image order or option order changes. No threshold tuning.

Reuse all20frozenQA and exact saved Change-4+4 manifests from run
change-memory-v1-0fbce55d18ea. Decode causal prefixes once perQA, retain only8
specified frames, closing other decoded images immediately. Compare fourMCQ
conditions, rotating first position5/5/5/5: original; same-size RGB(127,127,127)
blank images; reverse image order; original images with option permutation
[1,2,3,0] (new index -> old index). This fixed derangement moves every option.
Never rerank based on permuted options or GT. No timestamp text in any prompt.
All ModelRequests use empty timestamp metadata because it is omitted from the
prompt; actual causal source timestamps/input orders are stored separately.
No synthetic blank is represented as actual video evidence. Reverse is an
explicit nonchronological diagnostic, not a normal streaming policy.

Map permuted predictions back to original labels/counts before scoring. Track
accuracy, parse validity, content consistency, letter stickiness and raw outputs.
Invalid MCQ remains incorrect. One atomic four-response record perQA; strict
source/config/split/annotation hashes, same-key resume; reject incomplete/mixed
source manifests before inference. Source provenance is derived annotations from
older baseline records, as in previous pilots. These inspected20videos are not
an untouched test. No proof of ignoring images follows from equal outputs.

Files: new evaluation/visual_sensitivity.py, scripts/run_visual_sensitivity.py,
tests/test_visual_sensitivity.py. Keep old model/backend/prompt/policies unchanged.
At most8retained source images+8generated blank images+incoming decode; decoder
buffers/model processing excluded. No cache or extra dependency/model download.

- [x] Write failing evaluator tests: mapping, exact prompt/options/image order,
  nofuture before retention, strict source match, normal/error image cleanup.
- [x] Implement evaluator and summary; add failing CLI/resume/hash tests, implement
  CLI; full tests and independent review before real inference.
- [ ] Run4QA/16call smoke (all first positions); resume unchanged; extend20/80.
- [ ] Audit manifests/prompt controls/GT mapping/order and old original answers;
  publish measured artifacts and limitations, update status and sync branch.

Interpretation guard: total-count questions are often order-invariant, so identical
reverse-condition outputs alone are not evidence of a temporal reasoning defect.
Blank-image invariance and option-position concentration are observational
diagnostics, not proof that the network ignores all visual inputs. A perturbed
condition's higher accuracy is not a streaming-method improvement.

Pre-inference:513tests pass. Independent review found a too-permissive fresh-source
guard; regression reproduced acceptance of an alternative valid source, then
hard frozen ID/SHA256 checks fixed it before inference. Rereview approved.

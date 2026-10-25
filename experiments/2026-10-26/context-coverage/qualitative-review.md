# Sampled qualitative review (assistant observations)

Status: **pending human verification**. These are observations from causal 1-FPS
contact sheets, not official event annotations, evidence recall, or certified
counts. Selection was fixed before inference: minimum SHA256 of
`coverage-review-v1:{stratum}:{sample_id}` within query-time strata <=60s,
(60,180]s and >180s. Sheets contain timestamps, not options, GT or predictions.
The reviewer had already seen the earlier MCQ record for sample100_3; this is
not a fully blinded human assessment. Outcomes did not determine case selection.

## Short: Real-Time Visual Understanding_sample_84_1, t=18s

One prefix page: initial black frame followed by a presenter beside a board.
No inserted clip is apparent in the sampled frames. Window-bin and prefix-bin
selections coincide because both sample [0,18]. Recent-8 covers about10-17s;
the binned policies cover about1-17s. This is an input-equivalence control for
the two binned policies, not proof that all intervening events are absent.

## Medium: Real-Time Visual Understanding_sample_100_3, t=86s

Two prefix pages: initial black frame, then the presenter and board throughout
the sampled sequence. No inserted clip is apparent at1FPS. Recent-8 covers
about78-85s; window bins about30-82s; prefix bins about5-81s. All three sheets
show similar scene content despite different temporal extents. Broader temporal
span alone need not supply more useful evidence.

## Long: Real-Time Visual Understanding_sample_80_2, t=185s

Three prefix pages: presenter/board, a color dance-like inserted scene visible
around82-97s, a brief return to presenter around98-99s, then monochrome footage
around100-123s, followed by presenter again. Lighting changes around133-158s
are visible but are not independently labelled as distinct countable events.

Recent-8 (about177-184s) and window bins (about129-181s) omit the two earlier
inserted-scene intervals. Prefix bins include monochrome footage near104s but
miss the visible color interval: consecutive representatives near81s and104s
straddle it. Fixed eight-bin coverage can still miss a relatively short event.

This motivates testing event-change retention and explicit cumulative event
state after human evidence annotation. It does not establish a model accuracy
gain, exact insertion count, or the internal cause of any prediction.

## Material and limits

Local ignored artifacts: `results/context-coverage/pilot-v1-materials/` contains
six prefix pages and three selection sheets. Source video images are not
republished here. Notes refer to sampled visible timestamps; verify continuous
video boundaries and the benchmark's counting convention before creating
evidence labels. Three assistant-reviewed cases do not satisfy the planned
20-30-case human failure analysis.

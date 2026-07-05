# Ranker-Audit Handoff Trace Analysis, 2026-07-05

## Scope

This note analyzes the ranker-to-audit handoff traces for the current
certified ranker-audit method. The purpose is to decide what the next method
iteration should be, not to replace the main paired-window result.

Analyzed records:

- 14 of the 15 main OR2023 100-order test windows.
- Seeds: 1, 2, and 3.
- K: 10.
- Schedule: `0.25:1000`.
- Ranker frontier budgets: `20,30,40,50`.
- Safety policy: `none`.
- Oracle: Java/Gurobi MILP with `label_6ori`.
- Final quality: exact staged-greedy audit from the ranker result.

One main window, `seed3:test[100,200)`, is excluded from this trace table
because that run is currently stored as split ranker/audit summaries rather
than a `frontier_summary.json`. It is included in the main 15-window certified
result and should not be treated as missing quality evidence.

## Aggregate Diagnostic Result

The 14-window handoff trace shows:

- 13/14 windows match exact staged PF after exact audit.
- 1/14 windows improves over exact staged PF after exact audit:
  `seed3:test[400,500)` by `-0.0062213021` PF.
- 0/14 windows regress after exact audit.
- All analyzed windows have 100% coverage and zero uncovered orders.
- Total ranker+audit validations: `145250`.
- Total ranker+audit uncached MILP box queries: `16076`.
- Total ranker+audit Java/Gurobi subprocess seconds: `8789.570`.
- Total ranker+audit wall-clock seconds: `11626.324`.
- Mean ranker-only PF gap to audited PF: `0.131414`.
- Max ranker-only PF gap to audited PF: `0.516028`.

The mean ranker-only gap is important: the ranker is not a certified final
optimizer. The exact audit is still doing real recovery work, which is why the
paper-facing method should remain certified ranker-audit rather than
ranker-only search.

## Handoff Table

| label | ranker cap | ranker PF | audit PF | PF delta vs exact | validations | uncached | subprocess s | elapsed s | ranker iters | last improvement | tail PF improvement | tail validations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed2:test[0,100) | 180 | 2.0339151519 | 1.9380009674 | 0.0000000000 | 7800 | 853 | 442.3405 | 585.9211 | 52 | 51 | 0.0137523284 | 180 |
| seed3:test[0,100) | 900 | 1.9780321464 | 1.9780321464 | 0.0000000000 | 3730 | 685 | 369.5694 | 460.6840 | 143 | 142 | 0.0058067454 | 410 |
| seed1:test[0,100) | 180 | 1.9860883909 | 1.8908695471 | 0.0000000000 | 8620 | 932 | 492.2234 | 646.8383 | 51 | 50 | 0.0128199449 | 180 |
| seed1:test[100,200) | 180 | 2.7244467591 | 2.2084188336 | 0.0000000000 | 25120 | 2338 | 1269.9778 | 1742.7348 | 30 | 29 | 0.0225493819 | 180 |
| seed1:test[200,300) | 180 | 2.1610223146 | 2.0315201381 | 0.0000000000 | 8220 | 931 | 515.5745 | 671.0918 | 31 | 30 | 0.0156959570 | 180 |
| seed1:test[300,400) | 180 | 2.1800241201 | 1.8968685620 | 0.0000000000 | 13620 | 1323 | 799.2195 | 1038.2689 | 49 | 48 | 0.0219599131 | 180 |
| seed1:test[400,500) | 180 | 2.0168002641 | 1.9910783173 | 0.0000000000 | 4280 | 480 | 265.5042 | 349.4464 | 92 | 91 | 0.0047454684 | 180 |
| seed2:test[100,200) | 180 | 2.5059039828 | 2.2095968714 | 0.0000000000 | 15120 | 1523 | 824.3480 | 1088.9027 | 28 | 27 | 0.0100028069 | 180 |
| seed2:test[200,300) | 180 | 2.0375794362 | 1.9186301368 | 0.0000000000 | 8280 | 928 | 529.5180 | 694.4606 | 31 | 30 | 0.0094876038 | 180 |
| seed2:test[300,400) | 180 | 2.1118666146 | 1.9960143397 | 0.0000000000 | 5820 | 672 | 407.0212 | 530.2212 | 52 | 51 | 0.0129447673 | 180 |
| seed2:test[400,500) | 180 | 2.1429548463 | 1.9929555750 | 0.0000000000 | 21040 | 1943 | 977.5555 | 1367.2486 | 54 | 53 | 0.0072919274 | 180 |
| seed3:test[200,300) | 900 | 1.9335757289 | 1.9324569101 | 0.0000000000 | 6560 | 982 | 548.8925 | 699.0928 | 227 | 226 | 0.0070650335 | 500 |
| seed3:test[300,400) | 900 | 2.0389964317 | 2.0360690769 | 0.0000000000 | 4380 | 860 | 499.6697 | 619.9253 | 173 | 172 | 0.0165620810 | 380 |
| seed3:test[400,500) | 900 | 1.8342269911 | 1.8251207019 | -0.0062213021 | 12660 | 1626 | 848.1555 | 1131.4878 | 314 | 313 | 0.0010658413 | 450 |

## Interpretation

The trace does not support a naive early-stopping rule. In every analyzed
window, the last ranker improvement occurs one iteration before the stopping
row. Therefore a simple no-improvement patience rule would not stop the costly
time-limited runs earlier.

The trace also does not support simply widening the frontier. Previous focused
ablations on `seed3:test[400,500)` show that wider frontiers and extra safety
candidates reach the same certified PF but add oracle work. This means the
issue is not that the main ranker frontier is obviously too narrow.

The useful signal is the tradeoff between marginal ranker progress and
predicted exact-audit burden. Some windows leave a large ranker-to-audit PF gap
and require heavy audit recovery, while others are already close to the audited
solution and need less recovery. A fixed ranker cap cannot distinguish these
cases.

## Next Method Direction

The next scientifically meaningful method iteration should be an adaptive
ranker-audit controller:

1. Keep the exact staged audit as the certification layer.
2. During ranker search, estimate whether another ranker iteration is likely to
   reduce subsequent audit work enough to justify its own MILP validations.
3. Handoff to exact audit when the expected audit-cost reduction is smaller
   than the expected ranker-step cost.
4. Report final PF, coverage, and uncovered orders only after exact audit.

Candidate controller signals:

- recent PF improvement per validation;
- recent PF improvement per uncached MILP box query;
- ranker-to-geometric feature disagreement on accepted moves;
- number of rejected candidates before an accepted move;
- cache-hit and uncached-query profile in the current window;
- current ranker-only PF relative to the initial and best-seen PF;
- accepted move type and dimension-change magnitude.

This gives a more defensible research contribution than adding more fixed
budgets: learning is used to allocate exact-oracle effort, while exact MILP
audit preserves the optimization claim.

## Acceptance Criteria For The Next Iteration

Against the current certified ranker-audit main policy, the adaptive controller
should be accepted only if paired windows satisfy:

- no audited PF regression;
- coverage remains `1.0000`;
- uncovered orders remain zero;
- uncached MILP box queries decrease, or subprocess seconds decrease, without
  trading away the above quality guarantees;
- any wall-clock-only win is labeled as such and not overclaimed as reduced
  oracle work.

## Reproduction Command

The manifest-level trace analysis was generated with:

```bash
python BoxDesignSurrogateRL/scripts/analyze_ranker_handoff_traces.py \
  --frontier-summary seed2:test[0,100)=BoxDesignSurrogateRL/results/ranker_assignment_aware_seed2_manifest_20260705/manifest_frontier_20260705_091502/00_s2_o0/ranker/frontier_20260705_091502/frontier_summary.json \
  --frontier-summary seed3:test[0,100)=BoxDesignSurrogateRL/results/aa_s3_o0_ranker900/manifest_frontier_20260705_122155/00_s3_o0/ranker/frontier_20260705_122155/frontier_summary.json \
  --manifest BoxDesignSurrogateRL/results/aa_s1r/manifest_frontier_20260705_103217/ranker_window_manifest.json \
  --manifest BoxDesignSurrogateRL/results/aa_s2r/manifest_frontier_20260705_092835/ranker_window_manifest.json \
  --manifest BoxDesignSurrogateRL/results/aa_s3_ranker900_o200_o400/protocol_20260705_132216/ranker_window_manifest.json \
  --tail-iterations 10 \
  --out-json BoxDesignSurrogateRL/results/aa_main_handoff_trace_analysis/handoff_trace_analysis_14of15.json \
  --out-md BoxDesignSurrogateRL/results/aa_main_handoff_trace_analysis/handoff_trace_analysis_14of15.md
```

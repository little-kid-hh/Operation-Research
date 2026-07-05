# Certified Ranker-Audit Direction, 2026-07-05

## Current Research Position

The clean method direction is not to replace the MILP oracle outright. The
defensible method is learned candidate ordering with exact certification:

1. Use the assignment-aware ranker to prioritize local box-design moves.
2. Validate only the ranked frontier with the Java/Gurobi MILP oracle.
3. Run exact staged-greedy audit from the ranker result.
4. Claim final PF and coverage only after the exact audit.

This makes the paper claim conservative: quality is certified by the same MILP
oracle and staged local-search objective as the baseline, while the learned
ranker reduces oracle work and wall-clock time.

## Why This Matters

The seed1/seed2 paired-window result already supports this main claim:

- 10/10 windows matched exact staged PF at reported precision.
- 10/10 windows had 100% coverage.
- Aggregate reductions over the 10 windows:
  - validations: 13.5%
  - uncached MILP box queries: 6.7%
  - Java/Gurobi subprocess seconds: 6.9%
  - wall-clock seconds: 7.8%

The ranker-only stage is not a reliable final optimizer. It can produce worse
PF because it may stop after validating a limited ranked frontier. Therefore,
ranker-only numbers should be reported as diagnostic or ablation results, not
as the main algorithm.

After completing seed3, the broader paired-window result is stronger:

- 15/15 windows have no certified quality regression after exact audit.
- 14/15 windows match exact staged PF at reported precision.
- 1/15 windows improves on the original exact staged run after ranker+audit.
- Exact and ranker+audit both have 100% coverage in every window.
- Aggregate reductions over the 15 windows:
  - validations: 26.0%
  - uncached MILP box queries: 11.7%
  - Java/Gurobi subprocess seconds: 12.4%
  - wall-clock seconds: 13.9%

## Seed3 Diagnostic

To test robustness beyond seed1/seed2, seed3 was probed on OR2023 test windows
with the same data, K, schedule, ranker artifact, and MILP oracle.

Configuration:

- Dataset: OR2023 test split, 100-order windows.
- K: 10.
- Schedule: `0.25:1000`.
- Ranker artifact:
  `BoxDesignSurrogateRL/results/candidate_ranker_assignment_aware_20260705/candidate_ranker_20260705_084240/candidate_ranker.joblib`.
- Ranker budget sequence: `20,30,40,50`.
- Safety policy: `none`.
- Oracle: Java/Gurobi MILP with `label_6ori`.
- Coverage repair: `geometric_expand`.

### seed3:test[0,100)

With ranker max elapsed set to 180 seconds, ranker-only stopped early and
regressed:

- exact PF: 1.9780321464
- ranker-only PF: 2.0719083668
- ranker-only stop reason: `time_limit`

Re-running the same window with a 900 second ranker limit and exact audit
restored exact quality:

| metric | exact staged | ranker+audit | reduction |
| --- | ---: | ---: | ---: |
| PF | 1.9780321464 | 1.9780321464 | 0.0000000000 delta |
| coverage | 1.0000 | 1.0000 | n/a |
| uncovered orders | 0 | 0 | n/a |
| uncached boxes | 804 | 685 | 14.8% |
| Java/Gurobi subprocess seconds | 438.0489 | 369.5694 | 15.6% |
| wall-clock seconds | 600.8094 | 460.6840 | 23.3% |
| validations | 8580 | 3730 | 56.5% |

### seed3:test[100,200)

The ranker-only stage was still not enough even with a 900 second cap:

- exact PF: 2.1883726050
- ranker-only PF: 2.2882618308
- ranker-only stop reason: `time_limit`

However, exact audit from the ranker result recovered exact PF while preserving
a cost advantage:

| metric | exact staged | ranker-only | exact audit after ranker | ranker+audit | reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| PF | 2.1883726050 | 2.2882618308 | 2.1883726050 | 2.1883726050 | 0.0000000000 delta |
| coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a |
| uncovered orders | 0 | 0 | 0 | 0 | n/a |
| uncached boxes | 1894 | 1335 | 107 | 1442 | 23.9% |
| Java/Gurobi subprocess seconds | 1019.8061 | 745.0229 | 48.8933 | 793.9161 | 22.2% |
| wall-clock seconds | 1374.9555 | 900.2675 | 134.9794 | 1035.2469 | 24.7% |
| validations | 19980 | 5600 | 4860 | 10460 | 47.6% |

### seed3 full-window result

All five seed3 windows are now complete under the certified ranker+audit
protocol:

- 4 windows match exact staged PF.
- 1 window improves over exact staged PF: seed3:test[400,500), by
  `-0.0062213021` PF.
- 0 windows regress after exact audit.
- Exact and ranker+audit both have 100% coverage in all five windows.
- Aggregate seed3 reductions:
  - validations: 49.0%
  - uncached MILP box queries: 20.8%
  - Java/Gurobi subprocess seconds: 22.1%
  - wall-clock seconds: 25.0%

Full seed3 tables are in
`ASSIGNMENT_AWARE_SEED3_CERTIFIED_SUMMARY_20260705.md`, and the combined
seed1/seed2/seed3 summary is in
`ASSIGNMENT_AWARE_SEED1_SEED2_SEED3_CERTIFIED_SUMMARY_20260705.md`.

## Interpretation

The seed3 probe clarifies the right scientific framing:

- A hard time-capped ranker is not safe as the final optimizer.
- Exact audit is essential and should be part of the main algorithm.
- The learned ranker is still useful because it moves the search into a region
  where exact audit can finish with fewer new MILP box queries.
- The main metric should be exact-quality preservation plus reduced uncached
  MILP queries, subprocess time, and wall-clock time.

This is a stronger and more defensible contribution than claiming the ML model
directly replaces the MILP feasibility oracle.

## Next Experiments

The immediate frontier, safety, targeted-safety, and fixed time-budget
ablations have now been run on the heaviest seed3 window. None replaces the
current main policy of `20,30,40,50`, safety `none`, and exact audit:

- `20,50,100,200` reaches the same certified PF but costs more.
- `all_expansions` reaches the same certified PF but is substantially more
  expensive.
- `targeted_expansion_capture` reaches the same certified PF but is not a clean
  oracle-work improvement.
- fixed 300/600 second ranker caps reach the same certified PF, but shorter
  ranker phases shift too much work into exact audit; the 600-second cap is only
  a small wall-clock tradeoff, not a clean subprocess or uncached-query win.

The next useful experiment is therefore not another fixed frontier or fixed
time cap. It is an adaptive ranker-audit controller that estimates whether
another ranker iteration is likely to reduce subsequent exact-audit cost enough
to justify its own oracle work. Ranker-only should remain an ablation, and
final PF/coverage should still be claimed only after exact audit.

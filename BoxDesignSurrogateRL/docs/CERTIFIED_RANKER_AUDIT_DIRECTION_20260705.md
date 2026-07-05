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
| validations | 20210 | 5830 | 4860 | 10690 | 47.1% |

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

1. Finish seed3 windows under the certified ranker+audit protocol, but do not
   stop after ranker-only summaries.
2. Add a frontier ablation with larger adaptive budget sequences, for example
   `20,50,100,200`, to test whether audit work can be reduced further.
3. Test `ranker-safety-policy=all_expansions` as a robustness variant.
4. Report ranker-only as an ablation, not the main method.
5. If seed3 remains quality-preserving after audit, update the aggregate
   seed1/seed2/seed3 descriptive statistics.


# Candidate Classifier Ranker Test50 Exact Audit, 2026-07-03

## Purpose

Audit whether the held-out test50 RF accepted-classifier time-budget result is
already an exact staged-greedy local optimum. The time-budget probe showed a
large PF advantage for the ranker after 180 seconds, but that does not prove
convergence.

This audit restarts exact `staged_greedy` from the ranker final boxes and runs
without a wall-clock budget.

## Setup

- Dataset: first 50 orders from the independent OR2023 test split.
- K: 10.
- Initial boxes: RF accepted top50 time-budget final boxes.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: none.
- Code version label: `ce3c997-audit-test50`.

Input ranker run:

```text
BoxDesignSurrogateRL/results/test50_classifier_timebudget_20260703/ranker_filtered_greedy/run_20260703_150201_895205
```

Exact audit run:

```text
BoxDesignSurrogateRL/results/test50_classifier_exact_audit_20260703/staged_greedy/run_20260703_154327_814314
```

## Result

| metric | RF accepted top50 at 180s | exact audit from RF boxes |
| --- | ---: | ---: |
| PF | 2.0875865358 | 1.7180912327 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 990 | 16140 |
| uncached boxes | 319 | 1399 |
| subprocess seconds | 142.3415 | 548.2533 |
| elapsed seconds | 180.8429 | 821.3924 |
| stop reason | time limit | converged/no improvement |

The exact audit improves PF by `0.3694953032` from the ranker time-budget
endpoint, a relative reduction of about `17.7%`, while preserving 100%
coverage.

## Interpretation

This audit is strong evidence that the held-out test50 time-budget result was
not an exact local optimum. The ranker still remains better than exact staged
under the same 180-second budget (`2.0876` vs `2.1932`), but exact search can
continue much longer from the ranker boxes and find substantially better boxes.

Therefore the held-out test50 claim should remain:

> The learned ranker is a better anytime search policy under a fixed wall-clock
> budget.

It should not be reported as:

> The learned ranker reaches the same held-out exact local optimum with less
> oracle work.

## Cost Caveat

The audit summary reports `disk_hits=0`, so this run should be treated as a
standalone cold-cache convergence audit, not as an incremental audit sharing
the ranker run's disk cache. The runner has been updated after this experiment
to record `oracle_cache_dir` in future manifests, which should make future
incremental-audit cost accounting less ambiguous.

## Next Step

Do not run test100 exact audit blindly until the audit protocol is fixed:

1. ensure future runs record `oracle_cache_dir`;
2. choose whether held-out audits are cold-cache convergence audits or
   incremental audits sharing a ranker cache;
3. if incremental cost is needed, run the ranker and audit through
   `scripts/run_query_budgeted_ranker_frontier.py` so both stages share the
   same oracle cache by construction.

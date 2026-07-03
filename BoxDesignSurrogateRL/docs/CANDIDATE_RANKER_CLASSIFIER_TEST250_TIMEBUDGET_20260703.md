# Candidate Classifier Ranker Test250 Time-Budget, 2026-07-03

## Purpose

Extend the independent held-out time-budget probe to the first 250 orders of
the `limit2500` test split. Unlike test50 and test100, this slice includes
orders not covered by the dev500 `0.5` checkpoint box set, so both quality and
coverage must be interpreted together.

This is not a convergence proof. Both methods are stopped at the same
iteration-boundary wall-clock budget.

## Setup

- XML:
  `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`
- Orders: first `250`
- Initial boxes:
  `BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_010436_419840/best_boxes.json`
- Schedule: `0.25:1000`
- K: 10
- Oracle: Java/Gurobi MILP, `label_6ori`
- Wall-clock budget: `--max-elapsed-seconds 180`
- Candidate-status prefetch: disabled
- Coverage repair: none

Classifier ranker:

```text
BoxDesignSurrogateRL/results/candidate_ranker_classifier_dev300_to_dev500_20260703/candidate_ranker_20260703_035111/candidate_ranker.joblib
```

## Runs

Exact staged:

```text
BoxDesignSurrogateRL/results/test250_classifier_timebudget_20260703/staged_greedy/run_20260703_151637_325749
```

RF accepted classifier top50:

```text
BoxDesignSurrogateRL/results/test250_classifier_timebudget_20260703/ranker_filtered_greedy/run_20260703_151959_664893
```

## Results

| metric | exact staged, 180s | RF accepted top50, 180s |
| --- | ---: | ---: |
| initial PF | 12.7138977639 | 12.7138977639 |
| best PF at stop | 11.9165101124 | 11.3901214442 |
| PF improvement from start | 0.7973876515 | 1.3237763198 |
| coverage | 0.992 | 0.992 |
| uncovered orders | 2 | 2 |
| stop reason | time_limit | time_limit |
| trace rows | 18 | 45 |
| generated candidates | 960 | 2580 |
| MILP validations | 960 | 600 |
| candidate avoidance rate | 0.0000 | 0.7674 |
| uncached boxes | 145 | 132 |
| uncached Java/Gurobi batches | 136 | 123 |
| subprocess seconds | 162.1439 | 158.4936 |
| ranker eval seconds | 0.0000 | 4.3464 |
| elapsed seconds | 184.3934 | 180.3225 |

## Interpretation

On held-out test250 and the same 180s budget, the classifier ranker again finds
better exact-MILP-verified boxes: PF `11.3901` vs exact staged `11.9165`.
Coverage is unchanged at `0.992`, with two uncovered orders in both runs. The
PF values are dominated by the uncovered-order penalty, so this result should
be reported with coverage and uncovered count in the same sentence.

Unlike test50 and test100, test250 also shows a modest oracle-efficiency
advantage under the same time budget: fewer MILP validations, fewer uncached
boxes (`132` vs `145`), fewer uncached Java/Gurobi batches (`123` vs `136`),
lower subprocess time, and lower elapsed time. This is still a time-budgeted
anytime result, not a converged exact-equivalence result.

The uncovered orders indicate that full held-out evaluation needs either a
coverage-repair phase or an initial box set trained on the corresponding
training split before PF-only comparisons are meaningful.

A follow-up run with the same `geometric_expand` coverage repair for both
methods is recorded in
`CANDIDATE_RANKER_CLASSIFIER_TEST250_REPAIRED_TIMEBUDGET_20260703.md`.

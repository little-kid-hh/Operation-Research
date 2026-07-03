# Candidate Classifier Ranker Test50 Time-Budget, 2026-07-03

## Purpose

Probe whether the RF accepted-move classifier ranker transfers beyond dev
orders. This is an independent held-out `limit2500` test-split sanity check,
not a convergence proof.

The key question is anytime behavior under a fixed wall-clock budget: starting
from the same dev500 exact `0.5` checkpoint, does the classifier-guided search
find better exact-MILP-verified boxes than exact staged greedy within the same
time?

## Setup

- XML:
  `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`
- Orders: first `50`
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
BoxDesignSurrogateRL/results/test50_classifier_timebudget_20260703/staged_greedy/run_20260703_145842_494441
```

RF accepted classifier top50:

```text
BoxDesignSurrogateRL/results/test50_classifier_timebudget_20260703/ranker_filtered_greedy/run_20260703_150201_895205
```

## Results

| metric | exact staged, 180s | RF accepted top50, 180s |
| --- | ---: | ---: |
| initial PF | 2.3411865018 | 2.3411865018 |
| best PF at stop | 2.1931578038 | 2.0875865358 |
| PF improvement from start | 0.1480286980 | 0.2535999660 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| stop reason | time_limit | time_limit |
| trace rows | 49 | 101 |
| generated candidates | 2820 | 5940 |
| MILP validations | 2820 | 990 |
| candidate avoidance rate | 0.0000 | 0.8333 |
| uncached boxes | 300 | 319 |
| uncached Java/Gurobi batches | 291 | 310 |
| subprocess seconds | 129.3814 | 142.3415 |
| ranker eval seconds | 0.0000 | 9.9165 |
| elapsed seconds | 180.4813 | 180.8429 |

## Interpretation

On independent test orders and the same 180s wall-clock budget, the classifier
ranker found substantially better boxes: PF `2.0876` vs exact staged `2.1932`,
with identical 100% coverage and zero uncovered orders. The relative PF
improvement over exact staged at the same time budget is about `4.8%`.

This is not a query-count win on this held-out test50 run. The classifier
validated fewer candidate box sets, but it advanced through more search
iterations and touched slightly more distinct uncached order-box dimensions
(`319` vs `300`). Subprocess time also increased (`142.3s` vs `129.4s`) and
ranker inference added `9.9s`. The supported claim here is therefore an
anytime-search claim: the learned ranker spends the budget on more useful
verified moves earlier, not that it always reduces uncached MILP labels on
held-out orders.

This complements the dev500 result, where RF accepted top50 reached the exact
staged PF with fewer uncached boxes and lower wall-clock time after audit. The
next rigorous step is to run larger held-out or replicated splits with both
quality and uncached-oracle metrics, because test50 is deliberately small and
time-budgeted.

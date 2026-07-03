# Candidate Classifier Ranker Test100 Time-Budget, 2026-07-03

## Purpose

Extend the independent held-out time-budget probe from test50 to the first 100
orders of the `limit2500` test split. This checks whether the RF accepted-move
classifier ranker still improves anytime solution quality on a larger held-out
slice.

This is not a convergence proof. Both methods are stopped at the same
iteration-boundary wall-clock budget.

## Setup

- XML:
  `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`
- Orders: first `100`
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
BoxDesignSurrogateRL/results/test100_classifier_timebudget_20260703/staged_greedy/run_20260703_150731_255192
```

RF accepted classifier top50:

```text
BoxDesignSurrogateRL/results/test100_classifier_timebudget_20260703/ranker_filtered_greedy/run_20260703_151050_348784
```

## Results

| metric | exact staged, 180s | RF accepted top50, 180s |
| --- | ---: | ---: |
| initial PF | 2.3399322113 | 2.3399322113 |
| best PF at stop | 2.2313691575 | 2.1038561898 |
| PF improvement from start | 0.1085630537 | 0.2360760215 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| stop reason | time_limit | time_limit |
| trace rows | 36 | 99 |
| generated candidates | 2040 | 5820 |
| MILP validations | 2040 | 970 |
| candidate avoidance rate | 0.0000 | 0.8333 |
| uncached boxes | 235 | 237 |
| uncached Java/Gurobi batches | 226 | 228 |
| subprocess seconds | 140.9532 | 144.4647 |
| ranker eval seconds | 0.0000 | 9.8358 |
| elapsed seconds | 182.2379 | 181.9608 |

## Interpretation

On held-out test100 and the same 180s budget, the classifier ranker again finds
better exact-MILP-verified boxes: PF `2.1039` vs exact staged `2.2314`, with
100% coverage and zero uncovered orders for both methods. The relative PF
reduction versus exact staged at the same budget is about `5.7%`.

As in the test50 probe, this is an anytime-search advantage, not a strict
uncached-query reduction. The classifier validates fewer full candidate box
sets (`970` vs `2040`), but by taking many more local-search iterations it
touches essentially the same number of distinct uncached order-box dimensions
(`237` vs `235`). Subprocess time is slightly higher and ranker inference adds
about `9.8s`.

Together with the test50 result, this strengthens the held-out quality claim:
the learned ranker transfers to independent test orders as a better
time-budgeted search policy. The stronger dev500 claim remains separate:
there, RF accepted top50 reaches the exact staged PF and exact audit confirms
the same local optimum with fewer uncached boxes and lower end-to-end time.

# Candidate Classifier Ranker Test250 Repaired Time-Budget, 2026-07-03

## Purpose

Repeat the held-out test250 time-budget probe after applying the same
geometric coverage repair to both methods. The unrepaired test250 run left two
orders uncovered for both exact staged and classifier-guided search, so PF was
dominated by the uncovered-order penalty. This repaired run gives a cleaner
100% coverage comparison.

This is still a time-budgeted anytime result, not a convergence proof.

## Setup

- XML:
  `BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml`
- Orders: first `250`
- Initial boxes:
  `BoxDesignSurrogateRL/results/candidate_predictor_dev500_limit2500dev_20260703/staged_greedy/run_20260703_010436_419840/best_boxes.json`
- Coverage repair: `geometric_expand`
- Schedule after repair: `0.25:1000`
- K: 10
- Oracle: Java/Gurobi MILP, `label_6ori`
- Wall-clock budget: `--max-elapsed-seconds 180`
- Candidate-status prefetch: disabled

Classifier ranker:

```text
BoxDesignSurrogateRL/results/candidate_ranker_classifier_dev300_to_dev500_20260703/candidate_ranker_20260703_035111/candidate_ranker.joblib
```

## Runs

Exact staged:

```text
BoxDesignSurrogateRL/results/test250_repaired_classifier_timebudget_20260703/staged_greedy/run_20260703_152616_403242
```

RF accepted classifier top50:

```text
BoxDesignSurrogateRL/results/test250_repaired_classifier_timebudget_20260703/ranker_filtered_greedy/run_20260703_152953_181578
```

## Results

| metric | exact staged + repair | RF accepted top50 + repair |
| --- | ---: | ---: |
| raw initial PF | 12.7138977639 | 12.7138977639 |
| raw initial coverage | 0.992 | 0.992 |
| repaired search-initial PF | 2.4797228171 | 2.4797228171 |
| repaired search-initial coverage | 1.000 | 1.000 |
| best PF at stop | 2.4763186692 | 2.4559321217 |
| PF improvement after repair | 0.0034041479 | 0.0237906954 |
| final coverage | 1.000 | 1.000 |
| final uncovered orders | 0 | 0 |
| stop reason | time_limit | time_limit |
| trace rows | 5 | 16 |
| generated candidates | 60 | 720 |
| total candidate evaluations | 179 | 239 |
| MILP validations after repair | 60 | 120 |
| candidate avoidance rate | 0.0000 | 0.8333 |
| uncached boxes | 176 | 152 |
| uncached Java/Gurobi batches | 167 | 143 |
| subprocess seconds | 189.1293 | 169.6840 |
| ranker eval seconds | 0.0000 | 1.2121 |
| elapsed seconds | 198.6728 | 180.9510 |

## Interpretation

After applying the same coverage repair, both methods reach 100% coverage
before the fine-stage search. Under the same requested 180s wall-clock budget,
the RF accepted-classifier ranker finds lower PF (`2.4559` vs `2.4763`) with
the same coverage and zero uncovered orders.

This repaired test250 run also supports an oracle-efficiency advantage:
classifier-guided search uses fewer uncached boxes (`152` vs `176`), fewer
uncached Java/Gurobi batches (`143` vs `167`), lower subprocess time, and lower
elapsed time. The exact run exceeds 180s because the runner exits only at
iteration boundaries and coverage repair is included in elapsed time.

The magnitude of post-repair PF improvement is modest because coverage repair
itself consumes most of the budget and leaves little time for fine-stage local
search. Still, this is the cleanest held-out test250 comparison so far because
coverage is equal and complete for both methods.

# Candidate Classifier Ranker Held-Out Summary, 2026-07-03

## Scope

This summarizes independent test-split time-budget probes for the RF
accepted-move classifier ranker. These runs start from the same dev500 exact
`0.5` checkpoint and use the first `N` orders from:

```text
BoxDesignSurrogateRL/results/splits_calibration/or2023_seed20260701_limit2500/or2023_bsp_unique_orders_test.xml
```

All runs use:

- schedule `0.25:1000`;
- K=10;
- Java/Gurobi MILP oracle, `label_6ori`;
- `--max-elapsed-seconds 180`;
- no candidate-status prefetch;
- no coverage repair.

Classifier ranker:

```text
BoxDesignSurrogateRL/results/candidate_ranker_classifier_dev300_to_dev500_20260703/candidate_ranker_20260703_035111/candidate_ranker.joblib
```

## Summary Table

| held-out slice | method | PF at stop | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test50 | exact staged | 2.1931578038 | 1.000 | 0 | 2820 | 300 | 129.3814 | 180.4813 |
| test50 | RF accepted top50 | 2.0875865358 | 1.000 | 0 | 990 | 319 | 142.3415 | 180.8429 |
| test100 | exact staged | 2.2313691575 | 1.000 | 0 | 2040 | 235 | 140.9532 | 182.2379 |
| test100 | RF accepted top50 | 2.1038561898 | 1.000 | 0 | 970 | 237 | 144.4647 | 181.9608 |
| test250 | exact staged | 11.9165101124 | 0.992 | 2 | 960 | 145 | 162.1439 | 184.3934 |
| test250 | RF accepted top50 | 11.3901214442 | 0.992 | 2 | 600 | 132 | 158.4936 | 180.3225 |

## Takeaways

The RF accepted-move classifier improves time-budgeted solution quality on all
three independent held-out slices tested so far:

- test50: PF reduction versus exact staged at the same budget is about `4.8%`.
- test100: PF reduction versus exact staged at the same budget is about `5.7%`.
- test250: PF reduction is about `4.4%`, with the same coverage and uncovered
  count. Because two orders remain uncovered, PF is dominated by the uncovered
  penalty and must be interpreted together with coverage.

The oracle-efficiency picture is mixed:

- test50 and test100 are anytime-quality wins, not uncached-query wins. The
  ranker validates fewer full candidate box sets but advances through more
  iterations and touches about the same or slightly more distinct uncached
  boxes.
- test250 shows both better time-budgeted PF and modestly fewer uncached boxes,
  uncached batches, subprocess seconds, and elapsed seconds, but coverage is
  below 100% for both methods.

## Claim Boundary

These held-out probes support the claim that the learned ranker transfers as a
better time-budgeted search policy on independent orders. They do not by
themselves prove converged exact-equivalence or a universal uncached-query
reduction.

The strongest exact-equivalence evidence remains the dev500 result:
RF accepted top50 reaches the exact staged PF, and exact audit confirms no
remaining local-search improvement while using fewer uncached boxes and lower
wall-clock time.

For a main paper table, report dev500 exact-audited results separately from
held-out time-budget results. For larger held-out comparisons, add coverage
repair or train an initial box set on the train split so PF is not dominated by
uncovered-order penalties.

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
- no candidate-status prefetch.

The main summary table below uses no coverage repair. A separate repaired
test250 check follows because the unrepaired test250 slice leaves two orders
uncovered for both methods.

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

## Repaired Test250 Check

Both methods were rerun on test250 with the same `geometric_expand` coverage
repair before fine-stage search. This gives a cleaner 100% coverage comparison:

| held-out slice | method | PF at stop | coverage | uncovered | validations | uncached boxes | subprocess sec | elapsed sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test250 repaired | exact staged | 2.4763186692 | 1.000 | 0 | 60 | 176 | 189.1293 | 198.6728 |
| test250 repaired | RF accepted top50 | 2.4559321217 | 1.000 | 0 | 120 | 152 | 169.6840 | 180.9510 |

The repaired test250 result preserves the classifier advantage under equal
100% coverage and also reduces uncached boxes, uncached batches, subprocess
time, and elapsed time. The exact run exceeds 180 seconds because the runner
exits only at iteration boundaries and coverage repair is included in elapsed
time.

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
- test250 repaired is the cleanest held-out coverage-controlled comparison so
  far: both methods finish with 100% coverage, and the classifier has lower PF
  and lower oracle cost.

## Claim Boundary

These held-out probes support the claim that the learned ranker transfers as a
better time-budgeted search policy on independent orders. They do not by
themselves prove converged exact-equivalence or a universal uncached-query
reduction.

The strongest exact-equivalence evidence remains the dev500 result:
RF accepted top50 reaches the exact staged PF, and exact audit confirms no
remaining local-search improvement while using fewer uncached boxes and lower
wall-clock time.

A later cold-cache exact audit from the test50 ranker final boxes improved PF
from `2.0875865358` to `1.7180912327` with 100% coverage, using 16140 exact
candidate validations and 821.3924 elapsed seconds. This confirms that the
test50 held-out result should be interpreted as an anytime advantage, not as
held-out exact convergence. See
`CANDIDATE_RANKER_CLASSIFIER_TEST50_EXACT_AUDIT_20260703.md`.

The corresponding exact staged convergence baseline from the same initial
boxes reached PF `1.7182177961` with 100% coverage after 22080 exact candidate
validations and 1166.3584 elapsed seconds. A shared-cache ranker 180s path plus
exact audit reached essentially the same final PF with 16980 validations and
981.7580 elapsed seconds. This strengthens the test50 claim from pure
time-budget quality to a slice-level convergence-path result with lower
incremental oracle cost, while remaining single-slice evidence. See
`CANDIDATE_RANKER_CLASSIFIER_TEST50_CONVERGENCE_20260703.md`.

The same shared-cache convergence-path protocol was replicated on test100.
Exact staged convergence reached PF `1.8121077375` with 100% coverage after
31140 validations and 2164.8369 elapsed seconds. The ranker 180s path plus
exact audit reached the same PF and coverage with 26190 validations and
1842.0075 elapsed seconds. This supports the convergence-path claim on two
held-out slice sizes, test50 and test100, while still not proving full OR2023
or multi-seed superiority. See
`CANDIDATE_RANKER_CLASSIFIER_TEST100_CONVERGENCE_20260703.md`.

The coverage-controlled repaired test250 slice also reaches the same final PF
under shared-cache ranker plus exact audit: exact repaired convergence reaches
PF `2.1590677627` after 24180 validations and 2763.3551 elapsed seconds, while
the ranker 180s path plus exact audit reaches the same PF with 23630
validations and 2712.6450 elapsed seconds. The advantage is positive but much
smaller than test50/test100, so it should be reported as diminishing returns
on the larger repaired slice. See
`CANDIDATE_RANKER_CLASSIFIER_TEST250_REPAIRED_CONVERGENCE_20260703.md`.

For a main paper table, report dev500 exact-audited results separately from
held-out time-budget results. For larger held-out comparisons, use coverage
repair or train an initial box set on the train split so PF is not dominated by
uncovered-order penalties.

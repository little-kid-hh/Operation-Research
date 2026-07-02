# Held-Out Test50 Time-Budget Probe, 2026-07-03

## Purpose

The strongest current result is dev500 progressive top50 plus exact audit. To
probe whether the idea transfers beyond the dev orders, we attempted held-out
fine-stage validation on the `limit2500` test split, starting from the same
dev500 exact `0.5` checkpoint.

Full-convergence exact runs on test500 and test100 were too slow for an
interactive validation turn and were manually stopped before summary output.
To make this type of probe reproducible, `run_milp_box_algorithms.py` now
supports `--max-elapsed-seconds`, which exits at an iteration boundary and
writes a normal summary with `stop_reason: time_limit`.

This document reports a smaller held-out test50 time-budget sanity check. It
does not replace the dev500 exact-audited result, and it is not a full
held-out convergence claim.

## Shared Setup

- XML: `or2023_seed20260701_limit2500\or2023_bsp_unique_orders_test.xml`
- Orders: first `50`
- Initial boxes:
  `candidate_predictor_dev500_limit2500dev_20260703\staged_greedy\run_20260703_010436_419840\best_boxes.json`
- Schedule: `0.25:1000`
- Java/Gurobi MILP oracle, `label_6ori`
- `--max-elapsed-seconds 180`
- Coverage repair: `none`

Exact run:

```text
C:\Users\Lenovo\Downloads\Operation-Research\BoxDesignSurrogateRL\results\fine_stage_test50_timebudget_from_dev05_20260703\staged_greedy\run_20260703_031143_320708
```

Progressive ranker run:

```text
C:\Users\Lenovo\Downloads\Operation-Research\BoxDesignSurrogateRL\results\fine_stage_test50_timebudget_from_dev05_20260703\ranker_filtered_greedy\run_20260703_031507_802304
```

## Results

| Metric | Exact 0.25, 180s budget | Progressive top50, 180s budget |
|---|---:|---:|
| Initial PF | 2.3411865018 | 2.3411865018 |
| Best PF at stop | 2.1904874984 | 2.0993270042 |
| PF improvement from start | 0.1506990034 | 0.2418594976 |
| Coverage | 1.0000 | 1.0000 |
| Uncovered orders | 0 | 0 |
| Stop reason | time_limit | time_limit |
| Trace rows | 50 | 127 |
| Generated candidates | 2880 | 7500 |
| MILP-validated candidates | 2880 | 1950 |
| Candidate avoidance rate | 0.0000 | 0.7400 |
| Oracle uncached boxes | 305 | 337 |
| Oracle subprocess seconds | 128.7076 | 128.3025 |
| Wall-clock seconds | 182.7498 | 180.4786 |
| Ranker eval seconds | 0.0000 | 8.9324 |

## Interpretation

Under the same wall-clock budget, progressive top50 finds substantially better
boxes on this held-out test50 probe: PF `2.0993` vs exact's `2.1905`, with the
same 100% coverage. This supports an anytime-search interpretation: the ranker
can guide the search toward useful moves earlier in wall-clock time.

This is not a pure oracle-query win. Progressive top50 uses fewer full
candidate evaluations, but it explores more iterations and touches more
distinct uncached order-box pairs (`337` vs `305`). Its subprocess time is
nearly identical to exact (`128.3s` vs `128.7s`), while ranker scoring adds
about `8.9s`. The wall-clock advantage here comes from better search progress
within a fixed time budget, not fewer uncached MILP boxes.

The held-out result should therefore be reported as a bounded-time sanity check,
not as a converged exact-equivalent validation. The next rigorous step is to
run progressive top50 plus exact audit on another fully converged split or
seed, or to design a selective fallback that reduces audit cost without
increasing uncached-box count.

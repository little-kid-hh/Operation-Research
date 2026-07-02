# Surrogate Filter Calibration, 2026-07-02

This note records the first MILP-verified surrogate-filter calibration on the
Windows/Gurobi machine. The purpose is to separate three questions:

1. Does finer-step exact MILP search become expensive?
2. Can a learned surrogate reduce exact MILP candidate validation?
3. Does the surrogate preserve enough candidate quality to be competitive?

## Setup

- Dataset: OR2023 unique orders dev split
- Orders: 100
- K: 10
- Seed: 0
- Oracle: Java/Gurobi exact loadability oracle
- MILP time limit: 1.0 second per Java oracle solve batch
- Schedule: `0.5:40,0.25:80`
- Coverage repair: `geometric_expand`
- Repair margins: `1.0,1.05,1.1,1.25,1.5,2.0`
- Final validity requirement: `coverage_rate=1.0`, `uncovered_orders=0`,
  `unknown_pairs=0`

## Results

| Method | PF | Trace rows | Generated candidates | MILP-validated candidates | Avoidance rate | Surrogate seconds | MILP eval seconds | Oracle subprocess seconds | Elapsed seconds | Converged? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exact staged greedy | 1.989142 | 123 | 7200 | 7200 | 0.0% | 0.0 | 521.8 | 432.8 | 555.6 | No, still improving at iteration 120 |
| Surrogate top-10 | 2.185922 | 28 | 1500 | 250 | 83.3% | 225.7 | 37.3 | 62.9 | 297.1 | Stopped early by surrogate-filter no-op |
| Surrogate top-30, before dedup | 2.152415 | 38 | 2100 | 1050 | 50.0% | 315.3 | 98.8 | 110.0 | 448.0 | Stopped early by surrogate-filter no-op |
| Surrogate top-30, dedup box predictions | 2.152415 | 38 | 2100 | 1050 | 50.0% | 40.7 | 98.2 | 109.8 | 173.8 | Stopped early by surrogate-filter no-op |
| Surrogate adaptive `10,30` + no-op fallback | 2.000377 | 123 | 7200 | 2410 | 66.5% | 142.4 | 331.6 | 313.5 | 508.0 | No, still improving at iteration 120 |

All runs finished with `coverage_rate=1.0`, `uncovered_orders=0`, and
`unknown_pairs=0`.

## Interpretation

The exact staged greedy run did not converge under `0.5:40,0.25:80`: its last
ten trace rows were all improving `0.25` moves. More fine-step iterations are
therefore needed for an exact convergence baseline on this dev100 sample.

The top-k surrogate filter reduced exact MILP validation work, but top-10 and
top-30 both stopped much earlier than exact because the true improving
candidates were filtered out. Increasing the iteration budget alone will not fix
this failure mode: once the filtered candidate set yields `noop`, the current
runner moves on or stops the stage even if unvalidated candidates still contain
exact MILP improvements.

The box-dimension deduplication optimization is important. For the top-30 run it
kept the same PF and exact MILP validation count while reducing surrogate
scoring time from 315.3 seconds to 40.7 seconds, and total elapsed time from
448.0 seconds to 173.8 seconds. The computational direction is viable after
deduplication; the remaining blocker is surrogate ranking quality.

Adaptive widening plus exact no-op fallback directly addresses false local
optima. On dev100 it reduced PF from the filtered top-30 value of 2.1524 to
2.0004 while keeping exact final validity. It still did not converge within
`0.25:80`, matching the exact run's behavior that the search was still
improving at iteration 120. The runtime saving over exact was modest in
wall-clock time (508.0 seconds vs. 555.6 seconds), but the exact MILP candidate
count dropped from 7200 to 2410. The next engineering bottleneck is therefore
oracle process overhead and remaining surrogate scoring time, while the next
modeling question is whether a better ranking surrogate can reduce fallback
uses without losing PF.

## Next Steps

1. Add a missed-candidate audit mode: periodically validate the full candidate
   set and record the exact rank of the best surrogate-kept candidate.
2. Compare fixed top-30 + fallback against adaptive `10,30` + fallback to
   isolate whether early top-10 acceptance loses measurable PF.
3. Train or calibrate a candidate-ranking surrogate using MILP-labeled greedy
   candidate data, not only order-box feasibility labels.
4. Re-run dev100 with top-k values such as 30, 40, and 50 after the audit is
   available, then choose the smallest top-k that keeps PF close to exact.
5. Extend the exact dev100 run beyond `0.25:80` until a true no-improvement step
   or a documented convergence cap.
6. Only after dev100 behavior is understood, scale to dev500 and full OR2023
   with fixed acceptance criteria.

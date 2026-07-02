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

### Short dev100 budget: `0.5:40,0.25:80`

| Method | PF | Trace rows | Generated candidates | MILP-validated candidates | Avoidance rate | Surrogate seconds | MILP eval seconds | Oracle subprocess seconds | Elapsed seconds | Converged? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exact staged greedy | 1.989142 | 123 | 7200 | 7200 | 0.0% | 0.0 | 521.8 | 432.8 | 555.6 | No, still improving at iteration 120 |
| Surrogate top-10 | 2.185922 | 28 | 1500 | 250 | 83.3% | 225.7 | 37.3 | 62.9 | 297.1 | Stopped early by surrogate-filter no-op |
| Surrogate top-30, before dedup | 2.152415 | 38 | 2100 | 1050 | 50.0% | 315.3 | 98.8 | 110.0 | 448.0 | Stopped early by surrogate-filter no-op |
| Surrogate top-30, dedup box predictions | 2.152415 | 38 | 2100 | 1050 | 50.0% | 40.7 | 98.2 | 109.8 | 173.8 | Stopped early by surrogate-filter no-op |
| Surrogate adaptive `10,30` + no-op fallback | 2.000377 | 123 | 7200 | 2410 | 66.5% | 142.4 | 331.6 | 313.5 | 508.0 | No, still improving at iteration 120 |
| Surrogate top-30 + no-op fallback | 2.000377 | 123 | 7200 | 3810 | 47.1% | 139.8 | 384.4 | 344.7 | 557.8 | No, still improving at iteration 120 |

### Longer dev100 budget: `0.5:80,0.25:240`

| Method | PF | Trace rows | Generated candidates | MILP-validated candidates | Avoidance rate | Surrogate seconds | MILP eval seconds | Oracle subprocess seconds | Elapsed seconds | Converged? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exact staged greedy | 1.769646 | 323 | 19200 | 19200 | 0.0% | 0.0 | 1252.2 | 969.0 | 1286.1 | No, still improving at iteration 320 |
| Surrogate adaptive `10,30` + no-op fallback | 1.786019 | 323 | 19200 | 6120 | 68.1% | 374.3 | 736.6 | 648.6 | 1145.9 | No, still improving at iteration 320 |

### Convergence-gated dev100 protocol

This protocol first runs exact `0.5` until exact no-op, then starts the `0.25`
stage from the converged `0.5` checkpoint.

| Stage | Method | PF | Trace rows | Generated candidates | MILP-validated candidates | Avoidance rate | Surrogate seconds | MILP eval seconds | Oracle subprocess seconds | Elapsed seconds | Converged? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.5` | Exact staged greedy | 1.760638 | 223 | 13200 | 13200 | 0.0% | 0.0 | 841.5 | 657.8 | 875.6 | Yes, exact no-op at iteration 220 |
| `0.25` from exact `0.5` checkpoint | Exact staged greedy | 1.746790 | 13 | 720 | 720 | 0.0% | 0.0 | 66.2 | 54.9 | 68.7 | Yes, exact no-op at iteration 12 |
| `0.25` from exact `0.5` checkpoint | Surrogate adaptive `10,30` + no-op fallback | 1.746790 | 13 | 720 | 310 | 56.9% | 14.3 | 47.0 | 42.5 | 63.7 | Yes, exact no-op at iteration 12 |
| `0.1` from exact `0.25` checkpoint | Exact staged greedy | 1.739964 | 17 | 960 | 960 | 0.0% | 0.0 | 79.4 | 65.0 | 82.0 | Yes, exact no-op at iteration 16 |
| `0.1` from exact `0.25` checkpoint | Surrogate adaptive `10,30` + no-op fallback | 1.739964 | 17 | 960 | 470 | 51.0% | 20.4 | 61.5 | 53.9 | 84.3 | Yes, exact no-op at iteration 16 |

All runs finished with `coverage_rate=1.0`, `uncovered_orders=0`, and
`unknown_pairs=0`.

The dev100 XML dimensions are integer-valued: 100% of item dimensions and
100% of order max sorted dimensions are multiples of 1.0. Therefore the
motivation for `0.1` is not raw order-dimension granularity. Its value comes
from finer box-adjustment lattice points around continuous box dimensions after
KMeans/repair/local-search moves.

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

Fixed top-30 plus fallback reached the same PF trajectory as adaptive
`10,30` plus fallback, but it validated 3810 candidates and took 557.8 seconds.
The adaptive variant validated only 2410 candidates and took 508.0 seconds, so
there is no dev100 evidence that always validating top-30 before accepting a
move improves quality. The current default calibration candidate is therefore
adaptive `10,30` plus no-op fallback.

The longer schedule supports the scaling argument more strongly. Both exact and
adaptive runs continued improving through the final `0.25` iteration, so neither
should be called converged. Exact reached PF 1.7696 after validating 19200
candidates. Adaptive `10,30` plus fallback reached PF 1.7860, a gap of 0.0164,
while validating only 6120 candidates. The MILP-candidate reduction is large
(68.1%), but elapsed wall-clock reduction is modest (1286.1 seconds to 1145.9
seconds), mainly because surrogate scoring took 374.3 seconds and the Java
oracle subprocess overhead remains high.

The convergence-gated protocol changes the interpretation of the staged
baseline. In the `0.5:80,0.25:240` run, the `0.5` stage had not converged:
iteration 80 was still an improving exact move. Running exact `0.5` alone to
no-op required 220 iterations and reached PF 1.7606. Starting `0.25` from that
checkpoint converged in only 12 iterations, reaching PF 1.7468. The surrogate
adaptive `10,30` plus fallback matched the exact `0.25` continuation's final PF
while validating 310 instead of 720 fine-stage candidates.

Continuing from the exact `0.25` checkpoint with `0.1` is meaningful on dev100:
exact PF improves further from 1.7468 to 1.7400 before exact no-op. Surrogate
adaptive `10,30` plus fallback again matches the exact final PF while validating
470 instead of 960 candidates. Wall-clock is slightly slower for the surrogate
run at this small scale because surrogate scoring overhead dominates the saved
MILP validations.

## Next Steps

1. Add a missed-candidate audit mode: periodically validate the full candidate
   set and record the exact rank of the best surrogate-kept candidate.
2. Treat convergence-gated coarse-to-fine as the main dev protocol: exact
   `0.5` to no-op, then compare exact vs. surrogate-filtered `0.25` and `0.1`
   fine stages from the same checkpoint.
3. Reduce surrogate scoring and Java oracle overhead; otherwise MILP-call
   reduction does not translate cleanly into wall-clock speedup.
4. Train or calibrate a candidate-ranking surrogate using MILP-labeled greedy
   candidate data, not only order-box feasibility labels.
5. Re-run dev100 with top-k values such as 30, 40, and 50 after the audit is
   available, then choose the smallest top-k that keeps PF close to exact.
6. Only after dev100 behavior is understood, scale to dev500 and full OR2023
   with fixed acceptance criteria.

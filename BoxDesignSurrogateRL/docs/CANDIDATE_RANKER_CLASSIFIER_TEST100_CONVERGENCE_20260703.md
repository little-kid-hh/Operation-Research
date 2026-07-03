# Candidate Classifier Ranker Test100 Convergence Comparison, 2026-07-03

## Purpose

Replicate the held-out test50 convergence-path protocol on the first 100
orders of the independent OR2023 test split. This checks whether the
shared-cache ranker plus exact-audit result is a one-slice artifact or whether
it also holds on a larger held-out slice.

The comparison answers:

> From the same initial box set, can a time-budgeted learned ranker followed by
> exact audit reach the same local-search quality as exact staged convergence
> while using fewer exact MILP oracle resources?

## Setup

- Dataset: first 100 orders from the independent OR2023 test split.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: none.
- Exact convergence wall-clock budget: none.
- Ranker wall-clock budget: `--ranker-max-elapsed-seconds 180`.
- Audit wall-clock budget: none.
- Ranker budget sequence: `10,20,30,40,50`.

Exact convergence run:

```text
BoxDesignSurrogateRL/results/test100_exact_convergence_20260703/staged_greedy/run_20260703_165709_188504
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test100_shared_cache_ranker_frontier_20260703/frontier_20260703_173350
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test100_shared_cache_ranker_frontier_20260703/top10_20_30_40_50
```

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 1.8121077375 | 1.8121077375 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 31140 | 26190 |
| uncached boxes | 2647 | 2374 |
| subprocess seconds | 1560.0587 | 1366.4305 |
| elapsed seconds | 2164.8369 | 1842.0075 |

The RF top50 path plus exact audit reaches the same converged PF and coverage
as exact staged from the original initial boxes.

The cost comparison favors the ranker path:

- 15.9% fewer MILP candidate validations.
- 10.3% fewer uncached boxes.
- 12.4% lower Java/Gurobi subprocess time.
- 14.9% lower wall-clock time.

The shared-cache ranker run stopped by the 180-second budget at PF
`2.1003328338`, then exact audit continued from those boxes to PF
`1.8121077375`.

## Time-To-Quality

The exact convergence trace shows when exact staged first reaches key quality
thresholds:

| threshold | exact iteration | exact PF at crossing | cumulative MILP eval seconds | cumulative validations |
| --- | ---: | ---: | ---: | ---: |
| exact 180s PF `2.2313691575` | 34 | 2.2313691575 | 171.5265 | 2040 |
| shared-cache ranker 180s PF `2.1003328338` | 98 | 2.0995206568 | 443.4809 | 5880 |

This supports the same anytime-search interpretation as test50. The learned
ranker reaches PF `2.1003` in about 181.1 wall-clock seconds with 990 exact
candidate validations. Exact staged needs 5880 validations and about 443.5
cumulative MILP-evaluation seconds before crossing the same PF threshold.

## Interpretation

For held-out test100, the accepted-move classifier ranker reproduces the
test50 convergence-path result: under the same exact MILP oracle and exact
acceptance rule, a time-budgeted learned ranker followed by exact audit reaches
the same local-search quality as exact staged convergence with lower measured
oracle and wall-clock cost.

This is still slice-level evidence. It strengthens the held-out claim because
the result now appears on both test50 and test100, but it does not prove full
OR2023 or multi-seed superiority.

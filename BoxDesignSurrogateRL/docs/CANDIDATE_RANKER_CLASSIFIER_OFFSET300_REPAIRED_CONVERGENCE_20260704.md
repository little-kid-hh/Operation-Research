# Candidate Classifier Ranker Offset300 Repaired Convergence Comparison, 2026-07-04

## Purpose

Run a third non-prefix held-out order-window check for the shared-cache
convergence-path protocol. This extends the offset100 and offset200 evidence
and tests whether the learned candidate ranker preserves exact-audited final
quality on another independent slice of the OR2023 test split.

The comparison answers:

> On a third non-prefix held-out window, can a time-budgeted learned ranker
> followed by exact audit reach the same repaired local-search quality as exact
> staged convergence while using fewer exact MILP oracle resources?

## Setup

- Dataset: OR2023 test split orders with zero-based indices `[300, 400)`.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: `geometric_expand` for both exact and ranker+audit.
- Exact convergence wall-clock budget: none.
- Ranker wall-clock budget: `--ranker-max-elapsed-seconds 180`.
- Audit wall-clock budget: none.
- Ranker budget sequence: `10,20,30,40,50`.
- Code version: `6306430`.

The raw initial boxes have PF `13.9609471639`, coverage `0.990`, and one
uncovered order. The shared repair step raises coverage to `1.000` and sets
the repaired search-initial PF to `2.2389006033`.

Exact repaired convergence run:

```text
BoxDesignSurrogateRL/results/test100_offset300_exact_repaired_convergence_20260704/staged_greedy/run_20260704_230057_695730
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test100_offset300_repaired_shared_cache_ranker_frontier_20260704/frontier_20260704_233156
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test100_offset300_repaired_shared_cache_ranker_frontier_20260704/top10_20_30_40_50
```

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 1.8590426956 | 1.8590426956 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 28560 | 25460 |
| uncached boxes | 2478 | 2301 |
| subprocess seconds | 1275.1553 | 1143.9761 |
| elapsed seconds | 1830.4731 | 1596.6522 |

The RF top50 path plus exact audit reaches the same converged PF and coverage
as exact staged from the same repaired initial boxes.

The cost comparison favors the ranker path:

- 10.9% fewer MILP candidate validations.
- 7.1% fewer uncached boxes.
- 10.3% lower Java/Gurobi subprocess time.
- 12.8% lower wall-clock time.

The ranker-only 180-second phase stopped at PF `2.1224721126` with full
coverage. The exact audit then continued from those boxes to PF
`1.8590426956`.

## Interpretation

This third non-prefix window is consistent with offset100 and offset200:
ranker-guided search plus exact audit reaches identical final PF and coverage
while reducing all primary cost metrics. Among the three non-prefix windows so
far, offset300 shows the largest wall-clock reduction.

The current evidence now supports a stronger slice-level statement: on three
non-prefix held-out windows, exact-verified learned candidate ranking preserves
final local-search quality and reduces measured exact-oracle work. This still
requires additional seeds or windows before it becomes a statistical claim.

# Candidate Classifier Ranker Offset400 Repaired Convergence Comparison, 2026-07-05

## Purpose

Run the final non-prefix 100-order held-out window from the OR2023 test split.
Together with the prefix test100 and the offset100/offset200/offset300 windows,
this completes a five-window evaluation over all 500 test orders.

The comparison answers:

> On the final non-prefix held-out window, can a time-budgeted learned ranker
> followed by exact audit reach the same repaired local-search quality as exact
> staged convergence while using fewer exact MILP oracle resources?

## Setup

- Dataset: OR2023 test split orders with zero-based indices `[400, 500)`.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: `geometric_expand` for both exact and ranker+audit.
- Exact convergence wall-clock budget: none.
- Ranker wall-clock budget: `--ranker-max-elapsed-seconds 180`.
- Audit wall-clock budget: none.
- Ranker budget sequence: `10,20,30,40,50`.
- Code version: `e0979fd`.

The raw initial boxes have PF `30.6247136867`, coverage `0.970`, and three
uncovered orders. The shared repair step raises coverage to `1.000` and sets
the repaired search-initial PF to `2.7064905957`.

Exact repaired convergence run:

```text
BoxDesignSurrogateRL/results/test100_offset400_exact_repaired_convergence_20260705/staged_greedy/run_20260705_000145_173451
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test100_offset400_repaired_shared_cache_ranker_frontier_20260705/frontier_20260705_004335
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test100_offset400_repaired_shared_cache_ranker_frontier_20260705/top10_20_30_40_50
```

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 1.9889904647 | 1.9889904647 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 36360 | 31860 |
| uncached boxes | 3173 | 2853 |
| subprocess seconds | 1857.8720 | 1660.0325 |
| elapsed seconds | 2488.9937 | 2277.6879 |

The RF top50 path plus exact audit reaches the same converged PF and coverage
as exact staged from the same repaired initial boxes.

The cost comparison favors the ranker path:

- 12.4% fewer MILP candidate validations.
- 10.1% fewer uncached boxes.
- 10.6% lower Java/Gurobi subprocess time.
- 8.5% lower wall-clock time.

The ranker-only 180-second phase stopped at PF `2.5364186537` with full
coverage. The exact audit then continued from those boxes to PF
`1.9889904647`.

## Interpretation

This final non-prefix window is consistent with the previous offset windows:
ranker-guided search plus exact audit reaches identical final PF and coverage
while reducing all primary cost metrics. The test split is now fully covered by
five 100-order windows: `[0,100)`, `[100,200)`, `[200,300)`, `[300,400)`, and
`[400,500)`.

The five-window result supports a substantially stronger paper-facing claim
than the earlier prefix-only evidence: under this protocol, exact-verified
learned candidate ranking preserves converged local-search quality across the
entire 500-order test split when evaluated window-wise, while reducing measured
exact-oracle work and wall-clock time.

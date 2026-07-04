# Candidate Classifier Ranker Offset200 Repaired Convergence Comparison, 2026-07-04

## Purpose

Run a second non-prefix held-out order-window check for the shared-cache
convergence-path protocol. This complements the offset100 window and tests
whether the learned candidate ranker still preserves exact-audited final
quality on a harder independent slice.

The comparison answers:

> On a harder non-prefix held-out window, can a time-budgeted learned ranker
> followed by exact audit reach the same repaired local-search quality as exact
> staged convergence while using fewer exact MILP oracle resources?

## Setup

- Dataset: OR2023 test split orders with zero-based indices `[200, 300)`.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: `geometric_expand` for both exact and ranker+audit.
- Exact convergence wall-clock budget: none.
- Ranker wall-clock budget: `--ranker-max-elapsed-seconds 180`.
- Audit wall-clock budget: none.
- Ranker budget sequence: `10,20,30,40,50`.
- Code version: `952549e`.

The raw initial boxes have PF `25.9066669227`, coverage `0.980`, and two
uncovered orders. The shared repair step raises coverage to `1.000` and sets
the repaired search-initial PF to `2.4794986684`.

Exact repaired convergence run:

```text
BoxDesignSurrogateRL/results/test100_offset200_exact_repaired_convergence_20260704/staged_greedy/run_20260704_212204_089123
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test100_offset200_repaired_shared_cache_ranker_frontier_20260704/frontier_20260704_221144
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test100_offset200_repaired_shared_cache_ranker_frontier_20260704/top10_20_30_40_50
```

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 1.9098141289 | 1.9098141289 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 43020 | 39220 |
| uncached boxes | 3714 | 3485 |
| subprocess seconds | 2114.5624 | 1955.3046 |
| elapsed seconds | 2954.3013 | 2729.9786 |

The RF top50 path plus exact audit reaches the same converged PF and coverage
as exact staged from the same repaired initial boxes.

The cost comparison favors the ranker path:

- 8.8% fewer MILP candidate validations.
- 6.2% fewer uncached boxes.
- 7.5% lower Java/Gurobi subprocess time.
- 7.6% lower wall-clock time.

The ranker-only 180-second phase stopped at PF `2.3675964860` with full
coverage. The exact audit then continued from those boxes to PF
`1.9098141289`.

## Interpretation

This is a harder slice than offset100: exact staged convergence required
`43,020` MILP candidate validations and `2,954.3` wall-clock seconds. The
ranker path still reached the exact same audited final PF and coverage while
reducing all primary cost metrics.

Together with offset100, this strengthens the conservative paper claim that
exact-verified learned candidate ranking can reduce exact-oracle work on
non-prefix held-out windows. The result remains seed-0 and window-level
evidence, not a full-OR2023 or multi-seed statistical claim.

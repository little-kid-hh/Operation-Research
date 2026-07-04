# Candidate Classifier Ranker Offset100 Repaired Convergence Comparison, 2026-07-04

## Purpose

Run the shared-cache convergence-path protocol on a non-prefix held-out order
window. Earlier test50/test100/test250 evidence used nested prefixes of the
test XML; this run checks whether the result also holds on the independent
window with zero-based order indices `[100, 200)`.

The comparison answers:

> On a non-prefix held-out window, can a time-budgeted learned ranker followed
> by exact audit reach the same repaired local-search quality as exact staged
> convergence while using fewer exact MILP oracle resources?

## Setup

- Dataset: OR2023 test split orders with zero-based indices `[100, 200)`.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: `geometric_expand` for both exact and ranker+audit.
- Exact convergence wall-clock budget: none.
- Ranker wall-clock budget: `--ranker-max-elapsed-seconds 180`.
- Audit wall-clock budget: none.
- Ranker budget sequence: `10,20,30,40,50`.
- Code version: `57df423`.

The raw initial boxes have PF `17.3569047601`, coverage `0.990`, and one
uncovered order. The shared repair step raises coverage to `1.000` and sets
the repaired search-initial PF to `2.6767037661`.

Exact repaired convergence run:

```text
BoxDesignSurrogateRL/results/test100_offset100_exact_repaired_convergence_20260704/staged_greedy/run_20260704_201108_079489
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test100_offset100_repaired_shared_cache_ranker_frontier_20260704/frontier_20260704_204552
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test100_offset100_repaired_shared_cache_ranker_frontier_20260704/top10_20_30_40_50
```

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 2.0757308361 | 2.0757308361 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 28980 | 26200 |
| uncached boxes | 2521 | 2416 |
| subprocess seconds | 1469.8335 | 1419.6851 |
| elapsed seconds | 2047.5268 | 1944.8361 |

The RF top50 path plus exact audit reaches the same converged PF and coverage
as exact staged from the same repaired initial boxes.

The cost comparison favors the ranker path:

- 9.6% fewer MILP candidate validations.
- 4.2% fewer uncached boxes.
- 3.4% lower Java/Gurobi subprocess time.
- 5.0% lower wall-clock time.

The ranker-only 180-second phase stopped at PF `2.5724241839` with full
coverage. The exact audit then continued from those boxes to PF
`2.0757308361`.

## Interpretation

This run closes an important rigor gap in the earlier evidence. Because the
window is `[100, 200)` rather than a nested prefix, the result is less likely
to be an artifact of repeatedly evaluating the easiest early orders in the
same XML ordering.

The advantage is positive but modest on this repaired independent window. It
supports a conservative paper claim: exact-verified learned candidate ranking
can preserve final local-search quality and reduce measured exact-oracle work
on held-out order windows. It does not support a claim of large universal
speedup or full-OR2023 dominance.

# Candidate Classifier Ranker Test250 Repaired Convergence Comparison, 2026-07-03

## Purpose

Extend the shared-cache convergence-path protocol to a larger held-out slice
while controlling coverage. The unrepaired test250 slice leaves two orders
uncovered from the dev500 `0.5` checkpoint, so PF is dominated by the
uncovered-order penalty. This comparison uses the same `geometric_expand`
coverage repair for both exact staged search and the ranker-plus-audit path.

The comparison asks:

> After coverage is repaired to 100%, can a time-budgeted learned ranker
> followed by exact audit reach the same local-search quality as exact staged
> convergence while using fewer exact MILP oracle resources?

## Setup

- Dataset: first 250 orders from the independent OR2023 test split.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Coverage repair: `geometric_expand`.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Exact convergence wall-clock budget: none.
- Ranker wall-clock budget: `--ranker-max-elapsed-seconds 180`.
- Audit wall-clock budget: none.
- Ranker budget sequence: `10,20,30,40,50`.

Exact repaired convergence run:

```text
BoxDesignSurrogateRL/results/test250_exact_repaired_convergence_20260703/staged_greedy/run_20260703_181235_291147
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test250_repaired_shared_cache_ranker_frontier_20260703/frontier_20260703_185904
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test250_repaired_shared_cache_ranker_frontier_20260703/top10_20_30_40_50
```

## Coverage Repair

The raw initial boxes have PF `12.7138977639`, coverage `0.992`, and two
uncovered orders. The shared repair step raises coverage to `1.000` and sets
the repaired search-initial PF to `2.4797228171`.

This repair is part of the coverage-controlled protocol. PF comparisons below
should be read as post-repair local-search comparisons, not as unrepaired
initial-box comparisons.

## Main Comparison

| metric | exact repaired convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 2.1590677627 | 2.1590677627 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 24180 | 23630 |
| uncached boxes | 2175 | 2140 |
| subprocess seconds | 2284.5554 | 2236.8717 |
| elapsed seconds | 2763.3551 | 2712.6450 |

The RF top50 path plus exact audit reaches the same converged PF and coverage
as exact staged from the repaired initial boxes.

The cost comparison favors the ranker path, but the margin is modest:

- 2.3% fewer MILP candidate validations.
- 1.6% fewer uncached boxes.
- 2.1% lower Java/Gurobi subprocess time.
- 1.8% lower wall-clock time.

The shared-cache ranker run stopped by the 180-second budget at PF
`2.4579211581`, then exact audit continued from those boxes to PF
`2.1590677627`.

## Time-To-Quality

The exact convergence trace shows when exact staged first reaches key quality
thresholds after repair:

| threshold | exact iteration | exact PF at crossing | cumulative MILP eval seconds | cumulative validations |
| --- | ---: | ---: | ---: | ---: |
| exact 180s repaired PF `2.4763186692` | 1 | 2.4763186692 | 59.1291 | 60 |
| shared-cache ranker 180s PF `2.4579211581` | 10 | 2.4576505193 | 116.7118 | 600 |

Unlike test50 and test100, the test250 repaired slice begins much closer to a
post-repair local basin: exact staged reaches the 180-second exact time-budget
PF after one search iteration. The ranker still reaches a better 180-second PF
under the same wall-clock budget, but the time-to-quality separation is much
smaller than on test50/test100.

## Interpretation

For coverage-controlled held-out test250, the accepted-move classifier ranker
reproduces the final-quality result from test50/test100: a time-budgeted
learned ranker followed by exact audit reaches the same local-search quality as
exact staged convergence with lower measured oracle and wall-clock cost.

The cost reduction is much smaller on test250 repaired. This should be reported
as a positive but diminishing return, not as a large speedup. The likely reason
is that coverage repair already moves the search into a narrow local region
where there are fewer useful ranker decisions before exact audit dominates the
remaining work.

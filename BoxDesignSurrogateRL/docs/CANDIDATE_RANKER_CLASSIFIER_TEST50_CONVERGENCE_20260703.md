# Candidate Classifier Ranker Test50 Convergence Comparison, 2026-07-03

## Purpose

Complete the held-out test50 picture by running exact `staged_greedy` from the
same initial boxes until local convergence. Earlier test50 evidence compared
methods under a 180-second wall-clock budget and then audited the ranker final
boxes. This run answers a stricter question:

> From the same initial box set, what does exact staged greedy reach when it is
> allowed to converge, and how does its cost compare with the ranker path plus
> exact audit?

## Setup

- Dataset: first 50 orders from the independent OR2023 test split.
- K: 10.
- Initial boxes: dev500 exact `0.5` checkpoint.
- Schedule: `0.25:1000`.
- Oracle: Java/Gurobi MILP, `label_6ori`.
- Coverage repair: none.
- Wall-clock budget: none.
- Code version label: `1f49bad-exact-test50-convergence`.

Exact convergence run:

```text
BoxDesignSurrogateRL/results/test50_exact_convergence_20260703/staged_greedy/run_20260703_160207_147284
```

Shared-cache ranker plus audit frontier:

```text
BoxDesignSurrogateRL/results/test50_shared_cache_ranker_frontier_20260703/frontier_20260703_163254
```

The shared-cache frontier uses one oracle cache for the ranker run and the
subsequent exact audit:

```text
BoxDesignSurrogateRL/results/oracle_cache_test50_shared_cache_ranker_frontier_20260703/top10_20_30_40_50
```

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit, shared cache |
| --- | ---: | ---: |
| final PF | 1.7182177961 | 1.7180912327 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 22080 | 16980 |
| uncached boxes | 1892 | 1695 |
| subprocess seconds | 756.5954 | 675.3390 |
| elapsed seconds | 1166.3584 | 981.7580 |

The RF top50 path plus exact audit reaches essentially the same converged PF as
exact staged from the original initial boxes. The ranker-audit PF is lower by
`0.0001265634`, which is only `0.0074%` relative to the exact convergence PF.

The cost comparison favors the ranker path:

- 23.1% fewer MILP candidate validations.
- 10.4% fewer uncached boxes.
- 10.7% lower Java/Gurobi subprocess time.
- 15.8% lower wall-clock time.

The shared-cache ranker run stopped by the 180-second budget at PF
`2.0790315490`, then exact audit continued from those boxes to PF
`1.7180912327`.

## Time-To-Quality

The exact convergence trace shows when exact staged first reaches key quality
thresholds:

| threshold | exact iteration | exact PF at crossing | cumulative MILP eval seconds | cumulative validations |
| --- | ---: | ---: | ---: | ---: |
| exact 180s PF `2.1931578038` | 47 | 2.1931578038 | 178.9251 | 2820 |
| shared-cache ranker 180s PF `2.0790315490` | 91 | 2.0789416186 | 317.6653 | 5460 |

This supports a clear anytime-search advantage. The learned ranker reaches PF
`2.0790` in about 181.0 wall-clock seconds with 1020 exact candidate
validations. Exact staged needs 5460 validations and about 317.7 cumulative
MILP-evaluation seconds before crossing the same PF threshold.

## Cache And Accounting Caveat

The shared-cache frontier is the primary cost accounting for this comparison.
The ranker stage and exact audit use the same oracle cache directory, so audit
misses are incremental after the labels already computed by the ranker stage.

The exact staged convergence baseline used its own cold cache:

```text
BoxDesignSurrogateRL/results/oracle_cache_test50_exact_convergence_20260703
```

An earlier ranker/audit comparison was a conservative sum of two cold-cache
runs. It reached the same PF but reported 17130 validations, 1718 uncached
boxes, 690.5948 subprocess seconds, and 1002.2353 elapsed seconds. The
shared-cache frontier supersedes that cost accounting.

## Interpretation

For held-out test50, the strongest defensible claim is now:

> The accepted-move classifier ranker changes the search trajectory so that,
> under the same exact MILP oracle and exact acceptance rule, it reaches better
> boxes much earlier and, after exact audit, reaches essentially the same local
> quality as exact staged convergence with lower measured oracle and wall-clock
> cost on this slice.

This still should not be generalized to full OR2023 or multi-seed performance
without additional replicated runs.

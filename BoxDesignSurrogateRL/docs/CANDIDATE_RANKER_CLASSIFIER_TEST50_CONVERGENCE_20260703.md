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

## Main Comparison

| metric | exact staged convergence | RF top50 180s + exact audit |
| --- | ---: | ---: |
| final PF | 1.7182177961 | 1.7180912327 |
| coverage | 1.000 | 1.000 |
| uncovered orders | 0 | 0 |
| MILP validations | 22080 | 17130 |
| uncached boxes | 1892 | 1718 |
| subprocess seconds | 756.5954 | 690.5948 |
| elapsed seconds | 1166.3584 | 1002.2353 |

The RF top50 path plus exact audit reaches essentially the same converged PF as
exact staged from the original initial boxes. The ranker-audit PF is lower by
`0.0001265634`, which is only `0.0074%` relative to the exact convergence PF.

The cost comparison favors the ranker path:

- 22.4% fewer MILP candidate validations.
- 9.2% fewer uncached boxes.
- 8.7% lower Java/Gurobi subprocess time.
- 14.1% lower wall-clock time.

## Time-To-Quality

The exact convergence trace shows when exact staged first reaches key quality
thresholds:

| threshold | exact iteration | exact PF at crossing | cumulative MILP eval seconds | cumulative validations |
| --- | ---: | ---: | ---: | ---: |
| exact 180s PF `2.1931578038` | 47 | 2.1931578038 | 178.9251 | 2820 |
| ranker 180s PF `2.0875865358` | 88 | 2.0865167433 | 307.5474 | 5280 |

This supports a clear anytime-search advantage. The learned ranker reaches PF
`2.0876` in about 180.8 wall-clock seconds with 990 exact candidate
validations. Exact staged needs 5280 validations and about 307.5 cumulative
MILP-evaluation seconds before crossing the same PF threshold.

## Cache And Accounting Caveat

The RF top50 180s run and its later exact audit were separate cold-cache runs;
the original ranker run did not record an `oracle_cache_dir`, and the audit
reported `disk_hits=0`. Therefore the combined ranker-audit cost above is a
conservative sum of two standalone runs, not a shared-cache incremental audit.
A properly shared-cache frontier run should be no more expensive in uncached
oracle queries, but that remains to be measured explicitly.

The exact staged convergence baseline used its own cold cache:

```text
BoxDesignSurrogateRL/results/oracle_cache_test50_exact_convergence_20260703
```

## Interpretation

For held-out test50, the strongest defensible claim is now:

> The accepted-move classifier ranker changes the search trajectory so that,
> under the same exact MILP oracle and exact acceptance rule, it reaches better
> boxes much earlier and, after exact audit, reaches essentially the same local
> quality as exact staged convergence with lower measured oracle and wall-clock
> cost on this slice.

This still should not be generalized to full OR2023 or multi-seed performance
without additional replicated runs.

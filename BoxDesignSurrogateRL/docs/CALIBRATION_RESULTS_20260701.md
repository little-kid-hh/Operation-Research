# MILP Calibration Results, 2026-07-01

These results are development calibration evidence only. They are not final
held-out claims.

## Protocol

- Code version: `77864b9`
- Machine: Windows/Gurobi over Tailscale
- Split: `or2023_seed20260701_limit500`
- Evaluation set: hash-materialized dev XML, `100` orders
- `K`: `10`
- Seeds: `0,1,2`
- Feasibility: Java/Gurobi MILP oracle
- Orientation label: `label_6ori`
- MILP time limit: `1s`
- Coverage repair:

```text
--coverage-repair geometric_expand
--repair-margins 1.0,1.05,1.1,1.25,1.5,2.0
--repair-max-rounds 3
```

Comparison label:

```text
dev100_repaired_two_sweep_fixed_vs_staged
```

Methods:

- `fixed05_i2`: `paper_fixed_step`, `--fixed-step 0.5`, `--iterations 2`
- `staged05_025_i2`: `staged_greedy`, `--schedule 0.5:1,0.25:1`

Both methods start from the same KMeans boxes and the same MILP-verified
coverage repair for each seed.

## Paired Results

PF is lower-is-better. Delta is `staged05_025_i2 - fixed05_i2`.

| Seed | Fixed PF | Staged PF | Delta PF | Fixed uncovered | Staged uncovered | Fixed unknown | Staged unknown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.246383 | 2.245592 | -0.000791 | 0 | 0 | 0 | 0 |
| 1 | 2.218329 | 2.220528 | +0.002199 | 0 | 0 | 0 | 0 |
| 2 | 2.218475 | 2.220241 | +0.001765 | 0 | 0 | 0 | 0 |

Aggregate:

| Method | n | Mean PF | Best PF | Worst PF | Mean candidate evals | Mean elapsed seconds |
|---|---:|---:|---:|---:|---:|---:|
| `fixed05_i2` | 3 | 2.227729 | 2.218329 | 2.246383 | 272.33 | 76.29 |
| `staged05_025_i2` | 3 | 2.228787 | 2.220241 | 2.245592 | 272.33 | 31.29 |

Paired delta:

```text
n = 3
mean_delta_pf = +0.001058
sd_delta_pf = 0.001615
95% CI = [-0.002955, +0.005071]
```

## Interpretation

There is no evidence here that switching to `0.25` after one `0.5` sweep beats
the fixed `0.5` baseline. The staged schedule wins one seed by a very small
margin and loses two seeds. The confidence interval crosses zero.

The scientifically relevant finding is that common MILP coverage repair is
necessary before comparing volume optimization: raw KMeans starts can leave
orders uncovered under exact MILP feasibility, causing PF to be dominated by
the uncovered-order penalty.

Next dev experiments should treat `fixed05_i2` as the active strong baseline
and test more targeted variants, such as `0.5:2,0.25:1`, adaptive per-iteration
step selection with the same candidate-evaluation budget, or the learned
surrogate policy under the same repair and exact-MILP final evaluation.

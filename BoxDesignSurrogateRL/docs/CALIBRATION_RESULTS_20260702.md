# MILP Strict-Convergence Results, 2026-07-02

These results are development calibration evidence only. They are not final
held-out claims.

## Protocol

- Code version: `55c4ca4`
- Machine: Windows/Gurobi over Tailscale
- Split: `or2023_seed20260701_limit500`
- Evaluation set: hash-materialized dev XML, `100` orders
- `K`: `10`
- Seeds: `0,1,2`
- Feasibility: Java/Gurobi MILP oracle
- Orientation label: `label_6ori`
- MILP time limit: `1s`
- Result root:

```text
BoxDesignSurrogateRL/results/milp_box_algorithms_dev100_strict_converged_canonical_20260702
```

Comparison label:

```text
dev100_strict_converged_fixed_vs_refine_canonical
```

Methods:

- `fixed05_converged`: `paper_fixed_step`, `--fixed-step 0.5`,
  `--iterations 300`, stopped only after a `noop` non-improving action.
- `fixed05_then025_converged`: starts from the converged `fixed05_converged`
  boxes for the same seed, verifies no remaining `0.5` improvement, then runs
  `0.25` until a `noop` non-improving action.

The initial KMeans boxes use shared MILP-verified coverage repair:

```text
--coverage-repair geometric_expand
--repair-margins 1.0,1.05,1.1,1.25,1.5,2.0
--repair-max-rounds 3
```

For the `0.25` refinement runs, the same coverage-repair manifest is retained
for pairing, but the repair is a no-op because the fixed `0.5` checkpoint is
already MILP-covered.

## Paired Results

PF is lower-is-better. Delta is
`fixed05_then025_converged - fixed05_converged`.

| Seed | Fixed PF | Refined PF | Delta PF | Fixed uncovered | Refined uncovered | Fixed unknown | Refined unknown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.760638 | 1.746790 | -0.013848 | 0 | 0 | 0 | 0 |
| 1 | 1.843040 | 1.832082 | -0.010957 | 0 | 0 | 0 | 0 |
| 2 | 1.793945 | 1.781189 | -0.012756 | 0 | 0 | 0 | 0 |

Aggregate:

| Method | n | Mean PF | Best PF | Worst PF | Mean assigned box volume | Mean elapsed seconds |
|---|---:|---:|---:|---:|---:|---:|
| `fixed05_converged` | 3 | 1.799208 | 1.760638 | 1.843040 | 63752.35 | 316.46 |
| `fixed05_then025_converged` | 3 | 1.786687 | 1.746790 | 1.832082 | 63308.71 | 15.83 |

Paired delta:

```text
n = 3
mean_delta_pf = -0.012520
sd_delta_pf = 0.001460
95% CI = [-0.016147, -0.008894]
mean_delta_assigned_box_volume = -443.64
```

## Interpretation

On this dev100 split, switching from a strictly converged fixed `0.5` solution
to a finer `0.25` local refinement improves all three seeds without increasing
uncovered orders or unknown MILP pairs. This is a different conclusion from the
2026-07-01 two-sweep experiment: once the fixed `0.5` baseline is run to actual
no-op convergence, the finer step still finds a small but consistent additional
PF reduction.

This should become the active dev-set baseline protocol for the MILP-backed
Problem B experiments:

1. Run the paper-style fixed `0.5` greedy search to no-op convergence.
2. Treat that as the baseline.
3. Evaluate any finer-step or learned policy as an explicit continuation from
   the same MILP-feasible checkpoint.

The next step is to repeat this protocol on a larger calibration split before
using it as the full OR2023 comparison baseline.

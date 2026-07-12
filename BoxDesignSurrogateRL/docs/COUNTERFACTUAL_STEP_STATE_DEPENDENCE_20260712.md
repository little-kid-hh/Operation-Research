# Counterfactual Step State Dependence (2026-07-12)

## Finding

The query-efficient first scale reverses between an early and a late state of
the same fixed-step training trajectory. This is evidence that scale choice has
a state-dependent long-horizon compute value; it is not yet evidence that a
learned policy generalizes.

All interventions use OR2023 training orders `[0, 100)`, `K=10`, the same
frozen candidate ranker, exact six-orientation MILP verification, and the same
four-scale continuation `{2, 1, 0.5, 0.25}` to exact no-improvement audit.

| Source state | First scale | Terminal PF | Coverage | Uncached boxes | Subprocess | Wall |
|---|---:|---:|---:|---:|---:|---:|
| iteration 1 | 0.25 | 1.7253077469 | 100% | **850** | **452.44 s** | **546.44 s** |
| iteration 1 | 2.0 | 1.7253077469 | 100% | 881 | 549.24 s | 695.55 s |
| iteration 200 | 0.25 | 1.7253077469 | 100% | 417 | 361.29 s | 405.70 s |
| iteration 200 | 2.0 | 1.7253077469 | 100% | **409** | **352.35 s** | **399.46 s** |

Within each source state, the two branches terminate with byte-identical box
files. Thus the comparison isolates exact-oracle work while holding terminal
quality and feasibility fixed.

The coarse action is harmful early: `+31` uncached boxes and `+96.80` subprocess
seconds. It is beneficial late: `-8` uncached boxes and `-8.94` subprocess
seconds. Wall time has the same direction but is treated as secondary because
it contains machine noise.

A subsequent exact repeat of the iteration-100 fine branch found more than 20%
timing variation with identical trajectory and uncached-query count. Therefore
the single-run time differences in this table are exploratory only. Uncached
queries are the deterministic policy-training label; subprocess and wall time
require repeated runs and distributional reporting at the final evaluation
gate.

## Consequence for the learning problem

A fixed rule that always chooses coarse or always chooses fine cannot realize
both observed query minima. This justifies a state-conditioned high-level
controller. The first defensible model should use the binary action set
`{coarse=2.0, fine=0.25}` and Monte Carlo branch-to-convergence query cost as
the return. Terminal PF, uncovered orders, and coverage remain hard quality
constraints rather than being traded away for speed.

More training states and independent order windows are required before fitting
the controller. The next data gate is to label intermediate states (iterations
50, 100, and 150) under both actions and confirm that the cost difference is
not an isolated two-state artifact.

## Provenance

- Branch runner code: `1352895`
- Early runs: `results/counterfactual_step_pilot_20260712/multiscale_ranker_greedy/`
- Late runs: `results/counterfactual_step_late_20260712/multiscale_ranker_greedy/`

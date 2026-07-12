# Counterfactual Step-Value Pilot (2026-07-12)

## Question

At one fixed training state, does the first move scale affect converged solution
quality or exact-oracle cost when all later decisions use the same multiscale
ranker and MILP verification?

The state is iteration 1 of OR2023 training orders `[0, 100)`, `K=10`. The
high-level action restricts only the first decision to one scale. The frozen
ranker scores that scale's 60 coordinate moves, MILP selects the improving
shortlist candidate, and all branches then continue with action scales
`{2, 1, 0.5, 0.25}` until exact no-improvement audit convergence.

## Results

| First scale | MILP-selected first action | One-step PF | Terminal PF | Uncached boxes | Subprocess | Wall |
|---:|---|---:|---:|---:|---:|---:|
| 0.25 | box 1 height `-0.25` | 2.0319514265 | 1.7253077469 | 850 | 452.44 s | 546.44 s |
| 0.5 | box 7 height `-0.5` | 2.0297342984 | 1.7253077469 | **848** | 460.62 s | 559.98 s |
| 1.0 | box 4 height `-1.0` | 2.0251134534 | 1.7253077469 | 850 | **452.60 s** | **545.09 s** |
| 2.0 | box 0 height `-2.0` | **2.0172988455** | 1.7253077469 | 881 | 549.24 s | 695.55 s |

Coverage is 100% in every branch. All four terminal box files have identical
SHA256 `24053C23F18BE0FE0DD02060D922EC7AC3BF470E96298C7DE8FC0EEA6EBE9B48`.

## Interpretation

The scale with the best immediate PF (`2.0`) is the worst long-horizon compute
choice. Relative to the fastest branch (`1.0`), it adds 31 uncached box queries,
96.65 seconds of subprocess time, and 150.47 seconds of wall time without any
terminal PF or coverage gain. This is a concrete case where one-step PF is not
the correct high-level action objective.

However, this single state is not evidence that a learned scale policy
generalizes or beats a fixed scale rule. The differences among `0.25`, `0.5`,
and `1.0` are also small enough that runtime noise can change their ordering;
uncached queries are the more stable primary cost label. The next gate must
repeat the four-way intervention at later training states and test whether the
query-minimizing scale is state dependent. Only then is learning a scale policy
justified.

## Provenance

- Code: `1352895`
- Runs: `results/counterfactual_step_pilot_20260712/multiscale_ranker_greedy/`
- Infrastructure tests: 92 passing

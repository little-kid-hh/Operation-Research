# MILP Runtime Repeatability Check (2026-07-12)

## Purpose

Determine whether a single subprocess or wall-clock measurement is stable
enough to label high-level step actions.

The iteration-100 fine (`0.25`) counterfactual branch was rerun from an empty
in-memory oracle cache with the same code, state, orders, ranker, MILP settings,
and continuation configuration.

| Metric | Run 1 | Run 2 | Difference |
|---|---:|---:|---:|
| Terminal PF | 1.7253077469 | 1.7253077469 | 0 |
| Coverage | 100% | 100% | 0 pp |
| MILP validations | 2860 | 2860 | 0 |
| Uncached boxes | 651 | 651 | 0 |
| Subprocess time | 484.79 s | 381.50 s | -103.29 s (-21.3%) |
| Wall time | 583.13 s | 442.60 s | -140.53 s (-24.1%) |

The terminal boxes and search trajectory are identical. The structural work
counts are exactly reproducible, while timing changes by more than 20%.

## Protocol decision

- Train the high-level controller on uncached box queries, conditional on equal
  terminal PF, full coverage, and zero uncovered orders.
- Keep subprocess and wall-clock as required evaluation outcomes, but do not
  use one run as a target or definitive comparison.
- For final development and untouched-test timing claims, repeat each paired
  configuration at least three times on the same machine and report median,
  range (or IQR when enough repeats exist), and all structural query counts.
- Treat a timing gain without a structural query reduction as inconclusive
  unless replicated.

## Provenance

- First run: `counterfactual_step_mid_20260712/multiscale_ranker_greedy/run_20260712_172953_082589`
- Repeat: `counterfactual_step_mid_20260712/multiscale_ranker_greedy/run_20260712_174819_058354`
- Code: `1352895`

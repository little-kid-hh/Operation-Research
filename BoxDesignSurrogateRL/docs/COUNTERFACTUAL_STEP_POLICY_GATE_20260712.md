# Counterfactual Binary Step-Policy Gate (2026-07-12)

## Complete training-state labels

For five states from one OR2023 training trajectory, each action forces only
the first scale (`fine=0.25` or `coarse=2.0`). Both branches then use the same
unrestricted multiscale ranker plus exact MILP until convergence.

| Source iteration | Fine queries | Coarse queries | Better action | Query saving |
|---:|---:|---:|---|---:|
| 1 | **850** | 881 | fine | 31 |
| 50 | 739 | **725** | coarse | 14 |
| 100 | **651** | 653 | fine | 2 |
| 150 | 536 | **534** | coarse | 2 |
| 200 | 417 | **409** | coarse | 8 |
| Total | **3193** | 3202 | fine fixed rule | 9 |

Every pair reaches terminal PF `1.7253077468667117`, 100% coverage, zero
uncovered orders, and pairwise identical terminal boxes.

An oracle that selects the lower-query action independently at every sampled
state uses 3160 queries. Its maximum observed advantage is therefore only:

- 33 queries (1.03%) versus always fine;
- 42 queries (1.31%) versus always coarse.

## Gate decision

**Fail for model training.** State dependence is real, but the available
headroom from learning this one high-level first-step decision is too small.
A learned classifier would also be fitted on only five correlated states from
one trajectory, so any apparent accuracy would be statistically weak and its
inference/engineering complexity would exceed the proven query benefit.

This negative result prevents an unjustified RL claim. The strong current
result remains the supervised multiscale ranker, whose three-window development
comparison reduced uncached MILP boxes by 37.7% and wall time by 27.2% while
improving mean PF by 4.82% against exact fixed-fine staged search. Future RL work
must control a decision with materially larger oracle headroom, demonstrated by
an oracle policy before model fitting.

## Protocol lesson

Before training any controller:

1. Generate exact counterfactual returns for its actions.
2. Compute the oracle-policy upper bound against the strongest fixed rule.
3. Continue only if that bound is large enough to survive prediction error and
   evaluation variance.

## Provenance

- Branch code: `1352895`
- Early/late runs: `counterfactual_step_pilot_20260712`,
  `counterfactual_step_late_20260712`
- Middle runs: `counterfactual_step_mid_20260712`,
  `counterfactual_step_train_matrix_20260712`

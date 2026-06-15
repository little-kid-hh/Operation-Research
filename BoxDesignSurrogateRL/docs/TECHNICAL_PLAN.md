# Technical Plan

## Objective

Given a set of orders and a target assortment size `K`, find `K` carton sizes
that minimize total expected packaging cost while covering all or nearly all
orders.

A default cost can be:

```text
cost(order, box) = volume(box)
```

or a richer carton cost:

```text
cost(order, box) = alpha * volume(box) + beta * surface_area(box)
```

The evaluator should also report:

- coverage rate;
- mean assigned feasibility probability;
- number of low-margin assignments;
- exact verification pass rate for sampled/elite candidates.

## Surrogate-Aware Evaluation

For a candidate box set `B`, compute:

```text
p_ij = P(order i fits in box j)
```

Then assign each order:

```text
j* = argmin_j adjusted_cost(i, j)
subject to p_ij >= tau
```

where

```text
adjusted_cost(i, j) = carton_cost(j) + lambda_risk * risk(p_ij)
risk(p) = max(0, tau_high - p)^2
```

Recommended starting parameters:

- `tau = 0.95` for feasibility acceptance when using an uncalibrated model.
- `tau_high = 0.99` for risk shaping.
- `lambda_risk` large enough that a slightly larger but safer box is preferred.
- infeasible/uncovered orders get a large fixed penalty.

These parameters should be tuned from validation false-positive behavior, not
from AUC alone. For this problem, false positives are more damaging than false
negatives because they create invalid final assortments.

## Handling Model Error

Use three layers:

1. Conservative thresholding.
   Choose `tau` by precision-at-threshold or false-positive-rate constraints on
   held-out MILP labels.

2. Risk-aware optimization.
   The RL/search reward should include uncertainty penalties, not only carton
   volume.

3. Exact verification.
   Run MILP on:
   - final assigned order-box pairs;
   - assignments with `p` near the threshold;
   - elite candidates from different search seeds;
   - candidates that improve the incumbent by a large amount.

If exact verification finds false positives, either repair by expanding the
responsible box dimensions or reassign the order to a larger verified box.

## Search Strategy

Start with deterministic search before training RL:

1. k-means seed assortment.
2. Coordinate descent with adaptive step sizes:
   `1.0 -> 0.5 -> 0.1 -> 0.05 -> 0.01`.
3. Beam search over the best dimension moves.
4. Exact verification and repair.

Then add RL:

- state: current sorted `K x 3` box dimensions plus aggregate evaluator stats;
- action: choose box, dimension, direction, and step size;
- reward: improvement in verified-or-risk-adjusted objective;
- termination: no improvement or budget exhausted.

Tree search can be added as a policy improvement layer:

- use the learned policy to rank actions;
- expand top actions in a beam/MCTS-style tree;
- score nodes with the surrogate evaluator;
- verify elite leaves with MILP.

## Why 0.01 Resolution Is Feasible

Naive search over `0.01` increments is impossible. The project should use
coarse-to-fine search:

- optimize roughly at coarse steps;
- only refine dimensions around high-quality candidates;
- cache order-box feature matrices and probabilities;
- exploit monotonicity: increasing a box dimension should not reduce true
  feasibility, so suspect non-monotone surrogate predictions can be smoothed or
  overridden conservatively.

## SOTA Claim Requirements

To claim SOTA box sets, we need:

- comparison to Kandula-style k-means + RL/tree search where possible;
- comparison to existing OR 2023 package sets / candidate boxes;
- ablations:
  - no surrogate, fixed-fit simplification;
  - surrogate without risk penalty;
  - surrogate with risk penalty;
  - surrogate + tree search;
  - surrogate + exact repair;
- final exact verification of all reported assigned pairs or a statistically
  defensible verification protocol.


# Difference from Kandula et al.

## What Kandula Does

Kandula et al. solve e-commerce box-size design:

- input: item/SKU dimensions and demand;
- output: `K` standard carton sizes;
- objective: reduce packaging factor / wasted carton volume;
- method: k-means initialization, RL box-sizing game, tree search policy
  improvement.

Their framework is powerful, but the feasibility evaluation is simplified by
fixing the effective assignment/packing abstraction early.

## Our Change

We keep the optimization scaffold but replace the simplified evaluator with a
learned loadability evaluator.

For every candidate box set:

1. Generate features for every order-box pair.
2. Predict feasibility probabilities with the learned model.
3. Assign each order to the cheapest box whose feasibility probability clears a
   conservative threshold.
4. Penalize uncertainty and coverage failures.
5. Verify final and boundary candidates with exact MILP.

This lets the optimizer change box dimensions freely after initialization
instead of being locked to the initial packing/assignment simplification.

## Why This Is a Paperable Angle

The technical claim becomes:

> A learned feasibility surrogate can turn box-size design into a fast,
> confidence-aware sequential optimization problem, enabling much finer
> dimension moves and stronger box assortments than candidate-box or fixed-fit
> simplifications.

The risk is model error. The paper must therefore make uncertainty handling and
exact verification part of the method, not an afterthought.


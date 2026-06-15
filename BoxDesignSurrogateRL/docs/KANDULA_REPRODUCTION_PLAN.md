# Kandula-Style Reproduction Plan

This is a framework reproduction, not an exact numerical reproduction of
Kandula et al. The original paper uses proprietary e-commerce SKU/order demand
data and implementation details that are not available in this workspace.

## Available Local Data

We use the OR 2023 / Fontaine-Minner BSP data:

- `assets/or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml`
- `assets/or2023_bsp_data/order_geometry_map.csv`
- `assets/or2023_bsp_data/packages.txt`
- `assets/milp_labels/or2023_bsp_unique_package_labels.csv`

The local data has 12,864 unique order geometries and 90 existing candidate
packages.

## Reproduction Scope

The Kandula paper has three main algorithmic stages:

1. k-means initialization;
2. sequential box-dimension improvement;
3. tree search / policy improvement.

The first reproduction target is a deterministic, inspectable version:

1. Generate `K` initial boxes by k-means clustering order geometry vectors.
2. Evaluate a box set with a simplified geometric feasibility rule.
3. Improve boxes with coordinate descent / beam search.

This gives us a working baseline that follows the shape of Kandula's framework.
The paper-aligned reproduction now lives beside the earlier deterministic
baseline:

- `box_design_surrogate/kandula_paper.py`: Kandula Section 4.2 box-sizing game.
- `scripts/train_kandula_paper_policy.py`: Stage 2 policy-network training with
  a local PPO actor-critic implementation.
- `scripts/run_kandula_paper_paas.py`: Stage 3 policy-assisted active search
  (PAAS) using weighted packaging factor rollouts.

The exact online-companion hyperparameters and proprietary SKU data are not in
this workspace, so numerical reproduction is still out of scope. The current
target is algorithmic-framework reproduction on local OR2023 BSP-derived data.

## Simplified Feasibility Rule

Kandula's original formulation is closer to SKU-to-box sizing. Our data has
orders that may contain multiple items. For a first framework reproduction, an
order is treated through aggregate geometric requirements:

- each item's sorted dimensions must be bounded by the box dimensions;
- total item volume must not exceed box volume.

This is intentionally weaker than exact 3D packing. It is the baseline
simplification we later improve with the learned loadability surrogate.

## Objective

For a set of boxes `B`, each order is assigned to the smallest-volume feasible
box. The objective is:

```text
mean_assigned_box_volume / mean_order_volume
```

with a large penalty for uncovered orders. This is a local analog of the
packaging-factor objective in Kandula et al.

## Reproduction Milestones

1. `kmeans`: initial box set only.
2. `local`: k-means plus coordinate descent at a paper-like step such as `0.5`.
3. `beam`: local improvement with a tree-search-like beam.
4. `paper_policy`: paper-aligned box-sizing game with `K x 3` state, `6K + 1`
   actions, and PF-difference reward.
5. `paas`: policy-assisted active search using the learned policy for rollouts.

The project should report all stages separately, because Kandula's paper also
emphasizes the incremental value of initialization, learned improvement, and
tree search.

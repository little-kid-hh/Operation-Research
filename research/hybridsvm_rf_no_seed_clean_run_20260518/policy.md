# Feature Search Policy

Objective:
Improve the linear SVM baseline by adding a small number of interpretable,
per-dispatch numeric features.

Hard constraints:
- Do not change the train/test split.
- Do not change the model family: keep the stage-1 model as linear SVM.
- Do not use labels or any target-derived statistics inside feature generation.
- Do not read files or call external services in candidate feature code.
- Return one row per dispatch and keep feature names ASCII.

Preferred feature families:
- item-level quantiles / tails rather than only mean/std
- thresholded local bottleneck counts, not only global averages
- slack / spare-capacity interactions with large-piece pressure
- repeated-type structure and concentration
- extreme-piece bottlenecks against vehicle dimensions
- face-area / edge-pressure style packing stress signals
- heterogeneity / multimodality / long-tail measures
- features that are easy to explain to a human reviewer

Avoid:
- duplicating obvious existing aggregates unless the new version captures a
  different shape signal
- opaque embeddings
- huge feature sets; prefer a compact hypothesis with clear semantics

Tree-inspired search bias:
- assume trees are winning partly because they exploit local thresholds and
  interactions on features like `spare_capacity`, `wl_to_vehicle_wl_max`,
  `wl_to_vehicle_wl_total`, `sku_average_volume`, and dimension variances
- try to convert those nonlinear effects into explicit numeric features that
  a linear SVM can use

## Iteration 1
- Focus on the TPR@FPR=1% gap (0.17 below RF): the SVM misses hard positives because it cannot express nonlinear interactions and threshold effects on the top features.
- Priority 1: explicit interactions between `spare_capacity` and the next-highest RF features (`wl_to_vehicle_wl_total`, `sku_counts`) to linearize the conditional splits trees make.
- Priority 2: a threshold-based item-level signal (`big_piece_share`) that trees create naturally via binary splits but SVM cannot.
- Priority 3: cross-dimension tightness product (`hH × lL`) and a quadratic spare-capacity term to capture curvature near the decision boundary.

## Iteration 5
- Shift from spare_capacity interactions (already well-covered in active bank) to **dispersion and constraint signals** that trees exploit via sequential splits on wl-family spread and item-level orientation restrictions.
- Add **wl_cv** (H3) to capture whether weight-length utilization is dominated by a single SKU or uniformly spread — a pattern trees pick up via depth-2 splits on wl_std conditional on wl_avg.
- Introduce **fragile_share** from items_df as a genuinely new direction: fragile items cannot be reoriented, tightening the effective packing search space in a way not captured by any existing aggregate.
- Extend the spare_capacity polynomial family with **spare_cap_cubed** (H1): the accepted quadratic term helped, but the TPR@1% gap suggests the feasibility boundary is steeper than quadratic near zero slack.

## Iteration 9
- Target the stubborn TPR@FPR=1% gap by adding features that linearize the conditional structure trees exploit via depth-2 splits on wl-family dispersion conditional on spare capacity.
- Introduce a **spare_capacity × wl_cv** interaction: trees split first on spare_capacity then on wl spread; this product lets the linear SVM approximate that conditional regime.
- Add a **thresholded vertical bottleneck** signal (near_h_limit_share) from items_df: fraction of items whose largest sorted dimension exceeds 70% of vehicle height. This is a count-based signal distinct from the smooth h_to_H_ratio_avg already in the base.
- Add **wl_total_sq**: a quadratic term on the #2 RF-importance feature (wl_to_vehicle_wl_total), mirroring how spare_cap_sq already captured curvature on the #1 feature. Feasibility drops sharply as wl_total approaches 1.0.

## Iteration 10
- Pivot from pure spatial-constraint features to **weight-capacity pressure**, an entirely orthogonal constraint axis not yet represented in the active bank.
- Add a **per-item multi-dimensional tightness** threshold count: trees naturally isolate cases where a single item stresses 2+ vehicle axes simultaneously, which no existing feature captures.
- Introduce a **dim_l upper-tail ratio** (p90/p50) to capture long-item heterogeneity within a dispatch — distinct from the smooth `sku_std_length` and the binary `long_item_share` already in the bank.

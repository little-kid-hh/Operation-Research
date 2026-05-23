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

## Iteration 8
- Add `spare_cap_x_sku_counts` to capture the nonlinear interaction between slack and item count (XGB guidance §3.2, not yet in active bank despite spare_cap interactions with conc/wl_total/wl_max).
- Add `n_items_near_vL` as a threshold count of items whose longest sorted dimension exceeds 50% of vehicle length — multiple long items create axis-alignment conflicts that a single max-ratio cannot express.
- Add `n_tall_items` as a threshold count of items whose original height exceeds 50% of vehicle height — tall items limit vertical stacking; height pressure has high XGB gain (h_to_H_ratio_max rank 8) but no threshold-count representation in the bank.

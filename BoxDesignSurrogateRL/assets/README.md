# Asset Index

This folder contains symbolic links to existing workspace assets. The source
files remain in their original locations.

## Main Design Data

`or2023_bsp_data/`

- Use this for box-size design experiments.
- Important files:
  - `xml_unique/or2023_bsp_unique_orders.xml`
  - `order_geometry_map.csv`
  - `packages.txt`
- Existing labels:
  - `milp_labels/or2023_bsp_unique_package_labels.csv`

The BSP labels are geometry-deduplicated. Join through `geom_id` if expanded
order-level reporting is needed.

## Loadability Training / Validation Data

`or2023_bpp_data/`

- Use this for package feasibility model training and benchmark evaluation.
- Do not mix BPP rows into BSP design metrics unless the experiment is
  explicitly a transfer/generalization test.

## Feature Matrices

`processed_features/`

- `or2023_bpp_labeled_base40_package.csv`: base 40-feature matrix.
- `or2023_bpp_labeled_fe111_package.csv`: engineered 111-feature matrix.
- `permin_labeled_*`: legacy or differently scoped artifacts; verify before
  using in reported experiments.

## Models

`loadability_model_pack/`

Reusable model pack from `research/loadability_model_pack_20260526`.

- Linear SVM: interpretable, useful for conservative screening.
- Ensemble: stronger probability estimates, useful as default surrogate.
- TabTreeFormer/RF: high-capacity candidate evaluator.

The current pack has no standalone XGBoost artifact. XGBoost appears in the
workspace as training/search code and experiment summaries.


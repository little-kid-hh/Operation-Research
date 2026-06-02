# OR 2023 Data Usage Guide

This repository now keeps two Fontaine-Minner OR 2023 data tracks separate:

```text
Pirmin Fontaine, Stefan Minner (2023)
A Branch-and-Repair Method for Three-Dimensional Bin Selection and Packing in E-Commerce
Operations Research 71(1):273-288
DOI: 10.1287/opre.2022.2369
```

## Which Data To Use

| Data | Repository path | Source in supplement | Use for | Do not use for |
| --- | --- | --- | --- | --- |
| OR 2023 BPP benchmark | `or2023_bpp_data/` | `codes/BPP/dataset50/*.pkl` | ML loadability training/evaluation | Box-size design benchmark comparison |
| OR 2023 BSP/design XML | `or2023_bsp_data/` | `datasets_xml/BSP_*.xml` | Box-size design/order benchmark analysis; BSP-specific package-feasibility lookup | Default ML benchmark unless explicitly studying transfer/generalization |
| Legacy/stochastic BSP-derived files | `S3DBSP-main/performanceTest/`, `permin_*` outputs | Older/local legacy data | Auditing old runs only | Reporting new OR 2023 BPP/BSP results |
| Local source archive/PDF | `opre.2022.2369.sm1.zip`, `Permin+Fontaine-*.pdf` | Downloaded paper/supplement | Local reference only | Git-tracked reproducible output |

## BPP Benchmark Track

Use this track when the question is:

```text
Can a BPP instance/order be loaded into a candidate package?
```

Committed outputs:

- `or2023_bpp_data/xml/`: 450 converted BPP instances, 50 each for item counts 2 through 10.
- `or2023_bpp_data/packages.txt`: 90 candidate packages.
- `permin_dataset_processing/milp_labels/or2023_bpp_package_labels.csv`: 40,500 package-feasibility labels.
- `permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv`: Base40 ML feature table.
- `permin_dataset_processing/processed_features/or2023_bpp_labeled_fe111_package.csv`: FE111 engineered ML feature table.
- `permin_dataset_processing/experiments/cv_20260602_215731_label_6ori/`: Base40 CV for `label_6ori`.
- `permin_dataset_processing/experiments/cv_20260602_215818_label_2ori/`: Base40 CV for `label_2ori`.
- `permin_dataset_processing/experiments/cv_20260602_220214_label_6ori/`: FE111 SVM CV for `label_6ori`.
- `permin_dataset_processing/experiments/cv_20260602_220225_label_2ori/`: FE111 SVM CV for `label_2ori`.

Report:

- `permin_dataset_processing/reports/or2023_bpp_labeling_ml_report.md`

Evaluation rule:

- Use instance-level grouped splits for ML evaluation so all package rows from one BPP instance stay in the same fold.

## BSP/Design XML Track

Use this track when the question is:

```text
How do OR 2023 BSP/design benchmark orders interact with candidate package feasibility?
```

Committed outputs:

- `or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml`: 12,864 unique BSP order geometries.
- `or2023_bsp_data/order_geometry_map.csv`: mapping from 150,000 expanded BSP XML orders to unique `geom_id`.
- `or2023_bsp_data/packages.txt`: 90 candidate packages.
- `permin_dataset_processing/milp_labels/or2023_bsp_unique_package_labels.csv`: 1,157,760 unique geometry-package labels.

The fully expanded BSP label table has:

```text
150,000 orders * 90 packages = 13,500,000 rows
```

It is not committed directly. Reconstruct it by joining the mapping to unique labels:

```python
import pandas as pd

mapping = pd.read_csv("or2023_bsp_data/order_geometry_map.csv")
labels = pd.read_csv("permin_dataset_processing/milp_labels/or2023_bsp_unique_package_labels.csv")
labels = labels.rename(columns={"order_id": "geom_id", "instance_name": "label_source_xml"})
expanded = mapping.merge(labels, on="geom_id", how="left")
```

The `mapping.instance_name` and `mapping.order_id` columns are the original BSP
XML file and original order id. The renamed `label_source_xml` column is only
the deduplicated XML used for labeling.

Report:

- `permin_dataset_processing/reports/or2023_bsp_labeling_report.md`

## Label Meanings

Both BPP and BSP label files use the same binary label convention:

| Column | Meaning |
| --- | --- |
| `label_2ori` | Feasible under the two-orientation model: item height remains on package height; item length/width may swap horizontally. |
| `label_6ori` | Feasible under the six-orientation model: all axis permutations are allowed. |
| `time_2ori_ms`, `time_6ori_ms` | Recorded solve time in milliseconds. A zero value can mean the safe infeasibility pre-screen proved the pair impossible before calling Gurobi. |

The safe infeasibility pre-screen uses only necessary conditions:

- total item volume must not exceed package volume
- for `label_2ori`, every item must fit either as `(p, q, r)` or with horizontal
  swap `(q, p, r)`; item height `r` remains on package height
- for `label_6ori`, every item must fit under some axis permutation, checked by
  comparing sorted item dimensions against sorted package dimensions

For BSP geometry deduplication, item order inside an order is ignored because
the MILP feasibility model is symmetric in item indices. Each item's `(p, q, r)`
axis naming is preserved, so two-orientation height semantics are not collapsed
into six-orientation semantics.

## Reporting Guidance

Use this wording when summarizing the repository state:

```text
The OR 2023 BPP benchmark is used for ML loadability training and evaluation.
The OR 2023 BSP/design XML data is kept as a separate box-design/order benchmark track.
BSP labels are stored in geometry-deduplicated form and can be expanded to all order-package rows through order_geometry_map.csv.
Legacy S3DBSP/permin_* outputs are audit artifacts and are not part of the current OR 2023 BPP/BSP result set.
```

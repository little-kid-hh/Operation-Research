# OR 2023 BPP Labeling and ML Evaluation Report

Date: 2026-06-02

## Scope

This report covers the 3D-BPP benchmark data from the Fontaine and Minner OR 2023 supplement:

- Source supplement archive: `opre.2022.2369.sm1.zip` (not committed)
- BPP source files: `codes/BPP/dataset50/H3DBPP_<item_count>_<seed>.pkl`
- Package source file: `codes/BPP/data/packages.txt`
- Converted repository data: `or2023_bpp_data/`

The BPP benchmark contains 450 instances:

| Item count | Instances |
| ---: | ---: |
| 2 | 50 |
| 3 | 50 |
| 4 | 50 |
| 5 | 50 |
| 6 | 50 |
| 7 | 50 |
| 8 | 50 |
| 9 | 50 |
| 10 | 50 |

There are 90 candidate packages, so the full labeling task contains:

```text
450 BPP instances * 90 packages = 40,500 instance-package feasibility labels
```

## BPP vs BSP Data

The OR 2023 supplement contains two different data families that should not be mixed:

| Data family | Supplement location | Purpose in our project | Labeling need |
| --- | --- | --- | --- |
| BPP benchmark | `codes/BPP/dataset50/*.pkl` | Train/evaluate ML loadability classifiers for fixed order-package feasibility | Yes, label each instance-package pair |
| BSP/design benchmark | `datasets_xml/BSP_*.xml` | Box size design / bin selection and packing comparison against paper experiments | Usually no ML feasibility labeling needed unless a separate classifier experiment is defined |

The old `S3DBSP-main/performanceTest` XML data should be treated as legacy audit data. Some `O6` base XMLs match a subset of the OR 2023 BSP XMLs, but they are not the BPP benchmark used in this report.

## Labeling Configuration

Labeling command family:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File permin_dataset_processing\run_permin_label_batches.ps1 `
  -BatchSize 5000 `
  -OutputPath permin_dataset_processing\milp_labels\or2023_bpp_package_labels.csv `
  -XmlDir or2023_bpp_data\xml `
  -PackagesPath or2023_bpp_data\packages.txt `
  -TotalTasks 40500 `
  -TimeLimit2Ori 300 `
  -TimeLimit6Ori 300
```

The Java MILP loaders keep the original Fontaine-Minner normalized non-overlap formulation. Two-orientation and six-orientation labels are computed by separate loader classes:

- `MILP_Loading_2orientations.java`
- `MILP_Loading_6orientations.java`

## Labeling Result

Output:

- `permin_dataset_processing/milp_labels/or2023_bpp_package_labels.csv`

Summary:

| Target | Infeasible | Feasible | Timeout |
| --- | ---: | ---: | ---: |
| `label_2ori` | 31,755 | 8,745 | 0 |
| `label_6ori` | 30,930 | 9,570 | 0 |

The full labeling run took about 17 minutes on the local Gurobi setup.

## Feature Tables

Generated feature datasets:

| Dataset | Path | Rows | Columns |
| --- | --- | ---: | ---: |
| Base40 | `permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv` | 40,500 | 50 |
| FE111 | `permin_dataset_processing/processed_features/or2023_bpp_labeled_fe111_package.csv` | 40,500 | 121 |

The FE111 table contains 111 model features: 40 base features plus 71 active-bank engineered features, followed by labels and identifiers.

## ML Evaluation Protocol

Evaluation uses leakage-safe grouped splitting:

- Splitter: `StratifiedGroupKFold`
- Group level: `instance_name`
- Effect: all 90 package rows from the same BPP instance stay in the same fold

This prevents the same order geometry from appearing in both training and testing via different package rows.

## Base40 Results

### `label_6ori`

| Model | Accuracy | AUC | TPR@FPR1% | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| hist_gbdt | 0.9831 | 0.9987 | 0.9600 | 0.9608 | 0.9678 |
| xgboost | 0.9831 | 0.9987 | 0.9573 | 0.9587 | 0.9700 |
| random_forest | 0.9808 | 0.9982 | 0.9481 | 0.9494 | 0.9703 |
| linear_svm | 0.9456 | 0.9909 | 0.7802 | 0.8295 | 0.9686 |
| logreg | 0.9396 | 0.9896 | 0.7457 | 0.8109 | 0.9708 |

Experiment directory:

- `permin_dataset_processing/experiments/cv_20260602_215731_label_6ori/`

### `label_2ori`

| Model | Accuracy | AUC | TPR@FPR1% | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.9845 | 0.9989 | 0.9636 | 0.9601 | 0.9684 |
| hist_gbdt | 0.9844 | 0.9988 | 0.9626 | 0.9609 | 0.9671 |
| random_forest | 0.9825 | 0.9983 | 0.9491 | 0.9534 | 0.9660 |
| linear_svm | 0.9419 | 0.9904 | 0.7779 | 0.8043 | 0.9664 |
| logreg | 0.9357 | 0.9887 | 0.7453 | 0.7856 | 0.9662 |

Experiment directory:

- `permin_dataset_processing/experiments/cv_20260602_215818_label_2ori/`

## FE111 Linear SVM Results

| Target | Accuracy | AUC | TPR@FPR1% | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_6ori` | 0.9581 | 0.9953 | 0.8685 | 0.8583 | 0.9849 |
| `label_2ori` | 0.9540 | 0.9947 | 0.8466 | 0.8329 | 0.9840 |

Experiment directories:

- `permin_dataset_processing/experiments/cv_20260602_220214_label_6ori/`
- `permin_dataset_processing/experiments/cv_20260602_220225_label_2ori/`

## Takeaways

1. The OR 2023 BPP benchmark is now labeled and evaluated separately from BSP/design data.
2. Base40 tree models perform best overall, with AUC around 0.9987 to 0.9989.
3. Feature engineering substantially improves linear SVM over Base40 SVM, especially at low false-positive rates.
4. FE111 SVM is still weaker than the best non-linear tree models on this BPP benchmark.

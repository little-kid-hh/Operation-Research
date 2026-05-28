# Permin Package-Aware Loadability Dataset

This directory contains the scripts and experiment summaries for constructing
package-aware loadability labels from the Permin/S3DBSP performance instances.
Each row is an `(instance_name, order_id, package_id)` tuple and the labels
answer whether the order can be packed into that candidate package.

## Dataset Status

- Base XML instances only: `BSP_<n>_O6_<seed>.xml`
- Candidate package list: `S3DBSP-main/performanceTest/packages.txt`
- Exported labeling tasks: 675,000 package-order tasks
- Trusted labels: `milp_labels/ground_truth_package_labels_csv.zip`
- Label columns: `label_2ori`, `label_6ori`
- Label distribution:
  - `label_2ori`: 233,605 feasible, 441,375 infeasible, 20 solver failures
  - `label_6ori`: 238,083 feasible, 436,898 infeasible, 19 solver failures

The full labeled feature matrices are generated artifacts and are not tracked
because they are large:

- `processed_features/permin_labeled_base40_package.csv`
- `processed_features/permin_labeled_fe111_package.csv`

The uncompressed MILP label CSV is generated as
`milp_labels/ground_truth_package_labels.csv`; Git tracks the compressed copy
to keep pushes lightweight.

## Reproduce Labels

Generate the package-order task table:

```powershell
python permin_dataset_processing/export_labeling_tasks.py
```

Run the Java/Gurobi package labeler:

```powershell
javac -encoding UTF-8 -cp "C:\gurobi1300\win64\lib\gurobi.jar;MILP_3DBPP\src\main\java" MILP_3DBPP\src\main\java\org\example\GeneratePerminPackageLabels.java
java -cp "C:\gurobi1300\win64\lib\gurobi.jar;MILP_3DBPP\src\main\java" org.example.GeneratePerminPackageLabels
```

For batch labeling, use:

```powershell
powershell -ExecutionPolicy Bypass -File permin_dataset_processing/run_permin_label_batches.ps1
```

## Build Feature Matrices

Build base 40-feature package rows:

```powershell
python permin_dataset_processing/build_labeled_base40_dataset.py
```

Build the engineered 111-feature rows:

```powershell
python permin_dataset_processing/build_labeled_fe111_dataset.py
```

## Cross-Validation Results

Five-fold `StratifiedGroupKFold` results are stored in:

```text
experiments/all_model_summary_20260527.csv
```

Top results by target:

| Target | Feature set | Model | Accuracy | AUC | TPR at 1% FPR |
| --- | --- | --- | ---: | ---: | ---: |
| `label_2ori` | `base40` | random forest | 0.997964 | 0.999981 | 0.999898 |
| `label_6ori` | `base40` | random forest | 0.997326 | 0.999966 | 0.999675 |

Run cross-validation with:

```powershell
powershell -ExecutionPolicy Bypass -File permin_dataset_processing/run_cross_validation.ps1
```

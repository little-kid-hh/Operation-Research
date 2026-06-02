# OR 2023 3D-BPP Loadability Dataset

This directory is now scoped to the 3D-BPP data and model setting from:

```text
Pirmin Fontaine, Stefan Minner (2023)
A Branch-and-Repair Method for Three-Dimensional Bin Selection and Packing in E-Commerce
Operations Research 71(1):273-288
DOI: 10.1287/opre.2022.2369
```

The ML label target remains a BPP feasibility question:

```text
(instance_name, order_id, package_id) -> can this order be packed into this package?
```

## Data Boundary

Use the OR 2023 BPP e-companion data for new labeling and ML runs.

Expected local staging path:

```text
or2023_bpp_data/
  xml/
  packages.txt
```

The existing `S3DBSP-main/performanceTest` folder is treated as legacy
BSP/stochastic-derived data. Scripts in this directory now reject paths under
`S3DBSP-main` by default so new experiments do not silently index the wrong
paper/data package.

Use the legacy escape hatch only for audits of old results:

```powershell
--allow-bsp-derived-data
```

or, for PowerShell batch labeling:

```powershell
-AllowBspDerivedData
```

## Old Results

Previously generated files with names like `permin_*`, `BSP_*_O6_*`, or
`S3DBSP-main/performanceTest` are BSP-derived/legacy artifacts. They are useful
for debugging code paths, but should not be reported as OR 2023 BPP benchmark
results unless the data source is explicitly revalidated.

Examples of legacy artifacts:

```text
processed_features/permin_labeled_base40_package.csv
processed_features/permin_labeled_fe111_package.csv
milp_labels/ground_truth_package_labels.csv
experiments/cv_2026*_label_*
```

## Reproduce OR 2023 BPP Labels

Generate the package-order task table after placing the OR 2023 BPP data under
`or2023_bpp_data`:

```powershell
python permin_dataset_processing/export_labeling_tasks.py
```

If the e-companion uses a narrower XML naming scheme, pass it explicitly:

```powershell
python permin_dataset_processing/export_labeling_tasks.py --xml-name-regex ".*\.xml$"
```

Compile and run the Java/Gurobi package labeler:

```powershell
javac -encoding UTF-8 -cp "C:\gurobi1300\win64\lib\gurobi.jar;MILP_3DBPP\src\main\java" MILP_3DBPP\src\main\java\org\example\GeneratePerminPackageLabels.java
java -cp "C:\gurobi1300\win64\lib\gurobi.jar;MILP_3DBPP\src\main\java" org.example.GeneratePerminPackageLabels or2023_bpp_data\xml or2023_bpp_data\packages.txt permin_dataset_processing\milp_labels\or2023_bpp_package_labels.csv
```

For batch labeling, compute `TotalTasks = selected_orders * candidate_packages`
for the OR 2023 BPP input and run:

```powershell
powershell -ExecutionPolicy Bypass -File permin_dataset_processing/run_permin_label_batches.ps1 -XmlDir or2023_bpp_data\xml -PackagesPath or2023_bpp_data\packages.txt -TotalTasks <N>
```

The default MILP time limit is 300 seconds for both 2-orientation and
6-orientation models, matching the OR 2023 BPP computational setting. Shorter
time limits are acceptable only for smoke tests; timeout rows (`-1`) must stay
unlabeled/unknown and must not be treated as infeasible examples.

## Build Feature Matrices

Build base 40-feature package rows:

```powershell
python permin_dataset_processing/build_labeled_base40_dataset.py --labels-path permin_dataset_processing/milp_labels/or2023_bpp_package_labels.csv --output-path permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv
```

Build engineered 111-feature rows:

```powershell
python permin_dataset_processing/build_labeled_fe111_dataset.py --base-path permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv --output-path permin_dataset_processing/processed_features/or2023_bpp_labeled_fe111_package.csv
```

## Cross-Validation

The default evaluation input is now:

```text
processed_features/or2023_bpp_labeled_base40_package.csv
```

Run strict instance-holdout cross-validation with:

```powershell
powershell -ExecutionPolicy Bypass -File permin_dataset_processing/run_cross_validation.ps1
```

Use `-GroupLevel order` only for the less strict setting that keeps all
candidate packages for one order together but allows different orders from the
same XML instance to appear in train and test folds.

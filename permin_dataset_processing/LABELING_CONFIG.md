# Permin MILP Labeling Configuration

This file records the formal labeling contract for the Permin/S3DBSP
loadability dataset.

## Label Unit

One label row is one tuple:

```text
(instance_name, order_id, package_id)
```

The label answers whether all items in that order can be packed into that
single candidate package.

## Raw Inputs

- Orders: `S3DBSP-main/performanceTest/*.xml`
- Candidate packages: `S3DBSP-main/performanceTest/packages.txt`
- Package format: `package_id length width height`
- Order item format: XML `order -> item -> p, q, r`

By default, labeling should use only base performance instances matching:

```text
BSP_<n>_O6_<seed>.xml
```

Scenario/demand variants such as `BSP_100_O6_0_2_5.xml` should not be mixed
into the base loadability dataset unless we explicitly decide to label that
scenario-aware setting.

## Task Export

Generate the formal task table with:

```powershell
python permin_dataset_processing/export_labeling_tasks.py
```

Default output:

```text
permin_dataset_processing/milp_labels/labeling_tasks.csv
```

The expected columns are:

```text
instance_name,order_id,package_id,package_l,package_w,package_h,item_count,item_volume_sum,items_json
```

## Solver Contract

The trusted MILP implementation should consume `labeling_tasks.csv` or an
equivalent `(order, package)` task stream and emit:

```text
instance_name,order_id,package_id,orientation_mode,solver_status,feasible,runtime_sec
```

The orientation modes and all geometry constraints must follow the trusted
MILP implementation, not the smoke-test implementation.

## Current Status

`MILP_3DBPP` has been committed as a regular directory in this repository.
The package-aware label generator is:

```text
MILP_3DBPP/src/main/java/org/example/GeneratePerminPackageLabels.java
```

Full base-instance package labeling produced:

```text
permin_dataset_processing/milp_labels/ground_truth_package_labels.csv
```

The generated task table and labeled feature matrices are intentionally
reproducible artifacts. They can be regenerated with the scripts in this
directory and are excluded from Git when they are too large for convenient
GitHub storage.

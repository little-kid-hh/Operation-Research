# OR 2023 BPP MILP Labeling Configuration

This file records the labeling contract for the Fontaine & Minner OR 2023
3D-BPP loadability dataset.

## Scope

Target paper:

```text
Pirmin Fontaine, Stefan Minner (2023)
A Branch-and-Repair Method for Three-Dimensional Bin Selection and Packing in E-Commerce
Operations Research 71(1):273-288
DOI: 10.1287/opre.2022.2369
```

The current project should not default to the later `S3DBSP-main` stochastic
BSP data. Those files are legacy/debug data unless explicitly marked otherwise.

## Label Unit

One label row is one tuple:

```text
(instance_name, order_id, package_id)
```

The label answers whether all items in that order can be packed into that
single candidate package.

## Raw Inputs

Expected local staging path:

```text
or2023_bpp_data/
  xml/
  packages.txt
```

Package format:

```text
package_id length width height
```

Order item format:

```text
XML order -> item -> p, q, r
```

If the OR 2023 e-companion uses a different folder or XML naming convention,
pass it explicitly via `--xml-dir`, `--packages-path`, and `--xml-name-regex`.

## BSP-Derived Data Guard

Scripts reject paths containing `S3DBSP-main` by default. This prevents new
labeling/evaluation runs from silently indexing the stochastic BSP data package.

Only use the escape hatch for legacy audits:

```powershell
--allow-bsp-derived-data
```

or:

```powershell
-AllowBspDerivedData
```

## Task Export

Generate the formal task table with:

```powershell
python permin_dataset_processing/export_labeling_tasks.py
```

Default output:

```text
permin_dataset_processing/milp_labels/labeling_tasks.csv
```

Expected columns:

```text
instance_name,order_id,package_id,package_l,package_w,package_h,item_count,item_volume_sum,items_json
```

## Solver Contract

The trusted MILP implementation consumes an `(order, package)` task stream and
emits:

```text
instance_name,order_id,package_id,orientation_mode,solver_status,feasible,runtime_sec
```

The current package label generator is:

```text
MILP_3DBPP/src/main/java/org/example/GeneratePerminPackageLabels.java
```

It now requires explicit data arguments:

```powershell
java -cp "<gurobi-and-project-classpath>" org.example.GeneratePerminPackageLabels or2023_bpp_data\xml or2023_bpp_data\packages.txt permin_dataset_processing\milp_labels\or2023_bpp_package_labels.csv
```

The MILP non-overlap constraints should remain aligned with the normalized
Fontaine-Minner formulation unless a paper-level model review says otherwise.

The default solver time limit is 300 seconds for both orientation modes. A
feasible solution found before the limit is a positive label; proven infeasible
is a negative label; a time limit with no feasible solution is `-1` and remains
unknown for ML training/evaluation.

## Evaluation Contract

ML evaluation must use an OR 2023 BPP labeled feature matrix, defaulting to:

```text
permin_dataset_processing/processed_features/or2023_bpp_labeled_base40_package.csv
```

The evaluation split defaults to instance-level grouping to avoid train/test
leakage across orders from the same XML instance.

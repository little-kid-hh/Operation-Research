# OR 2023 BSP Package-Feasibility Labeling Report

Date: 2026-06-02

## Scope

This report covers the Fontaine-Minner OR 2023 BSP/design XML data:

- Source supplement archive: `opre.2022.2369.sm1.zip` (not committed)
- BSP source XMLs: `datasets_xml/BSP_*.xml`
- Prepared repository data: `or2023_bsp_data/`

The BSP XML data contains 160 files and 150,000 expanded orders.

| BSP variant | Files | Orders |
| --- | ---: | ---: |
| E6 | 40 | 37,500 |
| EL | 40 | 37,500 |
| EM | 40 | 37,500 |
| O6 | 40 | 37,500 |

| BSP size | Orders |
| ---: | ---: |
| 100 | 2,000 |
| 200 | 4,000 |
| 300 | 6,000 |
| 400 | 8,000 |
| 500 | 10,000 |
| 1000 | 20,000 |
| 2000 | 40,000 |
| 3000 | 60,000 |

## Why Geometry-Deduplicated Labels

Using 90 candidate packages, the fully expanded BSP labeling table would contain:

```text
150,000 BSP orders * 90 packages = 13,500,000 order-package labels
```

Many BSP orders have identical item geometries across XML files, sizes, variants, and seeds. To avoid committing a very large duplicated table, the repository stores:

- `or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml`
- `or2023_bsp_data/order_geometry_map.csv`
- `permin_dataset_processing/milp_labels/or2023_bsp_unique_package_labels.csv`

The expanded BSP label table can be reconstructed by joining:

```text
order_geometry_map.geom_id == or2023_bsp_unique_package_labels.order_id
```

This gives all 13,500,000 expanded order-package labels without storing repeated copies of identical geometry labels.

## BPP vs BSP

The OR 2023 supplement has two separate data families:

| Data family | Supplement location | Current repository output | Main use |
| --- | --- | --- | --- |
| BPP benchmark | `codes/BPP/dataset50/*.pkl` | `or2023_bpp_data/` and `or2023_bpp_package_labels.csv` | ML loadability benchmark |
| BSP/design XML | `datasets_xml/BSP_*.xml` | `or2023_bsp_data/` and `or2023_bsp_unique_package_labels.csv` | Box-size design/order benchmark labeling |

BPP has 450 instances with 2 to 10 items. BSP has 150,000 expanded orders, with at most 6 items per order in these XMLs.

## Labeling Configuration

Prepared unique BSP labeling data:

```powershell
python permin_dataset_processing\prepare_or2023_bsp_labeling_data.py `
  --bsp-xml-dir .codex_training_tmp\opre_sm1\datasets_xml `
  --packages-path or2023_bpp_data\packages.txt `
  --output-root or2023_bsp_data
```

Ran unique geometry-package labeling:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File permin_dataset_processing\run_permin_label_batches.ps1 `
  -BatchSize 50000 `
  -OutputPath permin_dataset_processing\milp_labels\or2023_bsp_unique_package_labels.csv `
  -XmlDir or2023_bsp_data\xml_unique `
  -PackagesPath or2023_bsp_data\packages.txt `
  -TotalTasks 1157760 `
  -TimeLimit2Ori 300 `
  -TimeLimit6Ori 300
```

The Java labeler uses a safe infeasibility pre-screen before invoking Gurobi:

- total item volume exceeds package volume
- at least one item cannot fit in the package by allowed orientation

These checks only short-circuit cases that are certainly infeasible.

## Labeling Result

| Quantity | Value |
| --- | ---: |
| Expanded BSP orders | 150,000 |
| Candidate packages | 90 |
| Expanded order-package labels | 13,500,000 |
| Unique order geometries | 12,864 |
| Stored unique geometry-package labels | 1,157,760 |

Unique label distribution:

| Target | Infeasible | Feasible |
| --- | ---: | ---: |
| `label_2ori` | 760,115 | 397,645 |
| `label_6ori` | 752,602 | 405,158 |

The full unique-label run took about 26 minutes on the local Gurobi setup. No timeout markers are stored because the labeler records final binary feasibility labels and solve times; the maximum observed recorded solve time was below 1 second after pre-screening and deduplication.

## Repository Files

| Path | Description |
| --- | --- |
| `or2023_bsp_data/README.md` | Data layout note |
| `or2023_bsp_data/packages.txt` | 90 candidate packages |
| `or2023_bsp_data/order_geometry_map.csv` | Expanded BSP order to unique geometry mapping |
| `or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml` | Canonical unique BSP order geometries |
| `permin_dataset_processing/milp_labels/or2023_bsp_unique_package_labels.csv` | Unique geometry-package feasibility labels |

## Reconstructing Expanded Labels

Example:

```python
import pandas as pd

mapping = pd.read_csv("or2023_bsp_data/order_geometry_map.csv")
labels = pd.read_csv("permin_dataset_processing/milp_labels/or2023_bsp_unique_package_labels.csv")
labels = labels.rename(columns={"order_id": "geom_id", "instance_name": "label_source_xml"})
expanded = mapping.merge(labels, on="geom_id", how="left")
```

The resulting `expanded` table has 13,500,000 rows.

In the expanded table, `instance_name` and `order_id` from `mapping` identify the
original BSP XML order. `label_source_xml` identifies only the deduplicated XML
file used for MILP labeling.

The geometry key ignores item order but preserves each item's `(p, q, r)` axis
naming. This keeps two-orientation labels semantically valid: item height is not
deduplicated away or treated as interchangeable with length/width.

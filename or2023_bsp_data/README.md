# OR 2023 BSP Labeling Data

This directory contains geometry-deduplicated Fontaine-Minner OR 2023 BSP orders for package-feasibility labeling.

- `xml_unique/or2023_bsp_unique_orders.xml`: canonical unique order geometries.
- `order_geometry_map.csv`: mapping from each BSP XML order to `geom_id`.
- `packages.txt`: the 90 candidate packages used for feasibility labels.

The expanded BSP label table can be reconstructed by joining `order_geometry_map.csv` to the unique geometry-package labels on `geom_id`.

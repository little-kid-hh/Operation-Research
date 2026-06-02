# OR 2023 BSP Labeling Data

This directory contains geometry-deduplicated Fontaine-Minner OR 2023 BSP orders for package-feasibility labeling.

- `xml_unique/or2023_bsp_unique_orders.xml`: canonical unique order geometries.
- `order_geometry_map.csv`: mapping from each BSP XML order to `geom_id`.
- `packages.txt`: the 90 candidate packages used for feasibility labels.

The expanded BSP label table can be reconstructed by joining `order_geometry_map.csv` to the unique geometry-package labels on `geom_id`.

The unique BSP label file is:

```text
permin_dataset_processing/milp_labels/or2023_bsp_unique_package_labels.csv
```

In that file, `order_id` is the unique geometry id. Rename it to `geom_id` before joining:

```python
labels = labels.rename(columns={"order_id": "geom_id"})
labels = labels.rename(columns={"instance_name": "label_source_xml"})
expanded = order_geometry_map.merge(labels, on="geom_id", how="left")
```

`order_geometry_map.instance_name` and `order_geometry_map.order_id` are the
original BSP XML file and original order id. `label_source_xml` is only the
deduplicated XML file used to run the labels.

Geometry deduplication ignores item order but preserves each item's `(p, q, r)`
axis naming. This is important for `label_2ori`, where item height remains tied
to package height.

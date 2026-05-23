## POLICY_UPDATE
- Focus on the **width-dimension bottleneck**, which is underrepresented in the active bank (only `n_wide_items` uses original width; no sorted-dimension width threshold exists).
- Introduce a **smallest-dimension chunkiness** signal (`dim_s`), which is entirely absent from the bank and captures gap-filling flexibility lost when items are thick in every orientation.
- Add a **spare-capacity × width-bottleneck interaction** to linearize the nonlinear regime where moderate spare capacity is negated by multiple wide items competing for the same floor strip.

## FEATURE_CODE
```python
def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """Add width-bottleneck and smallest-dimension chunkiness features."""
    import pandas as pd
    import numpy as np

    # Vehicle dimensions from items_df (constant per dispatch)
    vdim = (
        items_df.groupby("dispatch_id")
        .agg(
            vehicle_length=("vehicle_length", "first"),
            vehicle_width=("vehicle_width", "first"),
            vehicle_height=("vehicle_height", "first"),
        )
        .reset_index()
    )

    # ── Feature 1: n_items_near_vW ──
    # Count items whose sorted medium dimension exceeds 50% of vehicle width.
    # This captures width bottleneck using orientation-aware sorted dims,
    # unlike n_wide_items which uses the original (unsorted) item_width.
    items_with_vw = items_df.merge(vdim, on="dispatch_id", how="left")
    items_with_vw["near_vw"] = (items_with_vw["dim_m"] > 0.5 * items_with_vw["vehicle_width"]).astype(int)
    n_near_vw = items_with_vw.groupby("dispatch_id")["near_vw"].sum().reset_index()
    n_near_vw.columns = ["dispatch_id", "n_items_near_vW"]

    # ── Feature 2: dim_s_to_vH_avg ──
    # Average of (dim_s / vehicle_height) across items.
    # dim_s is the smallest sorted dimension; high average means items are
    # "chunky" (thick even in their thinnest orientation), leaving little
    # gap-filling flexibility. This dimension is entirely unused in the bank.
    items_with_vh = items_df.merge(vdim, on="dispatch_id", how="left")
    items_with_vh["dim_s_ratio"] = items_with_vh["dim_s"] / items_with_vh["vehicle_height"]
    dim_s_avg = items_with_vh.groupby("dispatch_id")["dim_s_ratio"].mean().reset_index()
    dim_s_avg.columns = ["dispatch_id", "dim_s_to_vH_avg"]

    # ── Feature 3: spare_cap_x_n_near_vW ──
    # Interaction between spare capacity and width-bottleneck count.
    # When spare capacity is moderate but many items are near vehicle width,
    # packing is harder than spare capacity alone predicts because wide items
    # cannot share the same floor strip.
    spare_cap = agg_df[["dispatch_id", "spare_capacity"]].copy()
    interaction = spare_cap.merge(n_near_vw, on="dispatch_id", how="left")
    interaction["spare_cap_x_n_near_vW"] = interaction["spare_capacity"] * interaction["n_items_near_vW"]
    result = interaction[["dispatch_id", "spare_cap_x_n_near_vW"]].copy()

    # ── Merge all features ──
    result = result.merge(n_near_vw, on="dispatch_id", how="left")
    result = result.merge(dim_s_avg, on="dispatch_id", how="left")

    # Ensure correct column order
    result = result[["dispatch_id", "n_items_near_vW", "dim_s_to_vH_avg", "spare_cap_x_n_near_vW"]]

    return result
```

## RATIONALE
- **`n_items_near_vW`** fills a gap in the active bank: `n_wide_items` uses the raw `item_width` column, but after orientation sorting the *medium* dimension `dim_m` is the width-relevant measure. Multiple items with `dim_m > 0.5 × vehicle_width` cannot be placed side-by-side, creating a floor-strip bottleneck that a single max-ratio cannot express.
- **`dim_s_to_vH_avg`** introduces the entirely-unused `dim_s` (smallest sorted dimension) into the bank. Items with high `dim_s` relative to vehicle height are "chunky" in every orientation, leaving no thin dimension to tuck into gaps—this loss of packing flexibility is invisible to volume-only or max-dimension features.
- **`spare_cap_x_n_near_vW`** linearizes the tree-recovered interaction where moderate spare capacity is negated by width competition. XGB's dominant `spare_capacity` splitter (gain 16.9) interacts conditionally with dimension constraints; this term makes that interaction explicit for the SVM, targeting the TPR@FPR=1% operating point where the current bank underperforms.
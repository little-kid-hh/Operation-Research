import numpy as np
import pandas as pd

def build_candidate_features(agg_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    base = agg_df[[
        "dispatch_id",
        "vehicle_length",
        "vehicle_width",
        "vehicle_height",
        "vehicle_capacity",
        "spare_capacity",
    ]].copy()

    bin_sorted = np.sort(
        base[["vehicle_length", "vehicle_width", "vehicle_height"]].to_numpy(dtype=float),
        axis=1,
    )
    base["bin_s"] = bin_sorted[:, 0]
    base["bin_m"] = bin_sorted[:, 1]
    base["bin_l"] = bin_sorted[:, 2]
    base["slack_ratio"] = base["spare_capacity"] / np.maximum(base["vehicle_capacity"], 1.0)

    work = items_df.merge(
        base[["dispatch_id", "bin_s", "bin_m", "bin_l", "vehicle_length", "vehicle_width", "slack_ratio"]],
        on="dispatch_id",
        how="left",
    )

    g = work.groupby("dispatch_id", sort=False)

    out = pd.DataFrame({"dispatch_id": list(g.groups.keys())})
    out["dominant_type_share"] = (
        g.apply(
            lambda x: (
                x.groupby(["dim_s", "dim_m", "dim_l"]).size().max() / max(len(x), 1)
            )
        ).astype(float).values
    )
    out["p90_long_over_bin_long"] = (
        g.apply(lambda x: float(np.quantile(x["dim_l"], 0.90)) / max(float(x["bin_l"].iloc[0]), 1.0)).values
    )
    out["p90_mid_over_bin_mid"] = (
        g.apply(lambda x: float(np.quantile(x["dim_m"], 0.90)) / max(float(x["bin_m"].iloc[0]), 1.0)).values
    )
    out["thin_item_share"] = (
        g.apply(lambda x: float((x["dim_s"] / np.maximum(x["dim_l"], 1.0) < 0.35).mean())).values
    )
    out["max_face_area_load_over_floor"] = (
        g.apply(
            lambda x: float((x["dim_m"] * x["dim_l"]).sum())
            / max(float(x["vehicle_length"].iloc[0] * x["vehicle_width"].iloc[0]), 1.0)
        ).values
    )
    out["tight_bin_large_piece_interaction"] = (
        g.apply(
            lambda x: float(np.quantile(x["dim_l"], 0.90))
            / max(float(x["bin_l"].iloc[0]), 1.0)
            * max(0.0, 0.20 - float(x["slack_ratio"].iloc[0]))
        ).values
    )

    return out


# -*- coding: utf-8 -*-
def apply_rule_patch(svm_prob: float, features: dict) -> int:
    sc = features.get('sku_counts', 0)
    sp = features.get('spare_capacity', 0)
    l_var = features.get('sku_length_var', 0)
    w_var = features.get('sku_width_var', 0)
    h_var = features.get('sku_height_var', 0)
    max_asr = features.get('max_asr', 0)
    wl_std = features.get('wl_to_vehicle_wl_std', 0)
    l_std = features.get('l_to_L_ratio_std', 0)
    avg_vol = features.get('sku_average_volume', 0)
    
    # ── FN Correction (SVM predicts NO, actually YES) ──────────────────────
    # Physical rationale: SVM linearly penalizes moderate sku_counts (w=-10.6) 
    # and sku_average_volume (w=-16.4), causing under-confidence on uniform items.
    # FN pattern: spare_capacity ~9240 (below p25=10749), sku_counts ~11, 
    # low dimensional variance indicates brick-like items that tile efficiently.
    # These uniform items have minimal wasted space despite SVM's volume bias.
    if 0.20 <= svm_prob <= 0.50:
        if 7000 <= sp <= 13000:
            if 9 <= sc <= 14:
                # Low variance = uniform items pack with near-zero wasted space
                if l_var < 50 and w_var < 9.0 and h_var < 14:
                    if max_asr < 5.7 and avg_vol < 3500:
                        return 1

    # ── FP Correction (SVM predicts YES, actually NO) ──────────────────────
    # Physical rationale: SVM over-relies on spare_capacity (w=+13.6).
    # FP pattern: spare_capacity ~12277 (above median), sku_counts ~7-8 (few items),
    # high length_var ~48 causes geometric fragmentation - theoretical volume 
    # is unusable for tiling due to size mismatches and poor aspect ratios.
    # Few large/varied items create gaps that cannot be filled.
    if 0.55 <= svm_prob <= 0.92:
        if sp > 13000 and sc <= 8:
            # Fragmentation signature: high variance + poor aspect ratios
            # OR significant footprint length mismatch across items
            if (l_var > 46 and max_asr > 5.4) or (wl_std > 40 and l_std > 0.11):
                return 0

    return -1
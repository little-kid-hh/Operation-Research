# -*- coding: utf-8 -*-
'''
3D-BPP Rule Patch — SVM Hard Case Corrector

This function is called AFTER SVM prediction to correct hard cases.
Parameters:
    svm_prob : float  — P(y=1) from SVM (already sigmoid of decision score)
    features : dict   — raw feature dict with all 41 feature values

Returns:
    -1  : do not override SVM (rule not triggered)
    0   : override → predict NO (not feasible)
    1   : override → predict YES (feasible)

IMPORTANT:
  - fill_ratio is a dimensionless fraction (often ~0–1): loaded volume / vehicle capacity.
    It is NOT cubic meters; use the Feature scales table in the prompt for thresholds.
  - Base your decision on physical 3D packing constraints:
      * sku_counts: number of items
      * sku_average_volume: avg item volume
      * fill_ratio = total_skuvolume / vehicle_capacity (same as fraction of capacity used)
      * sku_min_length/max_length, sku_min_width/max_width, sku_min_height/max_height
      * aspect_ratio (max_asr): longest / shortest dimension of any item
      * spare_capacity: remaining vehicle volume
      * sku_length_var, sku_width_var, sku_height_var: variance in item sizes
      * l_to_L_ratio_avg/min/max: SKU length / vehicle length ratio
      * etc.
  - Rules should only fire in specific feature ranges, not globally
  - Do NOT just return 0 or 1 unconditionally
  - Test your rule mentally on these patterns before writing code:
      * High fill_ratio + many items → likely infeasible
      * Low fill_ratio → likely feasible
      * Very high aspect ratio items → hard to pack
'''
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    sc      = features.get('spare_capacity', 0)
    masr    = features.get('max_asr', 1.0)
    cnt     = features.get('sku_counts', 0)
    l_avg   = features.get('l_to_L_ratio_avg', 0)
    l_var   = features.get('sku_length_var', 0)
    wl_std  = features.get('wl_to_vehicle_wl_std', 0)

    # ─── FN Correction (SVM says NO, actually YES) ─────────────────────
    # SVM heavily penalizes sku_counts & sku_average_volume. It underestimates
    # feasibility when items are geometrically "friendly": low aspect ratio,
    # uniform lengths, and still have meaningful spare volume.
    # Thresholds: below p25 for ASR, below median for length variance,
    # above FN-mean for spare capacity.
    if svm_prob < 0.45:
        if sc > 8500 and masr < 4.5 and cnt <= 12 and l_var < 44:
            return 1

    # ─── FP Correction (SVM says YES, actually NO) ─────────────────────
    # SVM sees positive spare_capacity but misses geometric fragmentation.
    # Very elongated items (high ASR) or long items with highly variable
    # floor footprints create dead space that volume metrics ignore.
    # Thresholds: above p75 for ASR/length-ratio/std, targeting awkward layouts.
    if svm_prob > 0.55:
        if (masr > 5.8 or (l_avg > 0.42 and wl_std > 42)) and cnt <= 9:
            return 0

    # ─── Default: trust SVM ────────────────────────────────────────────
    return -1
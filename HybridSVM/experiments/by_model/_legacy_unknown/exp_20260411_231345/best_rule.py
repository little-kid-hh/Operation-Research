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
  - Rules target specific FN/FP patterns identified in hard-case analysis.
  - Thresholds are calibrated against the provided feature statistics table.
  - Physical justification: 
      * FN fix: SVM over-penalizes item count/volume. When items are highly uniform 
        (low size variance & ratio std), they pack efficiently despite moderate counts.
      * FP fix: SVM over-trusts spare capacity. High capacity is misleading when items 
        are highly fragmented (high length variance, long relative lengths, high aspect ratio).
'''
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    # Extract relevant features
    sku_counts    = features.get('sku_counts', 0)
    spare_cap     = features.get('spare_capacity', 0)
    sku_len_var   = features.get('sku_length_var', 0)
    sku_wid_var   = features.get('sku_width_var', 0)
    l_ratio_std   = features.get('l_to_L_ratio_std', 0)
    l_ratio_max   = features.get('l_to_L_ratio_max', 0)
    max_asr       = features.get('max_asr', 1.0)

    # ─── Rule 1: Fix FN (SVM under-confident → YES) ─────────────────────
    # SVM heavily penalizes sku_counts (-10.59) and sku_average_volume (-16.41).
    # If count is moderate (10-13) and spare capacity is decent, but items are 
    # highly uniform (low variance & low length-ratio std), fragmentation is minimal 
    # and packing is likely feasible.
    if svm_prob < 0.5:
        if (10 <= sku_counts <= 13 and
            spare_cap > 6000 and
            sku_len_var < 38.0 and
            sku_wid_var < 7.0 and
            l_ratio_std < 0.095):
            return 1

    # ─── Rule 2: Fix FP (SVM over-confident → NO) ───────────────────────
    # SVM trusts spare_capacity (+13.56) too much. When spare capacity is high 
    # but items have high length variance, large relative max lengths, and high 
    # aspect ratios, space fragmentation prevents feasible packing despite volume.
    if svm_prob > 0.5:
        if (spare_cap > 12000 and
            sku_len_var > 52.0 and
            l_ratio_max > 0.58 and
            max_asr > 5.5):
            return 0

    return -1
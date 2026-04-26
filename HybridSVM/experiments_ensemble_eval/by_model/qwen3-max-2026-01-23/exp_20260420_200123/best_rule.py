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
    # Extract key features
    sku_counts = features.get('sku_counts', 0)
    spare_capacity = features.get('spare_capacity', 0)
    sku_average_volume = features.get('sku_average_volume', 0)
    wl_to_vehicle_wl_total = features.get('wl_to_vehicle_wl_total', 0)
    wl_to_vehicle_wl_min = features.get('wl_to_vehicle_wl_min', 0)
    sku_length_var = features.get('sku_length_var', 0)
    max_asr = features.get('max_asr', 1.0)
    
    # FN Correction: SVM says NO (low prob) but actually feasible
    # Pattern: moderate spare capacity, reasonable item count, low footprint usage
    if svm_prob < 0.5 and spare_capacity > 8000 and sku_counts <= 12 and wl_to_vehicle_wl_total < 1250:
        # Check that minimum footprint usage isn't too constraining
        if wl_to_vehicle_wl_min < 55:
            return 1
    
    # FP Correction: SVM says YES (high prob) but actually infeasible  
    # Pattern: high probability but high length variance creates packing inefficiency
    if svm_prob > 0.65 and sku_counts <= 8 and sku_length_var > 45 and wl_to_vehicle_wl_total > 1050:
        # Additional check: ensure we're not overriding borderline TP cases
        if spare_capacity < 15000 or max_asr > 6.5:
            return 0
    
    return -1
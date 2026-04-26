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
    sku_counts = features.get('sku_counts', 0)
    fill_ratio = features.get('fill_ratio', 0)
    max_asr = features.get('max_asr', 1.0)
    spare_cap = features.get('spare_capacity', 0)
    wl_total = features.get('wl_to_vehicle_wl_total', 0)
    wl_min = features.get('wl_to_vehicle_wl_min', 0)
    sku_avg_vol = features.get('sku_average_volume', 0)
    vehicle_cap = 45000  # standard vehicle capacity
    
    # Correct FN cases: low fill_ratio but SVM under-confident
    # From FN analysis: typical fill_ratio around 5000-7000 (which is ~0.11-0.16 of 45000)
    # But note: fill_ratio in data appears to be absolute volume, not fraction!
    # Checking feature scales: wl_to_vehicle_wl_total max is 2111.25, but fill_ratio examples are 5282, 6228, 7126
    # This suggests fill_ratio might actually be total volume in cubic units, not fraction
    # Given vehicle_capacity = 45000, then actual fill fraction = fill_ratio / 45000
    actual_fill_frac = fill_ratio / vehicle_cap if vehicle_cap > 0 else 0
    
    # FN correction: when actual fill fraction is low (< 0.2) but SVM is uncertain (prob < 0.6)
    # This matches FN examples: fill_ratio ~5000-7000 → ~0.11-0.16 fraction
    if actual_fill_frac < 0.2 and svm_prob < 0.6:
        return 1
    
    # FP correction: high fill fraction with moderate item count but high aspect ratio items
    # FP examples show fill_ratio 18000-21000 (~0.4-0.47 fraction) with sku_counts 7-8
    # High aspect ratio items (>6.0) are harder to pack efficiently
    if actual_fill_frac > 0.4 and actual_fill_frac < 0.5 and sku_counts <= 10 and max_asr > 6.0:
        return 0
    
    # Additional FP correction: very high total wall area usage with low minimum wall area
    # This indicates items have varying footprints that may not pack well
    # From FP analysis: wl_to_vehicle_wl_min is quite low (mean ~47) while total is high (~1100)
    # Vehicle wall area capacity seems to be around 1500 based on feature scales
    if wl_total > 1000 and wl_min < 40 and sku_counts <= 10:
        return 0
    
    return -1
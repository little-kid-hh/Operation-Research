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
    fill_ratio = features.get('fill_ratio', 0)
    max_asr = features.get('max_asr', 1.0)
    spare_capacity = features.get('spare_capacity', 0)
    sku_average_volume = features.get('sku_average_volume', 0)
    wl_to_vehicle_wl_total = features.get('wl_to_vehicle_wl_total', 0)
    wl_to_vehicle_wl_min = features.get('wl_to_vehicle_wl_min', 0)
    sku_length_var = features.get('sku_length_var', 0)
    sku_width_var = features.get('sku_width_var', 0)
    sku_height_var = features.get('sku_height_var', 0)
    sku_length_avg = features.get('sku_length_avg', 0)
    sku_width_avg = features.get('sku_width_avg', 0)
    sku_height_avg = features.get('sku_height_avg', 0)
    
    # Correct FN cases: SVM says NO but it's actually feasible
    if svm_prob < 0.5:  # SVM predicts NO
        # Pattern from analysis: FN cases often have positive spare capacity,
        # moderate fill ratios, and are not extremely variant in dimensions.
        # They're "underestimated" by the linear model.
        
        # Case 1: Moderate fill ratio with sufficient spare capacity and low variance
        if (0.65 <= fill_ratio <= 0.85 and 
            spare_capacity > 8000 and
            sku_length_var < 50 and 
            sku_width_var < 10 and 
            sku_height_var < 12):
            return 1
            
        # Case 2: Lower fill ratio with positive spare capacity (conservative SVM)
        if fill_ratio < 0.7 and spare_capacity > 2000:
            return 1
            
        # Case 3: Items are small on average but not too many
        if (sku_average_volume < 3000 and 
            sku_counts <= 12 and 
            spare_capacity > 5000):
            return 1

    # Correct FP cases: SVM says YES but it's actually infeasible
    elif svm_prob >= 0.5:  # SVM predicts YES
        # Pattern from analysis: FP cases often have high fill ratios,
        # high item count, or extreme variations that make packing difficult.
        
        # Case 1: Very high fill ratio with many items
        if fill_ratio > 0.92 and sku_counts > 10:
            return 0
            
        # Case 2: High total area usage with significant item count
        if wl_to_vehicle_wl_total > 1150 and sku_counts > 9:
            return 0
            
        # Case 3: Large items (high avg dimensions) with moderate-high fill
        if (sku_length_avg > 24 and sku_width_avg > 10 and 
            fill_ratio > 0.8 and sku_counts >= 8):
            return 0
            
        # Case 4: High minimum WL ratio suggesting large indivisible items
        if wl_to_vehicle_wl_min > 55 and sku_length_var > 45:
            return 0

    return -1
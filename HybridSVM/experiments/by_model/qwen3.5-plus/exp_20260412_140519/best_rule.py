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
'''
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    # ─── Extract key features ────────────────────────────────────────────
    sku_counts = features.get('sku_counts', 0)
    spare_cap = features.get('spare_capacity', 0)
    sku_average_volume = features.get('sku_average_volume', 0)
    sku_length_var = features.get('sku_length_var', 0)
    sku_width_var = features.get('sku_width_var', 0)
    sku_std_length = features.get('sku_std_length', 0)
    max_asr = features.get('max_asr', 1.0)
    
    # Derived fill ratio from spare_capacity and typical vehicle volume
    # Vehicle volume = 60 * 25 * 30 = 45000
    vehicle_volume = 45000
    fill_ratio = vehicle_volume - spare_cap if spare_cap < vehicle_volume else 0
    
    # ─── FN Correction: SVM under-confident (predicts NO, should be YES) ───
    # Pattern: Moderate item count with decent spare capacity but SVM over-penalizes
    # Typical FN: sku_counts=11, spare_cap~9000, moderate volume items
    # Geometric: Similar-sized items (low variance) pack efficiently despite count
    if svm_prob < 0.45:
        # Moderate spare capacity (above FN mean ~9240 but not extreme)
        if 7500 <= spare_cap <= 14000:
            # Moderate item count (FN typical: 11 items)
            if 9 <= sku_counts <= 13:
                # Items not too large on average (below p75 of ~3198)
                if sku_average_volume < 3350:
                    # Low-moderate length variance enables efficient packing
                    if sku_length_var < 55:
                        return 1  # Override to YES
    
    # ─── FP Correction Pattern 1: High fill + few large items + high variance ───
    # Pattern: Few but large items with high size variance create unusable gaps
    # Typical FP: sku_counts=7-8, fill_ratio=18000-21000, length_var~48
    # Geometric: Large items with varying dimensions leave awkward voids
    if svm_prob > 0.65:
        if fill_ratio > 17000 and sku_counts <= 8:
            # High length variance means items don't nest/stack well
            if sku_length_var > 44 and sku_average_volume > 3000:
                return 0  # Override to NO
    
    # ─── FP Correction Pattern 2: High aspect ratio + high fill ────────────
    # Very elongated items (max_asr > 6) are geometrically difficult
    # Combined with high fill, creates infeasible configurations
    if svm_prob > 0.70:
        if max_asr > 6.0 and fill_ratio > 15000:
            # High std in length compounds aspect ratio packing difficulty
            if sku_std_length > 6.5:
                return 0  # Override to NO
    
    # ─── FP Correction Pattern 3: Extreme fill with dimension variance ─────
    # When fill is very high AND both length and width vary significantly,
    # packing becomes geometrically constrained despite volume availability
    if svm_prob > 0.75:
        if fill_ratio > 18500:
            # High variance in multiple dimensions = poor nesting
            if sku_length_var > 48 or (sku_length_var > 40 and sku_width_var > 8.5):
                return 0  # Override to NO
    
    return -1
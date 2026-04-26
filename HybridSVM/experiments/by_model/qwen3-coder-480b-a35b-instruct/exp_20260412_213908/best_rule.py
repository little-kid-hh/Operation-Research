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
    # ─── Your rule logic here ────────────────────────────────────────────
    sku_counts = features.get('sku_counts', 0)
    fill_ratio = features.get('fill_ratio', 0)
    max_asr    = features.get('max_asr', 1.0)
    spare_cap  = features.get('spare_capacity', 0)
    avg_vol    = features.get('sku_average_volume', 0)
    wl_total   = features.get('wl_to_vehicle_wl_total', 0)
    wl_avg     = features.get('wl_to_vehicle_wl_avg', 0)
    wl_std     = features.get('wl_to_vehicle_wl_std', 0)
    l_ratio_avg = features.get('l_to_L_ratio_avg', 0)
    w_ratio_avg = features.get('w_to_W_ratio_avg', 0)
    h_ratio_avg = features.get('h_to_H_ratio_avg', 0)
    sku_length_var = features.get('sku_length_var', 0)
    sku_width_var = features.get('sku_width_var', 0)
    sku_height_var = features.get('sku_height_var', 0)
    sku_max_length = features.get('sku_max_length', 0)
    sku_max_width = features.get('sku_max_width', 0)
    sku_max_height = features.get('sku_max_height', 0)
    vehicle_length = features.get('vehicle_length', 60)
    vehicle_width = features.get('vehicle_width', 25)
    vehicle_height = features.get('vehicle_height', 30)
    sku_min_length = features.get('sku_min_length', 0)
    sku_min_width = features.get('sku_min_width', 0)
    sku_min_height = features.get('sku_min_height', 0)

    # FN Correction: High spare capacity with moderate item count and low SVM confidence
    if svm_prob < 0.4 and spare_cap > 18000 and sku_counts <= 12:
        # Items are not too small and loading seems reasonable
        if avg_vol > 2600 and wl_total > 800:
            return 1

    # FN Correction: Low item count with high average volume and moderate fill ratio
    if sku_counts <= 8 and fill_ratio < 0.8 and avg_vol > 3200:
        if svm_prob < 0.5:
            # Check if items are not extremely long or wide which would make them hard to fit
            if sku_max_length < 0.7 * vehicle_length and sku_max_width < 0.7 * vehicle_width:
                return 1

    # FN Correction: Moderate fill ratio, decent spare capacity, but SVM is underconfident
    if fill_ratio < 0.75 and svm_prob < 0.35 and spare_cap > 12000:
        if wl_avg > 95 and sku_counts >= 6:
            return 1

    # FP Correction: High fill ratio with high dimensional variance
    if svm_prob > 0.7 and fill_ratio > 0.85:
        if (sku_length_var > 45 or sku_width_var > 8.5 or sku_height_var > 12):
            return 0

    # FP Correction: Very high fill ratio nearing capacity
    if fill_ratio > 0.92 and svm_prob > 0.65:
        # Even with few items, if they're large or have poor utilization, it's risky
        if sku_counts >= 6:
            # Poor dimensional utilization
            if l_ratio_avg < 0.28 or w_ratio_avg < 0.11 or h_ratio_avg < 0.16:
                return 0
            # Or very large items
            if (sku_max_length > 0.75 * vehicle_length or
                sku_max_width > 0.75 * vehicle_width or
                sku_max_height > 0.75 * vehicle_height):
                return 0

    # FP Correction: High SVM confidence but very high aspect ratio
    if svm_prob > 0.75 and max_asr > 6.5:
        # Especially if combined with high fill ratio
        if fill_ratio > 0.8:
            return 0

    # Edge case: Many small items with high fill ratio
    if sku_counts > 14 and avg_vol < 2500 and fill_ratio > 0.78:
        if svm_prob > 0.55:
            # Unless they have very low variance (easy to stack/pack)
            low_variance = (sku_length_var < 28 and 
                           sku_width_var < 5.5 and 
                           sku_height_var < 9.5)
            if not low_variance:
                return 0

    # New FN correction: Low SVM confidence with good dimensional utilization
    # Pattern: SVM underconfident despite efficient packing indicators
    if svm_prob < 0.4 and fill_ratio < 0.75:
        # Good average WL utilization suggests efficient packing
        if wl_avg > 90 and wl_std < 40:
            # And reasonable item count
            if sku_counts >= 8 and sku_counts <= 15:
                return 1

    # New FN correction: High spare capacity with moderate item count
    # Pattern: Enough space but SVM is conservative
    if svm_prob < 0.5 and spare_cap > 20000 and sku_counts <= 12:
        # Check if items are reasonably sized
        if avg_vol > 2500:
            return 1

    # New FP correction: High fill ratio with large maximum item dimensions
    # Pattern: Large items near capacity limit
    if fill_ratio > 0.9 and svm_prob > 0.7:
        # Check if any single item is very large relative to vehicle
        if (sku_max_length > 0.8 * vehicle_length or
            sku_max_width > 0.8 * vehicle_width or
            sku_max_height > 0.8 * vehicle_height):
            return 0
            
    # Additional FN correction: Low fill ratio with sufficient spare capacity
    # Pattern: SVM underconfident when there's clearly enough space
    if svm_prob < 0.3 and fill_ratio < 0.6 and spare_cap > 15000:
        # Reasonable item count
        if sku_counts >= 6 and sku_counts <= 14:
            return 1
            
    # Additional FP correction: High dimensional variance with moderate fill ratio
    # Pattern: SVM overconfident despite packing challenges from varied sizes
    if svm_prob > 0.6 and fill_ratio > 0.75:
        if (sku_length_var > 50 or sku_width_var > 9 or sku_height_var > 13):
            # Only correct if not too many items (packing gets harder with more items)
            if sku_counts <= 12:
                return 0
                
    # Additional FN correction: Moderate item count with good size distribution
    # Pattern: Balanced loading that SVM underestimates
    if svm_prob < 0.45 and sku_counts >= 8 and sku_counts <= 12:
        # Not too full, not too empty
        if fill_ratio > 0.5 and fill_ratio < 0.8:
            # Good size distribution (not too much variance)
            if (sku_length_var < 40 and sku_width_var < 7 and sku_height_var < 11):
                # Sufficient spare capacity
                if spare_cap > 10000:
                    return 1
                    
    # Additional FP correction: High aspect ratio with poor dimensional utilization
    # Pattern: Long/thin items that are hard to pack efficiently
    if svm_prob > 0.7 and max_asr > 5.5:
        if fill_ratio > 0.8:
            # Poor utilization in length dimension
            if l_ratio_avg < 0.35:
                return 0

    # New correction strategy: Focus on cases where SVM is very uncertain
    # FN pattern: Very low confidence but good physical indicators
    if svm_prob < 0.25:
        # Plenty of space left
        if spare_cap > 10000:
            # Not too many items
            if sku_counts <= 15:
                # Items are reasonably sized (not microscopic)
                if avg_vol > 2000:
                    return 1

    # FP pattern: High confidence but close to capacity with many items
    if svm_prob > 0.85 and fill_ratio > 0.88:
        # Many items make packing challenging
        if sku_counts > 10:
            return 0

    # FN correction: When there's substantial unused space and SVM is unsure
    if svm_prob < 0.35 and fill_ratio < 0.7:
        if spare_cap > 12000 and wl_total > 700:
            # Avoid very high variance cases that might actually be infeasible
            if (sku_length_var < 55 and sku_width_var < 10 and sku_height_var < 14):
                return 1

    # FP correction: High confidence but poor dimensional utilization metrics
    if svm_prob > 0.8 and fill_ratio > 0.85:
        # Check if utilization is poor in multiple dimensions
        poor_length_util = l_ratio_avg < 0.3
        poor_width_util = w_ratio_avg < 0.13
        poor_height_util = h_ratio_avg < 0.17
        
        if (poor_length_util and poor_width_util) or \
           (poor_length_util and poor_height_util) or \
           (poor_width_util and poor_height_util):
            return 0

    # Additional FN correction: Low fill ratio with good average volume and moderate count
    if fill_ratio < 0.65 and svm_prob < 0.4:
        if avg_vol > 2800 and sku_counts >= 7 and sku_counts <= 13:
            # Ensure items aren't too variable in size
            if sku_length_var < 50 and sku_width_var < 9 and sku_height_var < 13:
                return 1

    # Additional FP correction: High confidence with high maximum ASR and moderate fill
    if svm_prob > 0.8 and max_asr > 6.0 and fill_ratio > 0.75:
        # Especially problematic if also high variance
        if sku_length_var > 40 or sku_width_var > 7 or sku_height_var > 11:
            return 0

    # Additional FN correction: Moderate confidence with excellent space availability
    if svm_prob < 0.5 and spare_cap > 25000 and fill_ratio < 0.7:
        if sku_counts <= 10 and avg_vol > 3000:
            return 1

    # Additional FP correction: Very high confidence near full capacity
    if svm_prob > 0.9 and fill_ratio > 0.93:
        return 0

    # Additional FN correction: Low confidence with good WL metrics
    if svm_prob < 0.3 and wl_avg > 100 and wl_std < 50:
        if fill_ratio < 0.75 and spare_cap > 8000:
            return 1

    # Additional FP correction: High confidence with poor dimensional fit
    if svm_prob > 0.75:
        if (l_ratio_avg < 0.25 and w_ratio_avg < 0.12) or \
           (l_ratio_avg < 0.25 and h_ratio_avg < 0.15) or \
           (w_ratio_avg < 0.12 and h_ratio_avg < 0.15):
            if fill_ratio > 0.8:
                return 0

    # New FN correction: Very low SVM confidence with ample space and reasonable item count
    if svm_prob < 0.2 and spare_cap > 15000:
        if sku_counts >= 5 and sku_counts <= 16:
            # Items are not too small
            if avg_vol > 2200:
                return 1

    # New FP correction: High confidence with poor utilization and high variance
    if svm_prob > 0.85 and fill_ratio > 0.8:
        # Poor utilization in at least one dimension
        if l_ratio_avg < 0.27 or w_ratio_avg < 0.115 or h_ratio_avg < 0.165:
            # Combined with high variance makes it risky
            if sku_length_var > 40 or sku_width_var > 7.5 or sku_height_var > 11.5:
                return 0

    # New FN correction: Low confidence with good dimensional spread
    if svm_prob < 0.35 and fill_ratio < 0.75:
        # Good spread across dimensions (not concentrated in one)
        if (l_ratio_avg > 0.3 and w_ratio_avg > 0.12 and h_ratio_avg > 0.15):
            # And reasonable item count
            if sku_counts >= 6 and sku_counts <= 14:
                # With sufficient spare capacity
                if spare_cap > 10000:
                    return 1

    # New FP correction: Very high confidence with extreme item dimensions
    if svm_prob > 0.92:
        # Any item dimension close to vehicle limit is concerning
        if (sku_max_length > 0.85 * vehicle_length or
            sku_max_width > 0.85 * vehicle_width or
            sku_max_height > 0.85 * vehicle_height):
            return 0

    # New FN correction: Moderate confidence with low fill and good WL stats
    if svm_prob < 0.45 and fill_ratio < 0.68:
        if wl_avg > 95 and wl_std < 45:
            # And reasonable item characteristics
            if sku_counts >= 7 and sku_counts <= 13:
                if avg_vol > 2600:
                    return 1

    return -1
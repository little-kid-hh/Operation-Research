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
    spare_cap          = features.get('spare_capacity', 0)
    sku_counts         = features.get('sku_counts', 0)
    sku_length_var     = features.get('sku_length_var', 0)
    wl_to_vehicle_wl_std = features.get('wl_to_vehicle_wl_std', 0)
    max_asr            = features.get('max_asr', 1.0)
    l_to_L_ratio_max   = features.get('l_to_L_ratio_max', 0)
    sku_concentration  = features.get('sku_concentration', 0)
    h_to_H_ratio_std   = features.get('h_to_H_ratio_std', 0)
    l_to_L_ratio_std   = features.get('l_to_L_ratio_std', 0)
    
    # ─────────────────────────────────────────────────────────────────────
    # FP Correction: SVM overestimates feasibility (predicts YES) when
    # spare capacity is high but items are geometrically fragmented.
    # The linear model over-weights spare_capacity (+13.56) and under-penalizes
    # dimensional variance. High variance + extreme aspect/length ratios cause
    # unavoidable dead space that volume metrics miss.
    # Triggers on upper-tail fragmentation metrics to protect easy TPs.
    # ─────────────────────────────────────────────────────────────────────
    if svm_prob > 0.55 and spare_cap > 10000 and sku_counts <= 10:
        frag_signals = 0
        if sku_length_var > 48:      frag_signals += 1  # FP mean ~48.2, p75 ~53.7
        if wl_to_vehicle_wl_std > 40: frag_signals += 1  # p75 ~43.2
        if max_asr > 5.5:            frag_signals += 1  # p75 ~5.67
        if l_to_L_ratio_max > 0.57:  frag_signals += 1  # p75 ~0.583
        
        if frag_signals >= 2:
            return 0

    # ─────────────────────────────────────────────────────────────────────
    # FN Correction: SVM underestimates feasibility (predicts NO) due to
    # heavy penalties on sku_counts (-10.59) and sku_average_volume (-16.41).
    # However, highly uniform orders pack efficiently in dense grid/brick
    # formations, neutralizing the count/volume penalty.
    # Triggers on lower-tail variance and upper-tail concentration to target
    # the uniform packing regime, avoiding overrides on genuinely messy TNs.
    # ─────────────────────────────────────────────────────────────────────
    if svm_prob < 0.46 and 6500 < spare_cap < 13500 and 9 <= sku_counts <= 13:
        uniform_signals = 0
        if sku_length_var < 35:      uniform_signals += 1  # p25 ~34.8
        if wl_to_vehicle_wl_std < 31: uniform_signals += 1  # p25 ~30.7
        if h_to_H_ratio_std < 0.050: uniform_signals += 1  # p25 ~0.049
        if sku_concentration > 0.27: uniform_signals += 1  # median ~0.275
        if l_to_L_ratio_std < 0.10:  uniform_signals += 1  # p25 ~0.098
        
        if uniform_signals >= 3:
            return 1

    return -1
# -*- coding: utf-8 -*-
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    sku_counts   = features.get('sku_counts', 0)
    fill_ratio   = features.get('fill_ratio', 1.0)
    max_asr      = features.get('max_asr', 1.0)
    sku_avg_vol  = features.get('sku_average_volume', 0)
    vehicle_cap  = features.get('vehicle_capacity', 45000)
    total_vol    = sku_counts * sku_avg_vol

    # Correct FP: very high fill_ratio with many small items — likely infeasible
    if fill_ratio > 0.93 and sku_counts > 15:
        return 0

    # Correct FN: low fill_ratio but SVM is uncertain
    if fill_ratio < 0.70 and svm_prob < 0.60:
        return 1

    return -1

# HybridSVM Single-Feature Ablation

| feature | delta_auc | delta_tpr_at_fpr1pct | auc | tpr_at_fpr1pct | accuracy |
| --- | --- | --- | --- | --- | --- |
| tight_bin_large_piece_interaction | 0.00535637 | 0.0855457 | 0.970496 | 0.720256 | 0.9336 |
| volume_tail_ratio | 0.00270614 | 0.0452311 | 0.967846 | 0.679941 | 0.9284 |
| p90_long_over_bin_long | 0.00119323 | 0.00639135 | 0.966333 | 0.641101 | 0.9316 |
| multi_dim_tight_share | 0.000774389 | -0.00540806 | 0.965914 | 0.629302 | 0.9284 |
| p90_mid_over_bin_mid | 0.000362929 | 0.0103245 | 0.965503 | 0.645034 | 0.9284 |
| max_2d_pressure | 9.49523e-05 | -0.00786627 | 0.965235 | 0.626844 | 0.9284 |
| p90_short_over_bin_short | 4.95862e-05 | 0.000491642 | 0.965189 | 0.635202 | 0.9276 |
| count_near_long_limit | 0 | 0 | 0.96514 | 0.63471 | 0.9276 |
| count_near_height_limit | 0 | 0 | 0.96514 | 0.63471 | 0.9276 |
| count_near_width_limit | 0 | 0 | 0.96514 | 0.63471 | 0.9276 |
| max_elongation | -1.05503e-06 | 0 | 0.965139 | 0.63471 | 0.9276 |
| short_dim_sum_ratio | -2.63756e-05 | -0.00245821 | 0.965113 | 0.632252 | 0.928 |
| spare_per_item | -3.79809e-05 | 0.00688299 | 0.965102 | 0.641593 | 0.9284 |
| max_face_area_load_over_floor | -4.95862e-05 | 0.0231072 | 0.96509 | 0.657817 | 0.9284 |
| dominant_type_share | -9.17873e-05 | 0.0122911 | 0.965048 | 0.647001 | 0.9284 |

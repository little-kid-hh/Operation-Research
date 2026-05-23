You are writing a frozen guidance markdown for an iterative LLM-driven feature-engineering workflow.

Task:
- Read the structured analysis artifacts below.
- Write one concise but rigorous guidance document that will later be embedded into a feature-search prompt.
- The target workflow is: stronger teacher model analysis -> guidance markdown -> LLM generates new SVM features.

Requirements:
- Output only one markdown document.
- Keep it factual and reproducible.
- Distinguish clearly between:
  1. reproducible evidence source
  2. current evidence
  3. feature directions to test
  4. what not to over-invest in
  5. immediate workflow implication
- Do not invent metrics or files not present in the inputs.
- Treat the analysis as hypothesis support, not ground truth.
- Keep the tone operational, not promotional.
- Mention exact local artifact paths when they are part of reproducibility.
- Do not mention that an assistant or LLM wrote this document.
- Return the result in this format:

## GUIDANCE_MARKDOWN
```markdown
<full markdown>
```

Document target:
- route name: RF base40 -> GLM guidance -> linear SVM feature search
- output filename: RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md

Structured analysis summary JSON (already condensed to the key fields needed for guidance):
```json
{
  "teacher_scope": "base40_only",
  "split": {
    "test_size": 0.25,
    "random_state": 42
  },
  "data_path": "/Users/zhongxiaochuan/Operation-Research/FunSearch_test/training_2orientations.csv",
  "feature_set": "base40",
  "active_bank_used": false,
  "base40_xgb_cfg_source": null,
  "base40_xgb_cfg": null,
  "base40_rf_cfg_source": "/Users/zhongxiaochuan/Operation-Research/Ensemble_baseline/run_ensemble_ablation.py",
  "base40_rf_cfg": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 2
  },
  "metrics": {
    "svm_base40": {
      "accuracy": 0.9276,
      "precision": 0.9443645083932853,
      "recall": 0.9680432645034415,
      "auc": 0.9651398331370984,
      "tpr_at_fpr1pct": 0.6347099311701082,
      "threshold_at_fpr1pct": 0.9344146517873981,
      "tn": 350,
      "fp": 116,
      "fn": 65,
      "tp": 1969,
      "fpr": 0.24892703862660945,
      "fnr": 0.0319567354965585
    },
    "rf_base40": {
      "accuracy": 0.9436,
      "precision": 0.9504997620180866,
      "recall": 0.9818092428711898,
      "auc": 0.9822655415870122,
      "tpr_at_fpr1pct": 0.8058013765978368,
      "threshold_at_fpr1pct": 0.8874083433614119,
      "tn": 362,
      "fp": 104,
      "fn": 37,
      "tp": 1997,
      "fpr": 0.22317596566523606,
      "fnr": 0.018190757128810225
    }
  },
  "recovery_meta": {
    "n_test_rows": 2500,
    "n_svm_wrong_rf_right": 66,
    "n_svm_right_rf_wrong": 26
  },
  "top_base40_features": [
    {
      "feature": "spare_capacity",
      "importance": 0.3117022095244916
    },
    {
      "feature": "wl_to_vehicle_wl_total",
      "importance": 0.140903744656589
    },
    {
      "feature": "sku_counts",
      "importance": 0.06526080097684163
    },
    {
      "feature": "sku_average_volume",
      "importance": 0.06380506228442517
    },
    {
      "feature": "wl_to_vehicle_wl_max",
      "importance": 0.0344938866157653
    },
    {
      "feature": "h_to_H_ratio_avg",
      "importance": 0.03003505396418698
    },
    {
      "feature": "sku_height_avg",
      "importance": 0.02750391761350604
    },
    {
      "feature": "wl_to_vehicle_wl_std",
      "importance": 0.024661796098711778
    },
    {
      "feature": "wl_to_vehicle_wl_avg",
      "importance": 0.0231487613958353
    },
    {
      "feature": "sku_length_avg",
      "importance": 0.018991120369299837
    },
    {
      "feature": "l_to_L_ratio_avg",
      "importance": 0.017739585279912987
    },
    {
      "feature": "sku_concentration",
      "importance": 0.014493961703958143
    }
  ]
}
```

Additional notes:
```text
analysis_dir=/Users/zhongxiaochuan/Operation-Research/research/rf_guidance_base40_20260514
summary_path=/Users/zhongxiaochuan/Operation-Research/research/rf_guidance_base40_20260514/summary.json
The generated guidance will later be passed into HybridSVM/scripts/run_feature_search_agent.py via --tree-guidance-path.
This step must be reproducible: save prompt, raw response, and final markdown.
Only use the base40 teacher scope; do not leak any later active-bank features into the guidance.
```

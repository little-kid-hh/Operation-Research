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
- route name: xgb_guided_hybridsvm_base40
- output filename: XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md

Structured analysis summary JSON (already condensed to the key fields needed for guidance):
```json
{
  "teacher_scope": "base40_only",
  "split": {
    "test_size": 0.25,
    "random_state": 42
  },
  "data_path": "repro_bundle/data/training_2orientations.csv",
  "feature_set": "base40",
  "active_bank_used": false,
  "base40_xgb_cfg_source": "repro_bundle/reference/Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json",
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
    "xgb_base40": {
      "accuracy": 0.9532,
      "precision": 0.9637155297532656,
      "recall": 0.9793510324483776,
      "auc": 0.9867931853764965,
      "tpr_at_fpr1pct": 0.8289085545722714,
      "threshold_at_fpr1pct": 0.9913399815559387,
      "tn": 391,
      "fp": 75,
      "fn": 42,
      "tp": 1992,
      "fpr": 0.1609442060085837,
      "fnr": 0.02064896755162242
    }
  },
  "recovery_meta": {
    "n_test_rows": 2500,
    "n_svm_wrong_xgb_right": 92,
    "n_svm_right_xgb_wrong": 28
  },
  "top_base40_features": [
    {
      "feature": "spare_capacity",
      "gain": 16.896759033203125,
      "weight": 1860.0,
      "split_used": 1860
    },
    {
      "feature": "sku_average_volume",
      "gain": 4.758613109588623,
      "weight": 1306.0,
      "split_used": 1306
    },
    {
      "feature": "wl_to_vehicle_wl_total",
      "gain": 3.1910274028778076,
      "weight": 1058.0,
      "split_used": 1058
    },
    {
      "feature": "wl_to_vehicle_wl_max",
      "gain": 2.4920620918273926,
      "weight": 816.0,
      "split_used": 816
    },
    {
      "feature": "sku_counts",
      "gain": 2.2792465686798096,
      "weight": 99.0,
      "split_used": 99
    },
    {
      "feature": "l_to_L_ratio_std",
      "gain": 2.2692439556121826,
      "weight": 14.0,
      "split_used": 14
    },
    {
      "feature": "sku_max_width",
      "gain": 2.028628349304199,
      "weight": 184.0,
      "split_used": 184
    },
    {
      "feature": "h_to_H_ratio_max",
      "gain": 1.9987821578979492,
      "weight": 32.0,
      "split_used": 32
    },
    {
      "feature": "sku_concentration",
      "gain": 1.9354865550994873,
      "weight": 1073.0,
      "split_used": 1073
    },
    {
      "feature": "w_to_W_ratio_avg",
      "gain": 1.8583167791366577,
      "weight": 93.0,
      "split_used": 93
    },
    {
      "feature": "w_to_W_ratio_min",
      "gain": 1.8066898584365845,
      "weight": 23.0,
      "split_used": 23
    },
    {
      "feature": "sku_min_width",
      "gain": 1.7052764892578125,
      "weight": 194.0,
      "split_used": 194
    }
  ]
}
```

Additional notes:
```text
analysis_dir=repro_bundle/teacher_analysis/xgb_guidance_base40_20260514
summary_path=repro_bundle/teacher_analysis/xgb_guidance_base40_20260514/summary.json
The generated guidance will later be passed into code/HybridSVM/scripts/run_feature_search_agent.py via --tree-guidance-path.
This step must be reproducible: save prompt, raw response, and final markdown.
Only use the base40 teacher scope; do not leak any later active-bank features into the guidance.
```

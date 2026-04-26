# -*- coding: utf-8 -*-
"""Smoke test for HybridSVM pipeline."""
import sys, pathlib
sys.path.insert(0, '.')

import numpy as np
from sklearn.model_selection import train_test_split as tts

from src.svm_train import train_svm, load_raw_data, DROP_COLS
from src.hard_cases import (
    mine_hard_cases,
    summarize_for_evolution_prompt,
    render_hard_case_prompt,
)

data = pathlib.Path(r'C:\Operation Research\FunSearch_test\training_2orientations.csv')
pipeline, X_train, X_test, y_train, y_test = train_svm(
    data_csv=data,
    test_size=0.25,
    random_state=42,
    C=10.0,
)

decision_test = pipeline.model.decision_function(X_test)
svm_probs = 1.0 / (1.0 + np.exp(-np.clip(decision_test, -500, 500)))
svm_pred = (svm_probs >= 0.5).astype(int)

print(f'SVM Accuracy: {(svm_pred == y_test).mean():.4f}')
print(f'y_test dist: {np.bincount(y_test.astype(int))}')

df = load_raw_data(data)
cols_drop = [c for c in DROP_COLS if c in df.columns]
X_full = df.drop(columns=cols_drop)
X_full = X_full.drop(columns=['if_loaded'], errors='ignore')

idx = np.arange(len(df))
_, test_idx = tts(idx, test_size=0.25, random_state=42)
X_test_df = X_full.iloc[test_idx].reset_index(drop=True)
y_test_aligned = df['if_loaded'].values[test_idx]
disp_ids = df['发车号'].values[test_idx] if '发车号' in df.columns else None

fn_c, fp_c, easy_tn, easy_tp = mine_hard_cases(
    X_df=X_test_df,
    y_true=y_test_aligned,
    svm_probs=svm_probs,
    threshold=0.5,
    dispatch_ids=disp_ids,
)
print(f'FN={len(fn_c)}, FP={len(fp_c)}, EasyTN={len(easy_tn)}, EasyTP={len(easy_tp)}')

summary = summarize_for_evolution_prompt(
    fn_c,
    fp_c,
    easy_tn,
    easy_tp,
    y_test_aligned,
    list(X_test_df.columns),
    n_near_hard_easy=2,
)
md = render_hard_case_prompt(summary)
print(f'Prompt length: {len(md)} chars')
print('Sample:\n', md[:400])

print('Pipeline smoke test passed!')
# Real XGBoost Hyperparameter Search

- feature_set: `base40`
- data: `FunSearch_test/training_2orientations.csv`
- total rows: `10000`
- train rows: `7500`
- test rows: `2500`
- feature count: `40`
- n_trials: `24`

Files:
- `trials.csv`: all trials sorted by AUC / TPR@1% / Accuracy
- `summary.json`: full structured payload including every sampled config

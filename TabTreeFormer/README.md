# TabTreeFormer (RF-backed)

## 1. Background and goals

This project defines a TabTreeFormer-style baseline for tabular binary classification:

`x -> RF tree model -> leaf/bin tokens -> embedding -> transformer encoder -> MLP head -> prediction`

Goals:

- Keep a reproducible experiment path aligned with existing `HybridSVM` baselines.
- Reuse the current train/test split and metric protocol (`Accuracy`, `AUC`, `Recall`, `TPR@FPR=1%`).
- Add a new baseline mode `tabtreeformer_rf` without breaking existing `svm` and `ensemble`.

## 2. Architecture definition (project version)

The current version fixes tree model to `RandomForestClassifier` and uses an information-preserving token path.

- **Tree tokenizer**: per-sample, per-tree leaf id from `rf.apply(X)` (shape: `n_samples x n_trees`).
- **Leaf stats side-channel**: per `(tree, leaf)` statistics from train split:
  - `leaf_pos_rate`, `log1p(leaf_count)`, `leaf_depth_norm`, `leaf_impurity`.
- **Token mapping**: each tree has its own leaf-id vocabulary and a dedicated offset in global token ids.
- **Model body**: token embedding + tree-position embedding + leaf-stats projection + transformer encoder.
- **Head**: pooled representation + RF logit residual -> MLP -> binary logit.

Optional branch:

- Bin tokens for selected continuous features (`disabled` by default, interface reserved).

## 3. Data flow (train / inference)

### Train

1. Split data with the existing `run_experiment.py` split strategy.
2. Fit RF on training rows (`X_train_df`, `y_train`).
3. Convert RF leaves to token ids using training-fitted vocabularies.
4. Build leaf statistics table.
5. Train fused model with `BCEWithLogitsLoss`.
5. Save tokenizer state, RF model, network weights, and config into `models/`.

### Inference

1. Load saved RF tokenizer + Transformer model.
2. Tokenize input rows via RF leaves and lookup leaf stats.
3. Predict `P(y=1)` with RF residual fusion; convert to hard label with threshold 0.5.

## 4. RF leaf tokenization + metadata

Leaf tokenization details:

- RF apply output is integer leaf indices for each tree.
- For each tree `t`, build `leaf_to_local_id[t]` from training leaves.
- Use reserved id `0` for unknown leaves.
- Global token id:
  - `global_id = offset[t] + local_id`
  - offset ensures different trees do not collide.

Data interface:

- Input: pandas DataFrame with the same engineered feature columns as baseline runs.
- Output:
  - token ids: `np.ndarray[int64]` `[n_samples, n_trees]`
  - leaf stats: `np.ndarray[float32]` `[n_samples, n_trees, 4]`

## 5. Model structure (Embedding + Transformer + MLP)

Recommended default (v1):

- RF: `n_estimators=128`, `max_depth=12`, `min_samples_leaf=2`, `random_state=42`
- Transformer:
  - `d_model=96`
  - `nhead=4`
  - `num_layers=2`
  - `dim_feedforward=192`
  - `dropout=0.1`
- MLP head: hidden size `64`
- Pooling: mean pooling across tree-token sequence
- RF residual: RF logit concatenated before final head

## 6. Objective and metrics

- Loss: `BCEWithLogitsLoss`
- Primary metrics:
  - `Accuracy`
  - `AUC`
  - `Recall`
  - `TPR@FPR=1%`

## 7. Baseline comparison protocol

Compare on the same split and metrics:

1. `svm`
2. `ensemble` (stacking baseline)
3. `tabtreeformer_rf` (new baseline)

Output alignment:

- Stage 1 must always expose:
  - `baseline_probs_test`
  - `baseline_pred_test`

## 8. Runtime and configs

Planned CLI integration into `HybridSVM/run_experiment.py`:

- `--baseline-mode tabtreeformer_rf`
- Core RF/tokenizer options:
  - `--tabtree-rf-n-estimators`
  - `--tabtree-rf-max-depth`
  - `--tabtree-rf-min-samples-leaf`
- Core training options:
  - `--tabtree-epochs`
  - `--tabtree-batch-size`
  - `--tabtree-lr`
  - `--tabtree-weight-decay`
  - `--tabtree-d-model`
  - `--tabtree-nhead`
  - `--tabtree-n-layers`

## 9. Definition of done (DoD)

- README can guide reproduction from zero.
- `tabtreeformer_rf` runs end-to-end at least through Stage 1 + Stage 2.
- `results.json` logs tabtree config and baseline metadata.
- Metrics are directly comparable with `svm` and `ensemble`.

## 10. Risks and troubleshooting

- **Token sparsity**: RF leaves can be highly fragmented; reduce RF depth or increase `min_samples_leaf`.
- **Overfitting**: monitor train/val loss gap; increase dropout/weight decay, reduce model width.
- **Speed / memory**: reduce trees, batch size, or transformer width/layers.
- **Unknown leaves in inference**: handled by `UNK=0`, but too many unknowns indicate train/test shift.

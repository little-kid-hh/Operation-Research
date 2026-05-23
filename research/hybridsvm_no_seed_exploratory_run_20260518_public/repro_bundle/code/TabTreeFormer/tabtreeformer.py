# -*- coding: utf-8 -*-
"""TabTreeFormer tokenizer-only: RF leaf tokens + leaf stats + RF residual."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .tabtree_data import RFTreeTokenizer


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("TabTreeFormer requires PyTorch. Install with: pip install torch") from e


@dataclass
class TabTreeFormerConfig:
    rf_n_estimators: int = 128
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 2
    d_model: int = 96
    nhead: int = 4
    n_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1
    mlp_hidden: int = 64
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42
    validation_size: float = 0.15
    leaf_stats_smoothing: float = 1.0
    oof_rf_folds: int = 5
    correction_scale: float = 1.0
    refit_full_after_validation: bool = True
    use_raw_features: bool = True
    raw_hidden: int = 64
    threshold_target_fpr: float = 0.01


def _config_from_dict(data: dict[str, Any]) -> TabTreeFormerConfig:
    """Load configs written by older experiments without failing on extra keys."""
    valid = set(TabTreeFormerConfig.__dataclass_fields__)
    return TabTreeFormerConfig(**{k: v for k, v in data.items() if k in valid})


def _make_rf(cfg: TabTreeFormerConfig) -> RandomForestClassifier:
    max_depth = None if cfg.rf_max_depth is None else int(cfg.rf_max_depth)
    return RandomForestClassifier(
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
        n_jobs=-1,
        random_state=int(cfg.random_state),
    )


def _prob_to_logit(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    return np.log(np.clip(p, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p, 1e-6, 1.0)).astype(
        np.float32
    )


def _rf_oof_logits(
    X_df: pd.DataFrame,
    y: np.ndarray,
    cfg: TabTreeFormerConfig,
) -> np.ndarray:
    """OOF RF logits for training rows, avoiding in-sample RF confidence leakage."""
    y_arr = np.asarray(y, dtype=np.int64).ravel()
    class_counts = np.bincount(y_arr, minlength=2)
    n_splits = min(int(cfg.oof_rf_folds), int(class_counts.min()))
    if n_splits < 2:
        rf = _make_rf(cfg)
        rf.fit(X_df.values, y_arr)
        return _prob_to_logit(rf.predict_proba(X_df.values)[:, 1])

    probs = np.zeros(len(y_arr), dtype=np.float64)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(cfg.random_state),
    )
    for tr_idx, va_idx in skf.split(X_df.values, y_arr):
        rf = _make_rf(cfg)
        rf.fit(X_df.iloc[tr_idx].values, y_arr[tr_idx])
        probs[va_idx] = rf.predict_proba(X_df.iloc[va_idx].values)[:, 1]
    return _prob_to_logit(probs)


def _calibrate_threshold_at_fpr(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_fpr: float,
) -> tuple[float, dict[str, float]]:
    """Pick the threshold with the best TPR subject to FPR <= target_fpr."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return 0.5, {
            "target_fpr": float(target_fpr),
            "actual_fpr": 0.0,
            "tpr": 0.0,
            "source": "fallback_single_class",
        }

    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.where(fpr <= float(target_fpr))[0]
    if len(valid) == 0:
        idx = int(np.argmin(fpr))
    else:
        idx = int(valid[np.argmax(tpr[valid])])
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(np.max(finite)) if len(finite) else 1.0
    return threshold, {
        "target_fpr": float(target_fpr),
        "actual_fpr": float(fpr[idx]),
        "tpr": float(tpr[idx]),
        "source": "inner_validation",
    }


def _build_model(
    vocab_size: int,
    n_trees: int,
    leaf_stats_dim: int,
    raw_dim: int,
    cfg: TabTreeFormerConfig,
):
    torch, nn, _, _ = _lazy_torch()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(vocab_size, cfg.d_model)
            self.pos = nn.Embedding(n_trees, cfg.d_model)
            self.leaf_stats_proj = nn.Sequential(
                nn.Linear(leaf_stats_dim, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.raw = None
            raw_out_dim = 0
            if cfg.use_raw_features and raw_dim > 0:
                raw_out_dim = int(cfg.raw_hidden)
                self.raw = nn.Sequential(
                    nn.Linear(raw_dim, raw_out_dim),
                    nn.LayerNorm(raw_out_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(raw_out_dim, raw_out_dim),
                    nn.GELU(),
                )
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model + raw_out_dim + 1, cfg.mlp_hidden),  # +1 for RF residual logit
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.mlp_hidden, 1),
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        def forward(self, x_tok, x_leaf_stats, x_rf_logit, x_raw=None):
            bsz, seq = x_tok.shape
            pos_ids = torch.arange(seq, device=x_tok.device).unsqueeze(0).expand(bsz, seq)
            x = self.tok(x_tok) + self.pos(pos_ids) + self.leaf_stats_proj(x_leaf_stats)
            x = self.enc(x)
            pooled = x.mean(dim=1)
            parts = [pooled]
            if self.raw is not None and x_raw is not None:
                parts.append(self.raw(x_raw))
            parts.append(x_rf_logit.unsqueeze(1))
            z = torch.cat(parts, dim=1)
            correction = self.head(z).squeeze(-1)
            return x_rf_logit + float(cfg.correction_scale) * correction

    return _Model()


@dataclass
class TabTreeFormerPipeline:
    rf_model: RandomForestClassifier
    tokenizer: RFTreeTokenizer
    config: TabTreeFormerConfig
    feature_names: list[str] = field(default_factory=list)
    raw_scaler: StandardScaler | None = None
    torch_state: dict[str, Any] = field(default_factory=dict)
    calibrated_threshold: float | None = None
    calibration_info: dict[str, float] = field(default_factory=dict)

    def _tokens(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.tokenizer.transform(self.rf_model, X_df[self.feature_names])

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        torch, _, DataLoader, TensorDataset = _lazy_torch()
        raw = self._raw_features(X_df)
        model = _build_model(
            self.tokenizer.vocab_size,
            self.tokenizer.n_trees,
            self.tokenizer.leaf_stats_dim,
            raw.shape[1],
            self.config,
        )
        model.load_state_dict(self.torch_state)
        model.eval()

        toks = self._tokens(X_df).astype(np.int64)
        leaf_stats = self.tokenizer.lookup_leaf_stats(toks).astype(np.float32)
        rf_prob = self.rf_model.predict_proba(X_df[self.feature_names].values)[:, 1]
        rf_logit = np.log(np.clip(rf_prob, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - rf_prob, 1e-6, 1.0)).astype(
            np.float32
        )
        ds = TensorDataset(
            torch.from_numpy(toks),
            torch.from_numpy(leaf_stats),
            torch.from_numpy(rf_logit),
            torch.from_numpy(raw),
        )
        loader = DataLoader(ds, batch_size=max(256, int(self.config.batch_size)), shuffle=False)
        probs = []
        with torch.no_grad():
            for xb_tok, xb_stats, xb_rf_logit, xb_raw in loader:
                logits = model(xb_tok, xb_stats, xb_rf_logit, xb_raw)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs).astype(np.float64)

    def _raw_features(self, X_df: pd.DataFrame) -> np.ndarray:
        if not self.config.use_raw_features or self.raw_scaler is None:
            return np.zeros((len(X_df), 0), dtype=np.float32)
        return self.raw_scaler.transform(X_df[self.feature_names].values).astype(np.float32)

    def decision_threshold(self) -> float:
        return float(self.calibrated_threshold) if self.calibrated_threshold is not None else 0.5

    def predict(self, X_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        t = float(threshold)
        return (self.predict_proba(X_df) >= t).astype(int)

    def save(self, path: pathlib.Path) -> None:
        info = {
            "schema_version": 3,
            "feature_names": self.feature_names,
            "config": self.config.__dict__,
            "calibrated_threshold": self.calibrated_threshold,
            "calibration_info": self.calibration_info,
            "tokenizer": {
                "feature_names": self.tokenizer.feature_names,
                "n_trees": self.tokenizer.n_trees,
                "leaf_to_local": self.tokenizer.leaf_to_local,
                "offsets": self.tokenizer.offsets,
                "vocab_size": self.tokenizer.vocab_size,
                "leaf_stats_dim": self.tokenizer.leaf_stats_dim,
                "token_leaf_stats": self.tokenizer.token_leaf_stats.tolist(),
                "token_leaf_counts": (
                    self.tokenizer.token_leaf_counts.tolist()
                    if self.tokenizer.token_leaf_counts is not None
                    else None
                ),
                "token_leaf_pos_sums": (
                    self.tokenizer.token_leaf_pos_sums.tolist()
                    if self.tokenizer.token_leaf_pos_sums is not None
                    else None
                ),
                "global_pos_rate": self.tokenizer.global_pos_rate,
            },
        }
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        joblib.dump(self.rf_model, path.with_suffix(".rf.joblib"))
        if self.raw_scaler is not None:
            joblib.dump(self.raw_scaler, path.with_suffix(".raw_scaler.joblib"))
        torch, _, _, _ = _lazy_torch()
        torch.save(self.torch_state, path.with_suffix(".pt"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "TabTreeFormerPipeline":
        meta = json.loads(path.read_text(encoding="utf-8"))
        rf_model = joblib.load(path.with_suffix(".rf.joblib"))
        cfg = _config_from_dict(meta["config"])
        raw_scaler_path = path.with_suffix(".raw_scaler.joblib")
        raw_scaler = joblib.load(raw_scaler_path) if raw_scaler_path.exists() else None
        if raw_scaler is None:
            cfg.use_raw_features = False
        torch, _, _, _ = _lazy_torch()
        torch_state = torch.load(path.with_suffix(".pt"), map_location="cpu")
        tok_meta = meta["tokenizer"]
        tokenizer = RFTreeTokenizer(
            feature_names=list(tok_meta["feature_names"]),
            n_trees=int(tok_meta["n_trees"]),
            leaf_to_local=[{int(k): int(v) for k, v in d.items()} for d in tok_meta["leaf_to_local"]],
            offsets=[int(x) for x in tok_meta["offsets"]],
            vocab_size=int(tok_meta["vocab_size"]),
            leaf_stats_dim=int(tok_meta["leaf_stats_dim"]),
            token_leaf_stats=np.asarray(tok_meta["token_leaf_stats"], dtype=np.float32),
            token_leaf_counts=(
                np.asarray(tok_meta["token_leaf_counts"], dtype=np.float32)
                if tok_meta.get("token_leaf_counts") is not None
                else None
            ),
            token_leaf_pos_sums=(
                np.asarray(tok_meta["token_leaf_pos_sums"], dtype=np.float32)
                if tok_meta.get("token_leaf_pos_sums") is not None
                else None
            ),
            global_pos_rate=float(tok_meta.get("global_pos_rate", 0.5)),
        )
        return cls(
            rf_model=rf_model,
            tokenizer=tokenizer,
            config=cfg,
            feature_names=list(meta["feature_names"]),
            raw_scaler=raw_scaler,
            torch_state=torch_state,
            calibrated_threshold=meta.get("calibrated_threshold"),
            calibration_info=dict(meta.get("calibration_info", {})),
        )


def fit_tabtreeformer_pipeline(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    config: TabTreeFormerConfig | None = None,
) -> TabTreeFormerPipeline:
    cfg = config or TabTreeFormerConfig()
    y = np.asarray(y_train, dtype=np.int64).ravel()
    feature_names = list(X_train_df.columns)

    tr_idx, va_idx = train_test_split(
        np.arange(len(y)),
        test_size=float(cfg.validation_size),
        random_state=int(cfg.random_state),
        stratify=y,
    )

    torch, nn, DataLoader, TensorDataset = _lazy_torch()
    torch.manual_seed(int(cfg.random_state))

    def fit_raw_scaler(X_df: pd.DataFrame) -> StandardScaler | None:
        if not bool(cfg.use_raw_features):
            return None
        scaler = StandardScaler()
        scaler.fit(X_df[feature_names].values)
        return scaler

    def raw_features(raw_scaler: StandardScaler | None, X_df: pd.DataFrame) -> np.ndarray:
        if raw_scaler is None:
            return np.zeros((len(X_df), 0), dtype=np.float32)
        return raw_scaler.transform(X_df[feature_names].values).astype(np.float32)

    def build_inputs(
        rf: RandomForestClassifier,
        tokenizer: RFTreeTokenizer,
        raw_scaler: StandardScaler | None,
        X_df: pd.DataFrame,
        y_arr: np.ndarray,
        *,
        training_rows: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tokens = tokenizer.transform(rf, X_df).astype(np.int64)
        if training_rows:
            leaf_stats = tokenizer.lookup_leaf_stats_leave_one_out(
                tokens,
                y_arr,
                smoothing=float(cfg.leaf_stats_smoothing),
            )
            rf_logit = _rf_oof_logits(X_df, y_arr, cfg)
        else:
            leaf_stats = tokenizer.lookup_leaf_stats(tokens).astype(np.float32)
            rf_logit = _prob_to_logit(rf.predict_proba(X_df.values)[:, 1])
        raw = raw_features(raw_scaler, X_df)
        return tokens, leaf_stats.astype(np.float32), rf_logit.astype(np.float32), raw

    def train_model(
        tokenizer: RFTreeTokenizer,
        train_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        val_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        epochs: int,
    ) -> tuple[dict[str, Any], int]:
        raw_dim = int(train_inputs[3].shape[1])
        model = _build_model(
            tokenizer.vocab_size,
            tokenizer.n_trees,
            tokenizer.leaf_stats_dim,
            raw_dim,
            cfg,
        )
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        tr_tokens, tr_stats, tr_logit, tr_raw, tr_y = train_inputs
        ds_tr = TensorDataset(
            torch.from_numpy(tr_tokens),
            torch.from_numpy(tr_stats),
            torch.from_numpy(tr_logit),
            torch.from_numpy(tr_raw),
            torch.from_numpy(tr_y.astype(np.float32)),
        )
        tr_loader = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=True)

        va_loader = None
        if val_inputs is not None:
            va_tokens, va_stats, va_logit, va_raw, va_y = val_inputs
            ds_va = TensorDataset(
                torch.from_numpy(va_tokens),
                torch.from_numpy(va_stats),
                torch.from_numpy(va_logit),
                torch.from_numpy(va_raw),
                torch.from_numpy(va_y.astype(np.float32)),
            )
            va_loader = DataLoader(ds_va, batch_size=max(256, int(cfg.batch_size)), shuffle=False)

        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(max(1, int(epochs))):
            model.train()
            for xb_tok, xb_stats, xb_rf_logit, xb_raw, yb in tr_loader:
                optim.zero_grad()
                logits = model(xb_tok, xb_stats, xb_rf_logit, xb_raw)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

            if va_loader is None:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                continue

            model.eval()
            va_loss = 0.0
            va_n = 0
            with torch.no_grad():
                for xb_tok, xb_stats, xb_rf_logit, xb_raw, yb in va_loader:
                    logits = model(xb_tok, xb_stats, xb_rf_logit, xb_raw)
                    l = loss_fn(logits, yb)
                    va_loss += float(l.item()) * len(yb)
                    va_n += len(yb)
            va_loss = va_loss / max(1, va_n)
            if va_loss < best_loss:
                best_loss = va_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = max(1, int(epochs))
        return best_state, best_epoch

    def predict_with_state(
        tokenizer: RFTreeTokenizer,
        state: dict[str, Any],
        inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        tokens, stats, rf_logit, raw = inputs
        model = _build_model(
            tokenizer.vocab_size,
            tokenizer.n_trees,
            tokenizer.leaf_stats_dim,
            int(raw.shape[1]),
            cfg,
        )
        model.load_state_dict(state)
        model.eval()
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(tokens),
                torch.from_numpy(stats),
                torch.from_numpy(rf_logit),
                torch.from_numpy(raw),
            ),
            batch_size=max(256, int(cfg.batch_size)),
            shuffle=False,
        )
        probs = []
        with torch.no_grad():
            for xb_tok, xb_stats, xb_rf_logit, xb_raw in loader:
                logits = model(xb_tok, xb_stats, xb_rf_logit, xb_raw)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs).astype(np.float64)

    # Inner validation uses only inner-train labels for the RF tokenizer/statistics.
    rf_inner = _make_rf(cfg)
    rf_inner.fit(X_train_df.iloc[tr_idx][feature_names].values, y[tr_idx])
    tokenizer_inner = RFTreeTokenizer.fit(
        rf_inner,
        X_train_df.iloc[tr_idx][feature_names],
        y[tr_idx],
    )
    raw_scaler_inner = fit_raw_scaler(X_train_df.iloc[tr_idx][feature_names])
    tr_tokens, tr_stats, tr_logit, tr_raw = build_inputs(
        rf_inner,
        tokenizer_inner,
        raw_scaler_inner,
        X_train_df.iloc[tr_idx][feature_names],
        y[tr_idx],
        training_rows=True,
    )
    va_tokens = tokenizer_inner.transform(
        rf_inner,
        X_train_df.iloc[va_idx][feature_names],
    ).astype(np.int64)
    va_stats = tokenizer_inner.lookup_leaf_stats(va_tokens).astype(np.float32)
    va_logit = _prob_to_logit(
        rf_inner.predict_proba(X_train_df.iloc[va_idx][feature_names].values)[:, 1]
    )
    va_raw = raw_features(raw_scaler_inner, X_train_df.iloc[va_idx][feature_names])

    inner_state, best_epoch = train_model(
        tokenizer_inner,
        (tr_tokens, tr_stats, tr_logit, tr_raw, y[tr_idx]),
        (va_tokens, va_stats, va_logit, va_raw, y[va_idx]),
        int(cfg.epochs),
    )
    va_probs = predict_with_state(
        tokenizer_inner,
        inner_state,
        (va_tokens, va_stats, va_logit, va_raw),
    )
    calibrated_threshold, calibration_info = _calibrate_threshold_at_fpr(
        y[va_idx],
        va_probs,
        float(cfg.threshold_target_fpr),
    )

    if not bool(cfg.refit_full_after_validation):
        return TabTreeFormerPipeline(
            rf_model=rf_inner,
            tokenizer=tokenizer_inner,
            config=cfg,
            feature_names=feature_names,
            raw_scaler=raw_scaler_inner,
            torch_state=inner_state,
            calibrated_threshold=calibrated_threshold,
            calibration_info=calibration_info,
        )

    # Final artifact uses all training rows, but target-derived train features are
    # leave-one-out/OOF so the neural head does not see each row's own label.
    rf = _make_rf(cfg)
    rf.fit(X_train_df[feature_names].values, y)
    tokenizer = RFTreeTokenizer.fit(rf, X_train_df[feature_names], y)
    raw_scaler = fit_raw_scaler(X_train_df[feature_names])
    tokens_all, leaf_stats_all, rf_logit_all, raw_all = build_inputs(
        rf,
        tokenizer,
        raw_scaler,
        X_train_df[feature_names],
        y,
        training_rows=True,
    )

    best_state, _ = train_model(
        tokenizer,
        (tokens_all, leaf_stats_all, rf_logit_all, raw_all, y),
        None,
        best_epoch,
    )

    return TabTreeFormerPipeline(
        rf_model=rf,
        tokenizer=tokenizer,
        config=cfg,
        feature_names=feature_names,
        raw_scaler=raw_scaler,
        torch_state=best_state,
        calibrated_threshold=calibrated_threshold,
        calibration_info=calibration_info,
    )


def load_tabtreeformer_pipeline(model_path: pathlib.Path) -> TabTreeFormerPipeline:
    return TabTreeFormerPipeline.load(model_path)

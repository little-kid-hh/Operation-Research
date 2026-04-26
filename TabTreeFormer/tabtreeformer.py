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
from sklearn.model_selection import train_test_split

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


def _build_model(
    vocab_size: int,
    n_trees: int,
    leaf_stats_dim: int,
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
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model + 1, cfg.mlp_hidden),  # +1 for RF residual logit
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.mlp_hidden, 1),
            )

        def forward(self, x_tok, x_leaf_stats, x_rf_logit):
            bsz, seq = x_tok.shape
            pos_ids = torch.arange(seq, device=x_tok.device).unsqueeze(0).expand(bsz, seq)
            x = self.tok(x_tok) + self.pos(pos_ids) + self.leaf_stats_proj(x_leaf_stats)
            x = self.enc(x)
            pooled = x.mean(dim=1)
            z = torch.cat([pooled, x_rf_logit.unsqueeze(1)], dim=1)
            return self.head(z).squeeze(-1)

    return _Model()


@dataclass
class TabTreeFormerPipeline:
    rf_model: RandomForestClassifier
    tokenizer: RFTreeTokenizer
    config: TabTreeFormerConfig
    feature_names: list[str] = field(default_factory=list)
    torch_state: dict[str, Any] = field(default_factory=dict)

    def _tokens(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.tokenizer.transform(self.rf_model, X_df[self.feature_names])

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        torch, _, DataLoader, TensorDataset = _lazy_torch()
        model = _build_model(
            self.tokenizer.vocab_size,
            self.tokenizer.n_trees,
            self.tokenizer.leaf_stats_dim,
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
        )
        loader = DataLoader(ds, batch_size=max(256, int(self.config.batch_size)), shuffle=False)
        probs = []
        with torch.no_grad():
            for xb_tok, xb_stats, xb_rf_logit in loader:
                logits = model(xb_tok, xb_stats, xb_rf_logit)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs).astype(np.float64)

    def predict(self, X_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_df) >= threshold).astype(int)

    def save(self, path: pathlib.Path) -> None:
        info = {
            "feature_names": self.feature_names,
            "config": self.config.__dict__,
            "tokenizer": {
                "feature_names": self.tokenizer.feature_names,
                "n_trees": self.tokenizer.n_trees,
                "leaf_to_local": self.tokenizer.leaf_to_local,
                "offsets": self.tokenizer.offsets,
                "vocab_size": self.tokenizer.vocab_size,
                "leaf_stats_dim": self.tokenizer.leaf_stats_dim,
                "token_leaf_stats": self.tokenizer.token_leaf_stats.tolist(),
            },
        }
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        joblib.dump(self.rf_model, path.with_suffix(".rf.joblib"))
        torch, _, _, _ = _lazy_torch()
        torch.save(self.torch_state, path.with_suffix(".pt"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "TabTreeFormerPipeline":
        meta = json.loads(path.read_text(encoding="utf-8"))
        rf_model = joblib.load(path.with_suffix(".rf.joblib"))
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
        )
        return cls(
            rf_model=rf_model,
            tokenizer=tokenizer,
            config=TabTreeFormerConfig(**meta["config"]),
            feature_names=list(meta["feature_names"]),
            torch_state=torch_state,
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

    rf = RandomForestClassifier(
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=int(cfg.rf_max_depth),
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
        n_jobs=-1,
        random_state=int(cfg.random_state),
    )
    rf.fit(X_train_df[feature_names].values, y)

    tokenizer = RFTreeTokenizer.fit(rf, X_train_df[feature_names], y)
    tokens_all = tokenizer.transform(rf, X_train_df[feature_names]).astype(np.int64)
    leaf_stats_all = tokenizer.lookup_leaf_stats(tokens_all).astype(np.float32)
    rf_prob_all = rf.predict_proba(X_train_df[feature_names].values)[:, 1]
    rf_logit_all = np.log(
        np.clip(rf_prob_all, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - rf_prob_all, 1e-6, 1.0)
    ).astype(np.float32)

    tr_idx, va_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.15,
        random_state=int(cfg.random_state),
        stratify=y,
    )

    torch, nn, DataLoader, TensorDataset = _lazy_torch()
    model = _build_model(
        tokenizer.vocab_size,
        tokenizer.n_trees,
        tokenizer.leaf_stats_dim,
        cfg,
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    ds_tr = TensorDataset(
        torch.from_numpy(tokens_all[tr_idx]),
        torch.from_numpy(leaf_stats_all[tr_idx]),
        torch.from_numpy(rf_logit_all[tr_idx]),
        torch.from_numpy(y[tr_idx].astype(np.float32)),
    )
    ds_va = TensorDataset(
        torch.from_numpy(tokens_all[va_idx]),
        torch.from_numpy(leaf_stats_all[va_idx]),
        torch.from_numpy(rf_logit_all[va_idx]),
        torch.from_numpy(y[va_idx].astype(np.float32)),
    )
    tr_loader = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=max(256, int(cfg.batch_size)), shuffle=False)

    best_state = None
    best_loss = float("inf")
    for _ in range(int(cfg.epochs)):
        model.train()
        for xb_tok, xb_stats, xb_rf_logit, yb in tr_loader:
            optim.zero_grad()
            logits = model(xb_tok, xb_stats, xb_rf_logit)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb_tok, xb_stats, xb_rf_logit, yb in va_loader:
                logits = model(xb_tok, xb_stats, xb_rf_logit)
                l = loss_fn(logits, yb)
                va_loss += float(l.item()) * len(yb)
                va_n += len(yb)
        va_loss = va_loss / max(1, va_n)
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return TabTreeFormerPipeline(
        rf_model=rf,
        tokenizer=tokenizer,
        config=cfg,
        feature_names=feature_names,
        torch_state=best_state,
    )


def load_tabtreeformer_pipeline(model_path: pathlib.Path) -> TabTreeFormerPipeline:
    return TabTreeFormerPipeline.load(model_path)


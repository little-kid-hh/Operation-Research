# -*- coding: utf-8 -*-
"""
RF leaf-tokenization helpers for TabTreeFormer baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RFTreeTokenizer:
    """
    Convert RF leaf indices into dense token ids.

    Each tree has:
      - local ids (1..K) for known leaves from training
      - 0 reserved for unknown leaves at inference
    Global ids are offset per tree so tokens from different trees do not collide.
    """

    feature_names: list[str]
    n_trees: int
    leaf_to_local: list[dict[int, int]]
    offsets: list[int]
    vocab_size: int
    leaf_stats_dim: int
    token_leaf_stats: np.ndarray

    @classmethod
    def fit(
        cls,
        rf: RandomForestClassifier,
        X_train_df: pd.DataFrame,
        y_train: np.ndarray,
    ) -> "RFTreeTokenizer":
        feature_names = list(X_train_df.columns)
        leaves = rf.apply(X_train_df[feature_names].values).astype(np.int64)
        y = np.asarray(y_train, dtype=np.float64).ravel()
        n_trees = int(leaves.shape[1])
        leaf_to_local: list[dict[int, int]] = []
        offsets: list[int] = []
        stats_blocks: list[np.ndarray] = []
        running = 0

        # Per-tree node depth lookup for leaf statistics.
        tree_leaf_depth_maps: list[dict[int, float]] = []
        tree_leaf_imp_maps: list[dict[int, float]] = []
        for est in rf.estimators_:
            t = est.tree_
            children_left = t.children_left
            children_right = t.children_right
            impurity = t.impurity
            depth = np.zeros(t.node_count, dtype=np.int32)
            stack = [0]
            while stack:
                n = stack.pop()
                l = children_left[n]
                r = children_right[n]
                if l != -1:
                    depth[l] = depth[n] + 1
                    stack.append(l)
                if r != -1:
                    depth[r] = depth[n] + 1
                    stack.append(r)
            max_depth = float(max(1, depth.max()))
            leaf_depth = {}
            leaf_imp = {}
            for n in range(t.node_count):
                if children_left[n] == -1 and children_right[n] == -1:
                    leaf_depth[int(n)] = float(depth[n] / max_depth)
                    leaf_imp[int(n)] = float(impurity[n])
            tree_leaf_depth_maps.append(leaf_depth)
            tree_leaf_imp_maps.append(leaf_imp)

        for t in range(n_trees):
            uniq = sorted(int(x) for x in np.unique(leaves[:, t]).tolist())
            mapping = {leaf_id: i + 1 for i, leaf_id in enumerate(uniq)}
            leaf_to_local.append(mapping)
            offsets.append(running)
            # stats columns: pos_rate, log_count, depth_norm, impurity
            block = np.zeros((len(mapping) + 1, 4), dtype=np.float32)
            block[0, 0] = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
            block[0, 1] = 0.0
            block[0, 2] = 0.0
            block[0, 3] = 0.0
            for leaf_id, local_id in mapping.items():
                mask = leaves[:, t] == leaf_id
                n = int(mask.sum())
                p = float(y[mask].mean()) if n > 0 else float(y.mean())
                block[local_id, 0] = float(np.clip(p, 1e-4, 1.0 - 1e-4))
                block[local_id, 1] = float(np.log1p(n))
                block[local_id, 2] = float(tree_leaf_depth_maps[t].get(int(leaf_id), 0.0))
                block[local_id, 3] = float(tree_leaf_imp_maps[t].get(int(leaf_id), 0.0))
            stats_blocks.append(block)
            running += len(mapping) + 1

        token_leaf_stats = np.concatenate(stats_blocks, axis=0).astype(np.float32)
        return cls(
            feature_names=feature_names,
            n_trees=n_trees,
            leaf_to_local=leaf_to_local,
            offsets=offsets,
            vocab_size=running,
            leaf_stats_dim=4,
            token_leaf_stats=token_leaf_stats,
        )

    def transform(self, rf: RandomForestClassifier, X_df: pd.DataFrame) -> np.ndarray:
        X = X_df[self.feature_names].values
        leaves = rf.apply(X).astype(np.int64)
        if leaves.shape[1] != self.n_trees:
            raise ValueError(f"Unexpected tree count: got {leaves.shape[1]}, expected {self.n_trees}")
        out = np.zeros_like(leaves, dtype=np.int64)
        for t in range(self.n_trees):
            mapping = self.leaf_to_local[t]
            off = int(self.offsets[t])
            col = leaves[:, t]
            out[:, t] = np.array([off + mapping.get(int(v), 0) for v in col], dtype=np.int64)
        return out

    def lookup_leaf_stats(self, token_ids: np.ndarray) -> np.ndarray:
        return self.token_leaf_stats[token_ids]


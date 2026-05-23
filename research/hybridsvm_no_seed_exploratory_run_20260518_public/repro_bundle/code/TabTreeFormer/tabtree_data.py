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
    token_leaf_counts: np.ndarray | None = None
    token_leaf_pos_sums: np.ndarray | None = None
    global_pos_rate: float = 0.5

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
        count_blocks: list[np.ndarray] = []
        pos_sum_blocks: list[np.ndarray] = []
        running = 0
        global_pos_rate = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))

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
            counts = np.zeros(len(mapping) + 1, dtype=np.float32)
            pos_sums = np.zeros(len(mapping) + 1, dtype=np.float32)
            block[0, 0] = global_pos_rate
            block[0, 1] = 0.0
            block[0, 2] = 0.0
            block[0, 3] = 0.0
            for leaf_id, local_id in mapping.items():
                mask = leaves[:, t] == leaf_id
                n = int(mask.sum())
                pos = float(y[mask].sum()) if n > 0 else 0.0
                p = float(pos / n) if n > 0 else global_pos_rate
                counts[local_id] = float(n)
                pos_sums[local_id] = float(pos)
                block[local_id, 0] = float(np.clip(p, 1e-4, 1.0 - 1e-4))
                block[local_id, 1] = float(np.log1p(n))
                block[local_id, 2] = float(tree_leaf_depth_maps[t].get(int(leaf_id), 0.0))
                block[local_id, 3] = float(tree_leaf_imp_maps[t].get(int(leaf_id), 0.0))
            stats_blocks.append(block)
            count_blocks.append(counts)
            pos_sum_blocks.append(pos_sums)
            running += len(mapping) + 1

        token_leaf_stats = np.concatenate(stats_blocks, axis=0).astype(np.float32)
        token_leaf_counts = np.concatenate(count_blocks, axis=0).astype(np.float32)
        token_leaf_pos_sums = np.concatenate(pos_sum_blocks, axis=0).astype(np.float32)
        return cls(
            feature_names=feature_names,
            n_trees=n_trees,
            leaf_to_local=leaf_to_local,
            offsets=offsets,
            vocab_size=running,
            leaf_stats_dim=4,
            token_leaf_stats=token_leaf_stats,
            token_leaf_counts=token_leaf_counts,
            token_leaf_pos_sums=token_leaf_pos_sums,
            global_pos_rate=global_pos_rate,
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

    def lookup_leaf_stats_leave_one_out(
        self,
        token_ids: np.ndarray,
        y: np.ndarray,
        smoothing: float = 1.0,
    ) -> np.ndarray:
        """
        Lookup leaf stats for training rows without feeding a row's own label back
        into its ``leaf_pos_rate`` feature.

        Only the target-derived columns (positive rate and count) are adjusted;
        structural columns (depth, impurity) are copied unchanged.
        """
        stats = self.lookup_leaf_stats(token_ids).astype(np.float32, copy=True)
        if self.token_leaf_counts is None or self.token_leaf_pos_sums is None:
            return stats

        labels = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if labels.shape[0] != token_ids.shape[0]:
            raise ValueError(
                f"y length {labels.shape[0]} does not match token rows {token_ids.shape[0]}"
            )

        counts = self.token_leaf_counts[token_ids].astype(np.float32)
        pos_sums = self.token_leaf_pos_sums[token_ids].astype(np.float32)
        loo_counts = np.maximum(counts - 1.0, 0.0)
        loo_pos = np.maximum(pos_sums - labels, 0.0)

        alpha = float(max(smoothing, 0.0))
        if alpha > 0.0:
            rate = (loo_pos + alpha * float(self.global_pos_rate)) / (loo_counts + alpha)
        else:
            rate = np.divide(
                loo_pos,
                np.maximum(loo_counts, 1.0),
                out=np.full_like(loo_pos, float(self.global_pos_rate)),
                where=loo_counts > 0,
            )
        stats[:, :, 0] = np.clip(rate, 1e-4, 1.0 - 1e-4)
        stats[:, :, 1] = np.log1p(loo_counts)
        return stats.astype(np.float32)

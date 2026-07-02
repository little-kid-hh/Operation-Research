from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from box_design_surrogate.candidate_ranker import (
    CANDIDATE_CATEGORICAL_FEATURES,
    CANDIDATE_NUMERIC_FEATURES,
    CandidateRanker,
    FastCandidatePipeline,
)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), CANDIDATE_NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                CANDIDATE_CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def make_rows(n: int = 80) -> list[dict]:
    dims = ["length", "width", "height"]
    directions = ["shrink", "expand"]
    rows = []
    for idx in range(n):
        row = {}
        for feat_idx, feature in enumerate(CANDIDATE_NUMERIC_FEATURES):
            row[feature] = float((idx + 1) * (feat_idx + 2) % 17) / 7.0
        row["candidate_index"] = idx % 60
        row["generated_candidates"] = 60
        row["move_dimension"] = dims[idx % len(dims)]
        row["move_direction"] = directions[idx % len(directions)]
        rows.append(row)
    return rows


class CandidateRankerFastPathTest(unittest.TestCase):
    def assert_fast_matches_pipeline(self, model) -> None:
        rows = make_rows()
        frame = pd.DataFrame(rows)
        x_df = frame[CANDIDATE_NUMERIC_FEATURES + CANDIDATE_CATEGORICAL_FEATURES]
        y = np.asarray([row["candidate_box_volume"] + 0.1 * row["is_shrink"] for row in rows])
        pipeline = make_pipeline(model)
        pipeline.fit(x_df, y)

        fast = FastCandidatePipeline.from_pipeline(
            pipeline,
            numeric_features=CANDIDATE_NUMERIC_FEATURES,
            categorical_features=CANDIDATE_CATEGORICAL_FEATURES,
        )
        self.assertIsNotNone(fast)
        slow_scores = np.asarray(pipeline.predict(x_df), dtype=np.float64)
        fast_scores = fast.predict(rows)
        np.testing.assert_allclose(fast_scores, slow_scores, rtol=0.0, atol=1e-12)

        ranker = CandidateRanker(
            pipeline=pipeline,
            numeric_features=CANDIDATE_NUMERIC_FEATURES,
            categorical_features=CANDIDATE_CATEGORICAL_FEATURES,
            metadata={},
            fast_predictor=fast,
        )
        np.testing.assert_allclose(ranker.predict_scores(rows), slow_scores, rtol=0.0, atol=1e-12)

    def test_fast_path_matches_hist_gradient_boosting_pipeline(self) -> None:
        self.assert_fast_matches_pipeline(
            HistGradientBoostingRegressor(max_iter=20, random_state=0)
        )

    def test_fast_path_matches_random_forest_pipeline(self) -> None:
        self.assert_fast_matches_pipeline(
            RandomForestRegressor(n_estimators=10, random_state=0)
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from box_design_surrogate.candidate_ranker import (
    CANDIDATE_CATEGORICAL_FEATURES,
    CANDIDATE_NUMERIC_FEATURES,
    CandidateRanker,
    FastCandidatePipeline,
    candidate_feature_row,
    positive_class_probability,
)
from box_design_surrogate.evaluator import Box
from box_design_surrogate.features import summarize_items
from box_design_surrogate.milp_oracle import MilpBoxSetScore
from box_design_surrogate.search import BoxMove, apply_move


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

    def test_classifier_score_mode_returns_negative_positive_probability(self) -> None:
        rows = make_rows()
        frame = pd.DataFrame(rows)
        x_df = frame[CANDIDATE_NUMERIC_FEATURES + CANDIDATE_CATEGORICAL_FEATURES]
        y = np.asarray([1 if idx % 4 == 0 else 0 for idx, _ in enumerate(rows)])
        pipeline = make_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
        pipeline.fit(x_df, y)

        expected = -positive_class_probability(pipeline, x_df)
        ranker = CandidateRanker(
            pipeline=pipeline,
            numeric_features=CANDIDATE_NUMERIC_FEATURES,
            categorical_features=CANDIDATE_CATEGORICAL_FEATURES,
            metadata={"score_mode": "negative_positive_probability"},
            fast_predictor=FastCandidatePipeline.from_pipeline(
                pipeline,
                numeric_features=CANDIDATE_NUMERIC_FEATURES,
                categorical_features=CANDIDATE_CATEGORICAL_FEATURES,
            ),
        )

        np.testing.assert_allclose(ranker.predict_scores(rows), expected, rtol=0.0, atol=1e-12)

    def test_candidate_features_include_assignment_opportunity_stats(self) -> None:
        orders = [
            summarize_items("toy.xml", "small", [(4.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "served", [(2.0, 1.0, 1.0)]),
        ]
        current = [Box(0, 3.0, 3.0, 3.0), Box(1, 5.0, 5.0, 5.0)]
        current_score = MilpBoxSetScore(
            packaging_factor=10.0,
            mean_box_volume=10.0,
            mean_order_volume=1.0,
            coverage_rate=1.0,
            uncovered_orders=0,
            unknown_pairs=0,
            orders_with_unknown=0,
            assignments=(1, 0),
        )

        expansion = BoxMove(0, "length", 1.0)
        expansion_row = candidate_feature_row(
            current_boxes=current,
            candidate_boxes=apply_move(current, expansion),
            move=expansion,
            current_score=current_score,
            step=1.0,
            stage=1,
            iteration=1,
            candidate_index=1,
            generated_candidates=12,
            orders=orders,
        )

        self.assertEqual(expansion_row["assignment_moved_box_order_count"], 1.0)
        self.assertEqual(expansion_row["assignment_candidate_capture_count"], 1.0)
        self.assertEqual(expansion_row["assignment_candidate_capture_volume"], orders[0].total_volume)
        self.assertEqual(expansion_row["assignment_candidate_new_capture_count"], 1.0)
        self.assertGreater(expansion_row["assignment_candidate_capture_assigned_box_volume_delta"], 0.0)
        self.assertEqual(expansion_row["assignment_moved_box_at_risk_count"], 0.0)

        shrink = BoxMove(0, "length", -2.0)
        shrink_row = candidate_feature_row(
            current_boxes=current,
            candidate_boxes=apply_move(current, shrink),
            move=shrink,
            current_score=current_score,
            step=2.0,
            stage=1,
            iteration=2,
            candidate_index=0,
            generated_candidates=12,
            orders=orders,
        )

        self.assertEqual(shrink_row["assignment_candidate_capture_count"], 0.0)
        self.assertEqual(shrink_row["assignment_moved_box_at_risk_count"], 1.0)
        self.assertEqual(shrink_row["assignment_moved_box_at_risk_volume"], orders[1].total_volume)


if __name__ == "__main__":
    unittest.main()

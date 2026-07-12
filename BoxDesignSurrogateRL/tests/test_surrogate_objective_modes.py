from __future__ import annotations

import math
import unittest

import numpy as np

from box_design_surrogate.evaluator import BatchSurrogateEvaluator, Box, SurrogateEvaluator
from box_design_surrogate.features import summarize_items
from box_design_surrogate.surrogate_game import SurrogateBoxSizingGame


class LengthProbabilityModel:
    def predict_proba(self, x_df):
        probs = np.where(x_df["vehicle_length"].to_numpy(dtype=float) < 3.0, 0.96, 0.99)
        return np.column_stack([1.0 - probs, probs])


class CountingLengthProbabilityModel(LengthProbabilityModel):
    def __init__(self) -> None:
        self.row_counts: list[int] = []

    def predict_proba(self, x_df):
        self.row_counts.append(len(x_df))
        return super().predict_proba(x_df)


class SurrogateObjectiveModeTest(unittest.TestCase):
    def test_multiscale_actions_decode_step_before_local_transform(self) -> None:
        game = SurrogateBoxSizingGame(
            self.orders,
            [Box(0, 4.0, 3.0, 2.0)],
            self.evaluator,
            step_size=0.5,
            action_steps=[2.0, 1.0],
            objective_mode="paper_pf_surrogate",
        )

        large_step = game.apply_action(game.boxes, 0)
        small_step = game.apply_action(game.boxes, 6)

        self.assertEqual(game.action_count, 13)
        self.assertEqual(game.resign_action, 12)
        self.assertEqual((large_step[0].length, large_step[0].width, large_step[0].height), (3.0, 2.0, 2.0))
        self.assertEqual((small_step[0].length, small_step[0].width, small_step[0].height), (3.0, 3.0, 2.0))

    def setUp(self) -> None:
        self.orders = [summarize_items("unit", "order0", [(1.0, 1.0, 1.0)])]
        self.boxes = [
            Box(0, 2.0, 2.0, 2.0),
            Box(1, 3.0, 3.0, 3.0),
        ]
        self.evaluator = SurrogateEvaluator(
            model=LengthProbabilityModel(),
            tau=0.95,
            tau_high=0.99,
            lambda_risk=1_000_000.0,
        )

    def test_paper_pf_surrogate_uses_min_feasible_volume(self) -> None:
        game = SurrogateBoxSizingGame(
            self.orders,
            self.boxes,
            self.evaluator,
            objective_mode="paper_pf_surrogate",
        )

        assignment = game.evaluation.assignments[0]
        breakdown = game.objective_breakdown(game.evaluation)

        self.assertEqual(assignment.box_id, 0)
        self.assertEqual(game.evaluation.uncovered_orders, 0)
        self.assertTrue(math.isclose(breakdown.mean_assigned_box_volume, 8.0))
        self.assertTrue(math.isclose(breakdown.mean_order_volume, 1.0))
        self.assertTrue(math.isclose(breakdown.packaging_factor, 8.0))
        self.assertTrue(math.isclose(game.current_objective, 8.0))

    def test_paper_pf_surrogate_ignores_risk_weights(self) -> None:
        base_game = SurrogateBoxSizingGame(
            self.orders,
            self.boxes,
            self.evaluator,
            objective_mode="paper_pf_surrogate",
            low_margin_weight=1.0,
            probability_weight=1.0,
        )
        weighted_game = SurrogateBoxSizingGame(
            self.orders,
            self.boxes,
            self.evaluator,
            objective_mode="paper_pf_surrogate",
            low_margin_weight=1_000_000.0,
            probability_weight=1_000_000.0,
        )

        self.assertTrue(math.isclose(base_game.current_objective, weighted_game.current_objective))
        self.assertEqual(base_game.evaluation.assignments[0].box_id, weighted_game.evaluation.assignments[0].box_id)

    def test_risk_aware_surrogate_keeps_adjusted_cost_choice(self) -> None:
        game = SurrogateBoxSizingGame(
            self.orders,
            self.boxes,
            self.evaluator,
            objective_mode="risk_aware_surrogate",
            low_margin_weight=0.0,
            probability_weight=0.0,
        )

        self.assertEqual(game.evaluation.assignments[0].box_id, 1)
        self.assertTrue(math.isclose(game.objective_breakdown(game.evaluation).mean_adjusted_cost, 27.0))

    def test_chunked_candidate_evaluation_matches_unchunked(self) -> None:
        evaluator = BatchSurrogateEvaluator.from_evaluator(self.evaluator, self.orders)
        candidates = [
            self.boxes,
            [Box(0, 1.5, 1.5, 1.5), Box(1, 3.0, 3.0, 3.0)],
            [Box(0, 2.0, 2.0, 2.0), Box(1, 2.5, 2.5, 2.5)],
        ]

        unchunked = evaluator.evaluate_many_box_sets(candidates, assignment_mode="min_volume")
        chunked = evaluator.evaluate_many_box_sets(
            candidates,
            assignment_mode="min_volume",
            candidate_batch_size=1,
        )

        self.assertEqual(len(unchunked), len(chunked))
        for left, right in zip(unchunked, chunked):
            self.assertEqual(left.uncovered_orders, right.uncovered_orders)
            self.assertEqual(left.low_margin_assignments, right.low_margin_assignments)
            self.assertTrue(math.isclose(left.total_base_cost, right.total_base_cost))
            self.assertTrue(math.isclose(left.total_adjusted_cost, right.total_adjusted_cost))
            self.assertEqual(
                [assignment.box_id for assignment in left.assignments],
                [assignment.box_id for assignment in right.assignments],
            )

    def test_candidate_batch_evaluation_deduplicates_box_dimensions(self) -> None:
        model = CountingLengthProbabilityModel()
        evaluator = BatchSurrogateEvaluator.from_evaluator(
            SurrogateEvaluator(model=model, tau=0.95, tau_high=0.99),
            self.orders,
        )
        candidates = [
            [Box(0, 2.0, 2.0, 2.0), Box(1, 4.0, 4.0, 4.0)],
            [Box(0, 1.5, 1.5, 1.5), Box(1, 4.0, 4.0, 4.0)],
            [Box(0, 2.0, 2.0, 2.0), Box(1, 3.0, 3.0, 3.0)],
        ]

        results = evaluator.evaluate_many_box_sets(candidates, assignment_mode="min_volume")

        self.assertEqual(len(results), 3)
        self.assertEqual(model.row_counts, [4])


if __name__ == "__main__":
    unittest.main()

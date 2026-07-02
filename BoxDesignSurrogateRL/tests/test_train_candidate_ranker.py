from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.train_candidate_ranker import add_training_columns, evaluate_predictions, target_values


def base_row(group: str, idx: int, pf: float, *, accepted: bool = False) -> dict:
    return {
        "trace_source_path": "trace.csv",
        "source_run_id": group,
        "phase": "staged_greedy",
        "stage": 1,
        "iteration": 1,
        "candidate_index": idx,
        "candidate_packaging_factor": pf,
        "candidate_uncovered_orders": 0,
        "candidate_unknown_pairs": 0,
        "is_accepted": int(accepted),
    }


class TrainCandidateRankerTargetTest(unittest.TestCase):
    def test_objective_gap_and_rank_targets_are_step_relative(self) -> None:
        data = pd.DataFrame(
            [
                base_row("a", 0, 3.0),
                base_row("a", 1, 2.0, accepted=True),
                base_row("b", 0, 20.0),
                base_row("b", 1, 10.0, accepted=True),
            ]
        )
        enriched = add_training_columns(data, uncovered_weight=1_000_000.0, unknown_weight=1_000.0)

        np.testing.assert_allclose(
            target_values(enriched, "objective_gap"),
            np.asarray([1.0, 0.0, 10.0, 0.0]),
        )
        np.testing.assert_allclose(
            target_values(enriched, "rank"),
            np.asarray([np.log1p(1.0), 0.0, np.log1p(1.0), 0.0]),
        )

    def test_evaluate_predictions_reports_accepted_capture(self) -> None:
        data = pd.DataFrame(
            [
                {**base_row("a", 0, 3.0), "candidate_objective": 3.0, "candidate_rank": 2},
                {**base_row("a", 1, 2.0, accepted=True), "candidate_objective": 2.0, "candidate_rank": 1},
                {**base_row("b", 0, 20.0), "candidate_objective": 20.0, "candidate_rank": 2},
                {**base_row("b", 1, 10.0, accepted=True), "candidate_objective": 10.0, "candidate_rank": 1},
            ]
        )
        enriched = add_training_columns(data, uncovered_weight=1_000_000.0, unknown_weight=1_000.0)
        enriched["predicted_objective"] = [0.0, 1.0, 1.0, 0.0]

        metrics = evaluate_predictions(enriched, [1, 2])
        self.assertEqual(metrics["eval_groups"], 2)
        self.assertEqual(metrics["accepted_eval_groups"], 2)
        self.assertEqual(metrics["exact_best_capture_at_1"], 0.5)
        self.assertEqual(metrics["accepted_move_capture_at_1"], 0.5)
        self.assertEqual(metrics["accepted_step_preservation_at_1"], 0.5)
        self.assertEqual(metrics["exact_best_capture_at_2"], 1.0)
        self.assertEqual(metrics["accepted_step_preservation_at_2"], 1.0)


if __name__ == "__main__":
    unittest.main()

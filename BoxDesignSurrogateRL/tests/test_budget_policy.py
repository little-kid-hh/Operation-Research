from __future__ import annotations

import unittest

import numpy as np

from box_design_surrogate.budget_policy import (
    BUDGET_STATE_FEATURES,
    budget_state_features,
    counterfactual_budget_preservation,
)
from scripts.train_budget_fqi_policy import evaluate_fixed_budgets
from box_design_surrogate.budget_policy import BudgetTransitionDataset


def candidate(index: int, objective: float, *, shrink: bool = True) -> dict[str, float | int]:
    return {
        "candidate_index": index,
        "candidate_uncovered_orders": 0,
        "candidate_unknown_pairs": 0,
        "candidate_objective": objective,
        "current_packaging_factor": 2.0,
        "step": 0.25,
        "current_set_mean_volume": 100.0,
        "current_set_std_volume": 20.0,
        "is_shrink": int(shrink),
        "relative_delta_volume": -0.1 if shrink else 0.1,
    }


class BudgetPolicyTest(unittest.TestCase):
    def test_fixed_budget_report_includes_oracle_minimum(self) -> None:
        dataset = BudgetTransitionDataset(
            states=np.zeros((2, 1)),
            preserves=np.asarray([[False, True], [True, True]]),
            next_indices=np.asarray([1, -1]),
            episode_ids=np.asarray(["run", "run"], dtype=object),
        )

        metrics = evaluate_fixed_budgets(dataset, (10, 20))

        self.assertEqual(metrics["10"]["preservation_rate"], 0.5)
        self.assertEqual(metrics["10"]["mean_effective_budget_with_full_audit_on_miss"], 15.0)
        self.assertEqual(metrics["oracle_minimum"]["mean_effective_budget_with_full_audit_on_miss"], 15.0)

    def test_state_features_are_fixed_and_finite(self) -> None:
        rows = [candidate(idx, 1.0 + idx) for idx in range(6)]
        features = budget_state_features(rows, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], iteration=2, max_iterations=10)

        self.assertEqual(features.shape, (len(BUDGET_STATE_FEATURES),))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_counterfactual_budget_detects_minimum_exact_preserving_tier(self) -> None:
        rows = [candidate(0, 5.0), candidate(1, 1.0), candidate(2, 3.0)]
        scores = [0.0, 2.0, 1.0]

        preserves = counterfactual_budget_preservation(rows, scores, [1, 2, 3])

        self.assertEqual(preserves.tolist(), [False, False, True])

    def test_equal_objective_is_exact_quality_preserving(self) -> None:
        rows = [candidate(0, 1.0), candidate(1, 1.0), candidate(2, 3.0)]
        scores = [1.0, 0.0, 2.0]

        preserves = counterfactual_budget_preservation(rows, scores, [1])

        self.assertEqual(preserves.tolist(), [True])


if __name__ == "__main__":
    unittest.main()

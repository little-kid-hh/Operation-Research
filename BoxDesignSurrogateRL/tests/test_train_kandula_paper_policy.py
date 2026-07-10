from __future__ import annotations

import unittest

from box_design_surrogate.policy_context import ORDER_CONTEXT_SCHEMA, order_distribution_context
from box_design_surrogate.features import summarize_items

from scripts.train_kandula_paper_policy import (
    make_training_order_windows,
    policy_training_reward,
    shuffled_environment_indices,
)


class PolicyTrainingRewardTest(unittest.TestCase):
    def test_order_context_has_fixed_normalized_schema(self) -> None:
        orders = [
            summarize_items("toy.xml", "0", [(2.0, 4.0, 8.0)]),
            summarize_items("toy.xml", "1", [(1.0, 3.0, 6.0)]),
        ]
        context = order_distribution_context(orders, scale_dim=10.0)

        self.assertEqual(len(context), len(ORDER_CONTEXT_SCHEMA))
        self.assertTrue(all(value >= 0.0 for value in context))
        self.assertLessEqual(float(max(context[:6])), 1.0)

    def test_training_windows_are_complete_and_non_overlapping_by_default_stride(self) -> None:
        windows = make_training_order_windows(list(range(10)), window_size=4, window_stride=0)

        self.assertEqual([(window_id, start, values) for window_id, start, values in windows], [
            (0, 0, [0, 1, 2, 3]),
            (1, 4, [4, 5, 6, 7]),
        ])

    def test_environment_schedule_visits_every_environment_per_cycle(self) -> None:
        indices = shuffled_environment_indices(count=3, episodes=7, seed=11)

        self.assertEqual(sorted(indices[:3]), [0, 1, 2])
        self.assertEqual(sorted(indices[3:6]), [0, 1, 2])
        self.assertIn(indices[6], [0, 1, 2])

    def test_paper_surrogate_preserves_pf_signal_at_fixed_coverage(self) -> None:
        reward = policy_training_reward(
            previous_metrics={"uncovered_orders": 2, "packaging_factor": 2.0},
            current_metrics={"uncovered_orders": 2, "packaging_factor": 1.9},
            environment_reward=1000.0,
            mode="surrogate",
            objective_mode="paper_pf_surrogate",
            coverage_reward_weight=1.0,
        )

        self.assertAlmostEqual(reward, 0.1)

    def test_paper_surrogate_coverage_loss_dominates_small_pf_gain(self) -> None:
        reward = policy_training_reward(
            previous_metrics={"uncovered_orders": 2, "packaging_factor": 2.0},
            current_metrics={"uncovered_orders": 3, "packaging_factor": 1.9},
            environment_reward=-20000.0,
            mode="surrogate",
            objective_mode="paper_pf_surrogate",
            coverage_reward_weight=1.0,
        )

        self.assertAlmostEqual(reward, -0.9)

    def test_other_modes_keep_environment_reward(self) -> None:
        reward = policy_training_reward(
            previous_metrics={"uncovered_orders": 2, "packaging_factor": 2.0},
            current_metrics={"uncovered_orders": 3, "packaging_factor": 1.9},
            environment_reward=-7.5,
            mode="paper",
            objective_mode="",
            coverage_reward_weight=1.0,
        )

        self.assertEqual(reward, -7.5)


if __name__ == "__main__":
    unittest.main()

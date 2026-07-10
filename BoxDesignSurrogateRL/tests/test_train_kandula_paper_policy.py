from __future__ import annotations

import unittest

from scripts.train_kandula_paper_policy import policy_training_reward


class PolicyTrainingRewardTest(unittest.TestCase):
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

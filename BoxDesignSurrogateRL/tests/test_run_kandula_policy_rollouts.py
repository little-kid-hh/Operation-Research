from __future__ import annotations

import unittest

from scripts.run_kandula_policy_rollouts import (
    action_type,
    aggregate_summaries,
    apply_checkpoint_environment_metadata,
    is_better,
    metric_rank,
)


class FakeEnv:
    normalize_observation = True
    scale_dim = 10.0


class RunKandulaPolicyRolloutsTest(unittest.TestCase):
    def test_checkpoint_scale_is_reused_and_step_must_match(self) -> None:
        env = FakeEnv()
        metadata = apply_checkpoint_environment_metadata(
            env,
            {"step_size": 0.25, "scale_dim": 105.0, "training_order_count": 500},
            requested_step=0.25,
        )

        self.assertEqual(env.scale_dim, 105.0)
        self.assertEqual(metadata["scale_dim_source"], "checkpoint")
        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_checkpoint_environment_metadata(env, {"step_size": 0.5}, requested_step=0.25)

    def test_action_type_matches_kandula_action_layout(self) -> None:
        self.assertEqual(action_type(0, 10), "decrement")
        self.assertEqual(action_type(29, 10), "decrement")
        self.assertEqual(action_type(30, 10), "increment")
        self.assertEqual(action_type(59, 10), "increment")
        self.assertEqual(action_type(60, 10), "resign")

    def test_metric_rank_prioritizes_uncovered_then_objective(self) -> None:
        self.assertEqual(metric_rank({"uncovered_orders": 0, "objective": 2.0}), (0.0, 2.0))
        self.assertTrue(
            is_better(
                {"uncovered_orders": 0, "objective": 3.0},
                {"uncovered_orders": 1, "objective": 1.0},
            )
        )
        self.assertTrue(
            is_better(
                {"uncovered_orders": 0, "objective": 1.5},
                {"uncovered_orders": 0, "objective": 2.0},
            )
        )

    def test_aggregate_summaries_selects_best_rollout_and_means(self) -> None:
        rows = [
            {
                "rollout": 0,
                "policy_mode": "sample",
                "seed": 10,
                "initial_objective": 2.0,
                "final_objective": 2.1,
                "best_objective": 2.0,
                "best_uncovered": 0,
                "initial_packaging_factor": 2.0,
                "final_packaging_factor": 2.1,
                "best_packaging_factor": 2.0,
                "terminal_reason": "max_steps",
            },
            {
                "rollout": 1,
                "policy_mode": "sample",
                "seed": 11,
                "initial_objective": 2.0,
                "final_objective": 1.8,
                "best_objective": 1.8,
                "best_uncovered": 0,
                "initial_packaging_factor": 2.0,
                "final_packaging_factor": 1.8,
                "best_packaging_factor": 1.8,
                "terminal_reason": "resign",
            },
        ]

        aggregate = aggregate_summaries(rows)

        self.assertEqual(aggregate["rollouts"], 2)
        self.assertEqual(aggregate["best_rollout"], 1)
        self.assertAlmostEqual(aggregate["final_objective_mean"], 1.95)
        self.assertEqual(aggregate["terminal_reasons"], {"max_steps": 1, "resign": 1})

    def test_aggregate_summaries_prefers_zero_uncovered_over_lower_bad_objective(self) -> None:
        rows = [
            {
                "rollout": 0,
                "policy_mode": "sample",
                "seed": 10,
                "best_objective": 1.0,
                "best_uncovered": 2,
                "best_packaging_factor": 1.0,
                "terminal_reason": "max_steps",
            },
            {
                "rollout": 1,
                "policy_mode": "sample",
                "seed": 11,
                "best_objective": 3.0,
                "best_uncovered": 0,
                "best_packaging_factor": 3.0,
                "terminal_reason": "max_steps",
            },
        ]

        aggregate = aggregate_summaries(rows)

        self.assertEqual(aggregate["best_rollout"], 1)


if __name__ == "__main__":
    unittest.main()

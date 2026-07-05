from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_ranker_window_protocol import exact_command, frontier_command, window_label


class RunRankerWindowProtocolTest(unittest.TestCase):
    def _args(self, *, initial_boxes_json: Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            xml_path=Path("test.xml"),
            candidate_ranker_path=Path("ranker.joblib"),
            initial_boxes_json=initial_boxes_json,
            orders_limit=100,
            k=10,
            schedule="0.25:1000",
            java_classes=Path("classes"),
            java_classpath="gurobi.jar",
            milp_time_limit_seconds=30.0,
            coverage_repair="geometric_expand",
            ranker_budget_sequence=["10,20,30,40,50"],
            ranker_safety_policy="all_expansions",
            ranker_safety_max_candidates=None,
            ranker_max_elapsed_seconds=180.0,
            code_version="test-version",
            config_prefix="protocol",
        )

    def test_window_label_uses_half_open_interval(self) -> None:
        self.assertEqual(window_label(200, 100), "test[200,300)")

    def test_exact_command_omits_initial_boxes_for_seed_specific_init(self) -> None:
        cmd = exact_command(
            args=self._args(initial_boxes_json=None),
            seed=2,
            offset=200,
            exact_out_root=Path("out/exact"),
            exact_cache_dir=Path("cache/exact"),
        )

        self.assertNotIn("--initial-boxes-json", cmd)
        self.assertIn("--seed", cmd)
        self.assertEqual(cmd[cmd.index("--seed") + 1], "2")
        self.assertEqual(cmd[cmd.index("--orders-offset") + 1], "200")

    def test_exact_command_includes_fixed_initial_boxes_when_supplied(self) -> None:
        cmd = exact_command(
            args=self._args(initial_boxes_json=Path("initial.json")),
            seed=2,
            offset=200,
            exact_out_root=Path("out/exact"),
            exact_cache_dir=Path("cache/exact"),
        )

        self.assertIn("--initial-boxes-json", cmd)
        self.assertEqual(cmd[cmd.index("--initial-boxes-json") + 1], "initial.json")

    def test_frontier_command_reuses_exact_initial_boxes_and_summary(self) -> None:
        cmd = frontier_command(
            args=self._args(initial_boxes_json=None),
            seed=1,
            offset=300,
            initial_boxes_json=Path("exact/run/initial_boxes.json"),
            exact_summary=Path("exact/run/summary.json"),
            frontier_out_root=Path("out/frontier"),
            frontier_cache_dir=Path("cache/frontier"),
        )

        self.assertEqual(Path(cmd[cmd.index("--initial-boxes-json") + 1]), Path("exact/run/initial_boxes.json"))
        self.assertEqual(Path(cmd[cmd.index("--exact-baseline-summary") + 1]), Path("exact/run/summary.json"))
        self.assertEqual(cmd[cmd.index("--ranker-budget-sequence") + 1], "10,20,30,40,50")
        self.assertEqual(cmd[cmd.index("--ranker-safety-policy") + 1], "all_expansions")
        self.assertEqual(cmd[cmd.index("--orders-offset") + 1], "300")

    def test_frontier_command_passes_targeted_safety_limit(self) -> None:
        args = self._args(initial_boxes_json=None)
        args.ranker_safety_policy = "targeted_expansion_capture"
        args.ranker_safety_max_candidates = 3

        cmd = frontier_command(
            args=args,
            seed=1,
            offset=300,
            initial_boxes_json=Path("exact/run/initial_boxes.json"),
            exact_summary=Path("exact/run/summary.json"),
            frontier_out_root=Path("out/frontier"),
            frontier_cache_dir=Path("cache/frontier"),
        )

        self.assertEqual(cmd[cmd.index("--ranker-safety-policy") + 1], "targeted_expansion_capture")
        self.assertEqual(cmd[cmd.index("--ranker-safety-max-candidates") + 1], "3")


if __name__ == "__main__":
    unittest.main()

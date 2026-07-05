from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_ranker_handoff_traces import analyze


class AnalyzeRankerHandoffTracesTest(unittest.TestCase):
    def test_analyze_frontier_trace_summarizes_handoff_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "ranker_run"
            run_dir.mkdir()
            trace_path = run_dir / "trace.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "phase",
                        "iteration",
                        "improved",
                        "packaging_factor",
                        "milp_validated_candidates",
                        "milp_eval_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "phase": "ranker_filtered_greedy",
                        "iteration": "1",
                        "improved": "True",
                        "packaging_factor": "2.0",
                        "milp_validated_candidates": "10",
                        "milp_eval_seconds": "1.5",
                    }
                )
                writer.writerow(
                    {
                        "phase": "ranker_filtered_greedy",
                        "iteration": "2",
                        "improved": "False",
                        "packaging_factor": "1.8",
                        "milp_validated_candidates": "20",
                        "milp_eval_seconds": "2.5",
                    }
                )

            frontier_path = tmp_path / "frontier_summary.json"
            frontier_path.write_text(
                json.dumps(
                    [
                        {
                            "ranker_run_dir": str(run_dir),
                            "ranker_max_elapsed_seconds": 300.0,
                            "ranker_stop_reason": "time_limit",
                            "ranker_pf": 1.8,
                            "audit_pf": 1.7,
                            "baseline_pf": 1.75,
                            "audit_coverage": 1.0,
                            "audit_uncovered": 0,
                            "ranker_milp_validated_candidates": 30,
                            "audit_milp_validated_candidates": 40,
                            "combined_milp_validated_candidates": 70,
                            "baseline_milp_validated_candidates": 100,
                            "combined_oracle_uncached_boxes": 25,
                            "baseline_oracle_uncached_boxes": 50,
                            "combined_oracle_subprocess_seconds": 8.0,
                            "baseline_oracle_subprocess_seconds": 10.0,
                            "combined_elapsed_seconds": 18.0,
                            "baseline_elapsed_seconds": 20.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = analyze([("cap300", frontier_path)], tail_iterations=2)

        row = result["frontiers"][0]
        self.assertEqual(row["label"], "cap300")
        self.assertEqual(row["ranker_iterations"], 2)
        self.assertEqual(row["ranker_improvements"], 1)
        self.assertEqual(row["last_improvement_iteration"], 1)
        self.assertEqual(row["iterations_since_last_improvement"], 1)
        self.assertAlmostEqual(row["trace_pf_improvement"], 0.2)
        self.assertAlmostEqual(row["tail_validations"], 30.0)
        self.assertAlmostEqual(row["tail_milp_eval_seconds"], 4.0)
        self.assertAlmostEqual(row["audit_pf_gap_vs_baseline"], -0.05)
        self.assertAlmostEqual(row["validation_reduction_pct"], 30.0)
        self.assertAlmostEqual(row["uncached_reduction_pct"], 50.0)
        self.assertAlmostEqual(row["subprocess_reduction_pct"], 20.0)
        self.assertAlmostEqual(row["elapsed_reduction_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_ranker_window_statistics import analyze_summary


class AnalyzeRankerWindowStatisticsTest(unittest.TestCase):
    def test_analyze_summary_counts_quality_and_cost_reductions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "label": "w0",
                                "pf_delta": 0.0,
                                "exact_coverage": 1.0,
                                "ranker_audit_coverage": 1.0,
                                "exact_uncovered": 0,
                                "ranker_audit_uncovered": 0,
                                "exact_validations": 100,
                                "ranker_audit_validations": 80,
                                "exact_uncached_boxes": 50,
                                "ranker_audit_uncached_boxes": 40,
                                "exact_subprocess_seconds": 30.0,
                                "ranker_audit_subprocess_seconds": 24.0,
                                "exact_elapsed_seconds": 60.0,
                                "ranker_audit_elapsed_seconds": 48.0,
                            },
                            {
                                "label": "w1",
                                "pf_delta": -0.1,
                                "exact_coverage": 1.0,
                                "ranker_audit_coverage": 1.0,
                                "exact_uncovered": 0,
                                "ranker_audit_uncovered": 0,
                                "exact_validations": 300,
                                "ranker_audit_validations": 240,
                                "exact_uncached_boxes": 150,
                                "ranker_audit_uncached_boxes": 120,
                                "exact_subprocess_seconds": 90.0,
                                "ranker_audit_subprocess_seconds": 72.0,
                                "exact_elapsed_seconds": 180.0,
                                "ranker_audit_elapsed_seconds": 144.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = analyze_summary(summary_path, bootstrap_iterations=10, seed=7)

        self.assertEqual(analysis["n_windows"], 2)
        self.assertEqual(analysis["pf"]["matched"], 1)
        self.assertEqual(analysis["pf"]["improved"], 1)
        self.assertEqual(analysis["pf"]["regressed"], 0)
        self.assertAlmostEqual(analysis["pf"]["mean_delta"], -0.05)
        self.assertEqual(analysis["coverage"]["total_ranker_audit_uncovered"], 0.0)

        validations = analysis["cost"]["validations"]
        self.assertEqual(validations["windows_reduced"], 2)
        self.assertEqual(validations["windows_increased"], 0)
        self.assertAlmostEqual(validations["aggregate_reduction_pct"], 20.0)
        self.assertEqual(validations["bootstrap"]["iterations"], 10)
        self.assertIsNotNone(validations["bootstrap"]["mean_reduction_pct_ci95"]["low"])
        self.assertIsNotNone(validations["bootstrap"]["mean_reduction_pct_ci95"]["high"])


if __name__ == "__main__":
    unittest.main()

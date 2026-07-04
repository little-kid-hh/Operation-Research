from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_ranker_window_results import load_window_rows, make_total_row


class SummarizeRankerWindowResultsTest(unittest.TestCase):
    def test_load_window_rows_and_total_reductions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exact_path = root / "exact_summary.json"
            frontier_path = root / "frontier_summary.json"
            manifest_path = root / "manifest.json"
            exact_path.write_text(
                json.dumps(
                    {
                        "best_score": {
                            "packaging_factor": 2.0,
                            "coverage_rate": 1.0,
                            "uncovered_orders": 0,
                        },
                        "milp_validated_candidates": 100,
                        "elapsed_seconds": 50.0,
                        "oracle_cache": {
                            "uncached_boxes": 20,
                            "subprocess_seconds": 30.0,
                        },
                        "run_dir": "exact/run",
                    }
                ),
                encoding="utf-8",
            )
            frontier_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "audit_pf": 2.0,
                                "audit_coverage": 1.0,
                                "audit_uncovered": 0,
                                "combined_milp_validated_candidates": 80,
                                "combined_oracle_uncached_boxes": 18,
                                "combined_oracle_subprocess_seconds": 24.0,
                                "combined_elapsed_seconds": 40.0,
                                "audit_run_dir": "audit/run",
                                "ranker_pf": 2.5,
                                "ranker_elapsed_seconds": 10.0,
                                "ranker_stop_reason": "time_limit",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "label": "toy[0,1)",
                            "exact_summary": "exact_summary.json",
                            "frontier_summary": "frontier_summary.json",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = load_window_rows(manifest_path)
            total = make_total_row(rows, label="toy total")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "toy[0,1)")
        self.assertEqual(rows[0]["pf_delta"], 0.0)
        self.assertAlmostEqual(rows[0]["validation_reduction_pct"], 20.0)
        self.assertAlmostEqual(rows[0]["uncached_box_reduction_pct"], 10.0)
        self.assertAlmostEqual(rows[0]["subprocess_reduction_pct"], 20.0)
        self.assertAlmostEqual(rows[0]["elapsed_reduction_pct"], 20.0)
        self.assertEqual(total["label"], "toy total")
        self.assertEqual(total["exact_validations"], 100)
        self.assertEqual(total["ranker_audit_validations"], 80)
        self.assertAlmostEqual(total["validation_reduction_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()

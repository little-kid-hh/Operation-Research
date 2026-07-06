from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_ranker_budget_variants import compare


class CompareRankerBudgetVariantsTest(unittest.TestCase):
    def test_compare_computes_exact_and_fixed_reference_reductions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "window_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "label": "seed3:test[400,500)",
                                "exact_pf": 2.0,
                                "ranker_audit_pf": 1.9,
                                "exact_coverage": 1.0,
                                "ranker_audit_coverage": 1.0,
                                "exact_uncovered": 0,
                                "ranker_audit_uncovered": 0,
                                "exact_validations": 100,
                                "ranker_audit_validations": 80,
                                "exact_uncached_boxes": 50,
                                "ranker_audit_uncached_boxes": 40,
                                "exact_subprocess_seconds": 20.0,
                                "ranker_audit_subprocess_seconds": 16.0,
                                "exact_elapsed_seconds": 30.0,
                                "ranker_audit_elapsed_seconds": 24.0,
                                "ranker_pf": 2.2,
                                "ranker_elapsed_seconds": 10.0,
                                "ranker_stop_reason": "time_limit",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            frontier_path = tmp_path / "frontier_summary.json"
            frontier_path.write_text(
                json.dumps(
                    [
                        {
                            "audit_pf": 1.9,
                            "audit_coverage": 1.0,
                            "audit_uncovered": 0,
                            "combined_milp_validated_candidates": 60,
                            "combined_oracle_uncached_boxes": 30,
                            "combined_oracle_subprocess_seconds": 12.0,
                            "combined_elapsed_seconds": 18.0,
                            "ranker_pf": 2.1,
                            "ranker_elapsed_seconds": 12.0,
                            "ranker_stop_reason": "ranker_marginal_pf_handoff",
                            "ranker_handoff_policy": "marginal_pf_per_validation",
                            "ranker_handoff_min_pf_improvement_per_validation": 3e-6,
                            "ranker_max_elapsed_seconds": 90.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = compare(
                window_summary_json=summary_path,
                window_label="seed3:test[400,500)",
                fixed_label="fixed",
                frontier_summaries=[("adaptive", frontier_path)],
            )

        exact = result["exact"]
        fixed = result["variants"][0]
        adaptive = result["variants"][1]
        self.assertEqual(exact["label"], "exact_staged_baseline")
        self.assertEqual(fixed["label"], "fixed")
        self.assertEqual(adaptive["label"], "adaptive")
        self.assertAlmostEqual(fixed["pf_delta_vs_exact"], -0.1)
        self.assertAlmostEqual(fixed["validations_reduction_vs_exact_pct"], 20.0)
        self.assertAlmostEqual(adaptive["validations_reduction_vs_exact_pct"], 40.0)
        self.assertAlmostEqual(adaptive["validations_reduction_vs_reference_pct"], 25.0)
        self.assertAlmostEqual(adaptive["elapsed_seconds_reduction_vs_reference_pct"], 25.0)
        self.assertEqual(result["summary"]["best_elapsed_variant"], "adaptive")
        self.assertEqual(result["summary"]["n_pf_regressions_vs_exact"], 0)
        self.assertEqual(result["summary"]["n_pf_improvements_vs_exact"], 2)


if __name__ == "__main__":
    unittest.main()

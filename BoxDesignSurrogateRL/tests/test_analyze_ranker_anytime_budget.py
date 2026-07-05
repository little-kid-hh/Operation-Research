from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.analyze_ranker_anytime_budget import analyze


TRACE_HEADER = "phase,action,milp_validated_candidates,packaging_factor\n"


class AnalyzeRankerAnytimeBudgetTest(unittest.TestCase):
    def test_analyze_replays_matched_validation_budget_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "label": "seed1:test[0,100)",
                                "exact_run_dir": "BoxDesignSurrogateRL/results/exact_run",
                                "ranker_audit_run_dir": "BoxDesignSurrogateRL/results/audit_run",
                                "frontier_summary_path": "BoxDesignSurrogateRL/results/frontier/frontier_summary.json",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            zip_path = tmp_path / "traces.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "exact_run/trace.csv",
                    TRACE_HEADER + "exact,init,0,10.0\nexact,move,10,8.0\nexact,move,10,6.0\nexact,noop,10,6.0\n",
                )
                zf.writestr("frontier/frontier_summary.json", json.dumps([{"ranker_run_dir": "ranker_run"}]))
                zf.writestr(
                    "ranker_run/trace.csv",
                    TRACE_HEADER + "ranker,init,0,10.0\nranker,move,5,7.0\nranker,move,5,6.0\n",
                )
                zf.writestr("audit_run/trace.csv", TRACE_HEADER + "audit,init,0,6.0\naudit,noop,5,6.0\n")

            result = analyze(
                summary_json=summary_path,
                zip_paths=[zip_path],
                exact_trace_dir=None,
                extra_trace_dir=None,
                budget_fractions=[0.5, 1.0],
                pf_tolerance=1e-12,
            )

        self.assertEqual(result["trace_complete_windows"], 1)
        half_budget = result["budget_summary"][0]
        self.assertEqual(half_budget["ranker_better"], 1)
        self.assertEqual(half_budget["ranker_worse"], 0)
        self.assertAlmostEqual(half_budget["mean_pf_delta_ranker_minus_exact"], -2.0)
        self.assertEqual(result["time_to_exact_final_pf"]["exact_sum"], 20.0)
        self.assertEqual(result["time_to_exact_final_pf"]["ranker_audit_sum"], 10.0)
        self.assertAlmostEqual(result["time_to_exact_final_pf"]["aggregate_reduction_pct"], 50.0)

    def test_missing_trace_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "label": "seed1:test[0,100)",
                                "exact_run_dir": "BoxDesignSurrogateRL/results/missing_exact",
                                "ranker_audit_run_dir": "BoxDesignSurrogateRL/results/missing_audit",
                                "frontier_summary_path": None,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            zip_path = tmp_path / "empty.zip"
            with zipfile.ZipFile(zip_path, "w"):
                pass

            result = analyze(
                summary_json=summary_path,
                zip_paths=[zip_path],
                exact_trace_dir=None,
                extra_trace_dir=None,
                budget_fractions=[1.0],
                pf_tolerance=1e-12,
            )

        self.assertEqual(result["trace_complete_windows"], 0)
        self.assertEqual(result["missing_windows"][0]["missing"], ["exact_trace", "ranker_trace", "audit_trace"])


if __name__ == "__main__":
    unittest.main()

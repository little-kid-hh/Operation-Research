from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.combine_ranker_window_summaries import combine_summaries


def summary(path: Path, label: str, exact: int, ranker: int) -> None:
    path.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "label": label,
                        "pf_delta": 0.0,
                        "exact_coverage": 1.0,
                        "ranker_audit_coverage": 1.0,
                        "exact_uncovered": 0,
                        "ranker_audit_uncovered": 0,
                        "exact_validations": exact,
                        "ranker_audit_validations": ranker,
                        "exact_uncached_boxes": exact // 10,
                        "ranker_audit_uncached_boxes": ranker // 10,
                        "exact_subprocess_seconds": float(exact),
                        "ranker_audit_subprocess_seconds": float(ranker),
                        "exact_elapsed_seconds": float(exact * 2),
                        "ranker_audit_elapsed_seconds": float(ranker * 2),
                    }
                ],
                "total": {"label": "ignored"},
            }
        ),
        encoding="utf-8",
    )


class CombineRankerWindowSummariesTest(unittest.TestCase):
    def test_combines_windows_and_recomputes_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p0 = Path(tmp) / "s0.json"
            p1 = Path(tmp) / "s1.json"
            summary(p0, "w0", 100, 80)
            summary(p1, "w1", 200, 160)

            combined = combine_summaries([p0, p1], total_label="combined")

        self.assertEqual([row["label"] for row in combined["windows"]], ["w0", "w1"])
        self.assertEqual(combined["total"]["label"], "combined")
        self.assertEqual(combined["total"]["exact_validations"], 300)
        self.assertEqual(combined["total"]["ranker_audit_validations"], 240)
        self.assertAlmostEqual(combined["total"]["validation_reduction_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()

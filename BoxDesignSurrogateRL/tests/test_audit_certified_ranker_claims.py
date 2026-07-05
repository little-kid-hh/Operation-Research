from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_certified_ranker_claims import audit_claims


def window(label: str, *, pf_delta: float = 0.0, ranker_validations: int = 80) -> dict[str, object]:
    return {
        "label": label,
        "pf_delta": pf_delta,
        "exact_coverage": 1.0,
        "ranker_audit_coverage": 1.0,
        "exact_uncovered": 0,
        "ranker_audit_uncovered": 0,
        "exact_validations": 100,
        "ranker_audit_validations": ranker_validations,
        "exact_uncached_boxes": 50,
        "ranker_audit_uncached_boxes": 40,
        "exact_subprocess_seconds": 30.0,
        "ranker_audit_subprocess_seconds": 24.0,
        "exact_elapsed_seconds": 60.0,
        "ranker_audit_elapsed_seconds": 48.0,
    }


class AuditCertifiedRankerClaimsTest(unittest.TestCase):
    def write_summary(self, windows: list[dict[str, object]]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        summary_path = Path(tmp.name) / "summary.json"
        summary_path.write_text(json.dumps({"windows": windows}), encoding="utf-8")
        return summary_path

    def test_audit_passes_when_quality_feasibility_and_cost_checks_pass(self) -> None:
        summary_path = self.write_summary([window("w0"), window("w1", pf_delta=-0.01)])

        audit = audit_claims(
            summary_path,
            min_windows=2,
            bootstrap_iterations=20,
            min_bootstrap_low_pct=None,
            max_sign_test_p=None,
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["verdict"], "PASS")
        self.assertTrue(all(check["passed"] for check in audit["checks"]))

    def test_audit_fails_on_pf_regression(self) -> None:
        summary_path = self.write_summary([window("w0"), window("w1", pf_delta=0.01)])

        audit = audit_claims(
            summary_path,
            min_windows=2,
            bootstrap_iterations=20,
            min_bootstrap_low_pct=None,
            max_sign_test_p=None,
        )

        self.assertFalse(audit["passed"])
        failed = {check["name"] for check in audit["checks"] if not check["passed"]}
        self.assertIn("no_pf_regression_after_exact_audit", failed)

    def test_audit_fails_on_cost_increase(self) -> None:
        summary_path = self.write_summary([window("w0"), window("w1", ranker_validations=120)])

        audit = audit_claims(
            summary_path,
            min_windows=2,
            bootstrap_iterations=20,
            min_bootstrap_low_pct=None,
            max_sign_test_p=None,
        )

        self.assertFalse(audit["passed"])
        failed = {check["name"] for check in audit["checks"] if not check["passed"]}
        self.assertIn("cost_validations", failed)


if __name__ == "__main__":
    unittest.main()

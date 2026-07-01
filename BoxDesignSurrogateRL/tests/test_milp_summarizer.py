from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_summarizer_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "summarize_milp_box_runs.py"
    spec = importlib.util.spec_from_file_location("summarize_milp_box_runs", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(
    *,
    algorithm: str,
    comparison_label: str,
    config_label: str,
    seed: int,
    pf: float,
) -> dict:
    return {
        "algorithm": algorithm,
        "comparison_label": comparison_label,
        "config_label": config_label,
        "code_version": "test",
        "orders_limit": 50,
        "k": 10,
        "seed": seed,
        "orientation_label": "label_6ori",
        "xml_path": "dev.xml",
        "milp_time_limit_seconds": 1.0,
        "coverage_repair": "geometric_expand",
        "repair_margins": [1.0],
        "repair_max_rounds": 1,
        "initial_score": {"packaging_factor": 3.0, "uncovered_orders": 0},
        "best_score": {
            "packaging_factor": pf,
            "mean_box_volume": pf * 10.0,
            "mean_order_volume": 10.0,
            "coverage_rate": 1.0,
            "uncovered_orders": 0,
            "unknown_pairs": 0,
            "orders_with_unknown": 0,
        },
    }


class MilpSummarizerTest(unittest.TestCase):
    def test_pairs_by_config_label_within_comparison_label(self) -> None:
        summarizer = _load_summarizer_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = [
                _summary(
                    algorithm="paper_fixed_step",
                    comparison_label="two_sweep",
                    config_label="fixed05_i2",
                    seed=0,
                    pf=2.0,
                ),
                _summary(
                    algorithm="staged_greedy",
                    comparison_label="two_sweep",
                    config_label="staged05_025_i2",
                    seed=0,
                    pf=2.1,
                ),
                _summary(
                    algorithm="staged_greedy",
                    comparison_label="one_sweep",
                    config_label="staged025_i1",
                    seed=0,
                    pf=2.2,
                ),
            ]
            for idx, payload in enumerate(payloads):
                run_dir = root / f"run_{idx}"
                run_dir.mkdir()
                with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump(payload, f)

            runs = [summarizer.load_run(path) for path in summarizer.find_summary_files([root])]
            pairs, skipped = summarizer.paired_deltas(runs, "fixed05_i2", "staged05_025_i2")

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["comparison_label"], "two_sweep")
        self.assertAlmostEqual(pairs[0]["delta_pf"], 0.1)
        self.assertEqual(len(skipped), 1)


if __name__ == "__main__":
    unittest.main()

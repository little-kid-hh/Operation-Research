from __future__ import annotations

import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from box_design_surrogate.evaluator import Box
from box_design_surrogate.features import summarize_items
from box_design_surrogate.milp_oracle import score_milp_feasibility_matrix


def _load_runner_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_milp_box_algorithms.py"
    spec = importlib.util.spec_from_file_location("run_milp_box_algorithms", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AggregateOracle:
    def evaluate(self, orders, boxes):
        feasible = []
        for order in orders:
            row = []
            for box in boxes:
                row.append(
                    max(order.dim_l) <= box.length
                    and max(order.dim_m) <= box.width
                    and max(order.dim_s) <= box.height
                    and order.total_volume <= box.volume
                )
            feasible.append(row)
        return score_milp_feasibility_matrix(orders, boxes, np.asarray(feasible, dtype=bool))


class MilpRunnerRepairTest(unittest.TestCase):
    def test_boxes_from_json_loads_checkpoint_boxes(self) -> None:
        runner = _load_runner_module()
        payload = [
            {"box_id": 2, "length": "3.5", "width": 4, "height": 5, "volume": 70},
            {"box_id": 1, "length": 2, "width": 3, "height": 4, "volume": 24},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_boxes.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            boxes = runner.boxes_from_json(path)

        self.assertEqual([box.box_id for box in boxes], [1, 2])
        self.assertEqual((boxes[0].length, boxes[0].width, boxes[0].height), (2.0, 3.0, 4.0))
        self.assertEqual((boxes[1].length, boxes[1].width, boxes[1].height), (3.5, 4.0, 5.0))

    def test_geometric_repair_can_remove_uncovered_order(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(2.0, 2.0, 2.0)])]
        boxes = [Box(0, 1.0, 1.0, 1.0)]
        oracle = AggregateOracle()
        initial_score = oracle.evaluate(orders, boxes)

        repaired_boxes, repaired_score, trace = runner.repair_coverage_by_expansion(
            oracle=oracle,
            orders=orders,
            boxes=boxes,
            initial_score=initial_score,
            margins=[1.0],
            max_rounds=1,
        )

        self.assertEqual(initial_score.uncovered_orders, 1)
        self.assertEqual(repaired_score.uncovered_orders, 0)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["phase"], "coverage_repair")
        self.assertTrue(trace[0]["improved"])
        self.assertEqual(trace[0]["candidate_evaluations"], 1)
        self.assertEqual((repaired_boxes[0].length, repaired_boxes[0].width, repaired_boxes[0].height), (2.0, 2.0, 2.0))

    def test_staged_greedy_writes_time_limit_trace_row(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])]
        boxes = [Box(0, 2.0, 2.0, 2.0)]
        oracle = AggregateOracle()

        best_boxes, best_score, trace = runner.run_staged_greedy(
            oracle=oracle,
            orders=orders,
            boxes=boxes,
            schedule=[(0.25, 10)],
            deadline=time.perf_counter() - 1.0,
        )

        self.assertEqual([box.box_id for box in best_boxes], [0])
        self.assertEqual(best_score.uncovered_orders, 0)
        self.assertEqual(trace[-1]["action"], "time_limit")
        self.assertEqual(trace[-1]["stop_reason"], "time_limit")
        self.assertEqual(trace[-1]["candidate_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()

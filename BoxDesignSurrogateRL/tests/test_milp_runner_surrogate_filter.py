from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from box_design_surrogate.evaluator import Box, BoxSetEvaluation
from box_design_surrogate.features import summarize_items
from box_design_surrogate.milp_oracle import MilpBoxSetScore


def _load_runner_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_milp_box_algorithms.py"
    spec = importlib.util.spec_from_file_location("run_milp_box_algorithms", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _surrogate_eval(cost: float) -> BoxSetEvaluation:
    return BoxSetEvaluation(
        total_adjusted_cost=cost,
        total_base_cost=cost,
        uncovered_orders=0,
        coverage_rate=1.0,
        mean_assigned_probability=0.99,
        low_margin_assignments=0,
        assignments=tuple(),
    )


def _milp_score(pf: float) -> MilpBoxSetScore:
    return MilpBoxSetScore(
        packaging_factor=pf,
        mean_box_volume=pf,
        mean_order_volume=1.0,
        coverage_rate=1.0,
        uncovered_orders=0,
        unknown_pairs=0,
        orders_with_unknown=0,
        assignments=(0,),
    )


class RankingSurrogate:
    def __init__(self, preferred_dimension: str) -> None:
        self.preferred_dimension = preferred_dimension
        self.scored_candidates = 0

    def evaluate_many_box_sets(self, candidates, *, assignment_mode="risk_adjusted", candidate_batch_size=None):
        self.scored_candidates += len(candidates)
        out = []
        for candidate in candidates:
            box = candidate[0]
            if self.preferred_dimension == "width" and box.width < 3.0:
                out.append(_surrogate_eval(1.0))
            elif self.preferred_dimension == "length" and box.length > 3.0:
                out.append(_surrogate_eval(1.0))
            else:
                out.append(_surrogate_eval(100.0))
        return out


class WidthImprovesOracle:
    def __init__(self) -> None:
        self.evaluated_boxes: list[Box] = []

    def evaluate(self, orders, boxes):
        box = boxes[0]
        self.evaluated_boxes.append(box)
        return _milp_score(5.0 if box.width < 3.0 else 11.0)


class SurrogateFilterTest(unittest.TestCase):
    def test_filter_only_sends_top_k_candidates_to_milp(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])]
        current = [Box(0, 3.0, 3.0, 3.0)]
        oracle = WidthImprovesOracle()
        surrogate = RankingSurrogate(preferred_dimension="width")

        best_boxes, best_score, action, metrics = runner.best_single_action_surrogate_filtered(
            oracle=oracle,
            surrogate=surrogate,
            orders=orders,
            current=current,
            current_score=_milp_score(10.0),
            step=1.0,
            top_k=1,
            rank_mode="paper_pf_surrogate",
            candidate_batch_size=None,
        )

        self.assertEqual(surrogate.scored_candidates, 6)
        self.assertEqual(len(oracle.evaluated_boxes), 1)
        self.assertEqual(action, "0:width:-1.000000")
        self.assertEqual(best_score.packaging_factor, 5.0)
        self.assertEqual(best_boxes[0].width, 2.0)
        self.assertEqual(metrics["generated_candidates"], 6)
        self.assertEqual(metrics["milp_validated_candidates"], 1)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 5)

    def test_surrogate_top_candidate_is_not_accepted_without_milp_improvement(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])]
        current = [Box(0, 3.0, 3.0, 3.0)]
        oracle = WidthImprovesOracle()
        surrogate = RankingSurrogate(preferred_dimension="length")

        best_boxes, best_score, action, metrics = runner.best_single_action_surrogate_filtered(
            oracle=oracle,
            surrogate=surrogate,
            orders=orders,
            current=current,
            current_score=_milp_score(10.0),
            step=1.0,
            top_k=1,
            rank_mode="paper_pf_surrogate",
            candidate_batch_size=None,
        )

        self.assertEqual(len(oracle.evaluated_boxes), 1)
        self.assertEqual(action, "noop")
        self.assertEqual(best_score.packaging_factor, 10.0)
        self.assertEqual(best_boxes, current)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 5)


if __name__ == "__main__":
    unittest.main()

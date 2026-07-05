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
    return _milp_score_with_assignments(pf, (0,))


def _milp_score_with_assignments(pf: float, assignments: tuple[int | None, ...]) -> MilpBoxSetScore:
    return MilpBoxSetScore(
        packaging_factor=pf,
        mean_box_volume=pf,
        mean_order_volume=1.0,
        coverage_rate=1.0,
        uncovered_orders=0,
        unknown_pairs=0,
        orders_with_unknown=0,
        assignments=assignments,
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


class WidthShrinkRanker:
    def predict_scores(self, rows):
        return [
            0.0 if row["move_dimension"] == "width" and row["move_direction"] == "shrink" else 100.0
            for row in rows
        ]


class HeightExpansionBestOracle:
    def __init__(self) -> None:
        self.evaluated_boxes: list[Box] = []

    def evaluate(self, orders, boxes):
        box = boxes[0]
        self.evaluated_boxes.append(box)
        if box.height > 3.0:
            return _milp_score(4.0)
        if box.width < 3.0:
            return _milp_score(5.0)
        return _milp_score(11.0)


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
            adaptive_top_k=None,
            noop_fallback=False,
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
            adaptive_top_k=None,
            noop_fallback=False,
            rank_mode="paper_pf_surrogate",
            candidate_batch_size=None,
        )

        self.assertEqual(len(oracle.evaluated_boxes), 1)
        self.assertEqual(action, "noop")
        self.assertEqual(best_score.packaging_factor, 10.0)
        self.assertEqual(best_boxes, current)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 5)

    def test_noop_fallback_recovers_improvement_filtered_out_by_surrogate(self) -> None:
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
            adaptive_top_k=None,
            noop_fallback=True,
            rank_mode="paper_pf_surrogate",
            candidate_batch_size=None,
        )

        self.assertEqual(len(oracle.evaluated_boxes), 6)
        self.assertEqual(action, "0:width:-1.000000")
        self.assertEqual(best_score.packaging_factor, 5.0)
        self.assertEqual(best_boxes[0].width, 2.0)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 0)
        self.assertTrue(metrics["surrogate_noop_fallback_used"])

    def test_adaptive_top_k_widens_only_until_it_finds_an_improvement(self) -> None:
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
            adaptive_top_k=[1, 3, 6],
            noop_fallback=False,
            rank_mode="paper_pf_surrogate",
            candidate_batch_size=None,
        )

        self.assertEqual(len(oracle.evaluated_boxes), 3)
        self.assertEqual(action, "0:width:-1.000000")
        self.assertEqual(best_score.packaging_factor, 5.0)
        self.assertEqual(best_boxes[0].width, 2.0)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 3)
        self.assertEqual(metrics["surrogate_tiers_evaluated"], 2)
        self.assertFalse(metrics["surrogate_noop_fallback_used"])

    def test_exact_candidate_trace_marks_accepted_move(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])]
        current = [Box(0, 3.0, 3.0, 3.0)]
        oracle = WidthImprovesOracle()
        candidate_trace = []

        best_boxes, best_score, action, metrics = runner.best_single_action(
            oracle=oracle,
            orders=orders,
            current=current,
            current_score=_milp_score(10.0),
            step=1.0,
            candidate_trace=candidate_trace,
            candidate_trace_context={
                "source_run_id": "toy_run",
                "algorithm": "paper_fixed_step",
                "phase": "fixed_step",
                "stage": 1,
                "iteration": 1,
            },
        )

        self.assertEqual(action, "0:width:-1.000000")
        self.assertEqual(best_score.packaging_factor, 5.0)
        self.assertEqual(best_boxes[0].width, 2.0)
        self.assertEqual(metrics["generated_candidates"], 6)
        self.assertEqual(len(candidate_trace), 6)
        accepted = [row for row in candidate_trace if row["is_accepted"] == 1]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["move_dimension"], "width")
        self.assertEqual(accepted[0]["move_direction"], "shrink")
        self.assertEqual(accepted[0]["candidate_rank"], 1)
        self.assertGreater(accepted[0]["candidate_pf_delta"], 0.0)

    def test_ranker_filter_only_sends_top_k_candidates_to_milp(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])]
        current = [Box(0, 3.0, 3.0, 3.0)]
        oracle = WidthImprovesOracle()

        best_boxes, best_score, action, metrics = runner.best_single_action_ranker_filtered(
            oracle=oracle,
            ranker=WidthShrinkRanker(),
            orders=orders,
            current=current,
            current_score=_milp_score(10.0),
            step=1.0,
            stage=1,
            iteration=1,
            top_k=1,
            adaptive_top_k=None,
            noop_fallback=False,
        )

        self.assertEqual(len(oracle.evaluated_boxes), 1)
        self.assertEqual(action, "0:width:-1.000000")
        self.assertEqual(best_score.packaging_factor, 5.0)
        self.assertEqual(best_boxes[0].width, 2.0)
        self.assertEqual(metrics["generated_candidates"], 6)
        self.assertEqual(metrics["ranker_scored_candidates"], 6)
        self.assertEqual(metrics["milp_validated_candidates"], 1)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 5)

    def test_ranker_expansion_safety_recovers_low_ranked_expansion(self) -> None:
        runner = _load_runner_module()
        orders = [summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])]
        current = [Box(0, 3.0, 3.0, 3.0)]
        oracle = HeightExpansionBestOracle()

        best_boxes, best_score, action, metrics = runner.best_single_action_ranker_filtered(
            oracle=oracle,
            ranker=WidthShrinkRanker(),
            orders=orders,
            current=current,
            current_score=_milp_score(10.0),
            step=1.0,
            stage=1,
            iteration=1,
            top_k=1,
            adaptive_top_k=None,
            noop_fallback=False,
            safety_policy="all_expansions",
        )

        self.assertEqual(len(oracle.evaluated_boxes), 4)
        self.assertEqual(action, "0:height:+1.000000")
        self.assertEqual(best_score.packaging_factor, 4.0)
        self.assertEqual(best_boxes[0].height, 4.0)
        self.assertEqual(metrics["milp_validated_candidates"], 4)
        self.assertEqual(metrics["milp_candidate_evaluations_avoided"], 2)
        self.assertEqual(metrics["ranker_safety_policy"], "all_expansions")
        self.assertEqual(metrics["ranker_safety_candidates"], 3)


if __name__ == "__main__":
    unittest.main()

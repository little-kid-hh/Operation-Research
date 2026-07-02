from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from box_design_surrogate.evaluator import Box
from box_design_surrogate.features import summarize_items
from box_design_surrogate.milp_oracle import (
    JavaMilpOracle,
    MilpLabelTableOracle,
    score_milp_feasibility_matrix,
)


class MilpOracleTest(unittest.TestCase):
    def test_score_milp_feasibility_matrix_assigns_smallest_feasible_box(self) -> None:
        orders = [
            summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "1", [(2.0, 1.0, 1.0)]),
        ]
        boxes = [
            Box(10, 1.0, 1.0, 1.0),
            Box(20, 2.0, 2.0, 1.0),
        ]
        feasible = np.asarray(
            [
                [True, True],
                [False, True],
            ],
            dtype=bool,
        )

        score = score_milp_feasibility_matrix(orders, boxes, feasible)

        self.assertEqual(score.uncovered_orders, 0)
        self.assertEqual(score.unknown_pairs, 0)
        self.assertEqual(score.orders_with_unknown, 0)
        self.assertEqual(score.assignments, (10, 20))
        self.assertAlmostEqual(score.mean_order_volume, 1.5)
        self.assertAlmostEqual(score.mean_box_volume, 2.5)
        self.assertAlmostEqual(score.packaging_factor, 2.5 / 1.5)

    def test_score_milp_feasibility_matrix_penalizes_uncovered_orders(self) -> None:
        orders = [
            summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "1", [(2.0, 1.0, 1.0)]),
        ]
        boxes = [Box(10, 1.0, 1.0, 1.0)]
        feasible = np.asarray([[True], [False]], dtype=bool)

        score = score_milp_feasibility_matrix(orders, boxes, feasible)

        self.assertEqual(score.uncovered_orders, 1)
        self.assertEqual(score.coverage_rate, 0.5)
        self.assertEqual(score.unknown_pairs, 0)
        self.assertEqual(score.orders_with_unknown, 0)
        self.assertEqual(score.assignments, (10, None))
        self.assertAlmostEqual(score.mean_box_volume, 50.5)

    def test_score_milp_feasibility_matrix_tracks_unknown_labels(self) -> None:
        orders = [
            summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "1", [(2.0, 1.0, 1.0)]),
        ]
        boxes = [
            Box(10, 1.0, 1.0, 1.0),
            Box(20, 2.0, 2.0, 1.0),
        ]
        feasible = np.asarray([[True, False], [False, False]], dtype=bool)
        unknown = np.asarray([[False, False], [True, True]], dtype=bool)

        score = score_milp_feasibility_matrix(orders, boxes, feasible, unknown=unknown)

        self.assertEqual(score.uncovered_orders, 1)
        self.assertEqual(score.unknown_pairs, 2)
        self.assertEqual(score.orders_with_unknown, 1)

    def test_label_table_oracle_uses_precomputed_labels(self) -> None:
        orders = [
            summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "1", [(2.0, 1.0, 1.0)]),
        ]
        boxes = [
            Box(10, 1.0, 1.0, 1.0),
            Box(20, 2.0, 2.0, 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.csv"
            with labels_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "instance_name",
                        "order_id",
                        "package_id",
                        "package_l",
                        "package_w",
                        "package_h",
                        "label_2ori",
                        "time_2ori_ms",
                        "label_6ori",
                        "time_6ori_ms",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "instance_name": "toy.xml",
                            "order_id": "0",
                            "package_id": "10",
                            "package_l": "1.0",
                            "package_w": "1.0",
                            "package_h": "1.0",
                            "label_2ori": "1",
                            "time_2ori_ms": "1",
                            "label_6ori": "1",
                            "time_6ori_ms": "1",
                        },
                        {
                            "instance_name": "toy.xml",
                            "order_id": "0",
                            "package_id": "20",
                            "package_l": "2.0",
                            "package_w": "2.0",
                            "package_h": "1.0",
                            "label_2ori": "1",
                            "time_2ori_ms": "1",
                            "label_6ori": "1",
                            "time_6ori_ms": "1",
                        },
                        {
                            "instance_name": "toy.xml",
                            "order_id": "1",
                            "package_id": "10",
                            "package_l": "1.0",
                            "package_w": "1.0",
                            "package_h": "1.0",
                            "label_2ori": "0",
                            "time_2ori_ms": "1",
                            "label_6ori": "0",
                            "time_6ori_ms": "1",
                        },
                        {
                            "instance_name": "toy.xml",
                            "order_id": "1",
                            "package_id": "20",
                            "package_l": "2.0",
                            "package_w": "2.0",
                            "package_h": "1.0",
                            "label_2ori": "1",
                            "time_2ori_ms": "1",
                            "label_6ori": "1",
                            "time_6ori_ms": "1",
                        },
                    ]
                )

            score = MilpLabelTableOracle(labels_path=labels_path).evaluate(orders, boxes)

        self.assertEqual(score.uncovered_orders, 0)
        self.assertEqual(score.assignments, (10, 20))

    def test_label_table_oracle_rejects_unlabeled_box_dimensions(self) -> None:
        order = summarize_items("toy.xml", "0", [(1.0, 1.0, 1.0)])
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.csv"
            with labels_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "instance_name",
                        "order_id",
                        "package_id",
                        "package_l",
                        "package_w",
                        "package_h",
                        "label_2ori",
                        "time_2ori_ms",
                        "label_6ori",
                        "time_6ori_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "instance_name": "toy.xml",
                        "order_id": "0",
                        "package_id": "10",
                        "package_l": "1.0",
                        "package_w": "1.0",
                        "package_h": "1.0",
                        "label_2ori": "1",
                        "time_2ori_ms": "1",
                        "label_6ori": "1",
                        "time_6ori_ms": "1",
                    }
                )

            oracle = MilpLabelTableOracle(labels_path=labels_path)
            with self.assertRaisesRegex(ValueError, "pre-labeled package dimensions"):
                oracle.evaluate([order], [Box(99, 3.0, 3.0, 3.0)])

    def test_java_oracle_caches_feasibility_by_order_signature_and_box_dimensions(self) -> None:
        class FakeJavaOracle(JavaMilpOracle):
            def __init__(self) -> None:
                super().__init__(xml_path=Path("toy.xml"), java_classpath="unused")
                self.uncached_calls = 0

            def _validate_environment(self) -> None:
                return None

            def _evaluate_uncached(self, orders, boxes):
                self.uncached_calls += 1
                return np.ones((len(orders), len(boxes)), dtype=bool)

        orders = [
            summarize_items("toy.xml", "10", [(1.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "20", [(2.0, 1.0, 1.0)]),
        ]
        oracle = FakeJavaOracle()
        boxes = [Box(0, 2.0, 2.0, 2.0), Box(1, 3.0, 3.0, 3.0)]

        oracle.evaluate(orders, boxes)
        self.assertEqual(oracle.uncached_calls, 1)
        self.assertEqual(
            oracle.cache_info(),
            {
                "entries": 2,
                "hits": 0,
                "misses": 2,
                "disk_hits": 0,
                "evaluate_calls": 1,
                "prefetch_calls": 0,
                "prefetch_cache_hits": 0,
                "prefetch_disk_hits": 0,
                "prefetch_misses": 0,
                "uncached_batches": 1,
                "uncached_boxes": 2,
                "subprocess_seconds": 0.0,
            },
        )

        oracle.evaluate(orders, boxes)
        self.assertEqual(oracle.uncached_calls, 1)
        self.assertEqual(
            oracle.cache_info(),
            {
                "entries": 2,
                "hits": 2,
                "misses": 2,
                "disk_hits": 0,
                "evaluate_calls": 2,
                "prefetch_calls": 0,
                "prefetch_cache_hits": 0,
                "prefetch_disk_hits": 0,
                "prefetch_misses": 0,
                "uncached_batches": 1,
                "uncached_boxes": 2,
                "subprocess_seconds": 0.0,
            },
        )

        oracle.evaluate(orders, [boxes[0], Box(2, 4.0, 4.0, 4.0)])
        self.assertEqual(oracle.uncached_calls, 2)
        self.assertEqual(
            oracle.cache_info(),
            {
                "entries": 3,
                "hits": 3,
                "misses": 3,
                "disk_hits": 0,
                "evaluate_calls": 3,
                "prefetch_calls": 0,
                "prefetch_cache_hits": 0,
                "prefetch_disk_hits": 0,
                "prefetch_misses": 0,
                "uncached_batches": 2,
                "uncached_boxes": 3,
                "subprocess_seconds": 0.0,
            },
        )

    def test_java_oracle_prefetches_unique_box_dimensions(self) -> None:
        class FakeJavaOracle(JavaMilpOracle):
            def __init__(self) -> None:
                super().__init__(xml_path=Path("toy.xml"), java_classpath="unused")
                self.uncached_calls = 0

            def _validate_environment(self) -> None:
                return None

            def _evaluate_uncached(self, orders, boxes):
                self.uncached_calls += 1
                return np.ones((len(orders), len(boxes)), dtype=bool)

        orders = [
            summarize_items("toy.xml", "10", [(1.0, 1.0, 1.0)]),
            summarize_items("toy.xml", "20", [(2.0, 1.0, 1.0)]),
        ]
        oracle = FakeJavaOracle()
        duplicate_boxes = [
            Box(0, 2.0, 2.0, 2.0),
            Box(1, 2.0, 2.0, 2.0),
            Box(2, 3.0, 3.0, 3.0),
        ]

        oracle.prefetch_box_statuses(orders, duplicate_boxes)

        self.assertEqual(oracle.uncached_calls, 1)
        self.assertEqual(
            oracle.cache_info(),
            {
                "entries": 2,
                "hits": 0,
                "misses": 2,
                "disk_hits": 0,
                "evaluate_calls": 0,
                "prefetch_calls": 1,
                "prefetch_cache_hits": 0,
                "prefetch_disk_hits": 0,
                "prefetch_misses": 2,
                "uncached_batches": 1,
                "uncached_boxes": 2,
                "subprocess_seconds": 0.0,
            },
        )

        oracle.evaluate(orders, [Box(10, 2.0, 2.0, 2.0), Box(11, 3.0, 3.0, 3.0)])

        self.assertEqual(oracle.uncached_calls, 1)
        self.assertEqual(
            oracle.cache_info(),
            {
                "entries": 2,
                "hits": 2,
                "misses": 2,
                "disk_hits": 0,
                "evaluate_calls": 1,
                "prefetch_calls": 1,
                "prefetch_cache_hits": 0,
                "prefetch_disk_hits": 0,
                "prefetch_misses": 2,
                "uncached_batches": 1,
                "uncached_boxes": 2,
                "subprocess_seconds": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()

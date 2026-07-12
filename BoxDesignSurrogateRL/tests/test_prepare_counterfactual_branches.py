from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_counterfactual_branches import prepare_branches, reconstruct_state_boxes


def row(box_id: int, candidate_rank: int, *, accepted: int = 0) -> dict[str, object]:
    return {
        "source_run_id": "run_a",
        "phase": "staged_greedy",
        "stage": 1,
        "iteration": 7,
        "generated_candidates": 12,
        "candidate_index": candidate_rank - 1,
        "candidate_rank": candidate_rank,
        "move_box_id": box_id,
        "move_dimension": "length",
        "move_delta": -0.25,
        "current_box_length": 10 + box_id,
        "current_box_width": 8 + box_id,
        "current_box_height": 6 + box_id,
        "current_packaging_factor": 2.0,
        "candidate_packaging_factor": 1.9 + candidate_rank / 100,
        "candidate_coverage_rate": 1.0,
        "candidate_uncovered_orders": 0,
        "candidate_unknown_pairs": 0,
        "is_exact_best": int(candidate_rank == 1),
        "is_accepted": accepted,
    }


class CounterfactualBranchTest(unittest.TestCase):
    def test_reconstructs_each_box_from_candidate_rows(self) -> None:
        rows = [row(0, 1), row(0, 2), row(1, 3), row(1, 4)]
        for candidate_rank in range(5, 13):
            rows.append(row(candidate_rank % 2, candidate_rank))
        boxes = reconstruct_state_boxes([{key: str(value) for key, value in item.items()} for item in rows])
        self.assertEqual([box["box_id"] for box in boxes], [0, 1])
        self.assertEqual(boxes[1]["length"], 11.0)

    def test_writes_ranked_branch_manifest(self) -> None:
        rows = [row(rank % 2, rank, accepted=int(rank == 1)) for rank in range(1, 13)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.csv"
            with trace.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records = prepare_branches(
                trace_path=trace,
                out_dir=root / "branches",
                iterations=[7],
                candidate_ranks=[1, 5],
            )
            self.assertEqual([record["candidate_rank"] for record in records], [1, 5])
            self.assertEqual(records[0]["forced_first_move"], "1:length:-0.25")
            boxes = json.loads((root / "branches" / "state_staged_greedy_s1_i7.json").read_text())
            self.assertEqual(len(boxes), 2)


if __name__ == "__main__":
    unittest.main()

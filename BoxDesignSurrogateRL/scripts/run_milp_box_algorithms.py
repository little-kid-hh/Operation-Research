#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_design_surrogate.evaluator import BatchSurrogateEvaluator, Box, BoxSetEvaluation, SurrogateEvaluator
from box_design_surrogate.candidate_ranker import (
    CANDIDATE_TRACE_COLUMNS,
    CandidateRanker,
    candidate_feature_row,
    candidate_trace_rows as build_candidate_trace_rows,
)
from box_design_surrogate.features import read_order_summaries
from box_design_surrogate.kandula_repro import initial_boxes_kmeans, order_requirement
from box_design_surrogate.milp_oracle import (
    JavaMilpOracle,
    MilpBoxSetScore,
    MilpLabelTableOracle,
)
from box_design_surrogate.search import apply_move, coordinate_moves


class BoxSetOracle(Protocol):
    def evaluate(self, orders: list, boxes: list[Box]) -> MilpBoxSetScore:
        ...


class CandidateSurrogate(Protocol):
    def evaluate_many_box_sets(
        self,
        candidates: list[list[Box]],
        *,
        assignment_mode: str = "risk_adjusted",
        candidate_batch_size: int | None = None,
    ) -> list[BoxSetEvaluation]:
        ...


def parse_schedule(value: str) -> list[tuple[float, int]]:
    schedule: list[tuple[float, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step_s, iter_s = item.split(":", 1)
        step = float(step_s)
        iterations = int(iter_s)
        if step <= 0 or iterations < 0:
            raise argparse.ArgumentTypeError("schedule entries must be positive_step:nonnegative_iters")
        schedule.append((step, iterations))
    if not schedule:
        raise argparse.ArgumentTypeError("schedule must contain at least one step:iters entry")
    return schedule


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float value")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("all float-list values must be positive")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all integer-list values must be positive")
    return values


def score_rank(score: MilpBoxSetScore) -> tuple[int, int, float]:
    return score.uncovered_orders, score.unknown_pairs, score.packaging_factor


def boxes_to_rows(boxes: list[Box]) -> list[dict[str, float | int]]:
    return [
        {
            "box_id": box.box_id,
            "length": box.length,
            "width": box.width,
            "height": box.height,
            "volume": box.volume,
        }
        for box in sorted(boxes, key=lambda b: b.box_id)
    ]


def boxes_from_json(path: Path) -> list[Box]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of boxes")

    boxes: list[Box] = []
    seen_ids: set[int] = set()
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path} box row {idx} must be an object")
        try:
            box_id = int(row["box_id"])
            length = float(row["length"])
            width = float(row["width"])
            height = float(row["height"])
        except KeyError as exc:
            raise ValueError(f"{path} box row {idx} is missing {exc.args[0]!r}") from exc
        if box_id in seen_ids:
            raise ValueError(f"{path} contains duplicate box_id {box_id}")
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path} box_id {box_id} has non-positive dimensions")
        seen_ids.add(box_id)
        boxes.append(Box(box_id=box_id, length=length, width=width, height=height))
    if not boxes:
        raise ValueError(f"{path} must contain at least one box")
    return sorted(boxes, key=lambda b: b.box_id)


def score_to_dict(score: MilpBoxSetScore) -> dict:
    out = asdict(score)
    out["assignments"] = list(score.assignments)
    return out


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def time_limit_row(
    *,
    phase: str,
    iteration: int,
    step: float,
    score: MilpBoxSetScore,
    stage: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "iteration": iteration,
        "step": step,
        "action": "time_limit",
        "improved": False,
        "stop_reason": "time_limit",
        "candidate_evaluations": 0,
        "generated_candidates": 0,
        "surrogate_scored_candidates": 0,
        "ranker_scored_candidates": 0,
        "milp_validated_candidates": 0,
        "milp_candidate_evaluations_avoided": 0,
        "milp_avoidance_rate": 0.0,
        "surrogate_eval_seconds": 0.0,
        "ranker_eval_seconds": 0.0,
        "milp_eval_seconds": 0.0,
        **score_to_dict(score),
    }
    if stage is not None:
        row["stage"] = stage
    return row


def git_info(repo_root: Path) -> dict[str, str | bool | None]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def make_oracle(args: argparse.Namespace) -> BoxSetOracle:
    if args.oracle == "labels":
        return MilpLabelTableOracle(
            labels_path=args.labels_path,
            orientation_label=args.orientation_label,
        )

    classpath_parts = [str(args.java_classes)]
    gurobi_jar = os.environ.get("GUROBI_JAR")
    if gurobi_jar:
        classpath_parts.append(gurobi_jar)
    if args.java_classpath:
        classpath_parts.append(args.java_classpath)

    return JavaMilpOracle(
        xml_path=args.xml_path,
        java_classpath=os.pathsep.join(classpath_parts),
        orientation_label=args.orientation_label,
        time_limit_seconds=args.milp_time_limit_seconds,
        allow_bsp_derived_data=args.allow_bsp_derived_data,
        cache_dir=args.oracle_cache_dir,
    )


def make_surrogate_evaluator(args: argparse.Namespace, orders: list) -> BatchSurrogateEvaluator:
    hybrid_root = REPO_ROOT / "HybridSVM"
    if str(hybrid_root) not in sys.path:
        sys.path.insert(0, str(hybrid_root))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.ensemble_train import load_ensemble_pipeline

    model = load_ensemble_pipeline(args.model_path)
    patch_sklearn_model_compat(model)
    evaluator = SurrogateEvaluator(
        model=model,
        tau=args.tau,
        tau_high=args.tau_high,
        lambda_risk=args.lambda_risk,
    )
    return BatchSurrogateEvaluator.from_evaluator(evaluator, orders)


def make_candidate_ranker(args: argparse.Namespace) -> CandidateRanker:
    if args.candidate_ranker_path is None:
        raise ValueError("--candidate-ranker-path is required for ranker_filtered_greedy")
    return CandidateRanker.load(args.candidate_ranker_path)


def select_order_window(orders: list, *, offset: int = 0, limit: int | None = None) -> list:
    if offset < 0:
        raise ValueError("--orders-offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("--orders-limit must be non-negative when supplied")
    selected = orders[offset:] if limit is None else orders[offset : offset + limit]
    if not selected:
        raise ValueError(
            "selected order window is empty: "
            f"orders_offset={offset}, orders_limit={limit}, available_orders={len(orders)}"
        )
    return selected


def patch_sklearn_model_compat(model: object) -> None:
    """Patch narrow sklearn persistence gaps seen across minor versions."""

    estimators = []
    base_models = getattr(model, "base_models", None)
    if isinstance(base_models, dict):
        estimators.extend(base_models.values())
    meta_model = getattr(model, "meta_model", None)
    if meta_model is not None:
        estimators.append(meta_model)

    for estimator in estimators:
        if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
            setattr(estimator, "multi_class", "auto")


def replace_box(boxes: list[Box], replacement: Box) -> list[Box]:
    return [replacement if box.box_id == replacement.box_id else box for box in boxes]


def prefetch_candidate_statuses(
    *,
    oracle: BoxSetOracle,
    orders: list,
    candidates: list[list[Box]],
    enabled: bool,
) -> float:
    if not enabled or not candidates or not hasattr(oracle, "prefetch_box_statuses"):
        return 0.0
    unique: dict[tuple[float, float, float], Box] = {}
    for candidate in candidates:
        for box in candidate:
            key = (round(float(box.length), 6), round(float(box.width), 6), round(float(box.height), 6))
            unique.setdefault(key, box)
    started = time.perf_counter()
    oracle.prefetch_box_statuses(orders, list(unique.values()))  # type: ignore[attr-defined]
    return time.perf_counter() - started


def expand_box_for_order(box: Box, order, margin: float) -> Box:
    req_l, req_m, req_s, total_volume = order_requirement(order)
    length = max(box.length, req_l * margin)
    width = max(box.width, req_m * margin)
    height = max(box.height, req_s * margin)
    target_volume = total_volume * (margin**3)
    volume = length * width * height
    if volume < target_volume:
        scale = float((target_volume / volume) ** (1.0 / 3.0))
        length *= scale
        width *= scale
        height *= scale
    return Box(box.box_id, float(length), float(width), float(height))


def same_box_dimensions(left: Box, right: Box, tolerance: float = 1e-9) -> bool:
    return (
        abs(left.length - right.length) <= tolerance
        and abs(left.width - right.width) <= tolerance
        and abs(left.height - right.height) <= tolerance
    )


def repair_coverage_by_expansion(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    initial_score: MilpBoxSetScore,
    margins: list[float],
    max_rounds: int,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score
    trace: list[dict] = []
    for repair_round in range(1, max_rounds + 1):
        if current_score.uncovered_orders == 0:
            break

        best_boxes = current
        best_score = current_score
        best_action = "noop"
        candidate_evaluations = 0
        uncovered_indices = [idx for idx, assignment in enumerate(current_score.assignments) if assignment is None]
        for order_idx in uncovered_indices:
            order = orders[order_idx]
            for box in current:
                for margin in margins:
                    replacement = expand_box_for_order(box, order, margin)
                    if same_box_dimensions(box, replacement):
                        continue
                    candidate = replace_box(current, replacement)
                    candidate_evaluations += 1
                    score = oracle.evaluate(orders, candidate)
                    if score_rank(score) < score_rank(best_score):
                        best_boxes = candidate
                        best_score = score
                        best_action = (
                            f"order={order.order_id};box={box.box_id};"
                            f"margin={margin:.6f}"
                        )

        improved = score_rank(best_score) < score_rank(current_score)
        trace.append(
            {
                "phase": "coverage_repair",
                "repair_round": repair_round,
                "iteration": 0,
                "action": best_action,
                "improved": improved,
                "candidate_evaluations": candidate_evaluations,
                **score_to_dict(best_score),
            }
        )
        if not improved:
            break
        current = best_boxes
        current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def best_single_action(
    *,
    oracle: BoxSetOracle,
    orders: list,
    current: list[Box],
    current_score: MilpBoxSetScore,
    step: float,
    candidate_trace: list[dict] | None = None,
    candidate_trace_context: dict[str, Any] | None = None,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, str, dict[str, Any]]:
    best_boxes = current
    best_score = current_score
    best_action = "noop"
    candidate_evaluations = 0
    milp_eval_seconds = 0.0
    prefetch_eval_seconds = 0.0
    moves = list(coordinate_moves(current, step))
    candidates: list[list[Box]] = []
    candidate_scores: list[MilpBoxSetScore] = []
    for move in moves:
        candidate = apply_move(current, move)
        candidates.append(candidate)
    prefetch_eval_seconds += prefetch_candidate_statuses(
        oracle=oracle,
        orders=orders,
        candidates=candidates,
        enabled=prefetch_candidate_statuses_enabled,
    )
    milp_eval_seconds += prefetch_eval_seconds
    for move, candidate in zip(moves, candidates):
        candidate_evaluations += 1
        eval_started = time.perf_counter()
        score = oracle.evaluate(orders, candidate)
        candidate_scores.append(score)
        milp_eval_seconds += time.perf_counter() - eval_started
        if score_rank(score) < score_rank(best_score):
            best_boxes = candidate
            best_score = score
            best_action = f"{move.box_id}:{move.dimension}:{move.delta:+.6f}"

    if candidate_trace is not None and candidate_trace_context is not None:
        candidate_trace.extend(
            build_candidate_trace_rows(
                source_run_id=str(candidate_trace_context.get("source_run_id", "")),
                algorithm=str(candidate_trace_context.get("algorithm", "")),
                phase=str(candidate_trace_context.get("phase", "")),
                stage=int(candidate_trace_context.get("stage", 0)),
                iteration=int(candidate_trace_context.get("iteration", 0)),
                step=step,
                current_boxes=current,
                current_score=current_score,
                moves=moves,
                candidates=candidates,
                candidate_scores=candidate_scores,
            )
        )
    metrics = {
        "candidate_evaluations": candidate_evaluations,
        "generated_candidates": candidate_evaluations,
        "surrogate_scored_candidates": 0,
        "milp_validated_candidates": candidate_evaluations,
        "milp_candidate_evaluations_avoided": 0,
        "milp_avoidance_rate": 0.0,
        "surrogate_eval_seconds": 0.0,
        "prefetch_eval_seconds": prefetch_eval_seconds,
        "milp_eval_seconds": milp_eval_seconds,
    }
    return best_boxes, best_score, best_action, metrics


def surrogate_rank(evaluation: BoxSetEvaluation, rank_mode: str) -> tuple[float, ...]:
    if rank_mode == "paper_pf_surrogate":
        return (
            float(evaluation.uncovered_orders),
            float(evaluation.total_base_cost),
            -float(evaluation.mean_assigned_probability),
        )
    if rank_mode == "risk_aware_surrogate":
        return (
            float(evaluation.uncovered_orders),
            float(evaluation.total_adjusted_cost),
            float(evaluation.total_base_cost),
            -float(evaluation.mean_assigned_probability),
        )
    raise ValueError(f"unknown surrogate rank mode: {rank_mode}")


def best_single_action_surrogate_filtered(
    *,
    oracle: BoxSetOracle,
    surrogate: CandidateSurrogate,
    orders: list,
    current: list[Box],
    current_score: MilpBoxSetScore,
    step: float,
    top_k: int,
    adaptive_top_k: list[int] | None,
    noop_fallback: bool,
    rank_mode: str,
    candidate_batch_size: int | None,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, str, dict[str, Any]]:
    moves = list(coordinate_moves(current, step))
    candidates = [apply_move(current, move) for move in moves]
    generated_candidates = len(candidates)
    if generated_candidates == 0:
        return current, current_score, "noop", {
            "candidate_evaluations": 0,
            "generated_candidates": 0,
            "surrogate_scored_candidates": 0,
            "milp_validated_candidates": 0,
            "milp_candidate_evaluations_avoided": 0,
            "milp_avoidance_rate": 0.0,
            "surrogate_eval_seconds": 0.0,
            "prefetch_eval_seconds": 0.0,
            "milp_eval_seconds": 0.0,
            "surrogate_top_k_sequence": "",
            "surrogate_tiers_evaluated": 0,
            "surrogate_noop_fallback_used": False,
        }

    assignment_mode = "min_volume" if rank_mode == "paper_pf_surrogate" else "risk_adjusted"
    surrogate_started = time.perf_counter()
    surrogate_evaluations = surrogate.evaluate_many_box_sets(
        candidates,
        assignment_mode=assignment_mode,
        candidate_batch_size=candidate_batch_size,
    )
    surrogate_eval_seconds = time.perf_counter() - surrogate_started
    if len(surrogate_evaluations) != generated_candidates:
        raise RuntimeError(
            "surrogate returned an unexpected number of candidate evaluations: "
            f"{len(surrogate_evaluations)} for {generated_candidates} candidates"
        )

    requested_top_k = adaptive_top_k if adaptive_top_k is not None else [top_k]
    top_k_sequence: list[int] = []
    for value in requested_top_k:
        keep = min(value, generated_candidates)
        if keep not in top_k_sequence:
            top_k_sequence.append(keep)
    if noop_fallback and generated_candidates not in top_k_sequence:
        top_k_sequence.append(generated_candidates)

    ranked_indices = sorted(
        range(generated_candidates),
        key=lambda idx: (surrogate_rank(surrogate_evaluations[idx], rank_mode), idx),
    )

    best_boxes = current
    best_score = current_score
    best_action = "noop"
    milp_eval_seconds = 0.0
    prefetch_eval_seconds = 0.0
    validated_indices: set[int] = set()
    tiers_evaluated = 0
    noop_fallback_used = False
    for tier_keep in top_k_sequence:
        tier_indices = ranked_indices[:tier_keep]
        new_indices = [idx for idx in tier_indices if idx not in validated_indices]
        if not new_indices:
            continue
        tiers_evaluated += 1
        if tier_keep == generated_candidates and len(validated_indices) > 0:
            noop_fallback_used = True
        prefetch_elapsed = prefetch_candidate_statuses(
            oracle=oracle,
            orders=orders,
            candidates=[candidates[idx] for idx in new_indices],
            enabled=prefetch_candidate_statuses_enabled,
        )
        prefetch_eval_seconds += prefetch_elapsed
        milp_eval_seconds += prefetch_elapsed
        for idx in new_indices:
            validated_indices.add(idx)
            eval_started = time.perf_counter()
            score = oracle.evaluate(orders, candidates[idx])
            milp_eval_seconds += time.perf_counter() - eval_started
            if score_rank(score) < score_rank(best_score):
                best_boxes = candidates[idx]
                best_score = score
                move = moves[idx]
                best_action = f"{move.box_id}:{move.dimension}:{move.delta:+.6f}"
        if score_rank(best_score) < score_rank(current_score):
            break

    avoided = generated_candidates - len(validated_indices)
    metrics = {
        "candidate_evaluations": len(validated_indices),
        "generated_candidates": generated_candidates,
        "surrogate_scored_candidates": generated_candidates,
        "milp_validated_candidates": len(validated_indices),
        "milp_candidate_evaluations_avoided": avoided,
        "milp_avoidance_rate": avoided / generated_candidates if generated_candidates else 0.0,
        "surrogate_eval_seconds": surrogate_eval_seconds,
        "prefetch_eval_seconds": prefetch_eval_seconds,
        "milp_eval_seconds": milp_eval_seconds,
        "surrogate_top_k_sequence": ",".join(str(value) for value in top_k_sequence),
        "surrogate_tiers_evaluated": tiers_evaluated,
        "surrogate_noop_fallback_used": noop_fallback_used,
    }
    return best_boxes, best_score, best_action, metrics


def best_single_action_ranker_filtered(
    *,
    oracle: BoxSetOracle,
    ranker: CandidateRanker,
    orders: list,
    current: list[Box],
    current_score: MilpBoxSetScore,
    step: float,
    stage: int,
    iteration: int,
    top_k: int,
    adaptive_top_k: list[int] | None,
    noop_fallback: bool,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, str, dict[str, Any]]:
    moves = list(coordinate_moves(current, step))
    candidates = [apply_move(current, move) for move in moves]
    generated_candidates = len(candidates)
    if generated_candidates == 0:
        return current, current_score, "noop", {
            "candidate_evaluations": 0,
            "generated_candidates": 0,
            "ranker_scored_candidates": 0,
            "milp_validated_candidates": 0,
            "milp_candidate_evaluations_avoided": 0,
            "milp_avoidance_rate": 0.0,
            "ranker_eval_seconds": 0.0,
            "prefetch_eval_seconds": 0.0,
            "milp_eval_seconds": 0.0,
            "ranker_top_k_sequence": "",
            "ranker_tiers_evaluated": 0,
            "ranker_noop_fallback_used": False,
        }

    feature_rows = [
        candidate_feature_row(
            current_boxes=current,
            candidate_boxes=candidate,
            move=move,
            current_score=current_score,
            step=step,
            stage=stage,
            iteration=iteration,
            candidate_index=idx,
            generated_candidates=generated_candidates,
        )
        for idx, (move, candidate) in enumerate(zip(moves, candidates))
    ]
    ranker_started = time.perf_counter()
    predicted_scores = ranker.predict_scores(feature_rows)
    ranker_eval_seconds = time.perf_counter() - ranker_started
    if len(predicted_scores) != generated_candidates:
        raise RuntimeError(
            "ranker returned an unexpected number of scores: "
            f"{len(predicted_scores)} for {generated_candidates} candidates"
        )

    requested_top_k = adaptive_top_k if adaptive_top_k is not None else [top_k]
    top_k_sequence: list[int] = []
    for value in requested_top_k:
        keep = min(value, generated_candidates)
        if keep not in top_k_sequence:
            top_k_sequence.append(keep)
    if noop_fallback and generated_candidates not in top_k_sequence:
        top_k_sequence.append(generated_candidates)

    ranked_indices = sorted(range(generated_candidates), key=lambda idx: (float(predicted_scores[idx]), idx))

    best_boxes = current
    best_score = current_score
    best_action = "noop"
    milp_eval_seconds = 0.0
    prefetch_eval_seconds = 0.0
    validated_indices: set[int] = set()
    tiers_evaluated = 0
    noop_fallback_used = False
    for tier_keep in top_k_sequence:
        tier_indices = ranked_indices[:tier_keep]
        new_indices = [idx for idx in tier_indices if idx not in validated_indices]
        if not new_indices:
            continue
        tiers_evaluated += 1
        if tier_keep == generated_candidates and len(validated_indices) > 0:
            noop_fallback_used = True
        prefetch_elapsed = prefetch_candidate_statuses(
            oracle=oracle,
            orders=orders,
            candidates=[candidates[idx] for idx in new_indices],
            enabled=prefetch_candidate_statuses_enabled,
        )
        prefetch_eval_seconds += prefetch_elapsed
        milp_eval_seconds += prefetch_elapsed
        for idx in new_indices:
            validated_indices.add(idx)
            eval_started = time.perf_counter()
            score = oracle.evaluate(orders, candidates[idx])
            milp_eval_seconds += time.perf_counter() - eval_started
            if score_rank(score) < score_rank(best_score):
                best_boxes = candidates[idx]
                best_score = score
                move = moves[idx]
                best_action = f"{move.box_id}:{move.dimension}:{move.delta:+.6f}"
        if score_rank(best_score) < score_rank(current_score):
            break

    avoided = generated_candidates - len(validated_indices)
    metrics = {
        "candidate_evaluations": len(validated_indices),
        "generated_candidates": generated_candidates,
        "ranker_scored_candidates": generated_candidates,
        "milp_validated_candidates": len(validated_indices),
        "milp_candidate_evaluations_avoided": avoided,
        "milp_avoidance_rate": avoided / generated_candidates if generated_candidates else 0.0,
        "ranker_eval_seconds": ranker_eval_seconds,
        "prefetch_eval_seconds": prefetch_eval_seconds,
        "milp_eval_seconds": milp_eval_seconds,
        "ranker_top_k_sequence": ",".join(str(value) for value in top_k_sequence),
        "ranker_tiers_evaluated": tiers_evaluated,
        "ranker_noop_fallback_used": noop_fallback_used,
    }
    return best_boxes, best_score, best_action, metrics


def run_fixed_step(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    step: float,
    iterations: int,
    initial_score: MilpBoxSetScore | None = None,
    initial_phase: str = "initial",
    candidate_trace: list[dict] | None = None,
    source_run_id: str = "",
    deadline: float | None = None,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score if initial_score is not None else oracle.evaluate(orders, current)
    trace = [
        {
            "phase": initial_phase,
            "iteration": 0,
            "step": step,
            "action": "init",
            "candidate_evaluations": 0,
            **score_to_dict(current_score),
        }
    ]
    for iteration in range(1, iterations + 1):
        if deadline_reached(deadline):
            trace.append(
                time_limit_row(
                    phase="fixed_step",
                    iteration=iteration,
                    step=step,
                    score=current_score,
                    stage=1,
                )
            )
            break
        best_boxes, best_score, action, selection_metrics = best_single_action(
            oracle=oracle,
            orders=orders,
            current=current,
            current_score=current_score,
            step=step,
            candidate_trace=candidate_trace,
            candidate_trace_context={
                "source_run_id": source_run_id,
                "algorithm": "paper_fixed_step",
                "phase": "fixed_step",
                "stage": 1,
                "iteration": iteration,
            },
            prefetch_candidate_statuses_enabled=prefetch_candidate_statuses_enabled,
        )
        improved = score_rank(best_score) < score_rank(current_score)
        trace.append(
            {
                "phase": "fixed_step",
                "iteration": iteration,
                "step": step,
                "action": action,
                "improved": improved,
                **selection_metrics,
                **score_to_dict(best_score),
            }
        )
        if not improved:
            break
        current = best_boxes
        current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def run_staged_greedy(
    *,
    oracle: BoxSetOracle,
    orders: list,
    boxes: list[Box],
    schedule: list[tuple[float, int]],
    initial_score: MilpBoxSetScore | None = None,
    initial_phase: str = "initial",
    candidate_trace: list[dict] | None = None,
    source_run_id: str = "",
    deadline: float | None = None,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score if initial_score is not None else oracle.evaluate(orders, current)
    trace = [
        {
            "phase": initial_phase,
            "iteration": 0,
            "stage": 0,
            "step": schedule[0][0],
            "action": "init",
            "candidate_evaluations": 0,
            **score_to_dict(current_score),
        }
    ]
    global_iteration = 0
    for stage_idx, (step, iterations) in enumerate(schedule, start=1):
        for _ in range(iterations):
            if deadline_reached(deadline):
                trace.append(
                    time_limit_row(
                        phase="staged_greedy",
                        iteration=global_iteration + 1,
                        stage=stage_idx,
                        step=step,
                        score=current_score,
                    )
                )
                return sorted(current, key=lambda b: b.volume), current_score, trace
            global_iteration += 1
            best_boxes, best_score, action, selection_metrics = best_single_action(
                oracle=oracle,
                orders=orders,
                current=current,
                current_score=current_score,
                step=step,
                candidate_trace=candidate_trace,
                candidate_trace_context={
                    "source_run_id": source_run_id,
                    "algorithm": "staged_greedy",
                    "phase": "staged_greedy",
                    "stage": stage_idx,
                    "iteration": global_iteration,
                },
                prefetch_candidate_statuses_enabled=prefetch_candidate_statuses_enabled,
            )
            improved = score_rank(best_score) < score_rank(current_score)
            trace.append(
                {
                    "phase": "staged_greedy",
                    "iteration": global_iteration,
                    "stage": stage_idx,
                    "step": step,
                    "action": action,
                    "improved": improved,
                    **selection_metrics,
                    **score_to_dict(best_score),
                }
            )
            if not improved:
                break
            current = best_boxes
            current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def run_surrogate_filtered_greedy(
    *,
    oracle: BoxSetOracle,
    surrogate: CandidateSurrogate,
    orders: list,
    boxes: list[Box],
    schedule: list[tuple[float, int]],
    top_k: int,
    adaptive_top_k: list[int] | None,
    noop_fallback: bool,
    rank_mode: str,
    candidate_batch_size: int | None,
    initial_score: MilpBoxSetScore | None = None,
    initial_phase: str = "initial",
    deadline: float | None = None,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score if initial_score is not None else oracle.evaluate(orders, current)
    trace = [
        {
            "phase": initial_phase,
            "iteration": 0,
            "stage": 0,
            "step": schedule[0][0],
            "action": "init",
            "candidate_evaluations": 0,
            "generated_candidates": 0,
            "surrogate_scored_candidates": 0,
            "milp_validated_candidates": 0,
            "milp_candidate_evaluations_avoided": 0,
            "milp_avoidance_rate": 0.0,
            "surrogate_eval_seconds": 0.0,
            "milp_eval_seconds": 0.0,
            "surrogate_top_k_sequence": "",
            "surrogate_tiers_evaluated": 0,
            "surrogate_noop_fallback_used": False,
            **score_to_dict(current_score),
        }
    ]
    global_iteration = 0
    for stage_idx, (step, iterations) in enumerate(schedule, start=1):
        for _ in range(iterations):
            if deadline_reached(deadline):
                trace.append(
                    time_limit_row(
                        phase="surrogate_filtered_greedy",
                        iteration=global_iteration + 1,
                        stage=stage_idx,
                        step=step,
                        score=current_score,
                    )
                )
                return sorted(current, key=lambda b: b.volume), current_score, trace
            global_iteration += 1
            best_boxes, best_score, action, selection_metrics = best_single_action_surrogate_filtered(
                oracle=oracle,
                surrogate=surrogate,
                orders=orders,
                current=current,
                current_score=current_score,
                step=step,
                top_k=top_k,
                adaptive_top_k=adaptive_top_k,
                noop_fallback=noop_fallback,
                rank_mode=rank_mode,
                candidate_batch_size=candidate_batch_size,
                prefetch_candidate_statuses_enabled=prefetch_candidate_statuses_enabled,
            )
            improved = score_rank(best_score) < score_rank(current_score)
            trace.append(
                {
                    "phase": "surrogate_filtered_greedy",
                    "iteration": global_iteration,
                    "stage": stage_idx,
                    "step": step,
                    "action": action,
                    "improved": improved,
                    **selection_metrics,
                    **score_to_dict(best_score),
                }
            )
            if not improved:
                break
            current = best_boxes
            current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def run_ranker_filtered_greedy(
    *,
    oracle: BoxSetOracle,
    ranker: CandidateRanker,
    orders: list,
    boxes: list[Box],
    schedule: list[tuple[float, int]],
    top_k: int,
    adaptive_top_k: list[int] | None,
    noop_fallback: bool,
    initial_score: MilpBoxSetScore | None = None,
    initial_phase: str = "initial",
    deadline: float | None = None,
    prefetch_candidate_statuses_enabled: bool = False,
) -> tuple[list[Box], MilpBoxSetScore, list[dict]]:
    current = sorted(boxes, key=lambda b: b.box_id)
    current_score = initial_score if initial_score is not None else oracle.evaluate(orders, current)
    trace = [
        {
            "phase": initial_phase,
            "iteration": 0,
            "stage": 0,
            "step": schedule[0][0],
            "action": "init",
            "candidate_evaluations": 0,
            "generated_candidates": 0,
            "ranker_scored_candidates": 0,
            "milp_validated_candidates": 0,
            "milp_candidate_evaluations_avoided": 0,
            "milp_avoidance_rate": 0.0,
            "ranker_eval_seconds": 0.0,
            "milp_eval_seconds": 0.0,
            "ranker_top_k_sequence": "",
            "ranker_tiers_evaluated": 0,
            "ranker_noop_fallback_used": False,
            **score_to_dict(current_score),
        }
    ]
    global_iteration = 0
    for stage_idx, (step, iterations) in enumerate(schedule, start=1):
        for _ in range(iterations):
            if deadline_reached(deadline):
                trace.append(
                    time_limit_row(
                        phase="ranker_filtered_greedy",
                        iteration=global_iteration + 1,
                        stage=stage_idx,
                        step=step,
                        score=current_score,
                    )
                )
                return sorted(current, key=lambda b: b.volume), current_score, trace
            global_iteration += 1
            best_boxes, best_score, action, selection_metrics = best_single_action_ranker_filtered(
                oracle=oracle,
                ranker=ranker,
                orders=orders,
                current=current,
                current_score=current_score,
                step=step,
                stage=stage_idx,
                iteration=global_iteration,
                top_k=top_k,
                adaptive_top_k=adaptive_top_k,
                noop_fallback=noop_fallback,
                prefetch_candidate_statuses_enabled=prefetch_candidate_statuses_enabled,
            )
            improved = score_rank(best_score) < score_rank(current_score)
            trace.append(
                {
                    "phase": "ranker_filtered_greedy",
                    "iteration": global_iteration,
                    "stage": stage_idx,
                    "step": step,
                    "action": action,
                    "improved": improved,
                    **selection_metrics,
                    **score_to_dict(best_score),
                }
            )
            if not improved:
                break
            current = best_boxes
            current_score = best_score
    return sorted(current, key=lambda b: b.volume), current_score, trace


def write_trace(path: Path, trace: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "stage",
        "repair_round",
        "iteration",
        "step",
        "action",
        "improved",
        "stop_reason",
        "packaging_factor",
        "mean_box_volume",
        "mean_order_volume",
        "coverage_rate",
        "uncovered_orders",
        "unknown_pairs",
        "orders_with_unknown",
        "candidate_evaluations",
        "generated_candidates",
        "surrogate_scored_candidates",
        "milp_validated_candidates",
        "milp_candidate_evaluations_avoided",
        "milp_avoidance_rate",
        "surrogate_eval_seconds",
        "ranker_scored_candidates",
        "ranker_eval_seconds",
        "prefetch_eval_seconds",
        "milp_eval_seconds",
        "surrogate_top_k_sequence",
        "surrogate_tiers_evaluated",
        "surrogate_noop_fallback_used",
        "ranker_top_k_sequence",
        "ranker_tiers_evaluated",
        "ranker_noop_fallback_used",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace)


def write_candidate_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_TRACE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MILP-backed OR2023 box search entrypoint. "
            "Algorithms: paper_fixed_step uses the fixed paper action step; "
            "staged_greedy uses a manually adjusted step schedule; "
            "surrogate_filtered_greedy uses a learned surrogate to filter "
            "candidate actions before MILP-verified acceptance; "
            "ranker_filtered_greedy uses a candidate-level learned ranker "
            "with the same MILP-verified acceptance rule."
        )
    )
    parser.add_argument(
        "--algorithm",
        choices=["paper_fixed_step", "staged_greedy", "surrogate_filtered_greedy", "ranker_filtered_greedy"],
        required=True,
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--orders-offset",
        type=int,
        default=0,
        help="Number of orders to skip before applying --orders-limit; enables non-overlapping windows.",
    )
    parser.add_argument("--orders-limit", type=int, default=20)
    parser.add_argument("--fixed-step", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--schedule", type=parse_schedule, default=parse_schedule("0.5:2,0.25:2"))
    parser.add_argument(
        "--initial-boxes-json",
        type=Path,
        default=None,
        help="Optional best_boxes.json checkpoint to use instead of k-means initialization.",
    )
    parser.add_argument("--oracle", choices=["java", "labels"], default="java")
    parser.add_argument("--orientation-label", choices=["label_2ori", "label_6ori"], default="label_6ori")
    parser.add_argument("--xml-path", type=Path, default=ROOT / "assets/or2023_bsp_data/xml_unique/or2023_bsp_unique_orders.xml")
    parser.add_argument("--labels-path", type=Path, default=ROOT / "assets/milp_labels/or2023_bsp_unique_package_labels.csv")
    parser.add_argument("--java-classes", type=Path, default=REPO_ROOT / "MILP_3DBPP/target/classes")
    parser.add_argument("--java-classpath", default="")
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "assets/loadability_model_pack/models/ensemble/ensemble_baseline_20260526.json",
        help="Surrogate model metadata path used by surrogate_filtered_greedy.",
    )
    parser.add_argument(
        "--candidate-ranker-path",
        type=Path,
        default=None,
        help="Candidate ranker joblib artifact used by ranker_filtered_greedy.",
    )
    parser.add_argument("--tau", type=float, default=0.95, help="Surrogate feasibility probability threshold.")
    parser.add_argument("--tau-high", type=float, default=0.99, help="Surrogate risk-shaping high threshold.")
    parser.add_argument("--lambda-risk", type=float, default=1000.0, help="Surrogate risk penalty weight.")
    parser.add_argument(
        "--surrogate-top-k",
        type=int,
        default=10,
        help="Number of surrogate-ranked candidates per iteration to verify with MILP.",
    )
    parser.add_argument(
        "--surrogate-adaptive-top-k",
        type=parse_int_list,
        default=None,
        help=(
            "Optional comma-separated top-k widening sequence, e.g. 10,30. "
            "Each iteration evaluates the next tier only if smaller tiers have no MILP improvement."
        ),
    )
    parser.add_argument(
        "--surrogate-noop-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When all surrogate tiers have no MILP improvement, validate the remaining candidates before noop.",
    )
    parser.add_argument(
        "--surrogate-candidate-batch-size",
        type=int,
        default=None,
        help="Optional batch size for surrogate candidate scoring.",
    )
    parser.add_argument(
        "--surrogate-rank-mode",
        choices=["paper_pf_surrogate", "risk_aware_surrogate"],
        default="paper_pf_surrogate",
        help="Surrogate ranking objective used only for candidate filtering.",
    )
    parser.add_argument(
        "--ranker-top-k",
        type=int,
        default=10,
        help="Number of ranker-ranked candidates per iteration to verify with MILP.",
    )
    parser.add_argument(
        "--ranker-adaptive-top-k",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated ranker top-k widening sequence, e.g. 10,30.",
    )
    parser.add_argument(
        "--ranker-noop-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When all ranker tiers have no MILP improvement, validate remaining candidates before noop.",
    )
    parser.add_argument(
        "--prefetch-candidate-statuses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before validating a candidate set or tier, batch-prefetch exact "
            "MILP statuses for unique candidate box dimensions when the oracle "
            "supports it. This changes subprocess batching, not the accepted "
            "objective rule."
        ),
    )
    parser.add_argument(
        "--candidate-trace-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for exact candidate-level trace export. "
            "Supported for paper_fixed_step and staged_greedy."
        ),
    )
    parser.add_argument("--allow-bsp-derived-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle-cache-dir", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/milp_box_algorithms")
    parser.add_argument(
        "--comparison-label",
        default=None,
        help="Optional label shared by runs that should be paired in one comparison.",
    )
    parser.add_argument(
        "--config-label",
        default=None,
        help="Optional label for this concrete algorithm/configuration, e.g. fixed05_i2.",
    )
    parser.add_argument("--coverage-repair", choices=["none", "geometric_expand"], default="none")
    parser.add_argument(
        "--repair-margins",
        type=parse_float_list,
        default=parse_float_list("1.0,1.05,1.1,1.25,1.5,2.0"),
    )
    parser.add_argument("--repair-max-rounds", type=int, default=5)
    parser.add_argument(
        "--max-elapsed-seconds",
        type=float,
        default=None,
        help=(
            "Optional graceful wall-clock budget for the search phase. "
            "The runner checks this before starting each new local-search iteration "
            "and writes the current best boxes and summary when reached."
        ),
    )
    parser.add_argument(
        "--code-version",
        default=None,
        help="Optional manually supplied code version, e.g. the pushed GitHub commit used for this run.",
    )
    args = parser.parse_args()

    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if args.repair_max_rounds < 0:
        raise ValueError("--repair-max-rounds must be non-negative")
    if args.max_elapsed_seconds is not None and args.max_elapsed_seconds <= 0.0:
        raise ValueError("--max-elapsed-seconds must be positive when supplied")
    if args.orders_offset < 0:
        raise ValueError("--orders-offset must be non-negative")
    if args.orders_limit is not None and args.orders_limit < args.k:
        raise ValueError("--orders-limit must be >= --k for k-means initialization")
    if args.surrogate_top_k <= 0:
        raise ValueError("--surrogate-top-k must be positive")
    if args.surrogate_adaptive_top_k is not None and any(value <= 0 for value in args.surrogate_adaptive_top_k):
        raise ValueError("--surrogate-adaptive-top-k values must be positive")
    if args.surrogate_candidate_batch_size is not None and args.surrogate_candidate_batch_size <= 0:
        raise ValueError("--surrogate-candidate-batch-size must be positive")
    if args.ranker_top_k <= 0:
        raise ValueError("--ranker-top-k must be positive")
    if args.ranker_adaptive_top_k is not None and any(value <= 0 for value in args.ranker_adaptive_top_k):
        raise ValueError("--ranker-adaptive-top-k values must be positive")
    if args.candidate_trace_csv is not None and args.algorithm not in {"paper_fixed_step", "staged_greedy"}:
        raise ValueError("--candidate-trace-csv is currently supported only for exact algorithms")

    all_orders = read_order_summaries(args.xml_path)
    orders = select_order_window(all_orders, offset=args.orders_offset, limit=args.orders_limit)
    if args.initial_boxes_json is None:
        if len(orders) < args.k:
            raise ValueError(f"selected order window contains {len(orders)} orders, expected at least --k={args.k}")
        initial_boxes = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
    else:
        initial_boxes = boxes_from_json(args.initial_boxes_json)
        if len(initial_boxes) != args.k:
            raise ValueError(f"--initial-boxes-json contains {len(initial_boxes)} boxes, expected --k={args.k}")
    oracle = make_oracle(args)
    surrogate = make_surrogate_evaluator(args, orders) if args.algorithm == "surrogate_filtered_greedy" else None
    ranker = make_candidate_ranker(args) if args.algorithm == "ranker_filtered_greedy" else None

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = args.out_root / args.algorithm / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "algorithm": args.algorithm,
        "comparison_label": args.comparison_label,
        "config_label": args.config_label,
        "dataset": "OR2023 unique orders",
        "available_orders": len(all_orders),
        "orders_offset": args.orders_offset,
        "orders_limit": args.orders_limit,
        "selected_orders": len(orders),
        "orders_end_exclusive": args.orders_offset + len(orders),
        "k": args.k,
        "seed": args.seed,
        "oracle": args.oracle,
        "orientation_label": args.orientation_label,
        "fixed_step": args.fixed_step,
        "iterations": args.iterations,
        "schedule": [{"step": step, "iterations": iters} for step, iters in args.schedule],
        "initial_boxes_json": str(args.initial_boxes_json) if args.initial_boxes_json is not None else None,
        "xml_path": str(args.xml_path),
        "labels_path": str(args.labels_path),
        "java_classes": str(args.java_classes),
        "java_classpath_extra": args.java_classpath,
        "milp_time_limit_seconds": args.milp_time_limit_seconds,
        "oracle_cache_dir": str(args.oracle_cache_dir) if args.oracle_cache_dir is not None else None,
        "model_path": str(args.model_path) if args.algorithm == "surrogate_filtered_greedy" else None,
        "candidate_ranker_path": str(args.candidate_ranker_path) if args.algorithm == "ranker_filtered_greedy" else None,
        "tau": args.tau if args.algorithm == "surrogate_filtered_greedy" else None,
        "tau_high": args.tau_high if args.algorithm == "surrogate_filtered_greedy" else None,
        "lambda_risk": args.lambda_risk if args.algorithm == "surrogate_filtered_greedy" else None,
        "surrogate_top_k": args.surrogate_top_k if args.algorithm == "surrogate_filtered_greedy" else None,
        "surrogate_adaptive_top_k": (
            args.surrogate_adaptive_top_k if args.algorithm == "surrogate_filtered_greedy" else None
        ),
        "surrogate_noop_fallback": (
            args.surrogate_noop_fallback if args.algorithm == "surrogate_filtered_greedy" else None
        ),
        "surrogate_candidate_batch_size": (
            args.surrogate_candidate_batch_size if args.algorithm == "surrogate_filtered_greedy" else None
        ),
        "surrogate_rank_mode": args.surrogate_rank_mode if args.algorithm == "surrogate_filtered_greedy" else None,
        "ranker_top_k": args.ranker_top_k if args.algorithm == "ranker_filtered_greedy" else None,
        "ranker_adaptive_top_k": (
            args.ranker_adaptive_top_k if args.algorithm == "ranker_filtered_greedy" else None
        ),
        "ranker_noop_fallback": args.ranker_noop_fallback if args.algorithm == "ranker_filtered_greedy" else None,
        "prefetch_candidate_statuses": args.prefetch_candidate_statuses,
        "candidate_trace_csv": str(args.candidate_trace_csv) if args.candidate_trace_csv is not None else None,
        "coverage_repair": args.coverage_repair,
        "repair_margins": args.repair_margins,
        "repair_max_rounds": args.repair_max_rounds,
        "max_elapsed_seconds": args.max_elapsed_seconds,
        "code_version": args.code_version,
        "git": git_info(REPO_ROOT),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "initial_boxes.json", boxes_to_rows(initial_boxes))

    started = time.perf_counter()
    deadline = started + args.max_elapsed_seconds if args.max_elapsed_seconds is not None else None
    candidate_trace_rows: list[dict] | None = [] if args.candidate_trace_csv is not None else None
    pre_search_trace: list[dict] = []
    search_boxes = initial_boxes
    search_initial_score: MilpBoxSetScore | None = None
    if args.coverage_repair != "none":
        raw_initial_score = oracle.evaluate(orders, search_boxes)
        pre_search_trace.append(
            {
                "phase": "initial",
                "iteration": 0,
                "action": "init",
                "candidate_evaluations": 0,
                **score_to_dict(raw_initial_score),
            }
        )
        if args.coverage_repair == "geometric_expand":
            search_boxes, search_initial_score, repair_trace = repair_coverage_by_expansion(
                oracle=oracle,
                orders=orders,
                boxes=search_boxes,
                initial_score=raw_initial_score,
                margins=args.repair_margins,
                max_rounds=args.repair_max_rounds,
            )
            pre_search_trace.extend(repair_trace)
        else:
            raise ValueError(f"unknown coverage repair mode: {args.coverage_repair}")

    if args.algorithm == "paper_fixed_step":
        best_boxes, best_score, search_trace = run_fixed_step(
            oracle=oracle,
            orders=orders,
            boxes=search_boxes,
            step=args.fixed_step,
            iterations=args.iterations,
            initial_score=search_initial_score,
            initial_phase="search_initial" if pre_search_trace else "initial",
            candidate_trace=candidate_trace_rows,
            source_run_id=run_id,
            deadline=deadline,
            prefetch_candidate_statuses_enabled=args.prefetch_candidate_statuses,
        )
    elif args.algorithm == "staged_greedy":
        best_boxes, best_score, search_trace = run_staged_greedy(
            oracle=oracle,
            orders=orders,
            boxes=search_boxes,
            schedule=args.schedule,
            initial_score=search_initial_score,
            initial_phase="search_initial" if pre_search_trace else "initial",
            candidate_trace=candidate_trace_rows,
            source_run_id=run_id,
            deadline=deadline,
            prefetch_candidate_statuses_enabled=args.prefetch_candidate_statuses,
        )
    elif args.algorithm == "surrogate_filtered_greedy":
        if surrogate is None:
            raise RuntimeError("surrogate evaluator was not initialized")
        best_boxes, best_score, search_trace = run_surrogate_filtered_greedy(
            oracle=oracle,
            surrogate=surrogate,
            orders=orders,
            boxes=search_boxes,
            schedule=args.schedule,
            top_k=args.surrogate_top_k,
            adaptive_top_k=args.surrogate_adaptive_top_k,
            noop_fallback=args.surrogate_noop_fallback,
            rank_mode=args.surrogate_rank_mode,
            candidate_batch_size=args.surrogate_candidate_batch_size,
            initial_score=search_initial_score,
            initial_phase="search_initial" if pre_search_trace else "initial",
            deadline=deadline,
            prefetch_candidate_statuses_enabled=args.prefetch_candidate_statuses,
        )
    else:
        if ranker is None:
            raise RuntimeError("candidate ranker was not initialized")
        best_boxes, best_score, search_trace = run_ranker_filtered_greedy(
            oracle=oracle,
            ranker=ranker,
            orders=orders,
            boxes=search_boxes,
            schedule=args.schedule,
            top_k=args.ranker_top_k,
            adaptive_top_k=args.ranker_adaptive_top_k,
            noop_fallback=args.ranker_noop_fallback,
            initial_score=search_initial_score,
            initial_phase="search_initial" if pre_search_trace else "initial",
            deadline=deadline,
            prefetch_candidate_statuses_enabled=args.prefetch_candidate_statuses,
        )
    trace = pre_search_trace + search_trace
    elapsed_seconds = time.perf_counter() - started

    repair_rows = [row for row in trace if row.get("phase") == "coverage_repair"]
    search_initial_rows = [row for row in trace if row.get("phase") in {"search_initial", "initial"}]

    summary = {
        **manifest,
        "initial_score": trace[0],
        "search_initial_score": search_initial_rows[-1] if search_initial_rows else trace[0],
        "coverage_repair_score": repair_rows[-1] if repair_rows else None,
        "best_score": score_to_dict(best_score),
        "trace_rows": len(trace),
        "candidate_evaluations": int(sum(row.get("candidate_evaluations", 0) for row in trace)),
        "generated_candidates": int(sum(row.get("generated_candidates", 0) for row in trace)),
        "surrogate_scored_candidates": int(sum(row.get("surrogate_scored_candidates", 0) for row in trace)),
        "ranker_scored_candidates": int(sum(row.get("ranker_scored_candidates", 0) for row in trace)),
        "milp_validated_candidates": int(sum(row.get("milp_validated_candidates", 0) for row in trace)),
        "milp_candidate_evaluations_avoided": int(
            sum(row.get("milp_candidate_evaluations_avoided", 0) for row in trace)
        ),
        "milp_avoidance_rate": (
            float(sum(row.get("milp_candidate_evaluations_avoided", 0) for row in trace))
            / float(sum(row.get("generated_candidates", 0) for row in trace))
            if sum(row.get("generated_candidates", 0) for row in trace)
            else 0.0
        ),
        "surrogate_eval_seconds": float(sum(row.get("surrogate_eval_seconds", 0.0) for row in trace)),
        "ranker_eval_seconds": float(sum(row.get("ranker_eval_seconds", 0.0) for row in trace)),
        "prefetch_eval_seconds": float(sum(row.get("prefetch_eval_seconds", 0.0) for row in trace)),
        "milp_eval_seconds": float(sum(row.get("milp_eval_seconds", 0.0) for row in trace)),
        "surrogate_tiers_evaluated": int(sum(row.get("surrogate_tiers_evaluated", 0) for row in trace)),
        "surrogate_noop_fallback_uses": int(
            sum(1 for row in trace if str(row.get("surrogate_noop_fallback_used", "")).lower() == "true")
        ),
        "ranker_tiers_evaluated": int(sum(row.get("ranker_tiers_evaluated", 0) for row in trace)),
        "ranker_noop_fallback_uses": int(
            sum(1 for row in trace if str(row.get("ranker_noop_fallback_used", "")).lower() == "true")
        ),
        "elapsed_seconds": elapsed_seconds,
        "stop_reason": next(
            (
                str(row.get("stop_reason"))
                for row in trace
                if row.get("stop_reason") not in {None, ""}
            ),
            None,
        ),
        "oracle_cache": oracle.cache_info() if hasattr(oracle, "cache_info") else None,
        "run_dir": str(run_dir),
    }
    write_trace(run_dir / "trace.csv", trace)
    if candidate_trace_rows is not None:
        write_candidate_trace(args.candidate_trace_csv, candidate_trace_rows)
        write_candidate_trace(run_dir / "candidate_trace.csv", candidate_trace_rows)
        summary["candidate_trace_rows"] = len(candidate_trace_rows)
    write_json(run_dir / "best_boxes.json", boxes_to_rows(best_boxes))
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

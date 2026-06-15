from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
HYBRID_ROOT = REPO_ROOT / "HybridSVM"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from box_design_surrogate import Box, SurrogateEvaluator, read_order_summaries  # noqa: E402
from box_design_surrogate.kandula_paper import KandulaBoxSizingGame, weighted_packaging_factor  # noqa: E402
from box_design_surrogate.kandula_repro import evaluate_simple_box_set, initial_boxes_kmeans  # noqa: E402
from box_design_surrogate.surrogate_game import SurrogateBoxSizingGame, weighted_objective  # noqa: E402
from scripts.train_kandula_paper_policy import ensure_torch, make_actor_critic_class  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


TRACE_FIELDS = [
    "iteration",
    "mode",
    "objective_mode",
    "selected_action",
    "selected_mean_weighted_objective",
    "current_objective",
    "best_objective",
    "candidate_children",
    "current_uncovered",
    "best_uncovered",
    "current_low_margin",
    "best_low_margin",
    "current_mean_probability",
    "best_mean_probability",
    "current_packaging_factor",
    "best_packaging_factor",
    "current_mean_assigned_box_volume",
    "best_mean_assigned_box_volume",
    "current_mean_order_volume",
    "best_mean_order_volume",
    "current_mean_adjusted_cost",
    "best_mean_adjusted_cost",
    "current_uncovered_penalty",
    "best_uncovered_penalty",
    "current_low_margin_penalty",
    "best_low_margin_penalty",
    "current_probability_penalty",
    "best_probability_penalty",
    "candidate_eval_seconds",
    "iteration_seconds",
    "elapsed_seconds",
]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def jsonable_parameters(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def boxes_to_rows(boxes: list[Box]) -> list[dict[str, float | int]]:
    return [
        {
            "box_id": box.box_id,
            "length": box.length,
            "width": box.width,
            "height": box.height,
            "volume": box.volume,
        }
        for box in sorted(boxes, key=lambda b: b.volume)
    ]


def load_boxes_json(path: Path) -> list[Box]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    for row in rows:
        boxes.append(
            Box(
                int(row["box_id"]),
                float(row["length"]),
                float(row["width"]),
                float(row["height"]),
            )
        )
    return sorted(boxes, key=lambda b: b.box_id)


def load_policy(path: Path):
    torch, _, _ = ensure_torch()
    checkpoint = torch.load(path, map_location="cpu")
    ActorCritic = make_actor_critic_class()
    model = ActorCritic(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_count"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["hidden_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def policy_action(model: ActorCritic, obs: np.ndarray, *, sample: bool, rng: np.random.Generator) -> int:
    torch, _, Categorical = ensure_torch()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_t)
        if sample:
            dist = Categorical(logits=logits)
            return int(dist.sample().item())
        return int(torch.argmax(logits, dim=-1).item())


def evaluate_boxes(env, boxes: list[Box]) -> tuple[object, float, dict[str, object]]:
    if isinstance(env, SurrogateBoxSizingGame):
        evaluation = env.evaluate_box_set(boxes)
        return evaluation, env.objective(evaluation), surrogate_metrics(env, evaluation)
    score = evaluate_simple_box_set(env.orders, boxes)
    return score, score.packaging_factor, {
        "uncovered_orders": score.uncovered_orders,
        "low_margin_assignments": "",
        "mean_probability": "",
        "packaging_factor": score.packaging_factor,
        "mean_assigned_box_volume": score.mean_box_volume,
        "mean_order_volume": score.mean_order_volume,
        "mean_adjusted_cost": "",
        "uncovered_penalty": "",
        "low_margin_penalty": "",
        "probability_penalty": "",
    }


def surrogate_metrics(env: SurrogateBoxSizingGame, evaluation) -> dict[str, object]:
    breakdown = env.objective_breakdown(evaluation)
    return {
        "uncovered_orders": evaluation.uncovered_orders,
        "low_margin_assignments": evaluation.low_margin_assignments,
        "mean_probability": evaluation.mean_assigned_probability,
        "packaging_factor": breakdown.packaging_factor,
        "mean_assigned_box_volume": breakdown.mean_assigned_box_volume,
        "mean_order_volume": breakdown.mean_order_volume,
        "mean_adjusted_cost": breakdown.mean_adjusted_cost,
        "uncovered_penalty": breakdown.uncovered_penalty,
        "low_margin_penalty": breakdown.low_margin_penalty,
        "probability_penalty": breakdown.probability_penalty,
    }


def simulate_weighted_objective(
    *,
    env,
    model: ActorCritic,
    start_boxes: list[Box],
    beta: float,
    rollout_steps: int,
    rollout_samples: int,
    sample_policy: bool,
    rng: np.random.Generator,
) -> float:
    weighted_values = []
    for _ in range(rollout_samples):
        boxes = list(start_boxes)
        _, objective, _ = evaluate_boxes(env, boxes)
        values = [objective]
        for _step in range(rollout_steps):
            obs = env.observation_for_boxes(boxes)
            action = policy_action(model, obs, sample=sample_policy, rng=rng)
            if action == env.resign_action:
                break
            if isinstance(env, SurrogateBoxSizingGame):
                child, child_eval, child_objective = env.child_result(boxes, action)
                values.append(child_objective)
            else:
                child, child_score, feasible = env.child_result(boxes, action)
                values.append(child_score.packaging_factor)
                if not feasible:
                    break
            boxes = child
        if isinstance(env, SurrogateBoxSizingGame):
            weighted_values.append(weighted_objective(values, beta))
        else:
            weighted_values.append(weighted_packaging_factor(values, beta))
    return float(np.mean(weighted_values))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank_for(env, evaluation, objective: float) -> tuple[int, float]:
    if isinstance(env, SurrogateBoxSizingGame):
        return env.objective_rank(evaluation)
    return 0, objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--mode", choices=["paper", "surrogate"], default="paper")
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument(
        "--initial-boxes-json",
        type=Path,
        default=None,
        help="Optional boxes JSON from a previous PAAS run to continue search from.",
    )
    parser.add_argument("--search-iters", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--rollout-samples", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--sample-policy", action="store_true")
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--tau-high", type=float, default=0.99)
    parser.add_argument("--lambda-risk", type=float, default=1000.0)
    parser.add_argument("--uncovered-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--surrogate-uncovered-weight", type=float, default=10_000_000.0)
    parser.add_argument("--surrogate-low-margin-weight", type=float, default=10_000.0)
    parser.add_argument("--surrogate-probability-weight", type=float, default=1_000.0)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Number of candidate box sets to score per batch in surrogate mode.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=["paper_pf_surrogate", "risk_aware_surrogate"],
        default="paper_pf_surrogate",
        help="Surrogate objective: paper-style PF with ML feasibility, or risk-aware robustness variant.",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=ROOT / "assets" / "or2023_bsp_data" / "xml_unique" / "or2023_bsp_unique_orders.xml",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT
        / "assets"
        / "loadability_model_pack"
        / "models"
        / "ensemble"
        / "ensemble_baseline_20260526.json",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "kandula_paper_paas")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "paper_alignment": {
            "stage": "Stage 3 PAAS",
            "procedure": "expand all 6K transforming children, simulate policy rollouts, choose child with lowest mean weighted objective",
        },
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": jsonable_parameters(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        orders = read_order_summaries(args.xml_path)
        if args.orders_limit:
            orders = orders[: args.orders_limit]
        if args.initial_boxes_json is None:
            initial_boxes = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
            initial_boxes_source = "kmeans"
        else:
            initial_boxes = load_boxes_json(args.initial_boxes_json)
            if len(initial_boxes) != args.k:
                raise ValueError(
                    f"initial boxes count {len(initial_boxes)} does not match requested K={args.k}"
                )
            initial_boxes_source = str(args.initial_boxes_json)
        if args.mode == "paper":
            env = KandulaBoxSizingGame(orders, initial_boxes, step_size=args.step_size, max_steps=args.search_iters)
        else:
            model_pack = load_ensemble_pipeline(args.model_path)
            evaluator = SurrogateEvaluator(
                model=model_pack,
                tau=args.tau,
                tau_high=args.tau_high,
                lambda_risk=args.lambda_risk,
                uncovered_penalty=args.uncovered_penalty,
            )
            env = SurrogateBoxSizingGame(
                orders,
                initial_boxes,
                evaluator,
                step_size=args.step_size,
                max_steps=args.search_iters,
                uncovered_weight=args.surrogate_uncovered_weight,
                low_margin_weight=args.surrogate_low_margin_weight,
                probability_weight=args.surrogate_probability_weight,
                objective_mode=args.objective_mode,
                candidate_batch_size=args.candidate_batch_size,
            )
        model, checkpoint = load_policy(args.policy_path)
        if int(checkpoint["action_count"]) != env.action_count:
            raise ValueError("policy action_count does not match requested K")
        if checkpoint.get("mode", args.mode) != args.mode:
            raise ValueError(f"policy mode {checkpoint.get('mode')} does not match requested mode {args.mode}")
        if args.mode == "surrogate" and checkpoint.get("objective_mode", args.objective_mode) != args.objective_mode:
            raise ValueError(
                f"policy objective_mode {checkpoint.get('objective_mode')} does not match requested objective_mode {args.objective_mode}"
            )
        rng = np.random.default_rng(args.seed)
        current_boxes = list(env.initial_boxes)
        current_eval, current_objective, current_metrics = evaluate_boxes(env, current_boxes)
        best_boxes = list(current_boxes)
        best_eval = current_eval
        best_objective = current_objective
        best_metrics = dict(current_metrics)
        trace = []
        started_at = time.perf_counter()

        for iteration in range(args.search_iters):
            iteration_started_at = time.perf_counter()
            candidate_eval_seconds = 0.0
            child_scores = []
            candidate_children = 0
            if isinstance(env, SurrogateBoxSizingGame):
                candidate_eval_started_at = time.perf_counter()
                child_results = env.child_results(current_boxes, range(env.resign_action))
                candidate_eval_seconds += time.perf_counter() - candidate_eval_started_at
                candidate_children = len(child_results)
                for action, child, child_eval, child_objective in child_results:
                    child_metrics = surrogate_metrics(env, child_eval)
                    if args.rollout_steps == 0:
                        mean_objective = child_objective
                    else:
                        mean_objective = simulate_weighted_objective(
                            env=env,
                            model=model,
                            start_boxes=child,
                            beta=args.beta,
                            rollout_steps=args.rollout_steps,
                            rollout_samples=args.rollout_samples,
                            sample_policy=args.sample_policy,
                            rng=rng,
                        )
                    child_scores.append((mean_objective, action, child, child_eval, child_objective, child_metrics))
            else:
                for action in range(env.resign_action):
                    child, child_score, feasible = env.child_result(current_boxes, action)
                    if not feasible:
                        continue
                    child_objective = child_score.packaging_factor
                    child_metrics = {
                        "uncovered_orders": child_score.uncovered_orders,
                        "low_margin_assignments": "",
                        "mean_probability": "",
                        "packaging_factor": child_score.packaging_factor,
                        "mean_assigned_box_volume": child_score.mean_box_volume,
                        "mean_order_volume": child_score.mean_order_volume,
                        "mean_adjusted_cost": "",
                        "uncovered_penalty": "",
                        "low_margin_penalty": "",
                        "probability_penalty": "",
                    }
                    candidate_children += 1
                    mean_objective = simulate_weighted_objective(
                        env=env,
                        model=model,
                        start_boxes=child,
                        beta=args.beta,
                        rollout_steps=args.rollout_steps,
                        rollout_samples=args.rollout_samples,
                        sample_policy=args.sample_policy,
                        rng=rng,
                    )
                    child_scores.append((mean_objective, action, child, child_score, child_objective, child_metrics))
            if not child_scores:
                break
            (
                mean_objective,
                selected_action,
                current_boxes,
                current_eval,
                current_objective,
                current_metrics,
            ) = min(child_scores, key=lambda x: x[0])
            if rank_for(env, current_eval, current_objective) < rank_for(env, best_eval, best_objective):
                best_boxes = list(current_boxes)
                best_eval = current_eval
                best_objective = current_objective
                best_metrics = dict(current_metrics)
            trace.append(
                {
                    "iteration": iteration,
                    "mode": args.mode,
                    "objective_mode": args.objective_mode if args.mode == "surrogate" else "",
                    "selected_action": selected_action,
                    "selected_mean_weighted_objective": mean_objective,
                    "current_objective": current_objective,
                    "best_objective": best_objective,
                    "candidate_children": candidate_children,
                    "current_uncovered": current_metrics["uncovered_orders"],
                    "best_uncovered": best_metrics["uncovered_orders"],
                    "current_low_margin": current_metrics["low_margin_assignments"],
                    "best_low_margin": best_metrics["low_margin_assignments"],
                    "current_mean_probability": current_metrics["mean_probability"],
                    "best_mean_probability": best_metrics["mean_probability"],
                    "current_packaging_factor": current_metrics["packaging_factor"],
                    "best_packaging_factor": best_metrics["packaging_factor"],
                    "current_mean_assigned_box_volume": current_metrics["mean_assigned_box_volume"],
                    "best_mean_assigned_box_volume": best_metrics["mean_assigned_box_volume"],
                    "current_mean_order_volume": current_metrics["mean_order_volume"],
                    "best_mean_order_volume": best_metrics["mean_order_volume"],
                    "current_mean_adjusted_cost": current_metrics["mean_adjusted_cost"],
                    "best_mean_adjusted_cost": best_metrics["mean_adjusted_cost"],
                    "current_uncovered_penalty": current_metrics["uncovered_penalty"],
                    "best_uncovered_penalty": best_metrics["uncovered_penalty"],
                    "current_low_margin_penalty": current_metrics["low_margin_penalty"],
                    "best_low_margin_penalty": best_metrics["low_margin_penalty"],
                    "current_probability_penalty": current_metrics["probability_penalty"],
                    "best_probability_penalty": best_metrics["probability_penalty"],
                    "candidate_eval_seconds": candidate_eval_seconds,
                    "iteration_seconds": time.perf_counter() - iteration_started_at,
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
            )
            print(
                f"iter={iteration} action={selected_action} "
                f"objective={current_objective:.6f} best={best_objective:.6f} "
                f"children={candidate_children}",
                flush=True,
            )

        write_csv(out_dir / "trace.csv", TRACE_FIELDS, trace)
        (out_dir / "initial_boxes.json").write_text(json.dumps(boxes_to_rows(env.initial_boxes), indent=2), encoding="utf-8")
        (out_dir / "best_boxes.json").write_text(json.dumps(boxes_to_rows(best_boxes), indent=2), encoding="utf-8")
        _, initial_objective, initial_metrics = evaluate_boxes(env, env.initial_boxes)
        summary = {
            "mode": args.mode,
            "objective_mode": args.objective_mode if args.mode == "surrogate" else "",
            "initial_boxes_source": initial_boxes_source,
            "initial_objective": initial_objective,
            "best_objective": best_objective,
            "iterations": len(trace),
            "initial_uncovered": initial_metrics["uncovered_orders"],
            "best_uncovered": best_metrics["uncovered_orders"],
            "initial_low_margin": initial_metrics["low_margin_assignments"],
            "best_low_margin": best_metrics["low_margin_assignments"],
            "initial_mean_probability": initial_metrics["mean_probability"],
            "best_mean_probability": best_metrics["mean_probability"],
            "initial_packaging_factor": initial_metrics["packaging_factor"],
            "best_packaging_factor": best_metrics["packaging_factor"],
            "initial_mean_assigned_box_volume": initial_metrics["mean_assigned_box_volume"],
            "best_mean_assigned_box_volume": best_metrics["mean_assigned_box_volume"],
            "initial_mean_order_volume": initial_metrics["mean_order_volume"],
            "best_mean_order_volume": best_metrics["mean_order_volume"],
            "initial_mean_adjusted_cost": initial_metrics["mean_adjusted_cost"],
            "best_mean_adjusted_cost": best_metrics["mean_adjusted_cost"],
            "initial_uncovered_penalty": initial_metrics["uncovered_penalty"],
            "best_uncovered_penalty": best_metrics["uncovered_penalty"],
            "initial_low_margin_penalty": initial_metrics["low_margin_penalty"],
            "best_low_margin_penalty": best_metrics["low_margin_penalty"],
            "initial_probability_penalty": initial_metrics["probability_penalty"],
            "best_probability_penalty": best_metrics["probability_penalty"],
            "elapsed_seconds": time.perf_counter() - started_at,
            "candidate_batch_size": args.candidate_batch_size,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        manifest["summary"] = summary
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

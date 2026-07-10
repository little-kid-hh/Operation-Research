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
from typing import Any

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

from box_design_surrogate import SurrogateEvaluator, read_order_summaries  # noqa: E402
from box_design_surrogate.kandula_paper import KandulaBoxSizingGame  # noqa: E402
from box_design_surrogate.kandula_repro import initial_boxes_kmeans  # noqa: E402
from box_design_surrogate.surrogate_game import SurrogateBoxSizingGame  # noqa: E402
from scripts.run_kandula_paper_paas import boxes_to_rows, load_boxes_json, load_policy  # noqa: E402
from scripts.train_kandula_paper_policy import ensure_torch, environment_metrics, result_metrics  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


SUMMARY_FIELDS = [
    "rollout",
    "policy_mode",
    "mode",
    "objective_mode",
    "seed",
    "steps",
    "total_reward",
    "initial_objective",
    "final_objective",
    "best_objective",
    "best_step",
    "initial_uncovered",
    "final_uncovered",
    "best_uncovered",
    "initial_low_margin",
    "final_low_margin",
    "best_low_margin",
    "initial_mean_probability",
    "final_mean_probability",
    "best_mean_probability",
    "initial_packaging_factor",
    "final_packaging_factor",
    "best_packaging_factor",
    "initial_mean_assigned_box_volume",
    "final_mean_assigned_box_volume",
    "best_mean_assigned_box_volume",
    "initial_mean_order_volume",
    "final_mean_order_volume",
    "best_mean_order_volume",
    "initial_mean_adjusted_cost",
    "final_mean_adjusted_cost",
    "best_mean_adjusted_cost",
    "terminal_reason",
]

TRAJECTORY_FIELDS = [
    "rollout",
    "policy_mode",
    "mode",
    "objective_mode",
    "seed",
    "step",
    "action",
    "action_type",
    "action_probability",
    "action_log_prob",
    "reward",
    "objective",
    "best_objective",
    "uncovered_orders",
    "best_uncovered",
    "low_margin_assignments",
    "mean_probability",
    "packaging_factor",
    "mean_assigned_box_volume",
    "mean_order_volume",
    "mean_adjusted_cost",
    "terminal_reason",
]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def jsonable_parameters(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def action_type(action: int, k: int) -> str:
    if action == 6 * k:
        return "resign"
    return "decrement" if action < 3 * k else "increment"


def numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def metric_rank(metrics: dict[str, object]) -> tuple[float, float]:
    uncovered = numeric(metrics.get("uncovered_orders"))
    objective = numeric(metrics.get("objective"))
    return (
        float("inf") if uncovered is None else uncovered,
        float("inf") if objective is None else objective,
    )


def is_better(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
    return metric_rank(candidate) < metric_rank(incumbent)


def choose_action(
    *,
    model,
    obs: np.ndarray,
    policy_mode: str,
    temperature: float,
) -> tuple[int, float, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    torch, _, Categorical = ensure_torch()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _value = model(obs_t)
        logits = logits / temperature
        if policy_mode == "greedy":
            action_t = torch.argmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            action = int(action_t.item())
            return action, float(torch.exp(log_probs[0, action]).item()), float(log_probs[0, action].item())
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        action = int(action_t.item())
        return action, float(torch.exp(log_prob_t).item()), float(log_prob_t.item())


def build_env(args: argparse.Namespace, orders, initial_boxes):
    if args.mode == "paper":
        return KandulaBoxSizingGame(
            orders,
            initial_boxes,
            step_size=args.step_size,
            max_steps=args.max_steps,
        )
    model_pack = load_ensemble_pipeline(args.model_path)
    evaluator = SurrogateEvaluator(
        model=model_pack,
        tau=args.tau,
        tau_high=args.tau_high,
        lambda_risk=args.lambda_risk,
        uncovered_penalty=args.uncovered_penalty,
    )
    return SurrogateBoxSizingGame(
        orders,
        initial_boxes,
        evaluator,
        step_size=args.step_size,
        max_steps=args.max_steps,
        uncovered_weight=args.surrogate_uncovered_weight,
        low_margin_weight=args.surrogate_low_margin_weight,
        probability_weight=args.surrogate_probability_weight,
        objective_mode=args.objective_mode,
        terminate_on_worse_than_initial=args.terminate_on_worse_than_initial,
        candidate_batch_size=args.candidate_batch_size,
    )


def apply_checkpoint_environment_metadata(env, checkpoint: dict[str, object], *, requested_step: float) -> dict[str, object]:
    checkpoint_step = checkpoint.get("step_size")
    if checkpoint_step is not None and abs(float(checkpoint_step) - float(requested_step)) > 1e-12:
        raise ValueError(
            f"policy step_size {checkpoint_step} does not match requested step_size {requested_step}"
        )
    checkpoint_scale = checkpoint.get("scale_dim")
    if checkpoint_scale is not None and getattr(env, "normalize_observation", False):
        env.scale_dim = float(checkpoint_scale)
        scale_source = "checkpoint"
    else:
        scale_source = "evaluation_initial_boxes"
    return {
        "step_size": checkpoint_step,
        "scale_dim": float(env.scale_dim),
        "scale_dim_source": scale_source,
        "training_order_count": checkpoint.get("training_order_count"),
        "training_xml_sha256": checkpoint.get("training_xml_sha256", ""),
        "reward_definition": checkpoint.get("reward_definition", ""),
    }


def rollout_once(
    *,
    env,
    model,
    rollout: int,
    seed: int,
    policy_mode: str,
    mode: str,
    temperature: float,
) -> tuple[dict[str, object], list[dict[str, object]], list]:
    torch, _, _ = ensure_torch()
    torch.manual_seed(seed)
    env.reset()
    initial = environment_metrics(env)
    final = dict(initial)
    best = dict(initial)
    best_step = -1
    best_boxes = list(env.boxes)
    total_reward = 0.0
    terminal_reason = ""
    trajectory: list[dict[str, object]] = []

    for step in range(env.max_steps):
        obs = env.observation()
        action, action_probability, action_log_prob = choose_action(
            model=model,
            obs=obs,
            policy_mode=policy_mode,
            temperature=temperature,
        )
        result = env.step(action)
        metrics = result_metrics(result, mode, env)
        total_reward += float(result.reward)
        if is_better(metrics, best):
            best = dict(metrics)
            best_step = step
            best_boxes = list(result.boxes)
        final = dict(metrics)
        terminal_reason = result.terminal_reason
        trajectory.append(
            {
                "rollout": rollout,
                "policy_mode": policy_mode,
                "mode": mode,
                "objective_mode": getattr(env, "objective_mode", ""),
                "seed": seed,
                "step": step,
                "action": action,
                "action_type": action_type(action, env.k),
                "action_probability": action_probability,
                "action_log_prob": action_log_prob,
                "reward": result.reward,
                "objective": metrics["objective"],
                "best_objective": best["objective"],
                "uncovered_orders": metrics["uncovered_orders"],
                "best_uncovered": best["uncovered_orders"],
                "low_margin_assignments": metrics["low_margin_assignments"],
                "mean_probability": metrics["mean_probability"],
                "packaging_factor": metrics["packaging_factor"],
                "mean_assigned_box_volume": metrics["mean_assigned_box_volume"],
                "mean_order_volume": metrics["mean_order_volume"],
                "mean_adjusted_cost": metrics["mean_adjusted_cost"],
                "terminal_reason": result.terminal_reason,
            }
        )
        if result.done:
            break

    summary = {
        "rollout": rollout,
        "policy_mode": policy_mode,
        "mode": mode,
        "objective_mode": getattr(env, "objective_mode", ""),
        "seed": seed,
        "steps": len(trajectory),
        "total_reward": total_reward,
        "initial_objective": initial["objective"],
        "final_objective": final["objective"],
        "best_objective": best["objective"],
        "best_step": best_step,
        "initial_uncovered": initial["uncovered_orders"],
        "final_uncovered": final["uncovered_orders"],
        "best_uncovered": best["uncovered_orders"],
        "initial_low_margin": initial["low_margin_assignments"],
        "final_low_margin": final["low_margin_assignments"],
        "best_low_margin": best["low_margin_assignments"],
        "initial_mean_probability": initial["mean_probability"],
        "final_mean_probability": final["mean_probability"],
        "best_mean_probability": best["mean_probability"],
        "initial_packaging_factor": initial["packaging_factor"],
        "final_packaging_factor": final["packaging_factor"],
        "best_packaging_factor": best["packaging_factor"],
        "initial_mean_assigned_box_volume": initial["mean_assigned_box_volume"],
        "final_mean_assigned_box_volume": final["mean_assigned_box_volume"],
        "best_mean_assigned_box_volume": best["mean_assigned_box_volume"],
        "initial_mean_order_volume": initial["mean_order_volume"],
        "final_mean_order_volume": final["mean_order_volume"],
        "best_mean_order_volume": best["mean_order_volume"],
        "initial_mean_adjusted_cost": initial["mean_adjusted_cost"],
        "final_mean_adjusted_cost": final["mean_adjusted_cost"],
        "best_mean_adjusted_cost": best["mean_adjusted_cost"],
        "terminal_reason": terminal_reason,
    }
    return summary, trajectory, best_boxes


def aggregate_summaries(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one rollout summary is required")

    def rank_value(row: dict[str, object], field: str) -> float:
        value = numeric(row.get(field))
        return float("inf") if value is None else value

    best_row = min(rows, key=lambda row: (rank_value(row, "best_uncovered"), rank_value(row, "best_objective")))

    def mean_field(field: str) -> float | None:
        values = [value for row in rows for value in [numeric(row.get(field))] if value is not None]
        return float(np.mean(values)) if values else None

    return {
        "rollouts": len(rows),
        "policy_modes": sorted({str(row.get("policy_mode")) for row in rows}),
        "best_rollout": best_row["rollout"],
        "best_seed": best_row["seed"],
        "best_objective": best_row["best_objective"],
        "best_uncovered": best_row["best_uncovered"],
        "best_packaging_factor": best_row["best_packaging_factor"],
        "initial_objective_mean": mean_field("initial_objective"),
        "final_objective_mean": mean_field("final_objective"),
        "best_objective_mean": mean_field("best_objective"),
        "initial_packaging_factor_mean": mean_field("initial_packaging_factor"),
        "final_packaging_factor_mean": mean_field("final_packaging_factor"),
        "best_packaging_factor_mean": mean_field("best_packaging_factor"),
        "terminal_reasons": {
            reason: sum(1 for row in rows if row.get("terminal_reason") == reason)
            for reason in sorted({str(row.get("terminal_reason")) for row in rows})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct RL policy rollouts for Kandula-style box design.")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--mode", choices=["paper", "surrogate"], default="paper")
    parser.add_argument("--policy-mode", choices=["sample", "greedy", "both"], default="sample")
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--orders-offset", type=int, default=0)
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--initial-boxes-json", type=Path, default=None)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--tau-high", type=float, default=0.99)
    parser.add_argument("--lambda-risk", type=float, default=1000.0)
    parser.add_argument("--uncovered-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--surrogate-uncovered-weight", type=float, default=10_000_000.0)
    parser.add_argument("--surrogate-low-margin-weight", type=float, default=10_000.0)
    parser.add_argument("--surrogate-probability-weight", type=float, default=1_000.0)
    parser.add_argument(
        "--objective-mode",
        choices=["paper_pf_surrogate", "risk_aware_surrogate"],
        default="paper_pf_surrogate",
    )
    parser.add_argument("--terminate-on-worse-than-initial", action="store_true")
    parser.add_argument("--candidate-batch-size", type=int, default=None)
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
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "policy_rollouts")
    args = parser.parse_args()

    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.orders_offset < 0:
        raise ValueError("--orders-offset must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": jsonable_parameters(args),
    }
    write_json(out_dir / "manifest.json", manifest)
    started_at = time.perf_counter()

    try:
        orders = read_order_summaries(args.xml_path)
        orders = orders[args.orders_offset :]
        if args.orders_limit:
            orders = orders[: args.orders_limit]
        if not orders:
            raise ValueError("selected order window is empty")
        if args.initial_boxes_json is None:
            initial_boxes = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
            initial_boxes_source = "kmeans"
        else:
            initial_boxes = load_boxes_json(args.initial_boxes_json)
            initial_boxes_source = str(args.initial_boxes_json)
        if len(initial_boxes) != args.k:
            raise ValueError(f"initial boxes count {len(initial_boxes)} does not match K={args.k}")

        model, checkpoint = load_policy(args.policy_path)
        env = build_env(args, orders, initial_boxes)
        checkpoint_metadata = apply_checkpoint_environment_metadata(
            env,
            checkpoint,
            requested_step=args.step_size,
        )
        if int(checkpoint["action_count"]) != env.action_count:
            raise ValueError("policy action_count does not match requested K")
        if checkpoint.get("mode", args.mode) != args.mode:
            raise ValueError(f"policy mode {checkpoint.get('mode')} does not match requested mode {args.mode}")
        if args.mode == "surrogate" and checkpoint.get("objective_mode", args.objective_mode) != args.objective_mode:
            raise ValueError(
                f"policy objective_mode {checkpoint.get('objective_mode')} does not match requested objective_mode {args.objective_mode}"
            )

        policy_modes = ["sample", "greedy"] if args.policy_mode == "both" else [args.policy_mode]
        summary_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        best_boxes_by_rollout: dict[int, list] = {}
        rollout_idx = 0
        for policy_mode in policy_modes:
            repeats = 1 if policy_mode == "greedy" and args.policy_mode == "both" else args.rollouts
            for local_idx in range(repeats):
                rollout_seed = args.seed + rollout_idx * 1009 + local_idx
                summary, trajectory, best_boxes = rollout_once(
                    env=env,
                    model=model,
                    rollout=rollout_idx,
                    seed=rollout_seed,
                    policy_mode=policy_mode,
                    mode=args.mode,
                    temperature=args.temperature,
                )
                summary_rows.append(summary)
                trajectory_rows.extend(trajectory)
                best_boxes_by_rollout[rollout_idx] = best_boxes
                print(
                    f"rollout={rollout_idx} mode={policy_mode} steps={summary['steps']} "
                    f"best={float(summary['best_objective']):.6f} "
                    f"terminal={summary['terminal_reason']}",
                    flush=True,
                )
                rollout_idx += 1

        aggregate = aggregate_summaries(summary_rows)
        best_rollout = int(aggregate["best_rollout"])
        best_boxes = best_boxes_by_rollout[best_rollout]
        write_csv(out_dir / "rollout_summary.csv", SUMMARY_FIELDS, summary_rows)
        write_csv(out_dir / "rollout_trajectories.csv", TRAJECTORY_FIELDS, trajectory_rows)
        write_json(out_dir / "initial_boxes.json", boxes_to_rows(initial_boxes))
        write_json(out_dir / "best_boxes.json", boxes_to_rows(best_boxes))
        aggregate["initial_boxes_source"] = initial_boxes_source
        aggregate["policy_checkpoint"] = checkpoint_metadata
        aggregate["best_boxes_json"] = "best_boxes.json"
        aggregate["elapsed_seconds"] = time.perf_counter() - started_at
        write_json(out_dir / "summary.json", aggregate)
        manifest["status"] = "complete"
        manifest["summary"] = aggregate
        write_json(out_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        write_json(out_dir / "manifest.json", manifest)
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

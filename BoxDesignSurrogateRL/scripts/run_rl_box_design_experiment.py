from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
from box_design_surrogate.kandula_repro import initial_boxes_kmeans  # noqa: E402
from box_design_surrogate.rl_env import BoxDesignAction, SurrogateBoxDesignEnv  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


POLICIES = ("rule", "epsilon_greedy", "random")
SUMMARY_FIELDNAMES = [
    "run_id",
    "policy",
    "episode",
    "seed",
    "orders",
    "k",
    "tau",
    "steps",
    "total_reward",
    "initial_uncovered",
    "final_uncovered",
    "initial_low_margin",
    "final_low_margin",
    "initial_mean_base_cost",
    "final_mean_base_cost",
    "initial_mean_adjusted_cost",
    "final_mean_adjusted_cost",
    "initial_mean_assigned_probability",
    "final_mean_assigned_probability",
    "done",
]
TRAJECTORY_FIELDNAMES = [
    "run_id",
    "policy",
    "episode",
    "step",
    "action_index",
    "action_kind",
    "box_id",
    "dimension",
    "delta",
    "reward",
    "done",
    "uncovered_orders",
    "coverage_rate",
    "low_margin_assignments",
    "mean_base_cost",
    "mean_adjusted_cost",
    "mean_assigned_probability",
]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def write_assignments(path: Path, assignments) -> None:
    fieldnames = [
        "order_key",
        "box_id",
        "probability",
        "base_cost",
        "adjusted_cost",
        "covered",
        "best_box_id",
        "best_probability",
        "low_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(
                {
                    "order_key": assignment.order_key,
                    "box_id": "" if assignment.box_id is None else assignment.box_id,
                    "probability": assignment.probability,
                    "base_cost": assignment.base_cost,
                    "adjusted_cost": assignment.adjusted_cost,
                    "covered": int(assignment.covered),
                    "best_box_id": "" if assignment.best_box_id is None else assignment.best_box_id,
                    "best_probability": assignment.best_probability,
                    "low_margin": int(assignment.low_margin),
                }
            )


def mean_base_cost(env: SurrogateBoxDesignEnv) -> float:
    return env.evaluation.total_base_cost / len(env.orders)


def mean_adjusted_cost(env: SurrogateBoxDesignEnv) -> float:
    return env.evaluation.total_adjusted_cost / len(env.orders)


def select_action(
    env: SurrogateBoxDesignEnv,
    *,
    policy: str,
    epsilon: float,
    stop_probability: float,
) -> tuple[int, BoxDesignAction]:
    actions = env.available_actions()
    if policy == "rule":
        action = env.rule_policy_action()
    elif policy == "epsilon_greedy":
        action = env.epsilon_greedy_action(epsilon)
    elif policy == "random":
        action = env.random_policy_action(stop_probability)
    else:
        raise ValueError(f"unknown policy: {policy}")

    for idx, candidate in enumerate(actions):
        if candidate == action:
            return idx, action
    return -1, action


def init_writer(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["rule", "epsilon_greedy", "random"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--random-stop-probability", type=float, default=0.0)
    parser.add_argument("--step-schedule", type=float, nargs="+", default=[0.5, 0.1, 0.05, 0.01])
    parser.add_argument("--max-repair-orders", type=int, default=16)
    parser.add_argument("--max-actions-per-step", type=int, default=96)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--tau-high", type=float, default=0.99)
    parser.add_argument("--lambda-risk", type=float, default=1000.0)
    parser.add_argument("--uncovered-penalty", type=float, default=1_000_000.0)
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
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "rl_box_design")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    boxes_dir = out_dir / "boxes"
    assignments_dir = out_dir / "assignments"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    trajectory_path = out_dir / "trajectories.csv"
    init_writer(summary_path, SUMMARY_FIELDNAMES)
    init_writer(trajectory_path, TRAJECTORY_FIELDNAMES)

    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "orders_limit": args.orders_limit,
            "k": args.k,
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "policies": args.policies,
            "epsilon": args.epsilon,
            "random_stop_probability": args.random_stop_probability,
            "step_schedule": args.step_schedule,
            "max_repair_orders": args.max_repair_orders,
            "max_actions_per_step": args.max_actions_per_step,
            "tau": args.tau,
            "tau_high": args.tau_high,
            "lambda_risk": args.lambda_risk,
            "uncovered_penalty": args.uncovered_penalty,
            "xml_path": str(args.xml_path),
            "model_path": str(args.model_path),
            "out_root": str(args.out_root),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    try:
        orders = read_order_summaries(args.xml_path)
        if args.orders_limit:
            orders = orders[: args.orders_limit]
        model = load_ensemble_pipeline(args.model_path)
        evaluator = SurrogateEvaluator(
            model=model,
            tau=args.tau,
            tau_high=args.tau_high,
            lambda_risk=args.lambda_risk,
            uncovered_penalty=args.uncovered_penalty,
        )
        initial_boxes = initial_boxes_kmeans(orders, args.k, random_state=args.seed)
        write_json(boxes_dir / "initial_kmeans.json", boxes_to_rows(initial_boxes))

        for policy in args.policies:
            for episode in range(args.episodes):
                episode_seed = args.seed + 1009 * episode + 7919 * args.policies.index(policy)
                run_id = f"{policy}_episode{episode}_seed{episode_seed}"
                env = SurrogateBoxDesignEnv(
                    orders,
                    initial_boxes,
                    evaluator,
                    step_schedule=tuple(args.step_schedule),
                    max_steps=args.max_steps,
                    max_repair_orders=args.max_repair_orders,
                    max_actions_per_step=args.max_actions_per_step,
                    seed=episode_seed,
                )
                env.reset()
                initial_uncovered = env.evaluation.uncovered_orders
                initial_low_margin = env.evaluation.low_margin_assignments
                initial_mean_base = mean_base_cost(env)
                initial_mean_adjusted = mean_adjusted_cost(env)
                initial_mean_probability = env.evaluation.mean_assigned_probability
                total_reward = 0.0
                done = False
                steps = 0

                for step_idx in range(args.max_steps):
                    action_index, action = select_action(
                        env,
                        policy=policy,
                        epsilon=args.epsilon,
                        stop_probability=args.random_stop_probability,
                    )
                    result = env.step(action)
                    total_reward += result.reward
                    done = result.done
                    action_row = env.action_to_dict(action)
                    append_row(
                        trajectory_path,
                        TRAJECTORY_FIELDNAMES,
                        {
                            "run_id": run_id,
                            "policy": policy,
                            "episode": episode,
                            "step": step_idx,
                            "action_index": action_index,
                            "action_kind": action_row["kind"],
                            "box_id": action_row["box_id"],
                            "dimension": action_row["dimension"],
                            "delta": action_row["delta"],
                            "reward": result.reward,
                            "done": int(done),
                            "uncovered_orders": result.evaluation.uncovered_orders,
                            "coverage_rate": result.evaluation.coverage_rate,
                            "low_margin_assignments": result.evaluation.low_margin_assignments,
                            "mean_base_cost": result.evaluation.total_base_cost / len(orders),
                            "mean_adjusted_cost": result.evaluation.total_adjusted_cost / len(orders),
                            "mean_assigned_probability": result.evaluation.mean_assigned_probability,
                        },
                    )
                    steps = step_idx + 1
                    if done:
                        break

                append_row(
                    summary_path,
                    SUMMARY_FIELDNAMES,
                    {
                        "run_id": run_id,
                        "policy": policy,
                        "episode": episode,
                        "seed": episode_seed,
                        "orders": len(orders),
                        "k": args.k,
                        "tau": args.tau,
                        "steps": steps,
                        "total_reward": total_reward,
                        "initial_uncovered": initial_uncovered,
                        "final_uncovered": env.evaluation.uncovered_orders,
                        "initial_low_margin": initial_low_margin,
                        "final_low_margin": env.evaluation.low_margin_assignments,
                        "initial_mean_base_cost": initial_mean_base,
                        "final_mean_base_cost": mean_base_cost(env),
                        "initial_mean_adjusted_cost": initial_mean_adjusted,
                        "final_mean_adjusted_cost": mean_adjusted_cost(env),
                        "initial_mean_assigned_probability": initial_mean_probability,
                        "final_mean_assigned_probability": env.evaluation.mean_assigned_probability,
                        "done": int(done),
                    },
                )
                write_json(boxes_dir / f"{run_id}.json", boxes_to_rows(env.boxes))
                write_assignments(assignments_dir / f"{run_id}.csv", env.evaluation.assignments)
                print(
                    f"{run_id}: "
                    f"steps={steps} reward={total_reward:.3f} "
                    f"uncovered={initial_uncovered}->{env.evaluation.uncovered_orders} "
                    f"low_margin={initial_low_margin}->{env.evaluation.low_margin_assignments}",
                    flush=True,
                )

        manifest["status"] = "complete"
        write_json(out_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        write_json(out_dir / "manifest.json", manifest)
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

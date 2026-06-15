from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
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

from box_design_surrogate import SurrogateEvaluator, read_order_summaries  # noqa: E402
from box_design_surrogate.kandula_repro import initial_boxes_kmeans  # noqa: E402
from box_design_surrogate.rl_env import SurrogateBoxDesignEnv  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


SUMMARY_FIELDNAMES = [
    "episode",
    "seed",
    "steps",
    "total_reward",
    "baseline",
    "initial_uncovered",
    "final_uncovered",
    "initial_low_margin",
    "final_low_margin",
    "initial_mean_base_cost",
    "final_mean_base_cost",
    "initial_mean_adjusted_cost",
    "final_mean_adjusted_cost",
]
TRAJECTORY_FIELDNAMES = [
    "episode",
    "step",
    "action_index",
    "action_kind",
    "box_id",
    "dimension",
    "delta",
    "probability",
    "reward",
    "return",
    "done",
    "uncovered_orders",
    "coverage_rate",
    "low_margin_assignments",
    "mean_base_cost",
    "mean_adjusted_cost",
]


@dataclass
class LinearSoftmaxPolicy:
    weights: np.ndarray

    @classmethod
    def init(cls, obs_dim: int, action_count: int, rng: np.random.Generator) -> "LinearSoftmaxPolicy":
        weights = rng.normal(loc=0.0, scale=0.01, size=(action_count, obs_dim + 1))
        weights[0, -1] = -2.0
        weights[1, -1] = -0.5
        return cls(weights=weights)

    def probabilities(
        self,
        obs: np.ndarray,
        temperature: float = 1.0,
        action_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        x = with_bias(obs)
        logits = self.weights @ x
        logits = logits / max(temperature, 1e-6)
        if action_mask is not None:
            logits = np.where(action_mask, logits, -1e9)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def sample(
        self,
        obs: np.ndarray,
        rng: np.random.Generator,
        temperature: float,
        action_mask: np.ndarray | None,
    ) -> tuple[int, float, np.ndarray]:
        probs = self.probabilities(obs, temperature, action_mask)
        action = int(rng.choice(np.arange(len(probs)), p=probs))
        return action, float(probs[action]), probs

    def greedy(self, obs: np.ndarray, action_mask: np.ndarray | None) -> tuple[int, float, np.ndarray]:
        probs = self.probabilities(obs, temperature=1.0, action_mask=action_mask)
        action = int(np.argmax(probs))
        return action, float(probs[action]), probs


def with_bias(obs: np.ndarray) -> np.ndarray:
    return np.concatenate([obs, np.asarray([1.0], dtype=np.float64)])


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def init_writer(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def mean_base_cost(env: SurrogateBoxDesignEnv) -> float:
    return env.evaluation.total_base_cost / len(env.orders)


def mean_adjusted_cost(env: SurrogateBoxDesignEnv) -> float:
    return env.evaluation.total_adjusted_cost / len(env.orders)


def build_action_mask(action_count: int, *, allow_stop: bool, allow_noop: bool) -> np.ndarray:
    mask = np.ones(action_count, dtype=bool)
    mask[0] = allow_stop
    mask[1] = allow_noop
    if not np.any(mask):
        raise ValueError("action mask disables every action")
    return mask


def run_episode(
    env: SurrogateBoxDesignEnv,
    policy: LinearSoftmaxPolicy,
    *,
    rng: np.random.Generator,
    episode: int,
    seed: int,
    gamma: float,
    temperature: float,
    greedy: bool,
    action_mask: np.ndarray | None,
) -> tuple[list[dict[str, object]], dict[str, object], list[np.ndarray], list[int], np.ndarray, list[np.ndarray]]:
    env.reset()
    initial_uncovered = env.evaluation.uncovered_orders
    initial_low_margin = env.evaluation.low_margin_assignments
    initial_mean_base = mean_base_cost(env)
    initial_mean_adjusted = mean_adjusted_cost(env)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    probs_by_step: list[np.ndarray] = []
    rewards: list[float] = []
    rows: list[dict[str, object]] = []

    for step_idx in range(env.max_steps):
        obs = env.observation_vector()
        if greedy:
            action_index, action_probability, probs = policy.greedy(obs, action_mask)
        else:
            action_index, action_probability, probs = policy.sample(obs, rng, temperature, action_mask)
        action = env.fixed_action(action_index)
        result = env.step(action)
        action_row = env.action_to_dict(action)
        observations.append(obs)
        actions.append(action_index)
        probs_by_step.append(probs)
        rewards.append(result.reward)
        rows.append(
            {
                "episode": episode,
                "step": step_idx,
                "action_index": action_index,
                "action_kind": action_row["kind"],
                "box_id": action_row["box_id"],
                "dimension": action_row["dimension"],
                "delta": action_row["delta"],
                "probability": action_probability,
                "reward": result.reward,
                "return": "",
                "done": int(result.done),
                "uncovered_orders": result.evaluation.uncovered_orders,
                "coverage_rate": result.evaluation.coverage_rate,
                "low_margin_assignments": result.evaluation.low_margin_assignments,
                "mean_base_cost": result.evaluation.total_base_cost / len(env.orders),
                "mean_adjusted_cost": result.evaluation.total_adjusted_cost / len(env.orders),
            }
        )
        if result.done:
            break

    returns = discounted_returns(rewards, gamma)
    for row, value in zip(rows, returns):
        row["return"] = float(value)
    summary = {
        "episode": episode,
        "seed": seed,
        "steps": len(rows),
        "total_reward": float(sum(rewards)),
        "baseline": "",
        "initial_uncovered": initial_uncovered,
        "final_uncovered": env.evaluation.uncovered_orders,
        "initial_low_margin": initial_low_margin,
        "final_low_margin": env.evaluation.low_margin_assignments,
        "initial_mean_base_cost": initial_mean_base,
        "final_mean_base_cost": mean_base_cost(env),
        "initial_mean_adjusted_cost": initial_mean_adjusted,
        "final_mean_adjusted_cost": mean_adjusted_cost(env),
    }
    return rows, summary, observations, actions, returns, probs_by_step


def update_policy(
    policy: LinearSoftmaxPolicy,
    *,
    observations: list[np.ndarray],
    actions: list[int],
    returns: np.ndarray,
    probs_by_step: list[np.ndarray],
    baseline: float,
    learning_rate: float,
    grad_clip: float,
) -> None:
    if len(actions) == 0:
        return
    grad = np.zeros_like(policy.weights)
    for obs, action, value, probs in zip(observations, actions, returns, probs_by_step):
        advantage = float(value - baseline)
        x = with_bias(obs)
        one_hot = np.zeros(len(probs), dtype=np.float64)
        one_hot[action] = 1.0
        grad += advantage * np.outer(one_hot - probs, x)
    norm = float(np.linalg.norm(grad))
    if norm > grad_clip:
        grad *= grad_clip / norm
    policy.weights += learning_rate * grad / max(len(actions), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--baseline-beta", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1000.0)
    parser.add_argument("--allow-stop-action", action="store_true")
    parser.add_argument("--allow-noop-action", action="store_true")
    parser.add_argument("--step-schedule", type=float, nargs="+", default=[0.5, 0.1, 0.05, 0.01])
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
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "rl_policy")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_summary_path = out_dir / "train_summary.csv"
    train_trajectory_path = out_dir / "train_trajectories.csv"
    eval_summary_path = out_dir / "eval_summary.csv"
    eval_trajectory_path = out_dir / "eval_trajectories.csv"
    init_writer(train_summary_path, SUMMARY_FIELDNAMES)
    init_writer(train_trajectory_path, TRAJECTORY_FIELDNAMES)
    init_writer(eval_summary_path, SUMMARY_FIELDNAMES)
    init_writer(eval_trajectory_path, TRAJECTORY_FIELDNAMES)

    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": vars(args) | {"xml_path": str(args.xml_path), "model_path": str(args.model_path), "out_root": str(args.out_root)},
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
        env = SurrogateBoxDesignEnv(
            orders,
            initial_boxes,
            evaluator,
            step_schedule=tuple(args.step_schedule),
            max_steps=args.max_steps,
            seed=args.seed,
        )
        rng = np.random.default_rng(args.seed)
        policy = LinearSoftmaxPolicy.init(
            obs_dim=len(env.observation_vector()),
            action_count=env.fixed_action_count,
            rng=rng,
        )
        action_mask = build_action_mask(
            env.fixed_action_count,
            allow_stop=args.allow_stop_action,
            allow_noop=args.allow_noop_action,
        )
        baseline = 0.0

        for episode in range(args.episodes):
            episode_seed = args.seed + episode * 1009
            episode_rng = np.random.default_rng(episode_seed)
            rows, summary, observations, actions, returns, probs = run_episode(
                env,
                policy,
                rng=episode_rng,
                episode=episode,
                seed=episode_seed,
                gamma=args.gamma,
                temperature=args.temperature,
                greedy=False,
                action_mask=action_mask,
            )
            if episode == 0:
                baseline = float(summary["total_reward"])
            else:
                baseline = args.baseline_beta * baseline + (1.0 - args.baseline_beta) * float(summary["total_reward"])
            update_policy(
                policy,
                observations=observations,
                actions=actions,
                returns=returns,
                probs_by_step=probs,
                baseline=baseline,
                learning_rate=args.learning_rate,
                grad_clip=args.grad_clip,
            )
            summary["baseline"] = baseline
            append_rows(train_summary_path, SUMMARY_FIELDNAMES, [summary])
            append_rows(train_trajectory_path, TRAJECTORY_FIELDNAMES, rows)
            print(
                f"episode={episode} reward={float(summary['total_reward']):.3f} "
                f"baseline={baseline:.3f} uncovered={summary['initial_uncovered']}->{summary['final_uncovered']} "
                f"low_margin={summary['initial_low_margin']}->{summary['final_low_margin']}",
                flush=True,
            )

        eval_env = SurrogateBoxDesignEnv(
            orders,
            initial_boxes,
            evaluator,
            step_schedule=tuple(args.step_schedule),
            max_steps=args.max_steps,
            seed=args.seed,
        )
        for episode in range(args.eval_episodes):
            rows, summary, _, _, _, _ = run_episode(
                eval_env,
                policy,
                rng=np.random.default_rng(args.seed + 900_000 + episode),
                episode=episode,
                seed=args.seed + 900_000 + episode,
                gamma=args.gamma,
                temperature=1.0,
                greedy=True,
                action_mask=action_mask,
            )
            summary["baseline"] = baseline
            append_rows(eval_summary_path, SUMMARY_FIELDNAMES, [summary])
            append_rows(eval_trajectory_path, TRAJECTORY_FIELDNAMES, rows)

        np.savez(
            out_dir / "policy_final.npz",
            weights=policy.weights,
            obs_dim=np.asarray([policy.weights.shape[1] - 1], dtype=np.int64),
            action_count=np.asarray([policy.weights.shape[0]], dtype=np.int64),
        )
        manifest["status"] = "complete"
        manifest["policy"] = {
            "type": "linear_softmax_reinforce",
            "obs_dim": int(policy.weights.shape[1] - 1),
            "action_count": int(policy.weights.shape[0]),
            "checkpoint": "policy_final.npz",
        }
        write_json(out_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        write_json(out_dir / "manifest.json", manifest)
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from box_design_surrogate.policy_context import ORDER_CONTEXT_SCHEMA  # noqa: E402
from box_design_surrogate.surrogate_game import SurrogateBoxSizingGame  # noqa: E402
from src.ensemble_train import load_ensemble_pipeline  # noqa: E402


SUMMARY_FIELDS = [
    "episode",
    "environment_id",
    "mode",
    "objective_mode",
    "steps",
    "total_reward",
    "initial_objective",
    "final_objective",
    "best_objective",
    "initial_uncovered",
    "final_uncovered",
    "initial_low_margin",
    "final_low_margin",
    "initial_mean_probability",
    "final_mean_probability",
    "initial_packaging_factor",
    "final_packaging_factor",
    "initial_mean_assigned_box_volume",
    "final_mean_assigned_box_volume",
    "initial_mean_order_volume",
    "final_mean_order_volume",
    "initial_mean_adjusted_cost",
    "final_mean_adjusted_cost",
    "initial_uncovered_penalty",
    "final_uncovered_penalty",
    "initial_low_margin_penalty",
    "final_low_margin_penalty",
    "initial_probability_penalty",
    "final_probability_penalty",
    "terminal_reason",
]
TRAJECTORY_FIELDS = [
    "episode",
    "environment_id",
    "mode",
    "objective_mode",
    "step",
    "action",
    "action_type",
    "reward",
    "environment_reward",
    "objective",
    "uncovered_orders",
    "low_margin_assignments",
    "mean_probability",
    "packaging_factor",
    "mean_assigned_box_volume",
    "mean_order_volume",
    "mean_adjusted_cost",
    "uncovered_penalty",
    "low_margin_penalty",
    "probability_penalty",
    "terminal_reason",
]


TORCH = None
NN = None
CATEGORICAL = None


def ensure_torch():
    global TORCH, NN, CATEGORICAL
    if TORCH is None:
        import torch
        import torch.nn as nn
        from torch.distributions import Categorical

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        TORCH = torch
        NN = nn
        CATEGORICAL = Categorical
    return TORCH, NN, CATEGORICAL


def make_actor_critic_class():
    torch, nn, _ = ensure_torch()

    class ActorCritic(nn.Module):
        def __init__(self, obs_dim: int, action_count: int, hidden_dim: int, hidden_layers: int) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = obs_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Tanh())
                in_dim = hidden_dim
            self.body = nn.Sequential(*layers)
            self.policy_head = nn.Linear(in_dim, action_count)
            self.value_head = nn.Linear(in_dim, 1)

        def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.body(obs)
            return self.policy_head(x), self.value_head(x).squeeze(-1)

    return ActorCritic


@dataclass
class EpisodeBatch:
    observations: list[np.ndarray]
    actions: list[int]
    old_log_probs: list[float]
    rewards: list[float]
    raw_rewards: list[float]
    returns: list[float]
    advantages: list[float]
    summary: dict[str, object]
    trajectory: list[dict[str, object]]


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        out.append(running)
    return list(reversed(out))


def action_type(action: int, k: int) -> str:
    if action == 6 * k:
        return "resign"
    return "decrement" if action < 3 * k else "increment"


def policy_training_reward(
    *,
    previous_metrics: dict[str, object],
    current_metrics: dict[str, object],
    environment_reward: float,
    mode: str,
    objective_mode: str,
    coverage_reward_weight: float,
) -> float:
    if mode != "surrogate" or objective_mode != "paper_pf_surrogate":
        return float(environment_reward)
    previous_uncovered = int(previous_metrics["uncovered_orders"])
    current_uncovered = int(current_metrics["uncovered_orders"])
    previous_pf = float(previous_metrics["packaging_factor"])
    current_pf = float(current_metrics["packaging_factor"])
    coverage_delta = coverage_reward_weight * (previous_uncovered - current_uncovered)
    return float(coverage_delta + previous_pf - current_pf)


def make_training_order_windows(
    orders: list,
    *,
    window_size: int,
    window_stride: int,
) -> list[tuple[int, int, list]]:
    if window_size <= 0 or window_size >= len(orders):
        return [(0, 0, list(orders))]
    if window_stride <= 0:
        window_stride = window_size
    windows = [
        (window_id, start, list(orders[start : start + window_size]))
        for window_id, start in enumerate(range(0, len(orders) - window_size + 1, window_stride))
    ]
    if not windows:
        raise ValueError("training window configuration produced no complete windows")
    return windows


def shuffled_environment_indices(*, count: int, episodes: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("environment count must be positive")
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    while len(indices) < episodes:
        indices.extend(int(value) for value in rng.permutation(count))
    return indices[:episodes]


def collect_episode(
    *,
    env,
    model: ActorCritic,
    episode: int,
    environment_id: int,
    mode: str,
    gamma: float,
    reward_scale: float,
    coverage_reward_weight: float,
    device: torch.device,
) -> EpisodeBatch:
    torch, _, Categorical = ensure_torch()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    values: list[float] = []
    trajectory: list[dict[str, object]] = []
    obs = env.reset()
    initial_metrics = environment_metrics(env)
    best_objective = float(initial_metrics["objective"])
    previous_metrics = initial_metrics
    terminal_reason = ""

    for step in range(env.max_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
        action = int(action_tensor.item())
        result = env.step(action)
        metrics = result_metrics(result, mode, env)
        best_objective = min(best_objective, float(metrics["objective"]))
        observations.append(obs)
        actions.append(action)
        old_log_probs.append(float(log_prob.item()))
        environment_reward = float(result.reward)
        raw_reward = policy_training_reward(
            previous_metrics=previous_metrics,
            current_metrics=metrics,
            environment_reward=environment_reward,
            mode=mode,
            objective_mode=getattr(env, "objective_mode", ""),
            coverage_reward_weight=coverage_reward_weight,
        )
        raw_rewards.append(raw_reward)
        rewards.append(raw_reward * reward_scale)
        values.append(float(value.item()))
        terminal_reason = result.terminal_reason
        trajectory.append(
            {
                "episode": episode,
                "environment_id": environment_id,
                "mode": mode,
                "objective_mode": getattr(env, "objective_mode", ""),
                "step": step,
                "action": action,
                "action_type": action_type(action, env.k),
                "reward": raw_reward,
                "environment_reward": environment_reward,
                "objective": metrics["objective"],
                "uncovered_orders": metrics["uncovered_orders"],
                "low_margin_assignments": metrics["low_margin_assignments"],
                "mean_probability": metrics["mean_probability"],
                "packaging_factor": metrics["packaging_factor"],
                "mean_assigned_box_volume": metrics["mean_assigned_box_volume"],
                "mean_order_volume": metrics["mean_order_volume"],
                "mean_adjusted_cost": metrics["mean_adjusted_cost"],
                "uncovered_penalty": metrics["uncovered_penalty"],
                "low_margin_penalty": metrics["low_margin_penalty"],
                "probability_penalty": metrics["probability_penalty"],
                "terminal_reason": result.terminal_reason,
            }
        )
        previous_metrics = metrics
        obs = env.observation()
        if result.done:
            break

    returns = discounted_returns(rewards, gamma)
    advantages = [ret - val for ret, val in zip(returns, values)]
    final_metrics = environment_metrics(env)
    summary = {
        "episode": episode,
        "environment_id": environment_id,
        "mode": mode,
        "objective_mode": getattr(env, "objective_mode", ""),
        "steps": len(rewards),
        "total_reward": float(sum(raw_rewards)),
        "initial_objective": initial_metrics["objective"],
        "final_objective": final_metrics["objective"],
        "best_objective": best_objective,
        "initial_uncovered": initial_metrics["uncovered_orders"],
        "final_uncovered": final_metrics["uncovered_orders"],
        "initial_low_margin": initial_metrics["low_margin_assignments"],
        "final_low_margin": final_metrics["low_margin_assignments"],
        "initial_mean_probability": initial_metrics["mean_probability"],
        "final_mean_probability": final_metrics["mean_probability"],
        "initial_packaging_factor": initial_metrics["packaging_factor"],
        "final_packaging_factor": final_metrics["packaging_factor"],
        "initial_mean_assigned_box_volume": initial_metrics["mean_assigned_box_volume"],
        "final_mean_assigned_box_volume": final_metrics["mean_assigned_box_volume"],
        "initial_mean_order_volume": initial_metrics["mean_order_volume"],
        "final_mean_order_volume": final_metrics["mean_order_volume"],
        "initial_mean_adjusted_cost": initial_metrics["mean_adjusted_cost"],
        "final_mean_adjusted_cost": final_metrics["mean_adjusted_cost"],
        "initial_uncovered_penalty": initial_metrics["uncovered_penalty"],
        "final_uncovered_penalty": final_metrics["uncovered_penalty"],
        "initial_low_margin_penalty": initial_metrics["low_margin_penalty"],
        "final_low_margin_penalty": final_metrics["low_margin_penalty"],
        "initial_probability_penalty": initial_metrics["probability_penalty"],
        "final_probability_penalty": final_metrics["probability_penalty"],
        "terminal_reason": terminal_reason,
    }
    return EpisodeBatch(
        observations,
        actions,
        old_log_probs,
        rewards,
        raw_rewards,
        returns,
        advantages,
        summary,
        trajectory,
    )


def environment_metrics(env) -> dict[str, object]:
    if isinstance(env, SurrogateBoxSizingGame):
        breakdown = env.objective_breakdown(env.evaluation)
        return {
            "objective": env.current_objective,
            "uncovered_orders": env.evaluation.uncovered_orders,
            "low_margin_assignments": env.evaluation.low_margin_assignments,
            "mean_probability": env.evaluation.mean_assigned_probability,
            "packaging_factor": breakdown.packaging_factor,
            "mean_assigned_box_volume": breakdown.mean_assigned_box_volume,
            "mean_order_volume": breakdown.mean_order_volume,
            "mean_adjusted_cost": breakdown.mean_adjusted_cost,
            "uncovered_penalty": breakdown.uncovered_penalty,
            "low_margin_penalty": breakdown.low_margin_penalty,
            "probability_penalty": breakdown.probability_penalty,
        }
    return {
        "objective": env.score.packaging_factor,
        "uncovered_orders": env.score.uncovered_orders,
        "low_margin_assignments": "",
        "mean_probability": "",
        "packaging_factor": env.score.packaging_factor,
        "mean_assigned_box_volume": env.score.mean_box_volume,
        "mean_order_volume": env.score.mean_order_volume,
        "mean_adjusted_cost": "",
        "uncovered_penalty": "",
        "low_margin_penalty": "",
        "probability_penalty": "",
    }


def result_metrics(result, mode: str, env) -> dict[str, object]:
    if mode == "surrogate":
        breakdown = env.objective_breakdown(result.evaluation)
        return {
            "objective": result.objective,
            "uncovered_orders": result.evaluation.uncovered_orders,
            "low_margin_assignments": result.evaluation.low_margin_assignments,
            "mean_probability": result.evaluation.mean_assigned_probability,
            "packaging_factor": breakdown.packaging_factor,
            "mean_assigned_box_volume": breakdown.mean_assigned_box_volume,
            "mean_order_volume": breakdown.mean_order_volume,
            "mean_adjusted_cost": breakdown.mean_adjusted_cost,
            "uncovered_penalty": breakdown.uncovered_penalty,
            "low_margin_penalty": breakdown.low_margin_penalty,
            "probability_penalty": breakdown.probability_penalty,
        }
    return {
        "objective": result.score.packaging_factor,
        "uncovered_orders": result.score.uncovered_orders,
        "low_margin_assignments": "",
        "mean_probability": "",
        "packaging_factor": result.score.packaging_factor,
        "mean_assigned_box_volume": result.score.mean_box_volume,
        "mean_order_volume": result.score.mean_order_volume,
        "mean_adjusted_cost": "",
        "uncovered_penalty": "",
        "low_margin_penalty": "",
        "probability_penalty": "",
    }


def ppo_update(
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: list[EpisodeBatch],
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    device: torch.device,
) -> None:
    torch, _, Categorical = ensure_torch()
    observations = np.asarray([obs for ep in batch for obs in ep.observations], dtype=np.float32)
    actions = np.asarray([act for ep in batch for act in ep.actions], dtype=np.int64)
    old_log_probs = np.asarray([lp for ep in batch for lp in ep.old_log_probs], dtype=np.float32)
    returns = np.asarray([ret for ep in batch for ret in ep.returns], dtype=np.float32)
    advantages = np.asarray([adv for ep in batch for adv in ep.advantages], dtype=np.float32)
    if len(actions) == 0:
        return
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)

    for _ in range(epochs):
        logits, values = model(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        ratio = torch.exp(log_probs - old_log_probs_t)
        unclipped = ratio * advantages_t
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_t
        policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
        value_loss = torch.mean((values - returns_t) ** 2)
        entropy = torch.mean(dist.entropy())
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument(
        "--training-window-size",
        type=int,
        default=0,
        help="Train across complete order windows of this size; 0 uses one environment with all selected orders.",
    )
    parser.add_argument(
        "--training-window-stride",
        type=int,
        default=0,
        help="Stride between training windows; 0 defaults to the window size.",
    )
    parser.add_argument("--mode", choices=["paper", "surrogate"], default="paper")
    parser.add_argument(
        "--include-order-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a fixed order-distribution summary to the Kx3 box state for the learned variant.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.0,
        help="Scale rewards before PPO returns/advantages. Use 0 for mode-aware default.",
    )
    parser.add_argument(
        "--paper-surrogate-coverage-reward-weight",
        type=float,
        default=1.0,
        help=(
            "Coverage-count reward weight for paper_pf_surrogate PPO training. "
            "PF improvement remains unscaled in the same decomposed reward."
        ),
    )
    parser.add_argument("--skip-policy-update", action="store_true")
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
        help="Surrogate objective: paper-style PF with ML feasibility, or risk-aware robustness variant.",
    )
    parser.add_argument("--terminate-on-worse-than-initial", action="store_true")
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
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "kandula_paper_policy")
    args = parser.parse_args()
    reward_scale = args.reward_scale
    if reward_scale == 0.0:
        reward_scale = (
            1.0
            if args.mode != "surrogate" or args.objective_mode == "paper_pf_surrogate"
            else 1e-6
        )

    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "paper_alignment": {
            "stage": "Stage 2 box-sizing game",
            "state": "K x 3 box dimensions",
            "actions": "6K transforming actions plus one resignation action",
            "reward": (
                "paper mode uses PF reward; paper_pf_surrogate uses coverage-count delta plus PF delta; "
                "risk-aware surrogate mode uses ML surrogate objective difference"
            ),
            "learning": "PPO actor-critic local implementation; paper uses PPO and gives exact hyperparameters in online companion not present locally",
        },
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": vars(args)
        | {
            "effective_reward_scale": reward_scale,
            "xml_path": str(args.xml_path),
            "model_path": str(args.model_path),
            "out_root": str(args.out_root),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        orders = read_order_summaries(args.xml_path)
        if args.orders_limit:
            orders = orders[: args.orders_limit]
        order_windows = make_training_order_windows(
            orders,
            window_size=args.training_window_size,
            window_stride=args.training_window_stride,
        )
        if any(len(window_orders) < args.k for _window_id, _start, window_orders in order_windows):
            raise ValueError(f"every training window must contain at least K={args.k} orders")
        evaluator = None
        if args.mode == "surrogate":
            model_pack = load_ensemble_pipeline(args.model_path)
            evaluator = SurrogateEvaluator(
                model=model_pack,
                tau=args.tau,
                tau_high=args.tau_high,
                lambda_risk=args.lambda_risk,
                uncovered_penalty=args.uncovered_penalty,
            )
        envs = []
        window_manifest = []
        for window_id, start, window_orders in order_windows:
            initial_boxes = initial_boxes_kmeans(window_orders, args.k, random_state=args.seed + window_id)
            if args.mode == "paper":
                env = KandulaBoxSizingGame(
                    window_orders,
                    initial_boxes,
                    step_size=args.step_size,
                    max_steps=args.max_steps,
                )
            else:
                assert evaluator is not None
                env = SurrogateBoxSizingGame(
                    window_orders,
                    initial_boxes,
                    evaluator,
                    step_size=args.step_size,
                    max_steps=args.max_steps,
                    uncovered_weight=args.surrogate_uncovered_weight,
                    low_margin_weight=args.surrogate_low_margin_weight,
                    probability_weight=args.surrogate_probability_weight,
                    objective_mode=args.objective_mode,
                    terminate_on_worse_than_initial=args.terminate_on_worse_than_initial,
                    include_order_context=args.include_order_context,
                )
            envs.append(env)
            window_manifest.append(
                {
                    "environment_id": window_id,
                    "orders_start": start,
                    "orders_count": len(window_orders),
                    "initial_uncovered": environment_metrics(env)["uncovered_orders"],
                    "initial_packaging_factor": environment_metrics(env)["packaging_factor"],
                }
            )
        shared_scale_dim = max(float(env.scale_dim) for env in envs)
        for env in envs:
            env.scale_dim = shared_scale_dim
        env = envs[0]
        manifest["training_environments"] = window_manifest
        manifest["shared_scale_dim"] = shared_scale_dim
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        torch, _, _ = ensure_torch()
        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        ActorCritic = make_actor_critic_class()
        model = ActorCritic(env.observation_dim, env.action_count, args.hidden_dim, args.hidden_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        summary_path = out_dir / "train_summary.csv"
        trajectory_path = out_dir / "train_trajectories.csv"
        batch: list[EpisodeBatch] = []
        environment_indices = shuffled_environment_indices(
            count=len(envs),
            episodes=args.episodes,
            seed=args.seed + 104729,
        )
        for episode, environment_id in enumerate(environment_indices):
            env = envs[environment_id]
            ep = collect_episode(
                env=env,
                model=model,
                episode=episode,
                environment_id=environment_id,
                mode=args.mode,
                gamma=args.gamma,
                reward_scale=reward_scale,
                coverage_reward_weight=args.paper_surrogate_coverage_reward_weight,
                device=device,
            )
            batch.append(ep)
            write_rows(summary_path, SUMMARY_FIELDS, [ep.summary])
            write_rows(trajectory_path, TRAJECTORY_FIELDS, ep.trajectory)
            print(
                f"episode={episode} env={environment_id} steps={ep.summary['steps']} "
                f"reward={ep.summary['total_reward']:.6f} "
                f"obj={float(ep.summary['initial_objective']):.6f}->{float(ep.summary['final_objective']):.6f} "
                f"reason={ep.summary['terminal_reason']}",
                flush=True,
            )
            if len(batch) >= args.batch_episodes and not args.skip_policy_update:
                print(f"ppo_update_start episodes={len(batch)}", flush=True)
                ppo_update(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    clip_ratio=args.clip_ratio,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    epochs=args.ppo_epochs,
                    device=device,
                )
                print("ppo_update_done", flush=True)
                batch = []
        if batch and not args.skip_policy_update:
            print(f"ppo_update_start episodes={len(batch)}", flush=True)
            ppo_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                clip_ratio=args.clip_ratio,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                epochs=args.ppo_epochs,
                device=device,
            )
            print("ppo_update_done", flush=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "obs_dim": env.observation_dim,
                "action_count": env.action_count,
                "k": env.k,
                "step_size": args.step_size,
                "hidden_dim": args.hidden_dim,
                "hidden_layers": args.hidden_layers,
                "normalize_observation": env.normalize_observation,
                "scale_dim": shared_scale_dim,
                "include_order_context": args.include_order_context,
                "order_context_schema": list(ORDER_CONTEXT_SCHEMA) if args.include_order_context else [],
                "mode": args.mode,
                "objective_mode": getattr(env, "objective_mode", ""),
                "reward_scale": reward_scale,
                "reward_definition": (
                    "coverage_count_delta_plus_pf_delta"
                    if args.mode == "surrogate" and args.objective_mode == "paper_pf_surrogate"
                    else "environment_objective_delta"
                ),
                "paper_surrogate_coverage_reward_weight": args.paper_surrogate_coverage_reward_weight,
                "training_order_count": len(orders),
                "training_window_count": len(envs),
                "training_window_size": args.training_window_size,
                "training_window_stride": args.training_window_stride,
                "training_xml_path": str(args.xml_path),
                "training_xml_sha256": file_sha256(args.xml_path),
            },
            out_dir / "policy_final.pt",
        )
        manifest["status"] = "complete"
        manifest["checkpoint"] = "policy_final.pt"
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

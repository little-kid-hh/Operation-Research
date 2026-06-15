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


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]


COMPARISON_FIELDS = [
    "method",
    "schedule",
    "status",
    "total_iterations",
    "candidate_evaluations",
    "initial_objective",
    "best_objective",
    "objective_delta",
    "initial_uncovered",
    "best_uncovered",
    "initial_low_margin",
    "best_low_margin",
    "initial_mean_probability",
    "best_mean_probability",
    "initial_packaging_factor",
    "best_packaging_factor",
    "initial_mean_assigned_box_volume",
    "best_mean_assigned_box_volume",
    "initial_mean_order_volume",
    "best_mean_order_volume",
    "run_dir",
    "final_boxes_json",
]


@dataclass(frozen=True)
class StageSpec:
    step_size: float
    search_iters: int


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_dir(root: Path) -> Path:
    runs = sorted(path for path in root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run_* directory under {root}")
    return runs[-1]


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")


def parse_stages(value: str) -> list[StageSpec]:
    stages = []
    for part in value.split(","):
        if not part:
            continue
        step_size, search_iters = part.split(":", 1)
        stages.append(StageSpec(float(step_size), int(search_iters)))
    if not stages:
        raise ValueError("staged schedule must contain at least one stage")
    return stages


def write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def maybe_train_policy(args, out_dir: Path, log_root: Path) -> Path:
    if args.policy_path is not None:
        return args.policy_path
    policy_root = out_dir / "policy"
    cmd = [
        sys.executable,
        "BoxDesignSurrogateRL/scripts/train_kandula_paper_policy.py",
        "--mode",
        "surrogate",
        "--orders-limit",
        str(args.orders_limit),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.train_episodes),
        "--batch-episodes",
        str(args.train_batch_episodes),
        "--max-steps",
        str(args.train_max_steps),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--ppo-epochs",
        str(args.train_ppo_epochs),
        "--tau",
        str(args.tau),
        "--objective-mode",
        args.objective_mode,
        "--out-root",
        str(policy_root),
    ]
    print("train policy", flush=True)
    run_command(cmd, log_root / "train_policy.log")
    return latest_run_dir(policy_root) / "policy_final.pt"


def paas_row(
    *,
    method: str,
    schedule: str,
    run_dir: Path,
    k: int,
    total_iterations: int,
) -> dict[str, object]:
    manifest = read_json(run_dir / "manifest.json")
    summary = read_json(run_dir / "summary.json")
    initial = summary.get("initial_objective", "")
    best = summary.get("best_objective", "")
    delta = "" if initial == "" or best == "" else best - initial
    return {
        "method": method,
        "schedule": schedule,
        "status": manifest.get("status", ""),
        "total_iterations": total_iterations,
        "candidate_evaluations": total_iterations * 6 * k,
        "initial_objective": initial,
        "best_objective": best,
        "objective_delta": delta,
        "initial_uncovered": summary.get("initial_uncovered", ""),
        "best_uncovered": summary.get("best_uncovered", ""),
        "initial_low_margin": summary.get("initial_low_margin", ""),
        "best_low_margin": summary.get("best_low_margin", ""),
        "initial_mean_probability": summary.get("initial_mean_probability", ""),
        "best_mean_probability": summary.get("best_mean_probability", ""),
        "initial_packaging_factor": summary.get("initial_packaging_factor", ""),
        "best_packaging_factor": summary.get("best_packaging_factor", ""),
        "initial_mean_assigned_box_volume": summary.get("initial_mean_assigned_box_volume", ""),
        "best_mean_assigned_box_volume": summary.get("best_mean_assigned_box_volume", ""),
        "initial_mean_order_volume": summary.get("initial_mean_order_volume", ""),
        "best_mean_order_volume": summary.get("best_mean_order_volume", ""),
        "run_dir": str(run_dir),
        "final_boxes_json": str(run_dir / "best_boxes.json"),
    }


def staged_row(
    *,
    run_dir: Path,
    k: int,
    stages: list[StageSpec],
    schedule: str,
) -> dict[str, object]:
    manifest = read_json(run_dir / "manifest.json")
    total = read_json(run_dir / "total_summary.json") or manifest.get("total_summary", {})
    total_iterations = sum(stage.search_iters for stage in stages)
    final_stage_dirs = sorted((run_dir / "stages").rglob("run_*"))
    final_boxes = final_stage_dirs[-1] / "best_boxes.json" if final_stage_dirs else ""
    return {
        "method": "coarse_to_fine",
        "schedule": schedule,
        "status": manifest.get("status", ""),
        "total_iterations": total.get("total_iterations", total_iterations),
        "candidate_evaluations": total.get("candidate_evaluations", total_iterations * 6 * k),
        "initial_objective": total.get("initial_objective", ""),
        "best_objective": total.get("best_objective", ""),
        "objective_delta": total.get("objective_delta", ""),
        "initial_uncovered": total.get("initial_uncovered", ""),
        "best_uncovered": total.get("best_uncovered", ""),
        "initial_low_margin": total.get("initial_low_margin", ""),
        "best_low_margin": total.get("best_low_margin", ""),
        "initial_mean_probability": total.get("initial_mean_probability", ""),
        "best_mean_probability": total.get("best_mean_probability", ""),
        "initial_packaging_factor": total.get("initial_packaging_factor", ""),
        "best_packaging_factor": total.get("best_packaging_factor", ""),
        "initial_mean_assigned_box_volume": total.get("initial_mean_assigned_box_volume", ""),
        "best_mean_assigned_box_volume": total.get("best_mean_assigned_box_volume", ""),
        "initial_mean_order_volume": total.get("initial_mean_order_volume", ""),
        "best_mean_order_volume": total.get("best_mean_order_volume", ""),
        "run_dir": str(run_dir),
        "final_boxes_json": str(final_boxes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument(
        "--objective-mode",
        choices=["paper_pf_surrogate", "risk_aware_surrogate"],
        default="paper_pf_surrogate",
        help="Surrogate objective used by training and PAAS.",
    )
    parser.add_argument("--single-iters", type=int, default=60)
    parser.add_argument("--single-steps", default="0.5,0.25")
    parser.add_argument("--staged-schedule", default="0.5:10,0.25:20,0.1:30")
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--train-max-steps", type=int, default=2)
    parser.add_argument("--train-batch-episodes", type=int, default=1)
    parser.add_argument("--train-ppo-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "step_schedule_comparison")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    log_root = out_dir / "logs"
    single_root = out_dir / "single"
    staged_root = out_dir / "staged"
    out_dir.mkdir(parents=True, exist_ok=True)
    staged_specs = parse_stages(args.staged_schedule)
    single_steps = [float(part) for part in args.single_steps.split(",") if part]

    manifest = {
        "timestamp": timestamp,
        "status": "running",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short", "BoxDesignSurrogateRL"]),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "policy_path": str(args.policy_path) if args.policy_path else None,
            "orders_limit": args.orders_limit,
            "k": args.k,
            "seed": args.seed,
            "tau": args.tau,
            "objective_mode": args.objective_mode,
            "single_iters": args.single_iters,
            "single_steps": args.single_steps,
            "staged_schedule": args.staged_schedule,
            "rollout_steps": args.rollout_steps,
            "rollout_samples": args.rollout_samples,
            "train_episodes": args.train_episodes,
            "train_max_steps": args.train_max_steps,
            "train_batch_episodes": args.train_batch_episodes,
            "train_ppo_epochs": args.train_ppo_epochs,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "out_root": str(args.out_root),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    try:
        policy_path = maybe_train_policy(args, out_dir, log_root)
        for step in single_steps:
            run_root = single_root / f"step{step:g}_iters{args.single_iters}"
            cmd = [
                sys.executable,
                "BoxDesignSurrogateRL/scripts/run_kandula_paper_paas.py",
                "--mode",
                "surrogate",
                "--policy-path",
                str(policy_path),
                "--orders-limit",
                str(args.orders_limit),
                "--k",
                str(args.k),
                "--seed",
                str(args.seed),
                "--step-size",
                str(step),
                "--search-iters",
                str(args.single_iters),
                "--rollout-steps",
                str(args.rollout_steps),
                "--rollout-samples",
                str(args.rollout_samples),
                "--tau",
                str(args.tau),
                "--objective-mode",
                args.objective_mode,
                "--out-root",
                str(run_root),
            ]
            print(f"single step={step:g} iters={args.single_iters}", flush=True)
            run_command(cmd, log_root / f"single_step{step:g}.log")
            run_dir = latest_run_dir(run_root)
            rows.append(
                paas_row(
                    method=f"single_step_{step:g}",
                    schedule=f"{step:g}:{args.single_iters}",
                    run_dir=run_dir,
                    k=args.k,
                    total_iterations=args.single_iters,
                )
            )

        staged_cmd = [
            sys.executable,
            "BoxDesignSurrogateRL/scripts/run_coarse_to_fine_surrogate_paas.py",
            "--policy-path",
            str(policy_path),
            "--orders-limit",
            str(args.orders_limit),
            "--k",
            str(args.k),
            "--seed",
            str(args.seed),
            "--tau",
            str(args.tau),
            "--objective-mode",
            args.objective_mode,
            "--stages",
            args.staged_schedule,
            "--rollout-steps",
            str(args.rollout_steps),
            "--rollout-samples",
            str(args.rollout_samples),
            "--out-root",
            str(staged_root),
        ]
        print(f"staged {args.staged_schedule}", flush=True)
        run_command(staged_cmd, log_root / "staged.log")
        staged_run = latest_run_dir(staged_root)
        rows.append(
            staged_row(
                run_dir=staged_run,
                k=args.k,
                stages=staged_specs,
                schedule=args.staged_schedule,
            )
        )

        comparison_path = out_dir / "comparison_summary.csv"
        write_comparison(comparison_path, rows)
        manifest["status"] = "complete"
        manifest["policy_path"] = str(policy_path)
        manifest["comparison_summary"] = str(comparison_path)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

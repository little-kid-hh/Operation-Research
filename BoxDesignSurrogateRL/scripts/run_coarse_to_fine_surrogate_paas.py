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


STAGE_FIELDS = [
    "stage",
    "step_size",
    "search_iters",
    "run_dir",
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
    "elapsed_seconds",
    "candidate_batch_size",
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


def parse_stages(value: str) -> list[StageSpec]:
    stages = []
    for part in value.split(","):
        if not part:
            continue
        step, iters = part.split(":", 1)
        stages.append(StageSpec(float(step), int(iters)))
    if not stages:
        raise ValueError("at least one stage is required")
    return stages


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable_parameters(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


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


def write_stage_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STAGE_FIELDS)
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
    parser.add_argument("--stages", default="0.5:10,0.25:20,0.1:30")
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Number of candidate box sets to score per batch in surrogate PAAS stages.",
    )
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--train-max-steps", type=int, default=2)
    parser.add_argument("--train-batch-episodes", type=int, default=1)
    parser.add_argument("--train-ppo-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "coarse_to_fine_surrogate_paas")
    args = parser.parse_args()

    stages = parse_stages(args.stages)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    stage_root = out_dir / "stages"
    log_root = out_dir / "logs"
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
        "stages": [{"step_size": stage.step_size, "search_iters": stage.search_iters} for stage in stages],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        policy_path = maybe_train_policy(args, out_dir, log_root)
        current_boxes_json = None
        rows = []
        for idx, stage in enumerate(stages):
            stage_out = stage_root / f"stage{idx}_step{stage.step_size:g}_iters{stage.search_iters}"
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
                str(stage.step_size),
                "--search-iters",
                str(stage.search_iters),
                "--rollout-steps",
                str(args.rollout_steps),
                "--rollout-samples",
                str(args.rollout_samples),
                "--tau",
                str(args.tau),
                "--objective-mode",
                args.objective_mode,
                "--out-root",
                str(stage_out),
            ]
            if args.candidate_batch_size is not None:
                cmd.extend(["--candidate-batch-size", str(args.candidate_batch_size)])
            if current_boxes_json is not None:
                cmd.extend(["--initial-boxes-json", str(current_boxes_json)])
            print(f"stage={idx} step={stage.step_size:g} iters={stage.search_iters}", flush=True)
            run_command(cmd, log_root / f"stage{idx}_step{stage.step_size:g}.log")
            run_dir = latest_run_dir(stage_out)
            summary = read_json(run_dir / "summary.json")
            rows.append(
                {
                    "stage": idx,
                    "step_size": stage.step_size,
                    "search_iters": stage.search_iters,
                    "run_dir": str(run_dir),
                    "initial_objective": summary.get("initial_objective", ""),
                    "best_objective": summary.get("best_objective", ""),
                    "objective_delta": summary.get("best_objective", 0.0) - summary.get("initial_objective", 0.0),
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
                    "elapsed_seconds": summary.get("elapsed_seconds", ""),
                    "candidate_batch_size": summary.get("candidate_batch_size", ""),
                }
            )
            current_boxes_json = run_dir / "best_boxes.json"

        write_stage_rows(out_dir / "stage_summary.csv", rows)
        final_summary = read_json(latest_run_dir(stage_out) / "summary.json")
        total_summary = {}
        if rows:
            total_summary = {
                "initial_objective": rows[0]["initial_objective"],
                "best_objective": rows[-1]["best_objective"],
                "objective_delta": rows[-1]["best_objective"] - rows[0]["initial_objective"],
                "initial_uncovered": rows[0]["initial_uncovered"],
                "best_uncovered": rows[-1]["best_uncovered"],
                "initial_low_margin": rows[0]["initial_low_margin"],
                "best_low_margin": rows[-1]["best_low_margin"],
                "initial_mean_probability": rows[0]["initial_mean_probability"],
                "best_mean_probability": rows[-1]["best_mean_probability"],
                "initial_packaging_factor": rows[0]["initial_packaging_factor"],
                "best_packaging_factor": rows[-1]["best_packaging_factor"],
                "initial_mean_assigned_box_volume": rows[0]["initial_mean_assigned_box_volume"],
                "best_mean_assigned_box_volume": rows[-1]["best_mean_assigned_box_volume"],
                "initial_mean_order_volume": rows[0]["initial_mean_order_volume"],
                "best_mean_order_volume": rows[-1]["best_mean_order_volume"],
                "elapsed_seconds": sum(
                    float(row["elapsed_seconds"])
                    for row in rows
                    if row["elapsed_seconds"] != ""
                ),
                "candidate_batch_size": args.candidate_batch_size,
                "total_iterations": sum(stage.search_iters for stage in stages),
                "candidate_evaluations": sum(stage.search_iters * 6 * args.k for stage in stages),
            }
            (out_dir / "total_summary.json").write_text(
                json.dumps(total_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        manifest["status"] = "complete"
        manifest["policy_path"] = str(policy_path)
        manifest["stage_summary"] = str(out_dir / "stage_summary.csv")
        manifest["final_summary"] = final_summary
        manifest["total_summary"] = total_summary
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

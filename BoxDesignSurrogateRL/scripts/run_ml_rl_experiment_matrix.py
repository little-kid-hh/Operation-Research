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


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def split_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def split_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


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


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_index(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    fields = [
        "stage",
        "mode",
        "objective_mode",
        "orders_limit",
        "k",
        "step_size",
        "tau",
        "policy_run_dir",
        "paas_run_dir",
        "status",
        "initial_objective",
        "best_objective",
        "initial_uncovered",
        "best_uncovered",
        "initial_low_margin",
        "best_low_margin",
        "initial_mean_probability",
        "best_mean_probability",
        "initial_packaging_factor",
        "best_packaging_factor",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-limits", default="500")
    parser.add_argument("--ks", default="10")
    parser.add_argument("--step-sizes", default="0.5,0.25")
    parser.add_argument("--taus", default="0.95")
    parser.add_argument(
        "--objective-mode",
        choices=["paper_pf_surrogate", "risk_aware_surrogate"],
        default="paper_pf_surrogate",
        help="Surrogate objective used by training and PAAS.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--train-max-steps", type=int, default=2)
    parser.add_argument("--train-batch-episodes", type=int, default=1)
    parser.add_argument("--train-ppo-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--search-iters", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "ml_rl_matrix")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{timestamp}"
    policy_root = out_dir / "policies"
    paas_root = out_dir / "paas"
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
        "parameters": vars(args) | {"out_root": str(args.out_root)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    index_path = out_dir / "matrix_index.csv"
    try:
        for orders_limit in split_ints(args.orders_limits):
            for k in split_ints(args.ks):
                for tau in split_floats(args.taus):
                    train_out = policy_root / f"orders{orders_limit}_k{k}_tau{tau:g}"
                    train_cmd = [
                        sys.executable,
                        "BoxDesignSurrogateRL/scripts/train_kandula_paper_policy.py",
                        "--mode",
                        "surrogate",
                        "--orders-limit",
                        str(orders_limit),
                        "--k",
                        str(k),
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
                        str(tau),
                        "--objective-mode",
                        args.objective_mode,
                        "--out-root",
                        str(train_out),
                    ]
                    print(f"train orders={orders_limit} k={k} tau={tau:g}", flush=True)
                    run_command(train_cmd, log_root / f"train_orders{orders_limit}_k{k}_tau{tau:g}.log")
                    policy_run = latest_run_dir(train_out)
                    policy_path = policy_run / "policy_final.pt"

                    for step_size in split_floats(args.step_sizes):
                        paas_out = paas_root / f"orders{orders_limit}_k{k}_tau{tau:g}_step{step_size:g}"
                        paas_cmd = [
                            sys.executable,
                            "BoxDesignSurrogateRL/scripts/run_kandula_paper_paas.py",
                            "--mode",
                            "surrogate",
                            "--policy-path",
                            str(policy_path),
                            "--orders-limit",
                            str(orders_limit),
                            "--k",
                            str(k),
                            "--seed",
                            str(args.seed),
                            "--step-size",
                            str(step_size),
                            "--search-iters",
                            str(args.search_iters),
                            "--rollout-steps",
                            str(args.rollout_steps),
                            "--rollout-samples",
                            str(args.rollout_samples),
                            "--tau",
                            str(tau),
                            "--objective-mode",
                            args.objective_mode,
                            "--out-root",
                            str(paas_out),
                        ]
                        print(
                            f"paas orders={orders_limit} k={k} tau={tau:g} step={step_size:g}",
                            flush=True,
                        )
                        run_command(
                            paas_cmd,
                            log_root / f"paas_orders{orders_limit}_k{k}_tau{tau:g}_step{step_size:g}.log",
                        )
                        paas_run = latest_run_dir(paas_out)
                        summary = read_json(paas_run / "summary.json")
                        append_index(
                            index_path,
                            {
                                "stage": "surrogate_policy_paas",
                                "mode": "surrogate",
                                "objective_mode": args.objective_mode,
                                "orders_limit": orders_limit,
                                "k": k,
                                "step_size": step_size,
                                "tau": tau,
                                "policy_run_dir": str(policy_run),
                                "paas_run_dir": str(paas_run),
                                "status": read_json(paas_run / "manifest.json").get("status", ""),
                                "initial_objective": summary.get("initial_objective", ""),
                                "best_objective": summary.get("best_objective", ""),
                                "initial_uncovered": summary.get("initial_uncovered", ""),
                                "best_uncovered": summary.get("best_uncovered", ""),
                                "initial_low_margin": summary.get("initial_low_margin", ""),
                                "best_low_margin": summary.get("best_low_margin", ""),
                                "initial_mean_probability": summary.get("initial_mean_probability", ""),
                                "best_mean_probability": summary.get("best_mean_probability", ""),
                                "initial_packaging_factor": summary.get("initial_packaging_factor", ""),
                                "best_packaging_factor": summary.get("best_packaging_factor", ""),
                            },
                        )

        summary_out = out_dir / "summary.csv"
        summarize_cmd = [
            sys.executable,
            "BoxDesignSurrogateRL/scripts/summarize_experiment_runs.py",
            str(policy_root),
            str(paas_root),
            "--out",
            str(summary_out),
        ]
        run_command(summarize_cmd, log_root / "summarize.log")
        manifest["status"] = "complete"
        manifest["index"] = str(index_path)
        manifest["summary"] = str(summary_out)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

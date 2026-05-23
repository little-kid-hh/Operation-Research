# -*- coding: utf-8 -*-
"""
Main entry point for HybridSVM experiments.

Usage:
    python run_experiment.py                         # full pipeline
    python run_experiment.py --skip-evolution         # SVM only
    python run_experiment.py --evolve-only           # use existing SVM model
    python run_experiment.py --model-path models/linear_svm.json
    python run_experiment.py --resume-exp-dir experiments/.../exp_...  # continue (any layout)

Pipeline stages:
    1. Load data + train / load linear SVM
    2. Mine hard cases (FN + FP)
    3. [Optional] Run FunSearch/ReEvo evolution
    4. Compare SVM baseline vs Hybrid
    5. Save results
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split as tts

ROOT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
BUNDLE_ROOT = ROOT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.svm_train import (
    train_svm,
    fit_svm_pipeline,
    load_svm_pipeline,
    load_raw_data,
    engineer_features,
    DROP_COLS,
    DEDUP_KEY,
    print_evaluation,
    format_svm_linear_insights_for_prompt,
)
from src.ensemble_train import (
    EnsembleConfig,
    fit_ensemble_pipeline,
    load_ensemble_pipeline,
)
from TabTreeFormer.tabtreeformer import (
    TabTreeFormerConfig,
    fit_tabtreeformer_pipeline,
    load_tabtreeformer_pipeline,
)
from src.hard_cases import (
    mine_hard_cases,
    summarize_hard_cases,
    summarize_for_evolution_prompt,
    render_hard_case_prompt,
    format_feature_stats_for_prompt,
)
from src.evolution import (
    EvolutionEngine,
    EvolutionConfig,
    Rule,
    RULE_SKELETON,
    save_rules,
    load_rules,
    population_from_flat_rules,
    apply_rules,
    boundary_candidate_mask,
)
from src.evaluate import (
    compare_svm_vs_hybrid,
    evaluate_predictor,
)


# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SVM_DATA_SRC = BUNDLE_ROOT / "data" / "training_2orientations.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
# Default root for new runs; use ``--experiments-root experiments`` for the legacy layout.
DEFAULT_EXPERIMENTS_ROOT = ROOT / "experiments_v2"

EVOLUTION_CHECKPOINT_NAME = "evolution_checkpoint.json"


def llm_model_slug(llm_model: str) -> str:
    """Safe single path segment for experiment subfolders."""
    s = llm_model.strip().replace("\\", "_").replace("/", "_").replace(":", "_")
    s = "_".join(s.split())
    return s or "unknown_model"

# Must match Stage 4 hybrid evaluation and evolution iter snapshots
HYBRID_RULE_TOP_K = 3
# Must match ``EvolutionConfig(..., island_reset_every=...)`` in ``run_experiment`` (for checkpoint / resume warnings)
EVOLUTION_ISLAND_RESET_EVERY = 5


def infer_resume_iteration_from_snapshots(exp_dir: pathlib.Path) -> int:
    """Largest N from hard_case_summary_hybrid_iter_NNN.md in exp_dir (0 if none)."""
    best = 0
    for p in exp_dir.glob("hard_case_summary_hybrid_iter_*.md"):
        m = re.match(r"hard_case_summary_hybrid_iter_(\d+)\.md", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def resolve_experiments_root(path: pathlib.Path | str | None) -> pathlib.Path:
    """Default ``experiments_v2`` under HybridSVM; absolute paths accepted."""
    if path is None:
        return DEFAULT_EXPERIMENTS_ROOT
    p = pathlib.Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def guess_model_path_for_exp_dir(exp_dir: pathlib.Path) -> pathlib.Path | None:
    """exp_YYYYMMDD_HHMMSS -> models/linear_svm_YYYYMMDD_HHMMSS.json"""
    name = exp_dir.name
    if not name.startswith("exp_"):
        return None
    suffix = name[4:]
    p = MODEL_DIR / f"linear_svm_{suffix}.json"
    return p if p.exists() else None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))


def evolution_mechanism_tag(
    score_mode: str,
    *,
    harm_weight: float,
    score_hard_guard: bool,
) -> str:
    if score_mode == "legacy_f1":
        return "legacy_f1"
    if score_mode == "net_gain_explore_v1":
        guard_tag = "guard" if score_hard_guard else "no_guard"
        hw = str(float(harm_weight)).rstrip("0").rstrip(".")
        return f"{score_mode}_hw{hw}_{guard_tag}"
    guard_tag = "guard" if score_hard_guard else "no_guard"
    hw = str(float(harm_weight)).rstrip("0").rstrip(".")
    return f"{score_mode}_hw{hw}_{guard_tag}"


def build_results_payload(
    *,
    run_ts: str,
    baseline_mode: str,
    baseline_name: str,
    baseline_meta: dict,
    llm_model: str,
    llm_timeout_per_call: int,
    n_evolution_iters: int,
    resume_mode: bool,
    last_iter: int,
    exp_dir: pathlib.Path,
    mechanism_tag: str,
    evolution_score_mode: str,
    evolution_harm_weight: float,
    evolution_score_hard_guard: bool,
    n_hard_typical: int,
    n_easy_typical: int,
    n_near_hard_easy: int,
    rule_selection_split: str,
    rule_val_size: float,
    y_rule_eval: np.ndarray,
    y_test: np.ndarray,
    boundary_gate_active: bool,
    hybrid_boundary_margin: float | None,
    hybrid_fn_margin: float | None,
    hybrid_fp_margin: float | None,
    evolve_boundary_only: bool,
    boundary_mask: np.ndarray | None,
    baseline_pred_rule_eval: np.ndarray,
    baseline_eval_res,
    hybrid_eval_res,
    hybrid_effects: dict,
    fn_cases: list,
    fp_cases: list,
    easy_tn: list,
    easy_tp: list,
    fn_cases_all: list,
    fp_cases_all: list,
    easy_tn_all: list,
    easy_tp_all: list,
    fn_h: list,
    fp_h: list,
    best_rule_score: float | None,
    skip_evolution: bool,
) -> dict:
    return {
        "timestamp": run_ts,
        "skip_evolution": bool(skip_evolution),
        "baseline_mode": baseline_mode,
        "baseline_models": baseline_meta.get("base_models", []),
        "meta_model": baseline_meta.get("meta_model"),
        "xgb_fallback_used": bool(baseline_meta.get("xgb_fallback", False)),
        "decision_threshold": float(baseline_meta.get("decision_threshold", 0.5)),
        "calibrated_threshold": baseline_meta.get("calibrated_threshold"),
        "calibration_info": baseline_meta.get("calibration_info", {}),
        "boundary_gate": {
            "active": bool(boundary_gate_active),
            "boundary_margin": hybrid_boundary_margin,
            "fn_margin": hybrid_fn_margin,
            "fp_margin": hybrid_fp_margin,
            "evolve_boundary_only": bool(evolve_boundary_only),
            "candidate_count": int(boundary_mask.sum()) if boundary_mask is not None else None,
            "candidate_error_count": int(np.sum(boundary_mask & (baseline_pred_rule_eval != y_rule_eval)))
            if boundary_mask is not None
            else None,
        },
        "split_protocol": {
            "rule_selection_split": rule_selection_split,
            "rule_val_size": rule_val_size if rule_selection_split == "val" else None,
            "rule_search_rows": int(len(y_rule_eval)),
            "final_eval_rows": int(len(y_test)),
            "rule_search_role": "validation" if rule_selection_split == "val" else "test_shared_with_final_eval",
            "final_eval_role": "test",
        },
        "mechanism": {
            "tag": mechanism_tag,
            "evolution_score_mode": evolution_score_mode,
            "evolution_harm_weight": float(evolution_harm_weight),
            "evolution_score_hard_guard": bool(evolution_score_hard_guard),
            "n_hard_typical": int(n_hard_typical),
            "n_easy_typical": int(n_easy_typical),
            "n_near_hard_easy": int(n_near_hard_easy),
        },
        "hybrid_effects": hybrid_effects,
        "llm_model": llm_model,
        "llm_timeout_per_call": int(llm_timeout_per_call),
        "n_evolution_iters": n_evolution_iters,
        "resumed": resume_mode,
        "resume_start_iteration": last_iter if resume_mode else None,
        "experiment_dir": str(exp_dir),
        "baseline": {
            "name": baseline_name,
            "accuracy": float(baseline_eval_res.accuracy),
            "precision": float(baseline_eval_res.precision),
            "recall": float(baseline_eval_res.recall),
            "auc": float(baseline_eval_res.auc),
            "tn": int(baseline_eval_res.tn),
            "fp": int(baseline_eval_res.fp),
            "fn": int(baseline_eval_res.fn),
            "tp": int(baseline_eval_res.tp),
            "fpr": float(baseline_eval_res.fpr),
            "fnr": float(baseline_eval_res.fnr),
            "tpr_at_fpr1pct": float(baseline_eval_res.tpr_at_fpr1pct),
            "hard_case_acc": float(baseline_eval_res.hard_case_acc),
            "n_hard_cases": int(baseline_eval_res.n_hard_cases),
        },
        # Legacy key retained for compatibility with existing scripts.
        "svm": {
            "accuracy": float(baseline_eval_res.accuracy),
            "precision": float(baseline_eval_res.precision),
            "recall": float(baseline_eval_res.recall),
            "auc": float(baseline_eval_res.auc),
            "tn": int(baseline_eval_res.tn),
            "fp": int(baseline_eval_res.fp),
            "fn": int(baseline_eval_res.fn),
            "tp": int(baseline_eval_res.tp),
            "fpr": float(baseline_eval_res.fpr),
            "fnr": float(baseline_eval_res.fnr),
            "tpr_at_fpr1pct": float(baseline_eval_res.tpr_at_fpr1pct),
            "hard_case_acc": float(baseline_eval_res.hard_case_acc),
            "n_hard_cases": int(baseline_eval_res.n_hard_cases),
        },
        "hybrid": {
            "accuracy": float(hybrid_eval_res.accuracy),
            "precision": float(hybrid_eval_res.precision),
            "recall": float(hybrid_eval_res.recall),
            "auc": float(hybrid_eval_res.auc),
            "tn": int(hybrid_eval_res.tn),
            "fp": int(hybrid_eval_res.fp),
            "fn": int(hybrid_eval_res.fn),
            "tp": int(hybrid_eval_res.tp),
            "fpr": float(hybrid_eval_res.fpr),
            "fnr": float(hybrid_eval_res.fnr),
            "tpr_at_fpr1pct": float(hybrid_eval_res.tpr_at_fpr1pct),
            "hard_case_acc": float(hybrid_eval_res.hard_case_acc),
            "n_hard_cases": int(hybrid_eval_res.n_hard_cases),
            "best_rule_score": best_rule_score,
        },
        "n_hard_cases": {
            "fn": len(fn_cases_all),
            "fp": len(fp_cases_all),
            "easy_tn": len(easy_tn_all),
            "easy_tp": len(easy_tp_all),
        },
        "n_evolution_cases": {
            "fn": len(fn_cases),
            "fp": len(fp_cases),
            "easy_tn": len(easy_tn),
            "easy_tp": len(easy_tp),
        },
        "n_rule_search_cases_total": {
            "fn": len(fn_cases_all),
            "fp": len(fp_cases_all),
            "easy_tn": len(easy_tn_all),
            "easy_tp": len(easy_tp_all),
        },
        "n_hard_cases_hybrid": {
            "fn": len(fn_h),
            "fp": len(fp_h),
        },
    }


# ─── Seed rules ──────────────────────────────────────────────────────────────

TRIVIAL_SEED = """\
# -*- coding: utf-8 -*-
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    # Seed rule 0: never override (trivial baseline)
    return -1
"""

FILLED_SEED_1 = """\
# -*- coding: utf-8 -*-
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    sku_counts   = features.get('sku_counts', 0)
    fill_ratio   = features.get('fill_ratio', 1.0)
    max_asr      = features.get('max_asr', 1.0)
    sku_avg_vol  = features.get('sku_average_volume', 0)
    vehicle_cap  = features.get('vehicle_capacity', 45000)
    total_vol    = sku_counts * sku_avg_vol

    # Correct FP: very high fill_ratio with many small items — likely infeasible
    if fill_ratio > 0.93 and sku_counts > 15:
        return 0

    # Correct FN: low fill_ratio but SVM is uncertain
    if fill_ratio < 0.70 and svm_prob < 0.60:
        return 1

    return -1
"""


# ─── LLM wrapper (placeholder) ────────────────────────────────────────────────

def _read_env_file_value(name: str) -> str:
    """Read a single KEY=value from HybridSVM/.env without printing secrets."""
    env_file = pathlib.Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return ""
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == name:
            return val.strip().strip('"').strip("'")
    return ""


def _read_config_value(*names: str) -> str:
    import os

    for name in names:
        val = os.environ.get(name, "")
        if val:
            return val.strip().strip('"').strip("'")
        val = _read_env_file_value(name)
        if val:
            return val
    return ""


def _read_bailian_api_key() -> str:
    return _read_config_value("BAILIAN_API_KEY")


def make_llm_callable(
    model: str = "qwen2.5-32b-instruct",
    max_tokens: int = 4096,
    timeout: int | None = 300,
) -> callable:
    """
    Build an LLM callable that works with your configured API.

    Supports:
      - Generic OpenAI-compatible endpoint via ``LLM_API_KEY`` + ``LLM_BASE_URL`` + ``LLM_MODEL``.
      - Bailian / DashScope OpenAI-compatible (BAILIAN_API_KEY) — tried first if key is set;
        pass any compatible model id via ``model`` or ``BAILIAN_MODEL``.
      - OpenAI (via OPENAI_API_KEY env var)
      - DeepSeek (via DEEPSEEK_API_KEY env var)
      - Zhipu (via ZHIPU_API_KEY env var)
      - Local / Llama API (via LLAMA_API_KEY env var)
      - Qwen / DashScope native SDK (via DASHSCOPE_API_KEY env var)

    Returns a function: prompt(str) -> response(str)

    Parameters
    ----------
    max_tokens
        Completion length limit for all chat backends (raise if rules are truncated mid-fence).
    timeout
        Seconds for OpenAI-compatible HTTP clients (Bailian, OpenAI, DeepSeek). ``None`` uses
        300. DashScope native SDK ignores this (see README).
    """
    mt = int(max_tokens)
    http_timeout = 300.0 if timeout is None else float(timeout)
    explicit_model = model
    env_model = _read_config_value("LLM_MODEL", "BAILIAN_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL", "DASHSCOPE_MODEL")
    if env_model and explicit_model in {"", "qwen2.5-32b-instruct"}:
        model = env_model

    # ── Option 0: Generic OpenAI-compatible endpoint ───────────────────────
    generic_key = _read_config_value("LLM_API_KEY")
    generic_base_url = _read_config_value("LLM_BASE_URL")
    if generic_key and generic_base_url:
        try:
            import openai as generic_lib

            def _extract_text(msg) -> str:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return content
                reasoning = getattr(msg, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning
                return ""

            def generic_call(prompt: str) -> str:
                client = generic_lib.OpenAI(
                    api_key=generic_key,
                    base_url=generic_base_url,
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert 3D packing optimization engineer. "
                                "Return only the final answer requested by the user. "
                                "Do not expose chain-of-thought or hidden reasoning."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=mt,
                    timeout=http_timeout,
                )
                return _extract_text(resp.choices[0].message)

            print(f"LLM: OpenAI-compatible endpoint ({model}) selected")
            return generic_call
        except Exception as e:
            print(f"OpenAI-compatible endpoint ({model}) not available: {e}")

    # ── Option 1: Bailian / 百炼 (OpenAI-compatible) — preferred when key is set ──
    bailian_key = _read_bailian_api_key()
    if bailian_key:
        try:
            import openai as bailian_lib
            bailian_base_url = _read_config_value("BAILIAN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

            def bailian_call(prompt: str) -> str:
                client = bailian_lib.OpenAI(
                    api_key=bailian_key,
                    base_url=bailian_base_url,
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert 3D packing optimization engineer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=mt,
                    timeout=http_timeout,
                )
                return resp.choices[0].message.content

            bailian_call("say hello")
            print(f"LLM: Bailian / DashScope compatible ({model}) selected")
            return bailian_call
        except Exception as e:
            print(f"Bailian ({model}) not available: {e}")

    # ── Option 2: OpenAI ────────────────────────────────────────────────────
    try:
        import openai

        def openai_call(prompt: str) -> str:
            openai_base_url = _read_config_value("OPENAI_BASE_URL")
            client = openai.OpenAI(
                base_url=openai_base_url or None,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert 3D packing optimization engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=mt,
                timeout=http_timeout,
            )
            return resp.choices[0].message.content

        openai_call("say hello")
        print(f"LLM: OpenAI ({model}) selected")
        return openai_call
    except Exception:
        pass

    # ── Option 3: DeepSeek ─────────────────────────────────────────────────
    try:
        import openai as deepseek_lib

        def deepseek_call(prompt: str) -> str:
            client = deepseek_lib.OpenAI(
                api_key=_read_config_value("DEEPSEEK_API_KEY"),
                base_url=_read_config_value("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert 3D packing optimization engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=mt,
                timeout=http_timeout,
            )
            return resp.choices[0].message.content

        deepseek_call("say hello")
        print(f"LLM: DeepSeek ({model}) selected")
        return deepseek_call
    except Exception:
        pass

    # ── Option 4: Qwen / DashScope (direct SDK) ────────────────────────────
    try:
        import dashscope

        def qwen_call(prompt: str) -> str:
            # DashScope native SDK: timeout is not wired (use OpenAI-compatible path for timeout_per_call).
            resp = dashscope.Generation.call(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert 3D packing optimization engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=mt,
            )
            return resp.output.text

        qwen_call("say hello")
        print(f"LLM: Qwen ({model}) via DashScope selected")
        return qwen_call
    except Exception:
        pass

    # ── Fallback: random baseline ──────────────────────────────────────────
    print("WARNING: No LLM API detected. Using random rule baseline (NOT REAL LLM).")
    import random

    def random_baseline(prompt: str) -> str:
        # Parse a few values from the prompt for a semi-informed rule
        seed = int(time.time()) % 1000
        rng = random.Random(seed)
        fills = [0.85, 0.88, 0.91, 0.93, 0.95]
        thresh = rng.choice(fills)
        cnts = [5, 10, 15, 20]
        cnt = rng.choice(cnts)
        code = f'''\
# -*- coding: utf-8 -*-
import math

def apply_rule_patch(svm_prob: float, features: dict) -> int:
    sku_counts = features.get('sku_counts', 0)
    fill_ratio = features.get('fill_ratio', 1.0)
    max_asr    = features.get('max_asr', 1.0)

    if fill_ratio > {thresh} and sku_counts > {cnt}:
        return 0
    if fill_ratio < 0.70 and svm_prob < 0.55:
        return 1
    return -1
'''
        return code

    return random_baseline


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(
    data_path: pathlib.Path | None = None,
    model_path: pathlib.Path | None = None,
    skip_evolution: bool = False,
    evolve_only: bool = False,
    n_evolution_iters: int = 10,
    llm_model: str = "qwen3-max-2026-01-23",
    seed: int = 42,
    test_size: float = 0.25,
    random_state: int = 42,
    resume_exp_dir: pathlib.Path | None = None,
    flat_experiment_dir: bool = False,
    llm_max_tokens: int = 4096,
    llm_timeout_per_call: int = 180,
    rule_placement: str = "round_robin",
    island_reset_mode: str = "trim_top_k",
    island_reset_keep_top: int = 3,
    experiments_root: pathlib.Path | str | None = None,
    n_easy_typical: int = 3,
    n_near_hard_easy: int = 3,
    n_hard_typical: int = 3,
    baseline_mode: str = "svm",
    ensemble_base_models: str = "svm,lr,rf,xgb",
    ensemble_meta_model: str = "logreg",
    ensemble_cv_folds: int = 5,
    ensemble_random_state: int | None = None,
    tabtree_rf_n_estimators: int = 128,
    tabtree_rf_max_depth: int = 12,
    tabtree_rf_min_samples_leaf: int = 2,
    tabtree_d_model: int = 96,
    tabtree_nhead: int = 4,
    tabtree_n_layers: int = 2,
    tabtree_ff_dim: int = 192,
    tabtree_dropout: float = 0.1,
    tabtree_mlp_hidden: int = 64,
    tabtree_epochs: int = 12,
    tabtree_batch_size: int = 256,
    tabtree_lr: float = 1e-3,
    tabtree_weight_decay: float = 1e-4,
    hybrid_boundary_margin: float | None = None,
    hybrid_fn_margin: float | None = None,
    hybrid_fp_margin: float | None = None,
    evolve_boundary_only: bool = False,
    rule_selection_split: str = "test",
    rule_val_size: float = 0.2,
    evolution_score_mode: str = "net_gain_v1",
    evolution_harm_weight: float = 3.0,
    evolution_score_hard_guard: bool = True,
) -> dict:
    """
    Run the full HybridSVM experiment.

    Parameters
    ----------
    data_path    : override path to training CSV
    model_path   : load a pre-trained SVM model instead of training
    skip_evolution : skip the evolution loop (SVM baseline only)
    evolve_only  : skip SVM training, load model and just evolve rules
    n_evolution_iters : evolution iterations (additional iterations when resuming)
    seed         : random seed for the LLM random baseline (ignored with real API)
    resume_exp_dir : existing experiment dir with evolved_rules.json; continues same run
    flat_experiment_dir : if True, use <experiments_root>/exp_<ts>/. If False (default),
        use <experiments_root>/by_model/<llm_model>/exp_<ts>/ for new runs.
    experiments_root : root folder for new runs (default: ``experiments_v2`` under HybridSVM).
        Use ``experiments`` to match the legacy layout.
    n_easy_typical : borderline easy TN/TP rows in the evolution prompt.
    n_near_hard_easy : extra contrastive rows (easy TP near FN cloud, easy TN near FP); 0 disables.
    n_hard_typical : typical FN/FP rows shown per group in the evolution prompt.
    llm_max_tokens : max completion tokens for the active LLM backend(s).
    llm_timeout_per_call : timeout (seconds) for each OpenAI-compatible LLM call.
    rule_placement : how new rules are assigned to islands (``round_robin``, ``min_best_score``, ``min_rule_count``).
    island_reset_mode : ``trim_top_k`` keeps the top rules on the worst island before re-seeding; ``clear_and_clone`` clears it.
    island_reset_keep_top : when ``trim_top_k``, number of rules to retain before injecting the clone.
    baseline_mode : baseline predictor for Stage 1/2/4 (``svm`` or ``ensemble``).
    ensemble_base_models : comma-separated base learners for ensemble mode (e.g. ``svm,lr,rf,xgb``).
    ensemble_meta_model : linear combiner for ensemble mode (default: ``logreg``).
    ensemble_cv_folds : OOF folds for ensemble stacking.
    ensemble_random_state : random_state for ensemble learners/folds (defaults to ``random_state`` when None).
    hybrid_boundary_margin : symmetric SVM signed-margin gate for rule overrides.
    hybrid_fn_margin / hybrid_fp_margin : optional asymmetric margins for negative-side FN rescue and positive-side FP veto.
    evolve_boundary_only : if True, LLM evolution sees/scores only rows inside the configured boundary gate.
    rule_selection_split : ``test`` keeps the legacy behavior where rule search and final report share the same held-out split.
        ``val`` uses an outer test split for final reporting and a validation subset from the training portion for rule search.
    rule_val_size : when ``rule_selection_split='val'``, fraction of the outer-train portion used as rule-selection validation.
    evolution_score_mode : rule fitness objective. ``net_gain_v1`` (default) emphasizes corrected-errors minus weighted harm;
        ``legacy_f1`` preserves the previous FN/FP F1-style objective.
    evolution_harm_weight : weight on harmed easy-correct rows under ``net_gain_v1``.
    evolution_score_hard_guard : when True, rules with harmed_correct >= corrected_errors are clipped to non-positive score.
    """
    np.random.seed(seed)
    llm_model = _read_config_value(
        "LLM_MODEL",
        "BAILIAN_MODEL",
        "OPENAI_MODEL",
        "DEEPSEEK_MODEL",
        "DASHSCOPE_MODEL",
    ) or llm_model
    baseline_mode = str(baseline_mode).strip().lower()
    if baseline_mode not in {"svm", "ensemble", "tabtreeformer_rf"}:
        raise ValueError(
            f"Unsupported baseline_mode={baseline_mode!r}; use 'svm', 'ensemble', or 'tabtreeformer_rf'"
        )
    rule_selection_split = str(rule_selection_split).strip().lower()
    if rule_selection_split not in {"test", "val"}:
        raise ValueError("rule_selection_split must be 'test' or 'val'")
    if not (0.0 < float(rule_val_size) < 1.0):
        raise ValueError("rule_val_size must be in (0, 1)")
    evolution_score_mode = str(evolution_score_mode).strip().lower()
    if evolution_score_mode not in {"net_gain_v1", "net_gain_explore_v1", "legacy_f1"}:
        raise ValueError("evolution_score_mode must be 'net_gain_v1', 'net_gain_explore_v1', or 'legacy_f1'")
    if float(evolution_harm_weight) <= 0:
        raise ValueError("evolution_harm_weight must be > 0")
    if int(llm_timeout_per_call) <= 0:
        raise ValueError("llm_timeout_per_call must be > 0")
    mechanism_tag = evolution_mechanism_tag(
        evolution_score_mode,
        harm_weight=float(evolution_harm_weight),
        score_hard_guard=bool(evolution_score_hard_guard),
    )
    boundary_gate_active = (
        hybrid_boundary_margin is not None
        or hybrid_fn_margin is not None
        or hybrid_fp_margin is not None
    )
    if boundary_gate_active and baseline_mode != "svm":
        raise ValueError("Boundary-gated Hybrid currently requires --baseline-mode svm.")
    ensemble_seed = int(random_state if ensemble_random_state is None else ensemble_random_state)
    ensemble_base_models = ",".join(
        [x.strip().lower() for x in str(ensemble_base_models).split(",") if x.strip()]
    )

    resume_mode = resume_exp_dir is not None
    if resume_mode and skip_evolution:
        raise ValueError("--resume-exp-dir cannot be used with --skip-evolution")

    ckpt: dict | None = None
    last_iter = 0
    loaded_rules: list[Rule] | None = None

    if resume_mode:
        exp_dir = pathlib.Path(resume_exp_dir).resolve()
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Resume experiment dir not found: {exp_dir}")
        rules_json = exp_dir / "evolved_rules.json"
        if not rules_json.exists():
            raise FileNotFoundError(f"Missing evolved_rules.json in {exp_dir}")
        loaded_rules = load_rules(rules_json)
        if not loaded_rules:
            raise ValueError(f"No rules in {rules_json}")

        ckpt_path = exp_dir / EVOLUTION_CHECKPOINT_NAME
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            last_iter = int(ckpt.get("iteration", 0))
        else:
            last_iter = infer_resume_iteration_from_snapshots(exp_dir)
            if last_iter == 0 and len(loaded_rules) > 2:
                print(
                    "WARNING: No evolution_checkpoint.json and no hybrid_iter_*.md; "
                    "assuming iteration 0 — snapshot filenames may overlap."
                )

        if ckpt:
            if ckpt.get("test_size") is not None and float(ckpt["test_size"]) != float(test_size):
                raise ValueError(
                    f"test_size={test_size} does not match checkpoint {ckpt['test_size']}"
                )
            if ckpt.get("random_state") is not None and int(ckpt["random_state"]) != int(
                random_state
            ):
                raise ValueError(
                    f"random_state={random_state} does not match checkpoint {ckpt['random_state']}"
                )
            if ckpt.get("data_path"):
                cp_data = pathlib.Path(ckpt["data_path"]).resolve()
                if data_path is not None and pathlib.Path(data_path).resolve() != cp_data:
                    raise ValueError(
                        f"--data-path does not match checkpoint ({cp_data}); omit it or use the same CSV"
                    )
                data_path = cp_data
            if ckpt.get("model_path"):
                cp_model = pathlib.Path(ckpt["model_path"]).resolve()
                if model_path is not None and pathlib.Path(model_path).resolve() != cp_model:
                    raise ValueError(
                        f"--model-path does not match checkpoint ({cp_model}); omit it when resuming"
                    )
                model_path = cp_model

            _soft_resume_keys = (
                ("llm_model", llm_model),
                ("llm_max_tokens", llm_max_tokens),
                ("rule_placement", rule_placement),
                ("island_reset_mode", island_reset_mode),
                ("island_reset_keep_top", island_reset_keep_top),
                ("island_reset_every", EVOLUTION_ISLAND_RESET_EVERY),
                ("timeout_per_call", int(llm_timeout_per_call)),
                ("experiments_root", str(resolve_experiments_root(experiments_root))),
                ("n_hard_typical", n_hard_typical),
                ("n_easy_typical", n_easy_typical),
                ("n_near_hard_easy", n_near_hard_easy),
                ("baseline_mode", baseline_mode),
                ("ensemble_base_models", ensemble_base_models),
                ("ensemble_meta_model", ensemble_meta_model),
                ("ensemble_cv_folds", ensemble_cv_folds),
                ("ensemble_random_state", ensemble_seed),
                ("tabtree_rf_n_estimators", tabtree_rf_n_estimators),
                ("tabtree_rf_max_depth", tabtree_rf_max_depth),
                ("tabtree_rf_min_samples_leaf", tabtree_rf_min_samples_leaf),
                ("tabtree_d_model", tabtree_d_model),
                ("tabtree_nhead", tabtree_nhead),
                ("tabtree_n_layers", tabtree_n_layers),
                ("tabtree_ff_dim", tabtree_ff_dim),
                ("tabtree_dropout", tabtree_dropout),
                ("tabtree_mlp_hidden", tabtree_mlp_hidden),
                ("tabtree_epochs", tabtree_epochs),
                ("tabtree_batch_size", tabtree_batch_size),
                ("tabtree_lr", tabtree_lr),
                ("tabtree_weight_decay", tabtree_weight_decay),
                ("hybrid_boundary_margin", hybrid_boundary_margin),
                ("hybrid_fn_margin", hybrid_fn_margin),
                ("hybrid_fp_margin", hybrid_fp_margin),
                ("evolve_boundary_only", evolve_boundary_only),
                ("rule_selection_split", rule_selection_split),
                ("rule_val_size", rule_val_size),
                ("evolution_score_mode", evolution_score_mode),
                ("evolution_harm_weight", float(evolution_harm_weight)),
                ("evolution_score_hard_guard", bool(evolution_score_hard_guard)),
            )
            for key, current in _soft_resume_keys:
                if key not in ckpt:
                    continue
                previous = ckpt[key]
                if previous != current:
                    print(
                        f"WARNING: {key}={current!r} differs from checkpoint value {previous!r}; "
                        "behavior may differ from the original run."
                    )

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nResuming evolution in: {exp_dir}")
        print(f"Loaded {len(loaded_rules)} rules; engine iteration continues from {last_iter}")
        print(f"This session id: {run_ts}")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_root = resolve_experiments_root(experiments_root)
        if flat_experiment_dir:
            exp_dir = exp_root / "by_mechanism" / mechanism_tag / f"exp_{run_ts}"
        else:
            slug = llm_model_slug(llm_model)
            exp_dir = exp_root / "by_model" / slug / "by_mechanism" / mechanism_tag / f"exp_{run_ts}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExperiment directory: {exp_dir}")

    data_path = pathlib.Path(data_path) if data_path else SVM_DATA_SRC
    data_path = data_path.resolve()
    print(f"Data: {data_path}")

    saved_model_path: pathlib.Path | None = None
    load_pretrained = evolve_only or resume_mode

    # ── Stage 1: Baseline model (SVM or Ensemble) ───────────────────────────
    baseline_name = (
        "SVM"
        if baseline_mode == "svm"
        else ("Ensemble" if baseline_mode == "ensemble" else "TabTreeFormer(RF)")
    )
    print(f"\n═══ Stage 1: {baseline_name} baseline ═══")

    df = load_raw_data(data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full_df = df.drop(columns=cols_to_drop + ["if_loaded"])
    y_full = df["if_loaded"].values
    indices = np.arange(len(df))
    train_idx, test_idx = tts(indices, test_size=test_size, random_state=random_state)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    if rule_selection_split == "val":
        inner_rel_train_idx, inner_rel_val_idx = tts(
            np.arange(len(train_idx)),
            test_size=rule_val_size,
            random_state=random_state,
        )
        svm_train_idx = train_idx[np.asarray(inner_rel_train_idx)]
        rule_eval_idx = train_idx[np.asarray(inner_rel_val_idx)]
        final_eval_idx = test_idx
    else:
        svm_train_idx = train_idx
        rule_eval_idx = test_idx
        final_eval_idx = test_idx

    X_train_df = X_full_df.iloc[svm_train_idx].reset_index(drop=True)
    y_train = y_full[svm_train_idx]

    X_rule_eval_df = X_full_df.iloc[rule_eval_idx].reset_index(drop=True)
    y_rule_eval = y_full[rule_eval_idx]
    dispatch_ids_rule_eval = (
        df["发车号"].values[rule_eval_idx] if "发车号" in df.columns else None
    )

    X_test_df = X_full_df.iloc[final_eval_idx].reset_index(drop=True)
    y_test = y_full[final_eval_idx]
    dispatch_ids_test = (
        df["发车号"].values[final_eval_idx] if "发车号" in df.columns else None
    )
    print(
        "Split protocol: "
        f"rule_search={rule_selection_split} ({len(y_rule_eval)} rows), "
        f"final_test=({len(y_test)} rows)"
    )

    pipeline = None
    ensemble_pipeline = None
    tabtree_pipeline = None
    baseline_decision_rule_eval: np.ndarray | None = None
    baseline_decision_test: np.ndarray | None = None
    baseline_probs_rule_eval: np.ndarray | None = None
    baseline_pred_rule_eval: np.ndarray | None = None
    baseline_meta: dict = {
        "mode": baseline_mode,
        "base_models": ["svm"] if baseline_mode == "svm" else (["rf"] if baseline_mode == "tabtreeformer_rf" else []),
        "meta_model": None,
        "xgb_fallback": False,
        "decision_threshold": 0.5,
        "calibration_info": {},
    }

    if baseline_mode == "svm":
        if not load_pretrained:
            pipeline = fit_svm_pipeline(
                X_train_df,
                y_train,
                random_state=random_state,
                C=10.0,
            )
            pipeline.drop_cols = cols_to_drop
            model_save_path = MODEL_DIR / f"linear_svm_{run_ts}.json"
            pipeline.save(model_save_path)
            saved_model_path = model_save_path
            print(f"Saved SVM model: {model_save_path}")
        else:
            st = (
                "\n═══ Stage 1: Loading pre-trained SVM (resume) ═══"
                if resume_mode
                else "\n═══ Stage 1: Loading pre-trained SVM ═══"
            )
            print(st)
            if resume_mode:
                if model_path is None:
                    g = guess_model_path_for_exp_dir(exp_dir)
                    if g is not None:
                        model_path = g
                    else:
                        existing = sorted(MODEL_DIR.glob("linear_svm_*.json"))
                        if not existing:
                            raise FileNotFoundError(
                                "Could not find SVM model for resume; pass --model-path or add linear_svm_<exp_suffix>.json"
                            )
                        model_path = existing[-1]
            elif model_path is None:
                existing = sorted(MODEL_DIR.glob("linear_svm_*.json"))
                if not existing:
                    raise FileNotFoundError("No SVM model found. Run without --evolve-only first.")
                model_path = existing[-1]

            model_path = pathlib.Path(model_path).resolve()
            saved_model_path = model_path
            pipeline = load_svm_pipeline(model_path)
            print(f"Loaded: {model_path}")

        baseline_decision_rule_eval = pipeline.decision_score(X_rule_eval_df)
        baseline_probs_rule_eval = _sigmoid(baseline_decision_rule_eval)
        baseline_pred_rule_eval = (baseline_probs_rule_eval >= 0.5).astype(int)
        baseline_decision_test = pipeline.decision_score(X_test_df)
        baseline_probs_test = _sigmoid(baseline_decision_test)
        baseline_pred_test = (baseline_probs_test >= 0.5).astype(int)
    elif baseline_mode == "ensemble":
        if not load_pretrained:
            ensemble_cfg = EnsembleConfig(
                base_models=[x.strip() for x in ensemble_base_models.split(",") if x.strip()],
                meta_model=ensemble_meta_model,
                cv_folds=ensemble_cv_folds,
                random_state=ensemble_seed,
            )
            ensemble_pipeline = fit_ensemble_pipeline(
                X_train_df=X_train_df,
                y_train=y_train,
                config=ensemble_cfg,
            )
            model_save_path = MODEL_DIR / f"ensemble_baseline_{run_ts}.json"
            ensemble_pipeline.save(model_save_path)
            saved_model_path = model_save_path
            print(f"Saved Ensemble model: {model_save_path}")
        else:
            st = (
                "\n═══ Stage 1: Loading pre-trained Ensemble (resume) ═══"
                if resume_mode
                else "\n═══ Stage 1: Loading pre-trained Ensemble ═══"
            )
            print(st)
            if resume_mode and model_path is None:
                existing = sorted(MODEL_DIR.glob("ensemble_baseline_*.json"))
                if not existing:
                    raise FileNotFoundError(
                        "Could not find ensemble model for resume; pass --model-path or train ensemble first."
                    )
                model_path = existing[-1]
            elif model_path is None:
                existing = sorted(MODEL_DIR.glob("ensemble_baseline_*.json"))
                if not existing:
                    raise FileNotFoundError(
                        "No ensemble model found. Run once with --baseline-mode ensemble first."
                    )
                model_path = existing[-1]

            model_path = pathlib.Path(model_path).resolve()
            saved_model_path = model_path
            ensemble_pipeline = load_ensemble_pipeline(model_path)
            print(f"Loaded: {model_path}")

        baseline_probs_test = ensemble_pipeline.predict_proba(X_test_df)
        baseline_pred_test = (baseline_probs_test >= 0.5).astype(int)
        baseline_meta = {
            "mode": "ensemble",
            "base_models": list(ensemble_pipeline.resolved_base_models),
            "meta_model": ensemble_pipeline.config.meta_model,
            "xgb_fallback": bool(ensemble_pipeline.xgb_fallback_used),
        }
        print(
            f"Ensemble base models: {baseline_meta['base_models']} | "
            f"meta={baseline_meta['meta_model']} | "
            f"xgb_fallback={baseline_meta['xgb_fallback']}"
        )
        baseline_probs_rule_eval = ensemble_pipeline.predict_proba(X_rule_eval_df)
        baseline_pred_rule_eval = (baseline_probs_rule_eval >= 0.5).astype(int)
    else:
        if not load_pretrained:
            tab_cfg = TabTreeFormerConfig(
                rf_n_estimators=tabtree_rf_n_estimators,
                rf_max_depth=tabtree_rf_max_depth,
                rf_min_samples_leaf=tabtree_rf_min_samples_leaf,
                d_model=tabtree_d_model,
                nhead=tabtree_nhead,
                n_layers=tabtree_n_layers,
                ff_dim=tabtree_ff_dim,
                dropout=tabtree_dropout,
                mlp_hidden=tabtree_mlp_hidden,
                epochs=tabtree_epochs,
                batch_size=tabtree_batch_size,
                lr=tabtree_lr,
                weight_decay=tabtree_weight_decay,
                random_state=ensemble_seed,
            )
            tabtree_pipeline = fit_tabtreeformer_pipeline(
                X_train_df=X_train_df,
                y_train=y_train,
                config=tab_cfg,
            )
            model_save_path = MODEL_DIR / f"tabtreeformer_rf_{run_ts}.json"
            tabtree_pipeline.save(model_save_path)
            saved_model_path = model_save_path
            print(f"Saved TabTreeFormer(RF) model: {model_save_path}")
        else:
            st = (
                "\n═══ Stage 1: Loading pre-trained TabTreeFormer(RF) (resume) ═══"
                if resume_mode
                else "\n═══ Stage 1: Loading pre-trained TabTreeFormer(RF) ═══"
            )
            print(st)
            if model_path is None:
                existing = sorted(MODEL_DIR.glob("tabtreeformer_rf_*.json"))
                if not existing:
                    raise FileNotFoundError(
                        "No TabTreeFormer(RF) model found. Run once with --baseline-mode tabtreeformer_rf first."
                    )
                model_path = existing[-1]

            model_path = pathlib.Path(model_path).resolve()
            saved_model_path = model_path
            tabtree_pipeline = load_tabtreeformer_pipeline(model_path)
            print(f"Loaded: {model_path}")

        baseline_probs_test = tabtree_pipeline.predict_proba(X_test_df)
        baseline_pred_test = (baseline_probs_test >= 0.5).astype(int)
        baseline_meta = {
            "mode": "tabtreeformer_rf",
            "base_models": ["rf"],
            "meta_model": "transformer_mlp",
            "xgb_fallback": False,
            "decision_threshold": 0.5,
            "calibrated_threshold": tabtree_pipeline.calibrated_threshold,
            "calibration_info": dict(tabtree_pipeline.calibration_info),
        }
        baseline_probs_rule_eval = tabtree_pipeline.predict_proba(X_rule_eval_df)
        baseline_pred_rule_eval = (baseline_probs_rule_eval >= 0.5).astype(int)

    # Keep legacy variable names for downstream compatibility.
    svm_probs_test = baseline_probs_test
    svm_pred_test = baseline_pred_test
    print_evaluation(y_test, svm_pred_test, svm_probs_test)

    rule_drop_cols = [c for c in ("if_loaded", "orderid", "发车号") if c in df.columns]
    X_rule_df = df.iloc[rule_eval_idx].drop(columns=rule_drop_cols).reset_index(drop=True)
    X_test_rule_df = df.iloc[final_eval_idx].drop(columns=rule_drop_cols).reset_index(drop=True)
    if "fill_ratio" not in X_rule_df.columns and {
        "total_skuvolume",
        "vehicle_capacity",
    }.issubset(X_rule_df.columns):
        X_rule_df["fill_ratio"] = (
            X_rule_df["total_skuvolume"] / X_rule_df["vehicle_capacity"]
        )
    if "fill_ratio" not in X_test_rule_df.columns and {
        "total_skuvolume",
        "vehicle_capacity",
    }.issubset(X_test_rule_df.columns):
        X_test_rule_df["fill_ratio"] = (
            X_test_rule_df["total_skuvolume"] / X_test_rule_df["vehicle_capacity"]
        )
    boundary_mask: np.ndarray | None = None
    boundary_ids: set[int] = set()
    if baseline_decision_rule_eval is not None:
        X_rule_df["svm_decision_score"] = np.asarray(baseline_decision_rule_eval, dtype=float)
        X_rule_df["svm_abs_decision_score"] = np.abs(np.asarray(baseline_decision_rule_eval, dtype=float))
    if boundary_gate_active:
        boundary_mask = boundary_candidate_mask(
            baseline_decision_rule_eval,
            boundary_margin=hybrid_boundary_margin,
            fn_margin=hybrid_fn_margin,
            fp_margin=hybrid_fp_margin,
        )
        X_rule_df["svm_near_boundary"] = boundary_mask.astype(int)
        id_array = (
            dispatch_ids_rule_eval
            if dispatch_ids_rule_eval is not None
            else np.arange(len(y_rule_eval))
        )
        boundary_ids = {int(id_array[i]) for i in np.flatnonzero(boundary_mask)}
        boundary_errors = int(np.sum(boundary_mask & (baseline_pred_rule_eval != y_rule_eval)))
        boundary_fn = int(
            np.sum(boundary_mask & (baseline_pred_rule_eval == 0) & (y_rule_eval == 1))
        )
        boundary_fp = int(
            np.sum(boundary_mask & (baseline_pred_rule_eval == 1) & (y_rule_eval == 0))
        )
        total_errors = int(np.sum(baseline_pred_rule_eval != y_rule_eval))
        print(
            "\nBoundary gate: "
            f"rows={int(boundary_mask.sum())}/{len(y_rule_eval)} "
            f"({boundary_mask.mean()*100:.1f}%), "
            f"errors covered={boundary_errors}/{total_errors}, "
            f"FN={boundary_fn}, FP={boundary_fp}"
        )

    # ── Stage 2: Hard case mining ───────────────────────────────────────────
    print("\n═══ Stage 2: Hard Case Mining ═══")
    fn_cases_all, fp_cases_all, easy_tn_all, easy_tp_all = mine_hard_cases(
        X_df=X_rule_df,
        y_true=y_rule_eval,
        svm_probs=baseline_probs_rule_eval,
        threshold=0.5,
        dispatch_ids=dispatch_ids_rule_eval,
        y_pred=baseline_pred_rule_eval,
    )

    fn_cases, fp_cases, easy_tn, easy_tp = (
        fn_cases_all,
        fp_cases_all,
        easy_tn_all,
        easy_tp_all,
    )
    if boundary_gate_active and evolve_boundary_only:
        fn_cases = [c for c in fn_cases_all if c.dispatch_id in boundary_ids]
        fp_cases = [c for c in fp_cases_all if c.dispatch_id in boundary_ids]
        easy_tn = [c for c in easy_tn_all if c.dispatch_id in boundary_ids]
        easy_tp = [c for c in easy_tp_all if c.dispatch_id in boundary_ids]
        print(
            "Boundary-only evolution: "
            f"FN {len(fn_cases)}/{len(fn_cases_all)}, "
            f"FP {len(fp_cases)}/{len(fp_cases_all)}, "
            f"Easy TN {len(easy_tn)}/{len(easy_tn_all)}, "
            f"Easy TP {len(easy_tp)}/{len(easy_tp_all)}"
        )

    print(f"FN cases: {len(fn_cases)} ({len(fn_cases)/sum(y_rule_eval)*100:.1f}% of GT=YES)")
    print(f"FP cases: {len(fp_cases)} ({len(fp_cases)/(len(y_rule_eval)-sum(y_rule_eval))*100:.1f}% of GT=NO)")
    print(f"Easy TN: {len(easy_tn)}, Easy TP: {len(easy_tp)}")

    summary = summarize_for_evolution_prompt(
        fn_cases=fn_cases,
        fp_cases=fp_cases,
        easy_tn=easy_tn,
        easy_tp=easy_tp,
        all_y_true=y_rule_eval,
        feature_names=list(X_rule_df.columns),
        n_typical_hard=n_hard_typical,
        n_easy_typical=n_easy_typical,
        n_near_hard_easy=n_near_hard_easy,
    )
    base_prob_caption = f"{baseline_name} probability"
    hard_case_md = render_hard_case_prompt(
        summary,
        predictor_name=baseline_name,
        prob_column_caption=base_prob_caption,
        instruction_style="explore" if evolution_score_mode == "net_gain_explore_v1" else "conservative",
    )
    if boundary_gate_active:
        total_errors = int(np.sum(baseline_pred_rule_eval != y_rule_eval))
        boundary_errors = int(np.sum(boundary_mask & (baseline_pred_rule_eval != y_rule_eval)))
        boundary_fn = int(
            np.sum(boundary_mask & (baseline_pred_rule_eval == 0) & (y_rule_eval == 1))
        )
        boundary_fp = int(
            np.sum(boundary_mask & (baseline_pred_rule_eval == 1) & (y_rule_eval == 0))
        )
        hard_case_md += (
            "\n\n### Boundary-gated Hybrid setup\n"
            f"- Gate active: only invoke heuristic rules near the SVM hyperplane.\n"
            f"- Symmetric margin: `{hybrid_boundary_margin}`; FN-side margin: `{hybrid_fn_margin}`; "
            f"FP-side margin: `{hybrid_fp_margin}`.\n"
            f"- Candidate rows: {int(boundary_mask.sum())} / {len(y_rule_eval)} "
            f"({boundary_mask.mean()*100:.1f}%).\n"
            f"- Baseline errors inside gate: {boundary_errors} / {total_errors}; "
            f"FN={boundary_fn}, FP={boundary_fp}.\n"
            f"- Evolution uses boundary-only cases: `{bool(evolve_boundary_only)}`.\n"
            "- Rule features include synthetic fields: `svm_decision_score`, "
            "`svm_abs_decision_score`, `svm_near_boundary`.\n"
        )

    # Save summary
    summary_path = exp_dir / "hard_case_summary.md"
    summary_path.write_text(hard_case_md, encoding="utf-8")
    print(f"Saved: {summary_path}")

    # Baseline hard-case masks aligned with the final evaluation split.
    fn_mask = np.zeros(len(y_test), dtype=bool)
    fp_mask = np.zeros(len(y_test), dtype=bool)
    id_array = dispatch_ids_test if dispatch_ids_test is not None else np.arange(len(y_test))
    fn_cases_test_all, fp_cases_test_all, easy_tn_test_all, easy_tp_test_all = mine_hard_cases(
        X_df=X_test_rule_df,
        y_true=y_test,
        svm_probs=svm_probs_test,
        threshold=0.5,
        dispatch_ids=dispatch_ids_test,
        y_pred=svm_pred_test,
    )
    for c in fn_cases_test_all:
        idx = np.where(id_array == c.dispatch_id)[0]
        if len(idx):
            fn_mask[idx[0]] = True
    for c in fp_cases_test_all:
        idx = np.where(id_array == c.dispatch_id)[0]
        if len(idx):
            fp_mask[idx[0]] = True

    baseline_eval_res = evaluate_predictor(
        baseline_name,
        y_test,
        svm_pred_test,
        svm_probs_test,
        fn_mask=fn_mask,
        fp_mask=fp_mask,
    )

    if skip_evolution:
        print(f"\nSkipping evolution. {baseline_name} baseline only.")
        baseline_only_effects = {
            "n_overrides": 0,
            "corrected_errors": 0,
            "harmed_correct": 0,
            "boundary_overrides": 0,
            "non_boundary_overrides": 0,
            "boundary_corrected_errors": 0,
            "boundary_harmed_correct": 0,
        }
        results = build_results_payload(
            run_ts=run_ts,
            baseline_mode=baseline_mode,
            baseline_name=baseline_name,
            baseline_meta=baseline_meta,
            llm_model=llm_model,
            llm_timeout_per_call=int(llm_timeout_per_call),
            n_evolution_iters=0,
            resume_mode=resume_mode,
            last_iter=last_iter,
            exp_dir=exp_dir,
            mechanism_tag=mechanism_tag,
            evolution_score_mode=evolution_score_mode,
            evolution_harm_weight=float(evolution_harm_weight),
            evolution_score_hard_guard=bool(evolution_score_hard_guard),
            n_hard_typical=n_hard_typical,
            n_easy_typical=n_easy_typical,
            n_near_hard_easy=n_near_hard_easy,
            rule_selection_split=rule_selection_split,
            rule_val_size=rule_val_size,
            y_rule_eval=y_rule_eval,
            y_test=y_test,
            boundary_gate_active=boundary_gate_active,
            hybrid_boundary_margin=hybrid_boundary_margin,
            hybrid_fn_margin=hybrid_fn_margin,
            hybrid_fp_margin=hybrid_fp_margin,
            evolve_boundary_only=evolve_boundary_only,
            boundary_mask=boundary_mask,
            baseline_pred_rule_eval=baseline_pred_rule_eval,
            baseline_eval_res=baseline_eval_res,
            hybrid_eval_res=baseline_eval_res,
            hybrid_effects=baseline_only_effects,
            fn_cases=fn_cases,
            fp_cases=fp_cases,
            easy_tn=easy_tn,
            easy_tp=easy_tp,
            fn_cases_all=fn_cases_test_all,
            fp_cases_all=fp_cases_test_all,
            easy_tn_all=easy_tn_test_all,
            easy_tp_all=easy_tp_test_all,
            fn_h=fn_cases_test_all,
            fp_h=fp_cases_test_all,
            best_rule_score=None,
            skip_evolution=True,
        )
        results_path = exp_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {results_path}")
        return results

    # ── Stage 3: FunSearch / ReEvo Evolution ────────────────────────────────
    print("\n═══ Stage 3: Evolutionary Rule Generation ═══")

    # Numeric scales for the same rows as evaluation (test split) — avoids LLM threshold mistakes
    feature_stats_md = format_feature_stats_for_prompt(X_rule_df)
    if baseline_mode == "svm" and pipeline is not None:
        svm_insights_md = format_svm_linear_insights_for_prompt(pipeline)
    else:
        svm_insights_md = (
            "_Baseline mode is not linear SVM. Linear SVM coefficient table is not applicable in this run._\n"
        )
    svm_insights_path = exp_dir / "svm_linear_insights.md"
    svm_insights_path.write_text(svm_insights_md, encoding="utf-8")
    print(f"Saved SVM linear insights for LLM: {svm_insights_path}")

    config = EvolutionConfig(
        n_iterations=n_evolution_iters,
        n_islands=4,
        max_island_size=50,
        island_reset_every=EVOLUTION_ISLAND_RESET_EVERY,
        n_seed_rules_per_prompt=2,
        llm_temperature=0.7,
        include_prev_iter_in_prompt=True,
        rule_placement=rule_placement,
        island_reset_mode=island_reset_mode,
        island_reset_keep_top=island_reset_keep_top,
        n_easy_typical=n_easy_typical,
        n_near_hard_easy=n_near_hard_easy,
        score_mode=evolution_score_mode,
        harm_weight=float(evolution_harm_weight),
        score_hard_guard=bool(evolution_score_hard_guard),
        timeout_per_call=int(llm_timeout_per_call),
    )
    engine = EvolutionEngine(config)

    if resume_mode and loaded_rules is not None:
        print(
            f"Restoring population from evolved_rules.json ({len(loaded_rules)} rules), "
            f"continuing for {n_evolution_iters} more iterations."
        )
        engine.population = population_from_flat_rules(
            loaded_rules,
            n_islands=config.n_islands,
            max_island_size=config.max_island_size,
            rule_placement=config.rule_placement,
            island_reset_mode=config.island_reset_mode,
            island_reset_keep_top=config.island_reset_keep_top,
        )
        engine.iteration = last_iter
        engine._seeded = True
        for isl in engine.population.islands:
            for rule in isl.rules:
                engine._fit_rule(rule, fn_cases, fp_cases, easy_tn, easy_tp)
    else:
        engine.seed_initial_rules([TRIVIAL_SEED, FILLED_SEED_1])
        for isl in engine.population.islands:
            for rule in isl.rules:
                engine._fit_rule(rule, fn_cases, fp_cases, easy_tn, easy_tp)

    llm_callable = make_llm_callable(
        llm_model,
        max_tokens=llm_max_tokens,
        timeout=config.timeout_per_call,
    )

    checkpoint_meta = {
        "version": 1,
        "data_path": str(data_path),
        "model_path": str(saved_model_path) if saved_model_path else "",
        "test_size": test_size,
        "random_state": random_state,
        "seed": seed,
        "n_islands": config.n_islands,
        "max_island_size": config.max_island_size,
        "llm_model": llm_model,
        "llm_timeout_per_call": int(config.timeout_per_call),
        "n_evolution_iters": n_evolution_iters,
        "llm_max_tokens": llm_max_tokens,
        "rule_placement": rule_placement,
        "island_reset_mode": island_reset_mode,
        "island_reset_keep_top": island_reset_keep_top,
        "island_reset_every": config.island_reset_every,
        "timeout_per_call": config.timeout_per_call,
        "experiments_root": str(resolve_experiments_root(experiments_root)),
        "n_easy_typical": n_easy_typical,
        "n_near_hard_easy": n_near_hard_easy,
        "n_hard_typical": n_hard_typical,
        "baseline_mode": baseline_mode,
        "ensemble_base_models": ensemble_base_models,
        "ensemble_meta_model": ensemble_meta_model,
        "ensemble_cv_folds": ensemble_cv_folds,
        "ensemble_random_state": ensemble_seed,
        "ensemble_resolved_base_models": baseline_meta.get("base_models", []),
        "xgb_fallback_used": bool(baseline_meta.get("xgb_fallback", False)),
        "tabtree_rf_n_estimators": tabtree_rf_n_estimators,
        "tabtree_rf_max_depth": tabtree_rf_max_depth,
        "tabtree_rf_min_samples_leaf": tabtree_rf_min_samples_leaf,
        "tabtree_d_model": tabtree_d_model,
        "tabtree_nhead": tabtree_nhead,
        "tabtree_n_layers": tabtree_n_layers,
        "tabtree_ff_dim": tabtree_ff_dim,
        "tabtree_dropout": tabtree_dropout,
        "tabtree_mlp_hidden": tabtree_mlp_hidden,
        "tabtree_epochs": tabtree_epochs,
        "tabtree_batch_size": tabtree_batch_size,
        "tabtree_lr": tabtree_lr,
        "tabtree_weight_decay": tabtree_weight_decay,
        "tabtree_decision_threshold": baseline_meta.get("decision_threshold", 0.5),
        "tabtree_calibrated_threshold": baseline_meta.get("calibrated_threshold"),
        "tabtree_calibration_info": baseline_meta.get("calibration_info", {}),
        "hybrid_boundary_margin": hybrid_boundary_margin,
        "hybrid_fn_margin": hybrid_fn_margin,
        "hybrid_fp_margin": hybrid_fp_margin,
        "evolve_boundary_only": evolve_boundary_only,
        "rule_selection_split": rule_selection_split,
        "rule_val_size": rule_val_size,
        "mechanism_tag": mechanism_tag,
        "evolution_score_mode": evolution_score_mode,
        "evolution_harm_weight": float(evolution_harm_weight),
        "evolution_score_hard_guard": bool(evolution_score_hard_guard),
    }

    best_rule = engine.run(
        llm_callable=llm_callable,
        fn_cases=fn_cases,
        fp_cases=fp_cases,
        easy_tn=easy_tn,
        easy_tp=easy_tp,
        hard_case_md=hard_case_md,
        n_iterations=n_evolution_iters,
        feature_stats_md=feature_stats_md,
        svm_insights_md=svm_insights_md,
        snapshot_dir=exp_dir,
        snapshot_ctx=(
            X_rule_df,
            y_rule_eval,
            baseline_probs_rule_eval,
            dispatch_ids_rule_eval,
            HYBRID_RULE_TOP_K,
            baseline_decision_rule_eval,
            hybrid_boundary_margin,
            hybrid_fn_margin,
            hybrid_fp_margin,
        ),
        warm_start=resume_mode,
        checkpoint_path=exp_dir / EVOLUTION_CHECKPOINT_NAME,
        checkpoint_meta=checkpoint_meta,
    )

    # Save all rules
    all_rules = [r for island in engine.population.islands for r in island.rules]
    rules_path = exp_dir / "evolved_rules.json"
    save_rules(all_rules, rules_path)
    print(f"Saved {len(all_rules)} rules to {rules_path}")

    if best_rule:
        best_code_path = exp_dir / "best_rule.py"
        best_code_path.write_text(best_rule.code, encoding="utf-8")
        print(f"\nBest rule (score={best_rule.score:.4f}):")
        print(best_code_path.read_text(encoding="utf-8")[:500])

    # ── Stage 4: Compare SVM vs Hybrid ──────────────────────────────────────
    print("\n═══ Stage 4: Evaluation ═══")

    boundary_mask_test: np.ndarray | None = None
    if boundary_gate_active:
        boundary_mask_test = boundary_candidate_mask(
            baseline_decision_test,
            boundary_margin=hybrid_boundary_margin,
            fn_margin=hybrid_fn_margin,
            fp_margin=hybrid_fp_margin,
        )
        if boundary_mask_test is not None:
            X_test_rule_df["svm_near_boundary"] = boundary_mask_test.astype(int)
    if baseline_decision_test is not None:
        X_test_rule_df["svm_decision_score"] = np.asarray(baseline_decision_test, dtype=float)
        X_test_rule_df["svm_abs_decision_score"] = np.abs(np.asarray(baseline_decision_test, dtype=float))

    # Get rules for hybrid prediction
    hybrid_pred_test = apply_rules(
        rules=all_rules,
        X_df=X_test_rule_df,
        svm_probs=svm_probs_test,
        top_k=HYBRID_RULE_TOP_K,
        decision_scores=baseline_decision_test,
        boundary_margin=hybrid_boundary_margin,
        fn_margin=hybrid_fn_margin,
        fp_margin=hybrid_fp_margin,
    )

    fn_h, fp_h, _, _ = mine_hard_cases(
        X_test_rule_df,
        y_test,
        svm_probs_test,
        dispatch_ids=dispatch_ids_test,
        y_pred=hybrid_pred_test,
    )
    summary_h = summarize_hard_cases(
        fn_h,
        fp_h,
        y_test,
        list(X_test_rule_df.columns),
    )
    md_hybrid = render_hard_case_prompt(
        summary_h,
        analysis_title=f"## Hybrid hard cases ({baseline_name} + rules, test split)",
        predictor_name=f"Hybrid ({baseline_name} + rules)",
        prob_column_caption=f"{baseline_name} P(y=1) (reference)",
        include_llm_instruction=True,
    )
    hybrid_summary_path = exp_dir / "hard_case_summary_hybrid.md"
    hybrid_summary_path.write_text(md_hybrid, encoding="utf-8")
    print(f"Saved hybrid hard-case summary: {hybrid_summary_path}")

    diff_mask = hybrid_pred_test != svm_pred_test
    base_error_mask = svm_pred_test != y_test
    hybrid_correct_mask = hybrid_pred_test == y_test
    if boundary_mask_test is None:
        effective_boundary_mask = np.ones(len(y_test), dtype=bool)
    else:
        effective_boundary_mask = boundary_mask_test
    hybrid_effects = {
        "n_overrides": int(diff_mask.sum()),
        "corrected_errors": int((diff_mask & base_error_mask & hybrid_correct_mask).sum()),
        "harmed_correct": int((diff_mask & ~base_error_mask & ~hybrid_correct_mask).sum()),
        "boundary_overrides": int((diff_mask & effective_boundary_mask).sum()),
        "non_boundary_overrides": int((diff_mask & ~effective_boundary_mask).sum()),
        "boundary_corrected_errors": int(
            (diff_mask & effective_boundary_mask & base_error_mask & hybrid_correct_mask).sum()
        ),
        "boundary_harmed_correct": int(
            (diff_mask & effective_boundary_mask & ~base_error_mask & ~hybrid_correct_mask).sum()
        ),
    }
    print(
        "Hybrid override summary: "
        f"overrides={hybrid_effects['n_overrides']}, "
        f"corrected={hybrid_effects['corrected_errors']}, "
        f"harmed={hybrid_effects['harmed_correct']}, "
        f"boundary_overrides={hybrid_effects['boundary_overrides']}"
    )

    svm_res, hybrid_res = compare_svm_vs_hybrid(
        y_true=y_test,
        svm_pred=svm_pred_test,
        svm_probs=svm_probs_test,
        hybrid_pred=hybrid_pred_test,
        fn_mask=fn_mask,
        fp_mask=fp_mask,
    )

    # Save results
    results = build_results_payload(
        run_ts=run_ts,
        baseline_mode=baseline_mode,
        baseline_name=baseline_name,
        baseline_meta=baseline_meta,
        llm_model=llm_model,
        llm_timeout_per_call=int(config.timeout_per_call),
        n_evolution_iters=n_evolution_iters,
        resume_mode=resume_mode,
        last_iter=last_iter,
        exp_dir=exp_dir,
        mechanism_tag=mechanism_tag,
        evolution_score_mode=evolution_score_mode,
        evolution_harm_weight=float(evolution_harm_weight),
        evolution_score_hard_guard=bool(evolution_score_hard_guard),
        n_hard_typical=n_hard_typical,
        n_easy_typical=n_easy_typical,
        n_near_hard_easy=n_near_hard_easy,
        rule_selection_split=rule_selection_split,
        rule_val_size=rule_val_size,
        y_rule_eval=y_rule_eval,
        y_test=y_test,
        boundary_gate_active=boundary_gate_active,
        hybrid_boundary_margin=hybrid_boundary_margin,
        hybrid_fn_margin=hybrid_fn_margin,
        hybrid_fp_margin=hybrid_fp_margin,
        evolve_boundary_only=evolve_boundary_only,
        boundary_mask=boundary_mask,
        baseline_pred_rule_eval=baseline_pred_rule_eval,
        baseline_eval_res=svm_res,
        hybrid_eval_res=hybrid_res,
        hybrid_effects=hybrid_effects,
        fn_cases=fn_cases,
        fp_cases=fp_cases,
        easy_tn=easy_tn,
        easy_tp=easy_tp,
        fn_cases_all=fn_cases_test_all,
        fp_cases_all=fp_cases_test_all,
        easy_tn_all=easy_tn_test_all,
        easy_tp_all=easy_tp_test_all,
        fn_h=fn_h,
        fp_h=fp_h,
        best_rule_score=float(best_rule.score) if best_rule else None,
        skip_evolution=False,
    )

    results_path = exp_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {results_path}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HybridSVM: Linear SVM + Evolved Rules")
    parser.add_argument("--data-path", type=pathlib.Path, default=None)
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--evolve-only", action="store_true")
    parser.add_argument("--n-evolution-iters", type=int, default=10)
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3-max-2026-01-23",
        help="Chat model id for the backend in use (e.g. Bailian when BAILIAN_API_KEY is set; OpenAI-compatible id). With DEEPSEEK_API_KEY, this is passed to DeepSeek (e.g. deepseek-chat). New runs go under <experiments-root>/by_model/<model>/ (default experiments_v2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--resume-exp-dir",
        type=pathlib.Path,
        default=None,
        help="Continue evolution in an existing experiment folder (uses evolved_rules.json).",
    )
    parser.add_argument(
        "--flat-exp-dir",
        action="store_true",
        help="Put new runs in <experiments-root>/exp_<time>/ instead of by_model/<llm>/exp_<time>/.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default=None,
        help="Root directory for new experiment outputs (default: experiments_v2 under HybridSVM). Use 'experiments' for legacy layout.",
    )
    parser.add_argument(
        "--n-hard-typical",
        type=int,
        default=3,
        help="Typical FN/FP rows per group in the evolution LLM prompt.",
    )
    parser.add_argument(
        "--n-easy-typical",
        type=int,
        default=3,
        help="Borderline easy TN/TP example rows in the evolution LLM prompt.",
    )
    parser.add_argument(
        "--n-near-hard-easy",
        type=int,
        default=3,
        help="Contrastive easy rows near FN/FP feature clouds (0 disables).",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=4096,
        help="Max completion tokens for LLM calls (all backends). Increase if rules are truncated mid-code fence.",
    )
    parser.add_argument(
        "--llm-timeout-per-call",
        type=int,
        default=180,
        help="Timeout in seconds per OpenAI-compatible LLM call (Bailian/OpenAI/DeepSeek).",
    )
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="svm",
        choices=("svm", "ensemble", "tabtreeformer_rf"),
        help="Stage-1 baseline predictor: classic linear SVM, stacked ensemble, or RF-backed TabTreeFormer.",
    )
    parser.add_argument(
        "--hybrid-boundary-margin",
        type=float,
        default=None,
        help="If set, run Hybrid rules only when abs(SVM decision score) is within this margin.",
    )
    parser.add_argument(
        "--hybrid-fn-margin",
        type=float,
        default=None,
        help="Optional negative-side margin for FN rescue: -margin <= decision < 0.",
    )
    parser.add_argument(
        "--hybrid-fp-margin",
        type=float,
        default=None,
        help="Optional positive-side margin for FP veto: 0 <= decision <= margin.",
    )
    parser.add_argument(
        "--evolve-boundary-only",
        action="store_true",
        help="When a boundary gate is configured, show/evaluate only boundary candidates in the LLM evolution loop.",
    )
    parser.add_argument(
        "--rule-selection-split",
        type=str,
        default="test",
        choices=("test", "val"),
        help="Where to search/select rules: 'test' keeps the legacy shared held-out split; 'val' uses an inner validation split and reserves the outer test split for final evaluation.",
    )
    parser.add_argument(
        "--rule-val-size",
        type=float,
        default=0.2,
        help="When --rule-selection-split val, fraction of the outer-train portion used for rule search/selection.",
    )
    parser.add_argument(
        "--evolution-score-mode",
        type=str,
        default="net_gain_v1",
        choices=("net_gain_v1", "net_gain_explore_v1", "legacy_f1"),
        help="Rule fitness objective. net_gain_v1 (default) optimizes corrected-harmed net gain; net_gain_explore_v1 adds mild correction-first exploration pressure; legacy_f1 keeps older FN/FP F1-style scoring.",
    )
    parser.add_argument(
        "--evolution-harm-weight",
        type=float,
        default=3.0,
        help="Weight on harmed easy-correct rows under net_gain_v1 scoring.",
    )
    parser.add_argument(
        "--evolution-score-hard-guard",
        action="store_true",
        default=True,
        help="Enable hard guard under net_gain_v1: candidates with harmed_correct >= corrected_errors are clipped to non-positive score.",
    )
    parser.add_argument(
        "--no-evolution-score-hard-guard",
        dest="evolution_score_hard_guard",
        action="store_false",
        help="Disable the net_gain_v1 hard guard.",
    )
    parser.add_argument(
        "--ensemble-base-models",
        type=str,
        default="svm,lr,rf,xgb",
        help="Comma-separated base learners for ensemble mode (subset/order of: svm,lr,rf,xgb).",
    )
    parser.add_argument(
        "--ensemble-meta-model",
        type=str,
        default="logreg",
        help="Combiner for ensemble mode (default: logreg; also supports: mlp).",
    )
    parser.add_argument(
        "--ensemble-cv-folds",
        type=int,
        default=5,
        help="OOF CV folds for ensemble stacking.",
    )
    parser.add_argument(
        "--ensemble-random-state",
        type=int,
        default=None,
        help="Random state for ensemble base/meta training (default: use --test-size split random_state).",
    )
    parser.add_argument("--tabtree-rf-n-estimators", type=int, default=128)
    parser.add_argument("--tabtree-rf-max-depth", type=int, default=12)
    parser.add_argument("--tabtree-rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--tabtree-d-model", type=int, default=96)
    parser.add_argument("--tabtree-nhead", type=int, default=4)
    parser.add_argument("--tabtree-n-layers", type=int, default=2)
    parser.add_argument("--tabtree-ff-dim", type=int, default=192)
    parser.add_argument("--tabtree-dropout", type=float, default=0.1)
    parser.add_argument("--tabtree-mlp-hidden", type=int, default=64)
    parser.add_argument("--tabtree-epochs", type=int, default=12)
    parser.add_argument("--tabtree-batch-size", type=int, default=256)
    parser.add_argument("--tabtree-lr", type=float, default=1e-3)
    parser.add_argument("--tabtree-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--rule-placement",
        type=str,
        default="round_robin",
        choices=("round_robin", "min_best_score", "min_rule_count"),
        help="How to assign new rules to islands (default: round_robin avoids piling rules on the weakest island).",
    )
    parser.add_argument(
        "--island-reset-mode",
        type=str,
        default="trim_top_k",
        choices=("trim_top_k", "clear_and_clone"),
        help="Worst-island reset: trim_top_k keeps top rules before re-seeding; clear_and_clone empties the island.",
    )
    parser.add_argument(
        "--island-reset-keep-top",
        type=int,
        default=3,
        help="With trim_top_k, how many best rules to keep on the worst island before cloning the global best.",
    )

    args = parser.parse_args()
    results = run_experiment(
        data_path=args.data_path,
        model_path=args.model_path,
        skip_evolution=args.skip_evolution,
        evolve_only=args.evolve_only,
        n_evolution_iters=args.n_evolution_iters,
        llm_model=args.llm_model,
        seed=args.seed,
        test_size=args.test_size,
        resume_exp_dir=args.resume_exp_dir,
        flat_experiment_dir=args.flat_exp_dir,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout_per_call=args.llm_timeout_per_call,
        rule_placement=args.rule_placement,
        island_reset_mode=args.island_reset_mode,
        island_reset_keep_top=args.island_reset_keep_top,
        experiments_root=args.experiments_root,
        n_hard_typical=args.n_hard_typical,
        n_easy_typical=args.n_easy_typical,
        n_near_hard_easy=args.n_near_hard_easy,
        baseline_mode=args.baseline_mode,
        ensemble_base_models=args.ensemble_base_models,
        ensemble_meta_model=args.ensemble_meta_model,
        ensemble_cv_folds=args.ensemble_cv_folds,
        ensemble_random_state=args.ensemble_random_state,
        tabtree_rf_n_estimators=args.tabtree_rf_n_estimators,
        tabtree_rf_max_depth=args.tabtree_rf_max_depth,
        tabtree_rf_min_samples_leaf=args.tabtree_rf_min_samples_leaf,
        tabtree_d_model=args.tabtree_d_model,
        tabtree_nhead=args.tabtree_nhead,
        tabtree_n_layers=args.tabtree_n_layers,
        tabtree_ff_dim=args.tabtree_ff_dim,
        tabtree_dropout=args.tabtree_dropout,
        tabtree_mlp_hidden=args.tabtree_mlp_hidden,
        tabtree_epochs=args.tabtree_epochs,
        tabtree_batch_size=args.tabtree_batch_size,
        tabtree_lr=args.tabtree_lr,
        tabtree_weight_decay=args.tabtree_weight_decay,
        hybrid_boundary_margin=args.hybrid_boundary_margin,
        hybrid_fn_margin=args.hybrid_fn_margin,
        hybrid_fp_margin=args.hybrid_fp_margin,
        evolve_boundary_only=args.evolve_boundary_only,
        rule_selection_split=args.rule_selection_split,
        rule_val_size=args.rule_val_size,
        evolution_score_mode=args.evolution_score_mode,
        evolution_harm_weight=args.evolution_harm_weight,
        evolution_score_hard_guard=args.evolution_score_hard_guard,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()

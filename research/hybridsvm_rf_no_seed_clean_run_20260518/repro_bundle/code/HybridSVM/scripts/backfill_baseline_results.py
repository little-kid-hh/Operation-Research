#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill missing results.json for baseline-only experiment folders.

This is intended for historical runs where `--skip-evolution` wrote
`hard_case_summary.md` but returned before saving `results.json`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split as tts

from src.evaluate import evaluate_predictor
from src.hard_cases import mine_hard_cases
from src.svm_train import DROP_COLS, load_raw_data, load_svm_pipeline
from src.ensemble_train import EnsembleConfig, fit_ensemble_pipeline, load_ensemble_pipeline
from TabTreeFormer.tabtreeformer import (
    TabTreeFormerConfig,
    fit_tabtreeformer_pipeline,
    load_tabtreeformer_pipeline,
)
from run_experiment import (
    SVM_DATA_SRC,
    build_results_payload,
    evolution_mechanism_tag,
)


def _timestamp_from_name(path: pathlib.Path) -> str:
    m = re.search(r"exp_(\d{8}_\d{6})$", path.name)
    if not m:
        raise ValueError(f"Cannot infer timestamp from {path}")
    return m.group(1)


def _summary_counts(summary_path: pathlib.Path) -> dict[str, int] | None:
    if not summary_path.is_file():
        return None
    text = summary_path.read_text(encoding="utf-8")
    matches = re.findall(r"- Count: (\d+) / (\d+) =", text)
    if len(matches) < 4:
        return None
    fn_count = int(matches[0][0])
    fp_count = int(matches[1][0])
    easy_tn = int(matches[2][0])
    easy_tp = int(matches[3][0])
    return {
        "fn": fn_count,
        "fp": fp_count,
        "easy_tn": easy_tn,
        "easy_tp": easy_tp,
    }


def _infer_predictor_from_summary(summary_path: pathlib.Path) -> tuple[str, str]:
    text = summary_path.read_text(encoding="utf-8")
    m = re.search(r"### 1\. FN Cases \((.+?) predicts ", text)
    if not m:
        raise ValueError(f"Cannot infer predictor from {summary_path}")
    predictor = m.group(1).strip()
    if predictor == "SVM":
        return "svm", "SVM"
    if predictor == "Ensemble":
        return "ensemble", "Ensemble"
    if predictor == "TabTreeFormer(RF)":
        return "tabtreeformer_rf", "TabTreeFormer(RF)"
    raise ValueError(f"Unsupported predictor in {summary_path}: {predictor}")


def _infer_model_path(mode: str, timestamp: str) -> pathlib.Path:
    prefix = {
        "svm": "linear_svm",
        "ensemble": "ensemble_baseline",
        "tabtreeformer_rf": "tabtreeformer_rf",
    }[mode]
    path = ROOT / "models" / f"{prefix}_{timestamp}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return path


def _infer_meta_model(exp_dir: pathlib.Path, model_path: pathlib.Path) -> str | None:
    if "ensemble_eval_logreg" in str(exp_dir):
        return "logreg"
    if "ensemble_eval_mlp" in str(exp_dir):
        return "mlp"
    if model_path.is_file():
        meta = json.loads(model_path.read_text(encoding="utf-8"))
        return meta.get("meta_model")
    return None


def _load_pipeline(mode: str, model_path: pathlib.Path):
    if mode == "svm":
        return load_svm_pipeline(model_path)
    if mode == "ensemble":
        return load_ensemble_pipeline(model_path)
    if mode == "tabtreeformer_rf":
        return load_tabtreeformer_pipeline(model_path)
    raise ValueError(f"Unsupported mode: {mode}")


def _baseline_meta(mode: str, pipeline, meta_model_hint: str | None) -> dict:
    if mode == "svm":
        return {
            "mode": "svm",
            "base_models": ["svm"],
            "meta_model": None,
            "xgb_fallback": False,
            "decision_threshold": 0.5,
            "calibration_info": {},
        }
    if mode == "ensemble":
        return {
            "mode": "ensemble",
            "base_models": list(pipeline.resolved_base_models),
            "meta_model": meta_model_hint or pipeline.config.meta_model,
            "xgb_fallback": bool(pipeline.xgb_fallback_used),
            "decision_threshold": 0.5,
            "calibration_info": {},
        }
    return {
        "mode": "tabtreeformer_rf",
        "base_models": ["rf"],
        "meta_model": "transformer_mlp",
        "xgb_fallback": False,
        "decision_threshold": 0.5,
        "calibrated_threshold": pipeline.calibrated_threshold,
        "calibration_info": dict(pipeline.calibration_info),
    }


def _collect_eval_context(data_path: pathlib.Path, test_size: float, random_state: int) -> dict:
    df = load_raw_data(data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_full_df = df.drop(columns=cols_to_drop + ["if_loaded"])
    y_full = df["if_loaded"].values
    indices = np.arange(len(df))
    train_idx, test_idx = tts(indices, test_size=test_size, random_state=random_state)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    X_train_df = X_full_df.iloc[train_idx].reset_index(drop=True)
    y_train = y_full[train_idx]
    X_test_df = X_full_df.iloc[test_idx].reset_index(drop=True)
    y_test = y_full[test_idx]
    dispatch_ids_test = df["发车号"].values[test_idx] if "发车号" in df.columns else None
    rule_drop_cols = [c for c in ("if_loaded", "orderid", "发车号") if c in df.columns]
    X_test_rule_df = df.iloc[test_idx].drop(columns=rule_drop_cols).reset_index(drop=True)
    if "fill_ratio" not in X_test_rule_df.columns and {
        "total_skuvolume",
        "vehicle_capacity",
    }.issubset(X_test_rule_df.columns):
        X_test_rule_df["fill_ratio"] = (
            X_test_rule_df["total_skuvolume"] / X_test_rule_df["vehicle_capacity"]
        )
    return {
        "df": df,
        "X_train_df": X_train_df,
        "y_train": y_train,
        "X_test_df": X_test_df,
        "y_test": y_test,
        "dispatch_ids_test": dispatch_ids_test,
        "X_test_rule_df": X_test_rule_df,
    }


def _rebuild_pipeline_from_model_info(mode: str, model_path: pathlib.Path, ctx: dict):
    meta = json.loads(model_path.read_text(encoding="utf-8"))
    X_train_df = ctx["X_train_df"]
    y_train = ctx["y_train"]
    if mode == "ensemble":
        cfg = EnsembleConfig(
            base_models=list(meta.get("base_models_requested", ["svm", "lr", "rf", "xgb"])),
            meta_model=str(meta.get("meta_model", "logreg")),
            cv_folds=int(meta.get("cv_folds", 5)),
            random_state=int(meta.get("random_state", 42)),
        )
        return fit_ensemble_pipeline(X_train_df=X_train_df, y_train=y_train, config=cfg)
    if mode == "tabtreeformer_rf":
        cfg_data = dict(meta.get("config", {}))
        valid = set(TabTreeFormerConfig.__dataclass_fields__)
        cfg = TabTreeFormerConfig(**{k: v for k, v in cfg_data.items() if k in valid})
        return fit_tabtreeformer_pipeline(X_train_df=X_train_df, y_train=y_train, config=cfg)
    raise ValueError(f"Rebuild not supported for mode={mode}")


def backfill_one(exp_dir: pathlib.Path, *, force: bool = False) -> tuple[bool, str]:
    exp_dir = exp_dir.resolve()
    results_path = exp_dir / "results.json"
    if results_path.exists() and not force:
        return False, f"skip existing {results_path}"

    timestamp = _timestamp_from_name(exp_dir)
    summary_path = exp_dir / "hard_case_summary.md"
    mode, baseline_name = _infer_predictor_from_summary(summary_path)
    model_path = _infer_model_path(mode, timestamp)
    meta_model_hint = _infer_meta_model(exp_dir, model_path)
    ctx = _collect_eval_context(SVM_DATA_SRC.resolve(), test_size=0.25, random_state=42)
    try:
        pipeline = _load_pipeline(mode, model_path)
        rebuild_note = None
    except Exception:
        if mode not in {"ensemble", "tabtreeformer_rf"}:
            raise
        pipeline = _rebuild_pipeline_from_model_info(mode, model_path, ctx)
        rebuild_note = "Primary binary artifact was missing; pipeline was deterministically rebuilt from saved model config."
    baseline_meta = _baseline_meta(mode, pipeline, meta_model_hint)

    X_test_df = ctx["X_test_df"]
    y_test = ctx["y_test"]
    dispatch_ids_test = ctx["dispatch_ids_test"]
    X_test_rule_df = ctx["X_test_rule_df"]
    if mode == "svm":
        decision = pipeline.decision_score(X_test_df)
        probs = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(decision, dtype=float), -500, 500)))
        pred = (probs >= 0.5).astype(int)
    else:
        probs = pipeline.predict_proba(X_test_df)
        pred = (probs >= 0.5).astype(int)

    fn_cases, fp_cases, easy_tn, easy_tp = mine_hard_cases(
        X_df=X_test_rule_df,
        y_true=y_test,
        svm_probs=probs,
        threshold=0.5,
        dispatch_ids=dispatch_ids_test,
        y_pred=pred,
    )

    summary_counts = _summary_counts(exp_dir / "hard_case_summary.md")
    if summary_counts is not None:
        observed = {
            "fn": len(fn_cases),
            "fp": len(fp_cases),
            "easy_tn": len(easy_tn),
            "easy_tp": len(easy_tp),
        }
        if observed != summary_counts:
            raise ValueError(
                f"Summary mismatch for {exp_dir.name}: observed={observed}, summary={summary_counts}"
            )

    id_array = dispatch_ids_test if dispatch_ids_test is not None else np.arange(len(y_test))
    fn_mask = np.zeros(len(y_test), dtype=bool)
    fp_mask = np.zeros(len(y_test), dtype=bool)
    for c in fn_cases:
        idx = np.where(id_array == c.dispatch_id)[0]
        if len(idx):
            fn_mask[idx[0]] = True
    for c in fp_cases:
        idx = np.where(id_array == c.dispatch_id)[0]
        if len(idx):
            fp_mask[idx[0]] = True

    eval_res = evaluate_predictor(
        baseline_name,
        y_test,
        pred,
        probs,
        fn_mask=fn_mask,
        fp_mask=fp_mask,
    )
    mechanism_tag = evolution_mechanism_tag(
        "net_gain_v1",
        harm_weight=3.0,
        score_hard_guard=True,
    )
    hybrid_effects = {
        "n_overrides": 0,
        "corrected_errors": 0,
        "harmed_correct": 0,
        "boundary_overrides": 0,
        "non_boundary_overrides": 0,
        "boundary_corrected_errors": 0,
        "boundary_harmed_correct": 0,
    }
    results = build_results_payload(
        run_ts=timestamp,
        baseline_mode=mode,
        baseline_name=baseline_name,
        baseline_meta=baseline_meta,
        llm_model="qwen3-max-2026-01-23",
        llm_timeout_per_call=180,
        n_evolution_iters=0,
        resume_mode=False,
        last_iter=0,
        exp_dir=exp_dir,
        mechanism_tag=mechanism_tag,
        evolution_score_mode="net_gain_v1",
        evolution_harm_weight=3.0,
        evolution_score_hard_guard=True,
        n_hard_typical=3,
        n_easy_typical=3,
        n_near_hard_easy=3,
        rule_selection_split="test",
        rule_val_size=0.2,
        y_rule_eval=y_test,
        y_test=y_test,
        boundary_gate_active=False,
        hybrid_boundary_margin=None,
        hybrid_fn_margin=None,
        hybrid_fp_margin=None,
        evolve_boundary_only=False,
        boundary_mask=None,
        baseline_pred_rule_eval=pred,
        baseline_eval_res=eval_res,
        hybrid_eval_res=eval_res,
        hybrid_effects=hybrid_effects,
        fn_cases=fn_cases,
        fp_cases=fp_cases,
        easy_tn=easy_tn,
        easy_tp=easy_tp,
        fn_cases_all=fn_cases,
        fp_cases_all=fp_cases,
        easy_tn_all=easy_tn,
        easy_tp_all=easy_tp,
        fn_h=fn_cases,
        fp_h=fp_cases,
        best_rule_score=None,
        skip_evolution=True,
    )
    results["backfilled_by"] = "scripts/backfill_baseline_results.py"
    results["model_path"] = str(model_path)
    results["notes"] = [
        "Backfilled from saved baseline model artifact and current deterministic evaluation pipeline.",
        "This is a baseline-only record; no evolution or Hybrid override was run in this directory.",
    ]
    if rebuild_note:
        results["notes"].append(rebuild_note)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, f"wrote {results_path}"


def discover_targets(root: pathlib.Path) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in sorted(root.rglob("hard_case_summary.md")):
        exp_dir = p.parent
        if not exp_dir.name.startswith("exp_"):
            continue
        try:
            _infer_predictor_from_summary(p)
        except ValueError:
            continue
        out.append(exp_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing baseline-only results.json files.")
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=ROOT,
        help="Search root for historical experiment folders (default: HybridSVM root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results.json if present.",
    )
    parser.add_argument(
        "--exp-dir",
        type=pathlib.Path,
        default=None,
        help="Backfill one specific experiment directory instead of scanning.",
    )
    args = parser.parse_args()

    targets = [args.exp_dir.resolve()] if args.exp_dir else discover_targets(args.root.resolve())
    changed = 0
    skipped = 0
    for exp_dir in targets:
        try:
            did_write, msg = backfill_one(exp_dir, force=args.force)
            print(msg)
            if did_write:
                changed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"ERROR {exp_dir}: {e}", file=sys.stderr)
    print(f"\nSummary: wrote={changed}, skipped={skipped}, total={len(targets)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Boundary-gated heuristic-agent harness for HybridSVM.

Purpose:
  1. Run the SVM first.
  2. Select only samples near the signed SVM decision boundary.
  3. Build a focused LLM prompt with boundary FN/FP and contrastive correct rows.
  4. Call the configured API for heuristic rule code.
  5. Extract, evaluate, and save candidate `apply_rule_patch` functions.

This is an exploratory harness. If the same boundary cases are used in the
prompt and in evaluation, the metrics are diagnostic, not a final holdout claim.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HYBRID_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = HYBRID_ROOT.parent
for p in (HYBRID_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_experiment import make_llm_callable, _read_config_value  # noqa: E402
from src.evolution import (  # noqa: E402
    Rule,
    apply_rules,
    boundary_candidate_mask,
    extract_code_from_response,
    rule_sort_key,
    save_rules,
)
from src.hard_cases import (  # noqa: E402
    format_feature_stats_for_prompt,
    mine_hard_cases,
    render_hard_case_prompt,
    summarize_for_evolution_prompt,
)
from src.svm_train import (  # noqa: E402
    DROP_COLS,
    format_svm_linear_insights_for_prompt,
    load_raw_data,
    load_svm_pipeline,
)


DEFAULT_DATA = PROJECT_ROOT / "FunSearch_test" / "training_2orientations.csv"
DEFAULT_OUT_ROOT = HYBRID_ROOT / "experiments_boundary_agent"
MODEL_DIR = HYBRID_ROOT / "models"


@dataclass
class CandidateReport:
    id: str
    score: float
    fn_corr: int
    fp_corr: int
    fn_miss: int
    fp_over: int
    easy_overrides: int
    accuracy: float
    precision: float
    recall: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    n_overrides: int
    corrected_errors: int
    harmed_correct: int
    boundary_overrides: int
    non_boundary_overrides: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _latest_svm_model() -> pathlib.Path:
    models = sorted(MODEL_DIR.glob("linear_svm_*.json"))
    if not models:
        raise FileNotFoundError("No linear_svm_*.json model found; pass --model-path.")
    return models[-1]


def _has_api_key() -> bool:
    return bool(
        _read_config_value("LLM_API_KEY")
        or _read_config_value("BAILIAN_API_KEY")
        or _read_config_value("OPENAI_API_KEY")
        or _read_config_value("DEEPSEEK_API_KEY")
        or _read_config_value("DASHSCOPE_API_KEY")
    )


def _effective_llm_model(cli_model: str) -> str:
    return (
        _read_config_value(
            "LLM_MODEL",
            "BAILIAN_MODEL",
            "OPENAI_MODEL",
            "DEEPSEEK_MODEL",
            "DASHSCOPE_MODEL",
        )
        or cli_model
    )


def _rule_feature_frame(df: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    drop_cols = [c for c in ("if_loaded", "orderid", "发车号") if c in df.columns]
    out = df.iloc[test_idx].drop(columns=drop_cols).reset_index(drop=True)
    if "fill_ratio" not in out.columns and {"total_skuvolume", "vehicle_capacity"}.issubset(out.columns):
        out["fill_ratio"] = out["total_skuvolume"] / out["vehicle_capacity"]
    return out


def _prepare_split(
    data_path: pathlib.Path,
    model_path: pathlib.Path,
    test_size: float,
    random_state: int,
    fn_margin: float | None,
    fp_margin: float | None,
    boundary_margin: float | None,
) -> dict:
    df = load_raw_data(data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X_svm = df.drop(columns=cols_to_drop + ["if_loaded"])
    y = df["if_loaded"].to_numpy()
    indices = np.arange(len(df))
    _, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    X_test_svm = X_svm.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    dispatch_ids = df["发车号"].values[test_idx] if "发车号" in df.columns else test_idx

    pipeline = load_svm_pipeline(model_path)
    decision = pipeline.decision_score(X_test_svm)
    probs = _sigmoid(decision)
    pred = (probs >= 0.5).astype(int)
    boundary_mask = boundary_candidate_mask(
        decision,
        boundary_margin=boundary_margin,
        fn_margin=fn_margin,
        fp_margin=fp_margin,
    )
    if boundary_mask is None:
        boundary_mask = np.ones(len(y_test), dtype=bool)

    X_rule = _rule_feature_frame(df, test_idx)
    X_rule["svm_decision_score"] = decision.astype(float)
    X_rule["svm_abs_decision_score"] = np.abs(decision.astype(float))
    X_rule["svm_near_boundary"] = boundary_mask.astype(int)

    fn_all, fp_all, tn_all, tp_all = mine_hard_cases(
        X_rule,
        y_test,
        probs,
        dispatch_ids=dispatch_ids,
        y_pred=pred,
    )
    boundary_ids = {int(dispatch_ids[i]) for i in np.flatnonzero(boundary_mask)}
    fn_cases = [c for c in fn_all if c.dispatch_id in boundary_ids]
    fp_cases = [c for c in fp_all if c.dispatch_id in boundary_ids]
    easy_tn = [c for c in tn_all if c.dispatch_id in boundary_ids]
    easy_tp = [c for c in tp_all if c.dispatch_id in boundary_ids]

    return {
        "pipeline": pipeline,
        "X_rule": X_rule,
        "X_test_svm": X_test_svm,
        "y_test": y_test,
        "dispatch_ids": dispatch_ids,
        "decision": decision,
        "probs": probs,
        "pred": pred,
        "boundary_mask": boundary_mask,
        "fn_all": fn_all,
        "fp_all": fp_all,
        "easy_tn_all": tn_all,
        "easy_tp_all": tp_all,
        "fn_cases": fn_cases,
        "fp_cases": fp_cases,
        "easy_tn": easy_tn,
        "easy_tp": easy_tp,
    }


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (
        len(a) + len(b) - 2
    )
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def _contrast_table(
    X: pd.DataFrame,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_name: str,
    right_name: str,
    top_k: int = 12,
) -> str:
    num = X.select_dtypes(include=[np.number])
    rows = []
    left = num.loc[left_mask]
    right = num.loc[right_mask]
    for col in num.columns:
        d = _cohens_d(left[col], right[col])
        rows.append(
            (
                abs(d),
                d,
                col,
                float(left[col].median()) if len(left) else float("nan"),
                float(right[col].median()) if len(right) else float("nan"),
            )
        )
    rows.sort(reverse=True)
    lines = [
        f"### {left_name} vs {right_name}",
        f"- n({left_name})={len(left)}, n({right_name})={len(right)}",
        "| feature | Cohen d | median left | median right |",
        "|---|---:|---:|---:|",
    ]
    for _, d, col, lm, rm in rows[:top_k]:
        lines.append(f"| `{col}` | {d:.3f} | {lm:.6g} | {rm:.6g} |")
    return "\n".join(lines)


def _build_prompt(ctx: dict, fn_margin: float | None, fp_margin: float | None, boundary_margin: float | None) -> str:
    X = ctx["X_rule"]
    y = ctx["y_test"]
    pred = ctx["pred"]
    boundary = ctx["boundary_mask"]
    errors = pred != y

    summary = summarize_for_evolution_prompt(
        fn_cases=ctx["fn_cases"],
        fp_cases=ctx["fp_cases"],
        easy_tn=ctx["easy_tn"],
        easy_tp=ctx["easy_tp"],
        all_y_true=y,
        feature_names=list(X.columns),
        n_typical_hard=5,
        n_easy_typical=5,
        n_near_hard_easy=5,
    )
    hard_case_md = render_hard_case_prompt(
        summary,
        analysis_title="## Boundary-Gated SVM Hard Case Analysis",
        predictor_name="SVM",
        prob_column_caption="SVM probability",
        include_llm_instruction=False,
    )

    fn_mask = boundary & (pred == 0) & (y == 1)
    fp_mask = boundary & (pred == 1) & (y == 0)
    tp_mask = boundary & (pred == 1) & (y == 1)
    tn_mask = boundary & (pred == 0) & (y == 0)

    contrasts = "\n\n".join(
        [
            _contrast_table(X, fn_mask, tp_mask, "Boundary FN", "Boundary TP"),
            _contrast_table(X, fp_mask, tn_mask, "Boundary FP", "Boundary TN"),
        ]
    )
    feature_stats = format_feature_stats_for_prompt(X.loc[boundary])
    svm_insights = format_svm_linear_insights_for_prompt(ctx["pipeline"])

    lines = [
        "# Boundary-Gated HybridSVM Heuristic Agent",
        "",
        "You are designing a narrow Python heuristic to run AFTER a linear SVM.",
        "Only rows selected by the signed-margin gate will call your rule.",
        "",
        "## Gate",
        f"- Symmetric margin: `{boundary_margin}`",
        f"- FN-side margin: `{fn_margin}` means `-margin <= svm_decision_score < 0`.",
        f"- FP-side margin: `{fp_margin}` means `0 <= svm_decision_score <= margin`.",
        f"- Candidate rows: {int(boundary.sum())} / {len(y)} ({boundary.mean()*100:.1f}%).",
        f"- Errors inside gate: {int((boundary & errors).sum())} / {int(errors.sum())}.",
        f"- FN inside gate: {int(fn_mask.sum())}; FP inside gate: {int(fp_mask.sum())}.",
        "",
        "## Rule interface",
        "Return `-1` to trust SVM, `0` to veto a positive SVM prediction, or `1` to rescue a negative SVM prediction.",
        "Use only deterministic Python and numeric fields in `features`.",
        "`svm_decision_score` is signed: negative means SVM predicts infeasible, positive means feasible.",
        "",
        "## Feature scales inside the gate",
        feature_stats,
        "",
        "## Linear SVM weights",
        svm_insights,
        "",
        hard_case_md,
        "",
        "## Contrastive feature differences inside the gate",
        contrasts,
        "",
        "## Output contract",
        "Return ONLY one Python code block. It must define:",
        "",
        "```python",
        "def apply_rule_patch(svm_prob: float, features: dict) -> int:",
        "    ...",
        "```",
        "",
        "Engineering constraints:",
        "- Optimize net Accuracy, not just hard-case correction count.",
        "- A candidate is bad if `harmed_correct >= corrected_errors`; abstain aggressively with `return -1`.",
        "- Prefer 1-3 very narrow conditions over broad rule sets.",
        "- Use signed margin/probability as guardrails, e.g. only fire for confident sub-regions inside the outer gate.",
        "- Do not globally flip all low/high fill_ratio rows.",
        "- Treat borderline correct TN/TP rows as guardrails.",
        "- A good first rule should fix more boundary errors than it harms correct boundary rows, ideally with at least 2:1 corrected:harmed.",
        "- If no physically convincing high-precision condition exists, return -1.",
    ]
    return "\n".join(lines)


def _evaluate_rule(
    rule: Rule,
    ctx: dict,
    boundary_margin: float | None,
    fn_margin: float | None,
    fp_margin: float | None,
) -> CandidateReport:
    rule.fit(
        ctx["fn_cases"],
        ctx["fp_cases"],
        ctx["easy_tn"],
        ctx["easy_tp"],
        default_pred=lambda c: c.svm_pred,
    )
    pred0 = ctx["pred"]
    y = ctx["y_test"]
    pred = apply_rules(
        [rule],
        ctx["X_rule"],
        ctx["probs"],
        top_k=1,
        decision_scores=ctx["decision"],
        boundary_margin=boundary_margin,
        fn_margin=fn_margin,
        fp_margin=fp_margin,
    )
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    diff = pred != pred0
    boundary = ctx["boundary_mask"]
    was_error = pred0 != y
    is_correct_now = pred == y
    return CandidateReport(
        id=rule.id,
        score=float(rule.score),
        fn_corr=int(rule.fn_corr),
        fp_corr=int(rule.fp_corr),
        fn_miss=int(rule.fn_miss),
        fp_over=int(rule.fp_over),
        easy_overrides=int(rule.easy_overrides),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        auc=float(roc_auc_score(y, ctx["probs"])),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        n_overrides=int(diff.sum()),
        corrected_errors=int((diff & was_error & is_correct_now).sum()),
        harmed_correct=int((diff & ~was_error & ~is_correct_now).sum()),
        boundary_overrides=int((diff & boundary).sum()),
        non_boundary_overrides=int((diff & ~boundary).sum()),
    )


def _write_json(path: pathlib.Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the boundary heuristic-agent harness.")
    parser.add_argument("--data-path", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    parser.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hybrid-boundary-margin", type=float, default=None)
    parser.add_argument("--hybrid-fn-margin", type=float, default=1.0)
    parser.add_argument("--hybrid-fp-margin", type=float, default=1.5)
    parser.add_argument("--llm-model", type=str, default="qwen3-max-2026-01-23")
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--n-calls", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Only write prompt and metadata; do not call the API.")
    parser.add_argument(
        "--allow-random-fallback",
        action="store_true",
        help="Allow run_experiment.make_llm_callable to use its random fallback when no API key is configured.",
    )
    args = parser.parse_args()

    model_path = args.model_path.resolve() if args.model_path else _latest_svm_model().resolve()
    out_dir = args.out_root.resolve() / f"boundary_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = _prepare_split(
        args.data_path.resolve(),
        model_path,
        args.test_size,
        args.random_state,
        args.hybrid_fn_margin,
        args.hybrid_fp_margin,
        args.hybrid_boundary_margin,
    )
    baseline_acc = float(accuracy_score(ctx["y_test"], ctx["pred"]))
    llm_model = _effective_llm_model(args.llm_model)
    prompt = _build_prompt(
        ctx,
        fn_margin=args.hybrid_fn_margin,
        fp_margin=args.hybrid_fp_margin,
        boundary_margin=args.hybrid_boundary_margin,
    )
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    metadata = {
        "data_path": str(args.data_path.resolve()),
        "model_path": str(model_path),
        "llm_model": llm_model,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "hybrid_boundary_margin": args.hybrid_boundary_margin,
        "hybrid_fn_margin": args.hybrid_fn_margin,
        "hybrid_fp_margin": args.hybrid_fp_margin,
        "baseline_accuracy": baseline_acc,
        "baseline_errors": int((ctx["pred"] != ctx["y_test"]).sum()),
        "boundary_candidate_count": int(ctx["boundary_mask"].sum()),
        "boundary_error_count": int(((ctx["pred"] != ctx["y_test"]) & ctx["boundary_mask"]).sum()),
        "boundary_fn_count": len(ctx["fn_cases"]),
        "boundary_fp_count": len(ctx["fp_cases"]),
        "boundary_easy_tn_count": len(ctx["easy_tn"]),
        "boundary_easy_tp_count": len(ctx["easy_tp"]),
        "diagnostic_note": "If prompt and evaluation use the same rows, candidate metrics are exploratory, not final holdout metrics.",
    }
    _write_json(out_dir / "metadata.json", metadata)
    print(f"Wrote prompt: {out_dir / 'prompt.md'}")
    print(
        "Baseline / gate: "
        f"acc={baseline_acc:.4f}, "
        f"boundary={metadata['boundary_candidate_count']}/{len(ctx['y_test'])}, "
        f"errors_in_gate={metadata['boundary_error_count']}/{metadata['baseline_errors']}"
    )

    if args.dry_run:
        print(f"Dry run complete: {out_dir}")
        return

    if not _has_api_key() and not args.allow_random_fallback:
        raise RuntimeError(
            "No supported API key found. Set BAILIAN_API_KEY / OPENAI_API_KEY / "
            "DEEPSEEK_API_KEY / DASHSCOPE_API_KEY, or pass --allow-random-fallback."
        )

    llm = make_llm_callable(
        llm_model,
        max_tokens=args.llm_max_tokens,
        timeout=args.timeout,
    )
    rules: list[Rule] = []
    reports: list[CandidateReport] = []
    for call_i in range(1, max(1, args.n_calls) + 1):
        response = llm(prompt)
        (out_dir / f"raw_response_{call_i:03d}.md").write_text(str(response), encoding="utf-8")
        code_blocks = extract_code_from_response(str(response))
        if not code_blocks:
            print(f"Call {call_i}: no extractable apply_rule_patch code.")
            continue
        for j, code in enumerate(code_blocks):
            rid = f"call{call_i:03d}_cand{j:02d}"
            rule = Rule(id=rid, code=code)
            report = _evaluate_rule(
                rule,
                ctx,
                boundary_margin=args.hybrid_boundary_margin,
                fn_margin=args.hybrid_fn_margin,
                fp_margin=args.hybrid_fp_margin,
            )
            rules.append(rule)
            reports.append(report)
            (out_dir / f"{rid}.py").write_text(code, encoding="utf-8")
            print(
                f"{rid}: acc={report.accuracy:.4f}, "
                f"corrected={report.corrected_errors}, harmed={report.harmed_correct}, "
                f"overrides={report.n_overrides}, score={report.score:.4f}"
            )

    reports_sorted = sorted(reports, key=lambda r: (r.accuracy, r.corrected_errors, -r.harmed_correct), reverse=True)
    _write_json(out_dir / "candidate_reports.json", [asdict(r) for r in reports_sorted])
    if rules:
        save_rules(rules, out_dir / "candidate_rules.json")
        best_report = reports_sorted[0]
        best_rule = next(r for r in rules if r.id == best_report.id)
        (out_dir / "best_rule.py").write_text(best_rule.code, encoding="utf-8")
        print(f"Best candidate: {best_report.id} acc={best_report.accuracy:.4f}")
    else:
        print("No candidate rules were produced.")
    print(f"Saved harness artifacts: {out_dir}")


if __name__ == "__main__":
    main()

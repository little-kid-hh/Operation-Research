# -*- coding: utf-8 -*-
"""
Iterative LLM-driven feature search for the linear-SVM route.

Workflow per iteration:
  1. read current context / policy / memory
  2. call LLM for candidate feature code
  3. materialize new dispatch-level features from aggregate + item-level tables
  4. train/evaluate linear SVM on the fixed split
  5. write trial artifacts, ablations, memory, and summary
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import sys
from datetime import datetime

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for p in (ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_experiment import make_llm_callable, _read_config_value  # noqa: E402
from src.feature_search import (  # noqa: E402
    DEFAULT_FEATURE_POLICY,
    SEED_FEATURE_CODE,
    SEED_POLICY_UPDATE,
    SEED_RATIONALE,
    _is_success_status,
    _metric_is_better,
    active_feature_bank_markdown,
    build_context_markdown,
    build_generation_prompt,
    build_memory_markdown,
    build_repair_prompt,
    build_summary_payload,
    coefficients_markdown,
    evaluate_candidate,
    evaluate_svm_features,
    extract_response_sections,
    failure_payload,
    load_search_data,
    records_to_csv_frame,
    run_feature_ablation,
    safe_trial_record,
    write_json,
)


DEFAULT_AGG = PROJECT_ROOT / "FunSearch_test" / "training_2orientations.csv"
DEFAULT_ITEMS = PROJECT_ROOT / "FunSearch_test" / "物品信息和dblf信息.csv"
DEFAULT_ROOT = ROOT / "experiments_feature_search"
DEFAULT_TREE_GUIDANCE = ROOT / "TREE_INSPIRED_FEATURE_HYPOTHESES.md"


TRANSIENT_LLM_ERROR_MARKERS = (
    "RateLimitError",
    "APIConnectionError",
    "InternalServerError",
    "Connection error",
    "timed out",
    "timeout",
    "429",
    "500",
)


def _is_transient_llm_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker.lower() in text.lower() for marker in TRANSIENT_LLM_ERROR_MARKERS)


def _call_llm_with_retries(llm, prompt: str, *, max_attempts: int = 4, sleep_seconds: float = 6.0) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm(prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _is_transient_llm_error(exc):
                raise
            print(
                f"[llm retry {attempt}/{max_attempts}] transient error: "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(sleep_seconds)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM call failed without an exception")


def _current_best_metrics(records: list, baseline_metrics: dict) -> dict:
    accepted = [r for r in records if _is_success_status(getattr(r, "status", None)) and getattr(r, "metrics", None)]
    if not accepted:
        return baseline_metrics
    accepted = sorted(
        accepted,
        key=lambda r: (
            float(r.metrics.get("auc", float("-inf"))),
            float(r.metrics.get("tpr_at_fpr1pct", float("-inf"))),
            float(r.metrics.get("accuracy", float("-inf"))),
        ),
        reverse=True,
    )
    return accepted[0].metrics


def _current_best_record(records: list):
    accepted = [r for r in records if _is_success_status(getattr(r, "status", None)) and getattr(r, "metrics", None)]
    if not accepted:
        return None
    accepted = sorted(
        accepted,
        key=lambda r: (
            float(r.metrics.get("auc", float("-inf"))),
            float(r.metrics.get("tpr_at_fpr1pct", float("-inf"))),
            float(r.metrics.get("accuracy", float("-inf"))),
        ),
        reverse=True,
    )
    return accepted[0]


def _best_candidate_prompt_block(records: list) -> str:
    best = _current_best_record(records)
    if best is None:
        return "## Current Best Accepted Candidate\n_No accepted candidate yet._\n"

    trial_dir = pathlib.Path(best.trial_dir)
    code_path = trial_dir / "feature_candidate.py"
    ablation_path = trial_dir / "ablation.json"
    code_text = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
    ablation = None
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    return best_candidate_markdown(
        feature_names=list(best.feature_names),
        metrics=best.metrics,
        code_text=code_text,
        ablation=ablation,
    )


def _accepted_records_in_order(records: list) -> list:
    return sorted(
        [r for r in records if _is_success_status(getattr(r, "status", None)) and getattr(r, "metrics", None)],
        key=lambda r: int(r.iteration),
    )


def _build_active_feature_bank(records: list, data, baseline_metrics: dict, svm_c: float, random_state: int):
    accepted = _accepted_records_in_order(records)
    bank_df = None
    bank_names: list[str] = []
    source_trials: list[int] = []
    bank_metrics = baseline_metrics

    for r in accepted:
        trial_dir = pathlib.Path(r.trial_dir)
        code_path = trial_dir / "feature_candidate.py"
        if not code_path.exists():
            continue
        code = code_path.read_text(encoding="utf-8")
        try:
            artifact = evaluate_candidate(
                code=code,
                data=data,
                baseline_metrics=baseline_metrics,
                svm_c=svm_c,
                random_state=random_state,
                existing_feature_df=bank_df,
                existing_feature_names=bank_names,
            )
            new_df = artifact.feature_frame[artifact.new_feature_names].reset_index(drop=True)
            bank_df = new_df if bank_df is None else pd.concat([bank_df.reset_index(drop=True), new_df], axis=1)
            bank_names.extend(list(artifact.new_feature_names))
        except Exception:
            # Backward compatibility for earlier experiments whose accepted candidates
            # were full feature snapshots rather than incremental additions.
            artifact = evaluate_candidate(
                code=code,
                data=data,
                baseline_metrics=baseline_metrics,
                svm_c=svm_c,
                random_state=random_state,
            )
            bank_df = artifact.feature_frame[artifact.new_feature_names].reset_index(drop=True)
            bank_names = list(artifact.new_feature_names)
        source_trials.append(int(r.iteration))
        bank_metrics = artifact.metrics

    return bank_df, bank_names, bank_metrics, source_trials


def _looks_like_feature_code_response(text: str) -> bool:
    body = (text or "").lower()
    return "## feature_code" in body or "build_candidate_features" in body


def _effective_llm_model(cli_model: str) -> str:
    if cli_model and cli_model != "qwen3coder":
        return cli_model
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


def _seed_trial(
    *,
    exp_dir: pathlib.Path,
    data,
    baseline_metrics: dict,
    svm_c: float,
    random_state: int,
    acceptance_auc_margin: float,
    acceptance_tpr_margin: float,
    acceptance_accuracy_margin: float,
) -> tuple[dict, object]:
    code = SEED_FEATURE_CODE
    artifact = evaluate_candidate(
        code=code,
        data=data,
        baseline_metrics=baseline_metrics,
        svm_c=svm_c,
        random_state=random_state,
    )
    trial_dir = exp_dir / "trials" / "iter_000_seed"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "feature_candidate.py").write_text(code + "\n", encoding="utf-8")
    (trial_dir / "policy_update.md").write_text(SEED_POLICY_UPDATE + "\n", encoding="utf-8")
    (trial_dir / "rationale.md").write_text(SEED_RATIONALE + "\n", encoding="utf-8")
    write_json(trial_dir / "metrics.json", artifact.metrics)
    write_json(trial_dir / "candidate_feature_coefficients.json", artifact.coefficients)
    artifact.feature_frame[artifact.new_feature_names].to_csv(
        trial_dir / "candidate_features.csv",
        index=False,
        encoding="utf-8",
    )
    (trial_dir / "candidate_feature_coefficients.md").write_text(
        coefficients_markdown(artifact.coefficients),
        encoding="utf-8",
    )
    ablation = run_feature_ablation(
        data=data,
        candidate_feature_df=artifact.feature_frame[artifact.new_feature_names],
        feature_names=artifact.new_feature_names,
        baseline_metrics=baseline_metrics,
        svm_c=svm_c,
        random_state=random_state,
    )
    write_json(trial_dir / "ablation.json", ablation)
    accepted = _metric_is_better(
        artifact.metrics,
        baseline_metrics,
        auc_margin=acceptance_auc_margin,
        tpr_margin=acceptance_tpr_margin,
        accuracy_margin=acceptance_accuracy_margin,
    )
    status = "accepted" if accepted else "rejected"
    write_json(
        trial_dir / "decision.json",
        {
            "status": status,
            "best_before_trial": baseline_metrics,
            "candidate_metrics": artifact.metrics,
            "acceptance_rule": {
                "auc_margin": acceptance_auc_margin,
                "tpr_margin": acceptance_tpr_margin,
                "accuracy_margin": acceptance_accuracy_margin,
            },
        },
    )
    record = safe_trial_record(
        iteration=0,
        trial_dir=trial_dir,
        policy_text=SEED_POLICY_UPDATE,
        rationale_text=SEED_RATIONALE,
        feature_names=artifact.new_feature_names,
        metrics=artifact.metrics,
        status=status,
    )
    return {"record": record, "artifact": artifact}


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative LLM feature search for linear SVM")
    parser.add_argument("--data-path", type=pathlib.Path, default=DEFAULT_AGG)
    parser.add_argument("--items-path", type=pathlib.Path, default=DEFAULT_ITEMS)
    parser.add_argument("--experiments-root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume-exp-dir", type=pathlib.Path, default=None)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument(
        "--tree-guidance-path",
        type=pathlib.Path,
        default=DEFAULT_TREE_GUIDANCE,
        help="Markdown note that captures RF/XGB-inspired feature-search guidance.",
    )
    parser.add_argument("--llm-model", type=str, default="qwen3coder")
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout-per-call", type=int, default=300)
    parser.add_argument("--max-new-features", type=int, default=6)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--skip-llm-probe",
        action="store_true",
        help="Skip the startup LLM availability probe and begin directly from the first iteration.",
    )
    parser.add_argument("--seed-trial", action="store_true", help="Always run the built-in seed candidate before LLM iterations.")
    parser.add_argument("--acceptance-auc-margin", type=float, default=5e-4)
    parser.add_argument("--acceptance-tpr-margin", type=float, default=5e-3)
    parser.add_argument("--acceptance-accuracy-margin", type=float, default=5e-4)
    args = parser.parse_args()

    llm_model = _effective_llm_model(args.llm_model)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume_exp_dir is not None:
        exp_dir = args.resume_exp_dir.resolve()
        if not exp_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {exp_dir}")
        print(f"Resuming feature-search experiment: {exp_dir}")
    else:
        root = (
            args.experiments_root.resolve()
            if args.experiments_root.is_absolute()
            else (PROJECT_ROOT / args.experiments_root).resolve()
        )
        exp_dir = root / "by_model" / llm_model.replace("/", "_").replace(":", "_") / f"exp_{ts}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"New feature-search experiment: {exp_dir}")

    trials_dir = exp_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    data = load_search_data(
        args.data_path,
        args.items_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    baseline_eval = evaluate_svm_features(
        feature_df=data.base_feature_df,
        y=data.y,
        train_idx=data.train_idx,
        test_idx=data.test_idx,
        svm_c=args.svm_c,
        random_state=args.random_state,
        new_feature_names=[],
    )
    baseline_metrics = baseline_eval.metrics
    print(
        "Baseline linear SVM: "
        f"AUC={baseline_metrics['auc']:.4f}, "
        f"TPR@1%={baseline_metrics['tpr_at_fpr1pct']:.4f}, "
        f"ACC={baseline_metrics['accuracy']:.4f}"
    )

    tree_guidance_path = (
        args.tree_guidance_path.resolve()
        if args.tree_guidance_path.is_absolute()
        else (PROJECT_ROOT / args.tree_guidance_path).resolve()
    )
    if not tree_guidance_path.exists():
        raise FileNotFoundError(f"Tree-guidance markdown not found: {tree_guidance_path}")
    tree_guidance_md = tree_guidance_path.read_text(encoding="utf-8")

    context_md = build_context_markdown(
        data,
        baseline_eval,
        tree_guidance_md=tree_guidance_md,
    )
    (exp_dir / "context.md").write_text(context_md, encoding="utf-8")
    (exp_dir / "tree_guidance.md").write_text(tree_guidance_md, encoding="utf-8")

    policy_path = exp_dir / "policy.md"
    if not policy_path.exists():
        policy_path.write_text(DEFAULT_FEATURE_POLICY + "\n", encoding="utf-8")

    records = []
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in old.get("trials", []):
            row_metrics = dict(row.get("metrics") or {})
            if row_metrics:
                if "accuracy" in row_metrics:
                    row_metrics["delta_accuracy"] = float(row_metrics["accuracy"] - baseline_metrics["accuracy"])
                if "recall" in row_metrics:
                    row_metrics["delta_recall"] = float(row_metrics["recall"] - baseline_metrics["recall"])
                if "auc" in row_metrics:
                    row_metrics["delta_auc"] = float(row_metrics["auc"] - baseline_metrics["auc"])
                if "tpr_at_fpr1pct" in row_metrics:
                    row_metrics["delta_tpr_at_fpr1pct"] = float(
                        row_metrics["tpr_at_fpr1pct"] - baseline_metrics["tpr_at_fpr1pct"]
                    )
            records.append(
                safe_trial_record(
                    iteration=row["iteration"],
                    trial_dir=pathlib.Path(row["trial_dir"]),
                    policy_text=row.get("policy_excerpt", ""),
                    rationale_text=row.get("rationale_excerpt", ""),
                    feature_names=row.get("feature_names", []),
                    metrics=row_metrics,
                    error=row.get("error"),
                    status=row.get("status"),
                )
            )

    if args.seed_trial and not any(r.iteration == 0 for r in records):
        seeded = _seed_trial(
            exp_dir=exp_dir,
            data=data,
            baseline_metrics=baseline_metrics,
            svm_c=args.svm_c,
            random_state=args.random_state,
            acceptance_auc_margin=args.acceptance_auc_margin,
            acceptance_tpr_margin=args.acceptance_tpr_margin,
            acceptance_accuracy_margin=args.acceptance_accuracy_margin,
        )
        records.append(seeded["record"])

    active_bank_df, active_bank_names, active_bank_metrics, active_bank_trials = _build_active_feature_bank(
        records,
        data,
        baseline_metrics,
        args.svm_c,
        args.random_state,
    )

    def persist_state() -> None:
        memory_md = build_memory_markdown(records, baseline_metrics)
        (exp_dir / "memory.md").write_text(memory_md, encoding="utf-8")
        bank_md = active_feature_bank_markdown(
            feature_names=active_bank_names,
            metrics=active_bank_metrics,
            source_trials=active_bank_trials,
        )
        (exp_dir / "active_feature_bank.md").write_text(bank_md, encoding="utf-8")
        if active_bank_df is not None and active_bank_names:
            active_bank_df.to_csv(exp_dir / "active_feature_bank.csv", index=False, encoding="utf-8")
        summary = build_summary_payload(
            baseline_metrics=baseline_metrics,
            records=records,
            config={
                "data_path": str(args.data_path),
                "items_path": str(args.items_path),
                "test_size": args.test_size,
                "random_state": args.random_state,
                "svm_c": args.svm_c,
                "tree_guidance_path": str(tree_guidance_path),
                "llm_model": llm_model,
                "llm_max_tokens": args.llm_max_tokens,
                "llm_timeout_per_call": args.llm_timeout_per_call,
                "max_new_features": args.max_new_features,
                "skip_llm_probe": bool(args.skip_llm_probe),
                "acceptance_rule": {
                    "auc_margin": args.acceptance_auc_margin,
                    "tpr_margin": args.acceptance_tpr_margin,
                    "accuracy_margin": args.acceptance_accuracy_margin,
                    "policy_update_only_if_accepted": True,
                },
                "feature_bank_mode": "cumulative_incremental",
                "active_feature_count": len(active_bank_names),
            },
        )
        write_json(summary_path, summary)
        records_to_csv_frame(records).to_csv(exp_dir / "trials.csv", index=False, encoding="utf-8")

    persist_state()

    llm = None
    if not args.skip_llm:
        llm = make_llm_callable(
            model=llm_model,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout_per_call,
        )
        if not args.skip_llm_probe:
            probe = _call_llm_with_retries(
                llm,
                "Return a minimal valid response with ## FEATURE_CODE and build_candidate_features.",
            )
            if not _looks_like_feature_code_response(probe):
                raise RuntimeError(
                    "Configured LLM endpoint is unavailable or returned a non-feature-search fallback response. "
                    "Check LLM_API_KEY / LLM_BASE_URL / provider configuration before running formal iterations."
                )

    start_iter = 1
    if records:
        start_iter = max(r.iteration for r in records) + 1

    for iteration in range(start_iter, start_iter + int(args.n_iters)):
        policy_md = policy_path.read_text(encoding="utf-8")
        memory_md = build_memory_markdown(records, baseline_metrics)
        (exp_dir / "memory.md").write_text(memory_md, encoding="utf-8")
        active_bank_md = active_feature_bank_markdown(
            feature_names=active_bank_names,
            metrics=active_bank_metrics,
            source_trials=active_bank_trials,
        )
        prompt = build_generation_prompt(
            context_md=context_md,
            policy_md=policy_md,
            memory_md=memory_md,
            best_candidate_md=active_bank_md,
            iteration=iteration,
            max_new_features=args.max_new_features,
            current_best_metrics=active_bank_metrics,
            acceptance_rule={
                "auc_margin": args.acceptance_auc_margin,
                "tpr_margin": args.acceptance_tpr_margin,
                "accuracy_margin": args.acceptance_accuracy_margin,
            },
        )

        trial_dir = trials_dir / f"iter_{iteration:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        if llm is None:
            print(f"[iter {iteration:03d}] skip-llm enabled; stopping before LLM generation")
            break

        try:
            response = _call_llm_with_retries(llm, prompt)
            (trial_dir / "raw_response.md").write_text(response or "", encoding="utf-8")

            policy_update, code, rationale = extract_response_sections(response or "")
            if not code.strip():
                raise ValueError("LLM response did not include FEATURE_CODE python block")

            (trial_dir / "policy_update.md").write_text((policy_update or "") + "\n", encoding="utf-8")
            (trial_dir / "rationale.md").write_text((rationale or "") + "\n", encoding="utf-8")
            (trial_dir / "feature_candidate.py").write_text(code + "\n", encoding="utf-8")

            artifact = evaluate_candidate(
                code=code,
                data=data,
                baseline_metrics=baseline_metrics,
                svm_c=args.svm_c,
                random_state=args.random_state,
                existing_feature_df=active_bank_df,
                existing_feature_names=active_bank_names,
            )
        except Exception as first_exc:
            fail = failure_payload(first_exc)
            if llm is not None:
                try:
                    repair_prompt = build_repair_prompt(
                        original_prompt=prompt,
                        original_response=response if "response" in locals() else "",
                        error_text=f"{fail['error_type']}: {fail['error']}",
                    )
                    (trial_dir / "repair_prompt.md").write_text(repair_prompt, encoding="utf-8")
                    repaired_response = _call_llm_with_retries(llm, repair_prompt)
                    (trial_dir / "repair_raw_response.md").write_text(repaired_response or "", encoding="utf-8")
                    policy_update, code, rationale = extract_response_sections(repaired_response or "")
                    if not code.strip():
                        raise ValueError("Repair response did not include FEATURE_CODE python block")
                    (trial_dir / "policy_update.md").write_text((policy_update or "") + "\n", encoding="utf-8")
                    (trial_dir / "rationale.md").write_text((rationale or "") + "\n", encoding="utf-8")
                    (trial_dir / "feature_candidate.py").write_text(code + "\n", encoding="utf-8")
                    artifact = evaluate_candidate(
                        code=code,
                        data=data,
                        baseline_metrics=baseline_metrics,
                        svm_c=args.svm_c,
                        random_state=args.random_state,
                        existing_feature_df=active_bank_df,
                        existing_feature_names=active_bank_names,
                    )
                except Exception as repair_exc:
                    fail = failure_payload(repair_exc)
                    write_json(trial_dir / "failure.json", fail)
                    record = safe_trial_record(
                        iteration=iteration,
                        trial_dir=trial_dir,
                        policy_text="",
                        rationale_text="",
                        feature_names=[],
                        metrics=None,
                        error=f"{fail['error_type']}: {fail['error']}",
                    )
                    records.append(record)
                    print(f"[iter {iteration:03d}] failed: {fail['error_type']}: {fail['error']}")
                    persist_state()
                    continue
            else:
                write_json(trial_dir / "failure.json", fail)
                record = safe_trial_record(
                    iteration=iteration,
                    trial_dir=trial_dir,
                    policy_text="",
                    rationale_text="",
                    feature_names=[],
                    metrics=None,
                    error=f"{fail['error_type']}: {fail['error']}",
                )
                records.append(record)
                print(f"[iter {iteration:03d}] failed: {fail['error_type']}: {fail['error']}")
                persist_state()
                continue

        try:
            artifact.feature_frame[artifact.new_feature_names].to_csv(
                trial_dir / "candidate_features.csv",
                index=False,
                encoding="utf-8",
            )
            write_json(trial_dir / "metrics.json", artifact.metrics)
            write_json(trial_dir / "candidate_feature_coefficients.json", artifact.coefficients)
            (trial_dir / "candidate_feature_coefficients.md").write_text(
                coefficients_markdown(artifact.coefficients),
                encoding="utf-8",
            )
            ablation = run_feature_ablation(
                data=data,
                candidate_feature_df=artifact.feature_frame[artifact.new_feature_names],
                feature_names=artifact.new_feature_names,
                baseline_metrics=baseline_metrics,
                svm_c=args.svm_c,
                random_state=args.random_state,
                existing_feature_df=active_bank_df,
                existing_feature_names=active_bank_names,
            )
            write_json(trial_dir / "ablation.json", ablation)
            (trial_dir / "svm_insights.md").write_text(
                json.dumps({"new_feature_coefficients": artifact.coefficients}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            best_before_trial = active_bank_metrics
            accepted = _metric_is_better(
                artifact.metrics,
                best_before_trial,
                auc_margin=args.acceptance_auc_margin,
                tpr_margin=args.acceptance_tpr_margin,
                accuracy_margin=args.acceptance_accuracy_margin,
            )
            trial_status = "accepted" if accepted else "rejected"
            write_json(
                trial_dir / "decision.json",
                {
                    "status": trial_status,
                    "best_before_trial": best_before_trial,
                    "candidate_metrics": artifact.metrics,
                    "acceptance_rule": {
                        "auc_margin": args.acceptance_auc_margin,
                        "tpr_margin": args.acceptance_tpr_margin,
                        "accuracy_margin": args.acceptance_accuracy_margin,
                    },
                },
            )

            record = safe_trial_record(
                iteration=iteration,
                trial_dir=trial_dir,
                policy_text=policy_update,
                rationale_text=rationale,
                feature_names=artifact.new_feature_names,
                metrics=artifact.metrics,
                status=trial_status,
            )
            records.append(record)
            print(
                f"[iter {iteration:03d}|{trial_status}] "
                f"AUC={artifact.metrics['auc']:.4f} "
                f"(Δ {artifact.metrics['delta_auc']:+.4f}), "
                f"TPR@1%={artifact.metrics['tpr_at_fpr1pct']:.4f} "
                f"(Δ {artifact.metrics['delta_tpr_at_fpr1pct']:+.4f}), "
                f"features={artifact.new_feature_names}"
            )

            if accepted and policy_update.strip():
                updated_policy = policy_md.rstrip() + "\n\n## Iteration " + str(iteration) + "\n" + policy_update.strip() + "\n"
                policy_path.write_text(updated_policy, encoding="utf-8")
            if accepted:
                new_df = artifact.feature_frame[artifact.new_feature_names].reset_index(drop=True)
                active_bank_df = new_df if active_bank_df is None else pd.concat(
                    [active_bank_df.reset_index(drop=True), new_df],
                    axis=1,
                )
                active_bank_names = list(active_bank_names) + list(artifact.new_feature_names)
                active_bank_metrics = artifact.metrics
                active_bank_trials = list(active_bank_trials) + [int(iteration)]
        except Exception as exc:
            fail = failure_payload(exc)
            write_json(trial_dir / "failure.json", fail)
            record = safe_trial_record(
                iteration=iteration,
                trial_dir=trial_dir,
                policy_text="",
                rationale_text="",
                feature_names=[],
                metrics=None,
                error=f"{fail['error_type']}: {fail['error']}",
            )
            records.append(record)
            print(f"[iter {iteration:03d}] failed: {fail['error_type']}: {fail['error']}")

        persist_state()

    print(f"Artifacts written to: {exp_dir}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Generate a frozen feature-search guidance markdown from structured analysis
artifacts using an LLM, and save the full prompt/response trail.

This is intended to remove manual summarization from the
tree/XGB-analysis -> guidance-markdown step.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for p in (ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_experiment import make_llm_callable  # noqa: E402


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_markdown_block(text: str) -> str:
    body = text or ""
    marker = "## GUIDANCE_MARKDOWN"
    if marker in body:
        body = body.split(marker, 1)[1].strip()
    if "```markdown" in body:
        block = body.split("```markdown", 1)[1]
        if "```" in block:
            return block.split("```", 1)[0].strip() + "\n"
    if "```md" in body:
        block = body.split("```md", 1)[1]
        if "```" in block:
            return block.split("```", 1)[0].strip() + "\n"
    if "```" in body:
        block = body.split("```", 1)[1]
        if "```" in block:
            return block.split("```", 1)[0].strip() + "\n"
    return body.strip() + ("\n" if body.strip() else "")


def build_prompt(
    *,
    route_name: str,
    output_filename: str,
    analysis_summary_json: str,
    analysis_notes: str,
) -> str:
    return f"""You are writing a frozen guidance markdown for an iterative LLM-driven feature-engineering workflow.

Task:
- Read the structured analysis artifacts below.
- Write one concise but rigorous guidance document that will later be embedded into a feature-search prompt.
- The target workflow is: stronger teacher model analysis -> guidance markdown -> LLM generates new SVM features.

Requirements:
- Output only one markdown document.
- Keep it factual and reproducible.
- Distinguish clearly between:
  1. reproducible evidence source
  2. current evidence
  3. feature directions to test
  4. what not to over-invest in
  5. immediate workflow implication
- Do not invent metrics or files not present in the inputs.
- Treat the analysis as hypothesis support, not ground truth.
- Keep the tone operational, not promotional.
- Mention exact local artifact paths when they are part of reproducibility.
- Do not mention that an assistant or LLM wrote this document.
- Return the result in this format:

## GUIDANCE_MARKDOWN
```markdown
<full markdown>
```

Document target:
- route name: {route_name}
- output filename: {output_filename}

Structured analysis summary JSON (already condensed to the key fields needed for guidance):
```json
{analysis_summary_json}
```

Additional notes:
```text
{analysis_notes}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature-search guidance markdown with LLM")
    parser.add_argument("--analysis-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-path", type=pathlib.Path, required=True)
    parser.add_argument("--route-name", type=str, required=True)
    parser.add_argument("--llm-model", type=str, default="glm-5.1")
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout-per-call", type=int, default=240)
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    output_path = args.output_path.resolve() if args.output_path.is_absolute() else (PROJECT_ROOT / args.output_path).resolve()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in analysis dir: {analysis_dir}")

    analysis_summary_obj = json.loads(_read_text(summary_path))
    compact_summary = {
        "teacher_scope": analysis_summary_obj.get("teacher_scope"),
        "split": analysis_summary_obj.get("split"),
        "data_path": analysis_summary_obj.get("data_path"),
        "feature_set": analysis_summary_obj.get("feature_set"),
        "active_bank_used": analysis_summary_obj.get("active_bank_used"),
        "base40_xgb_cfg_source": analysis_summary_obj.get("base40_xgb_cfg_source"),
        "base40_xgb_cfg": analysis_summary_obj.get("base40_xgb_cfg"),
        "base40_rf_cfg_source": analysis_summary_obj.get("base40_rf_cfg_source"),
        "base40_rf_cfg": analysis_summary_obj.get("base40_rf_cfg"),
        "metrics": analysis_summary_obj.get("metrics"),
        "recovery_meta": analysis_summary_obj.get("recovery_meta_base40") or analysis_summary_obj.get("recovery_meta_h21"),
        "top_base40_features": analysis_summary_obj.get("top_base40_features", [])[:12],
    }
    analysis_summary = json.dumps(compact_summary, indent=2, ensure_ascii=False)
    notes = "\n".join(
        [
            f"analysis_dir={analysis_dir}",
            f"summary_path={summary_path}",
            "The generated guidance will later be passed into HybridSVM/scripts/run_feature_search_agent.py via --tree-guidance-path.",
            "This step must be reproducible: save prompt, raw response, and final markdown.",
            "Only use the base40 teacher scope; do not leak any later active-bank features into the guidance.",
        ]
    )
    prompt = build_prompt(
        route_name=args.route_name,
        output_filename=output_path.name,
        analysis_summary_json=analysis_summary,
        analysis_notes=notes,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = output_dir / f"guidance_gen_{ts}_{args.llm_model.replace('/', '_').replace(':', '_')}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    llm = make_llm_callable(
        model=args.llm_model,
        max_tokens=args.llm_max_tokens,
        timeout=args.llm_timeout_per_call,
    )
    if not args.skip_probe:
        probe = llm("Reply exactly with: READY")
        (manifest_dir / "probe.txt").write_text(probe or "", encoding="utf-8")

    raw_response = llm(prompt)
    (manifest_dir / "raw_response.md").write_text(raw_response or "", encoding="utf-8")
    markdown = _extract_markdown_block(raw_response or "")
    if not markdown.strip():
        raise RuntimeError("LLM response did not yield a markdown guidance document")

    output_path.write_text(markdown, encoding="utf-8")
    manifest = {
        "timestamp": ts,
        "route_name": args.route_name,
        "analysis_dir": str(analysis_dir),
        "summary_path": str(summary_path),
        "output_path": str(output_path),
        "llm_model": args.llm_model,
        "llm_max_tokens": int(args.llm_max_tokens),
        "llm_timeout_per_call": int(args.llm_timeout_per_call),
        "skip_probe": bool(args.skip_probe),
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved guidance markdown: {output_path}")
    print(f"Saved generation trace: {manifest_dir}")


if __name__ == "__main__":
    main()

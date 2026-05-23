# Repro Bundle

This bundle contains the minimum local files needed to reproduce the XGB-guided
no-seed exploratory run recorded in the parent directory.

## Inputs

- `data/training_2orientations.csv`
- `data/物品信息和dblf信息.csv`
- `data/training_dispatch_item_types.csv`

## Teacher analysis

- `teacher_analysis/xgb_guidance_base40_20260514/`
  - base40-only XGB vs SVM analysis on the fixed split

## Frozen guidance

- `guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`
  - the frozen guidance markdown used by the feature-search run
- `guidance/guidance_gen_20260514_164953_glm-5.1/`
  - exact GLM prompt, raw response, and manifest used to produce the frozen
    XGB guidance markdown

## Main code

- `code/HybridSVM/run_experiment.py`
- `code/HybridSVM/requirements.txt`
- `code/HybridSVM/scripts/generate_guidance_with_llm.py`
- `code/HybridSVM/scripts/run_feature_search_agent.py`
- `code/HybridSVM/src/feature_search.py`
- `code/HybridSVM/src/svm_train.py`
- `code/research/analyze_xgb_guidance_base40.py`
- `code/TabTreeFormer/`

## Reference teacher config source

- `reference/Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json`
  - upstream XGB hyper-parameter search summary consumed by
    `analyze_xgb_guidance_base40.py`

## Recorded run

The actual run artifacts are stored in the parent directory:

- `../summary.json`
- `../trials.csv`
- `../context.md`
- `../tree_guidance.md`
- `../policy.md`
- `../memory.md`
- `../active_feature_bank.*`
- `../trials/`

## Repro commands

Teacher analysis:

```bash
python3 code/research/analyze_xgb_guidance_base40.py
```

Generate frozen XGB guidance with GLM:

```bash
python3 code/HybridSVM/scripts/generate_guidance_with_llm.py \
  --analysis-dir teacher_analysis/xgb_guidance_base40_20260514 \
  --output-path guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md \
  --route-name "XGB base40 -> GLM guidance -> linear SVM feature search" \
  --llm-model glm-5.1 \
  --llm-timeout-per-call 240 \
  --skip-probe
```

Run the no-seed XGB-guided search:

```bash
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY
python3 code/HybridSVM/scripts/run_feature_search_agent.py \
  --llm-model glm-5.1 \
  --n-iters 10 \
  --max-new-features 6 \
  --llm-timeout-per-call 240 \
  --tree-guidance-path guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md \
  --skip-llm-probe
```

Important:

- the recorded run in the parent directory is exploratory, not a formal clean
  run;
- it was resumed on the same `exp_dir`, so the recorded artifacts accumulated
  `15` trials rather than the original intended `10`.

## Environment

The LLM backend is resolved by `code/HybridSVM/run_experiment.py`.
Set one of the supported env groups there. In the recorded run, the model was:

- `glm-5.1`

The original environment also required unsetting local proxy env vars before
calling the script.

## What to inspect first

1. `teacher_analysis/xgb_guidance_base40_20260514/README.md`
2. `guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`
3. `guidance/guidance_gen_20260514_164953_glm-5.1/prompt.md`
4. `../summary.json`
5. `../trials.csv`
6. `../trials/iter_014/feature_candidate.py`

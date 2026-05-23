# Repro Bundle

This bundle contains the minimum local files needed to reproduce the RF-guided
no-seed run recorded in the parent directory.

## Inputs

- `data/training_2orientations.csv`
- `data/物品信息和dblf信息.csv`

## Teacher analysis

- `teacher_analysis/rf_guidance_base40_20260514/`
  - base40-only RF vs SVM analysis on the fixed split

## Frozen guidance

- `guidance/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`
  - the frozen guidance markdown used by the feature-search run
- `guidance/guidance_gen_20260514_181154_glm-5.1/`
  - exact GLM prompt, raw response, and manifest used to produce the frozen
    RF guidance markdown

## Main code

- `code/HybridSVM/run_experiment.py`
- `code/HybridSVM/scripts/generate_guidance_with_llm.py`
- `code/HybridSVM/scripts/run_feature_search_agent.py`
- `code/HybridSVM/src/feature_search.py`
- `code/HybridSVM/src/svm_train.py`
- `code/research/analyze_rf_guidance_base40.py`

## Formal run

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
python3 research/analyze_rf_guidance_base40.py
```

Generate frozen RF guidance with GLM:

```bash
python3 HybridSVM/scripts/generate_guidance_with_llm.py \
  --analysis-dir research/rf_guidance_base40_20260514 \
  --output-path HybridSVM/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md \
  --route-name "RF base40 -> GLM guidance -> linear SVM feature search" \
  --llm-model glm-5.1 \
  --llm-timeout-per-call 240 \
  --skip-probe
```

Run the clean no-seed RF-guided search:

```bash
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY
python3 HybridSVM/scripts/run_feature_search_agent.py \
  --llm-model glm-5.1 \
  --n-iters 10 \
  --llm-max-tokens 8192 \
  --max-new-features 6 \
  --llm-timeout-per-call 240 \
  --tree-guidance-path HybridSVM/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md \
  --skip-llm-probe
```

## Environment

The LLM backend is resolved by `HybridSVM/run_experiment.py`.
Set one of the supported env groups there. In the recorded run, the model was:

- `glm-5.1`

The run also required unsetting local proxy env vars before calling the script.

## What to inspect first

1. `teacher_analysis/rf_guidance_base40_20260514/README.md`
2. `guidance/RF_GUIDED_FEATURE_HYPOTHESES_BASE40.md`
3. `guidance/guidance_gen_20260514_181154_glm-5.1/prompt.md`
4. `../summary.json`
5. `../trials.csv`
6. `../trials/iter_010/feature_candidate.py`

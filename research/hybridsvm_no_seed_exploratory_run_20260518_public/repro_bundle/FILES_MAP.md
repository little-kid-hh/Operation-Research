# File Map

## Raw data

- `data/training_2orientations.csv`
  - raw dispatch-level table
  - one row per dispatch after the repository's internal deduplication logic
  - provides labels, base40 aggregate features, and vehicle-level context

- `data/物品信息和dblf信息.csv`
  - raw item-level table
  - multiple rows may belong to the same dispatch
  - used to derive item-level threshold, tail, share, and interaction features

- `data/training_dispatch_item_types.csv`
  - auxiliary raw-data file preserved with the bundle

## Teacher analysis

- `teacher_analysis/xgb_guidance_base40_20260514/summary.json`
  - compact XGB-vs-SVM comparison on base40 only
  - records split, teacher config, metrics, recovery counts, and top features

- `teacher_analysis/xgb_guidance_base40_20260514/base40_gain_importance.csv`
  - XGB gain importance on the fixed base40 feature set

- `teacher_analysis/xgb_guidance_base40_20260514/xgb_vs_svm_recovery_contrast_base40.csv`
  - feature-level contrast on test rows where XGB recovers SVM errors

- `teacher_analysis/xgb_guidance_base40_20260514/README.md`
  - concise human-readable summary of the teacher package

## Reference teacher config source

- `reference/Ensemble_baseline/experiments/xgb_search_20260511_202834_base40/summary.json`
  - upstream XGB hyper-parameter search summary used by
    `analyze_xgb_guidance_base40.py` to reconstruct the base40 teacher model

## Frozen guidance

- `guidance/XGB_GUIDED_FEATURE_HYPOTHESES_BASE40.md`
  - frozen markdown guidance written by GLM from the XGB teacher package
  - this exact file was inserted into the feature-search prompt

- `guidance/guidance_gen_20260514_164953_glm-5.1/prompt.md`
  - exact prompt sent to GLM to convert structured XGB analysis into guidance

- `guidance/guidance_gen_20260514_164953_glm-5.1/raw_response.md`
  - raw GLM response used to produce the frozen guidance file

- `guidance/guidance_gen_20260514_164953_glm-5.1/manifest.json`
  - metadata for the guidance-generation call

## Code

- `code/research/analyze_xgb_guidance_base40.py`
  - builds the base40-only XGB teacher package

- `code/HybridSVM/scripts/generate_guidance_with_llm.py`
  - converts the structured teacher package into frozen XGB guidance via GLM

- `code/HybridSVM/scripts/run_feature_search_agent.py`
  - main iterative driver
  - reads context and state, calls GLM, validates code, evaluates SVM, writes
    trial artifacts, updates memory and active bank

- `code/HybridSVM/src/feature_search.py`
  - prompt assembly, candidate execution, validation, ablation, run summaries

- `code/HybridSVM/src/svm_train.py`
  - raw data loading, deduplication, fixed-split SVM training, metric helpers

- `code/HybridSVM/run_experiment.py`
  - LLM backend resolution and API-call wrapper

- `code/HybridSVM/requirements.txt`
  - project dependency list preserved with the bundle

- `code/TabTreeFormer/`
  - local module copy kept because `run_experiment.py` imports it

## Recorded exploratory run

The parent directory is itself the recorded exploratory run:

- `../context.md`
  - assembled prompt context
  - includes data summary, schema summary, baseline metrics, linear SVM
    coefficients, and frozen XGB guidance snapshot

- `../tree_guidance.md`
  - the exact frozen XGB guidance text embedded into the run prompt

- `../policy.md`
  - persistent search policy and hard constraints

- `../memory.md`
  - compact record of accepted, rejected, and failed trials seen so far

- `../active_feature_bank.md` / `../active_feature_bank.csv`
  - accepted incremental features accumulated across accepted trials

- `../trials.csv`
  - one-line summary of every iteration

- `../summary.json`
  - final run summary, best accepted trial, and config

- `../trials/iter_XXX/prompt.md`
  - exact per-iteration prompt sent to GLM

- `../trials/iter_XXX/raw_response.md`
  - raw GLM output

- `../trials/iter_XXX/feature_candidate.py`
  - candidate feature code returned by GLM

- `../trials/iter_XXX/rationale.md`
  - GLM's explanation for the proposed features

- `../trials/iter_XXX/policy_update.md`
  - GLM's suggested policy update

- `../trials/iter_XXX/candidate_features.csv`
  - materialized new per-dispatch features

- `../trials/iter_XXX/metrics.json`
  - SVM metrics after adding the candidate features

- `../trials/iter_XXX/decision.json`
  - accept / reject / fail decision

- `../trials/iter_XXX/ablation.json`
  - single-feature and leave-one-out ablation for that iteration

- `../trials/iter_XXX/candidate_feature_coefficients.*`
  - SVM coefficients for the new feature block

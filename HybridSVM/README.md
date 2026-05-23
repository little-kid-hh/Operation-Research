# HybridSVM
# Linear SVM baseline + LLM-evolved rule heuristics for 3D bin packing feasibility

- `src/svm_train.py`    — Linear SVM training & inference pipeline
- `src/ensemble_train.py` — Project-grade stacked ensemble baseline (SVM/LR/RF/XGB + linear combiner)
- `../TabTreeFormer/tabtree_data.py` — RF leaf tokenization + leaf stats utilities
- `../TabTreeFormer/tabtreeformer.py` — RF-backed TabTreeFormer training/inference pipeline
- `src/hard_cases.py`    — FN/FP hard case mining & summarization
- `src/evolution.py`     — FunSearch + ReEvo evolutionary rule generation
- `src/evaluate.py`      — SVM vs Hybrid comparison harness
- `src/feature_search.py` — LLM-guided feature-engineering utilities for the linear-SVM route
- `run_experiment.py`    — Main entry point
- `scripts/analyze_svm_boundary.py` — SVM signed-margin error analysis for boundary-gated Hybrid
- `scripts/run_boundary_heuristic_agent.py` — one-shot/multi-shot API harness for boundary heuristic generation and scoring
- `scripts/run_feature_search_agent.py` — iterative LLM feature search on top of the linear-SVM baseline
- `scripts/run_manual_feature_trial.py` — manual incremental feature trial on top of an accepted feature bank

## Documentation map (where is the island model?)

| What you need | Where to find it |
|---------------|------------------|
| **FunSearch island model** (how many islands, rule placement, worst-island reset, `round_robin`, etc.) | In **this file**, under **[Architecture](#architecture)** → **[FunSearch island model: evolution pipeline (detailed)](#funsearch-island-model-evolution-pipeline-detailed)** (includes the “why multiple islands” sketch and `Rule.fit` scoring). The same area also has **[Stage 1](#stage-1-linear-svm-brief)** → **[Stage 2](#stage-2-hard-case-mining-detailed)** → **[Stage 4](#stage-4-hybrid-evaluation-and-top-k-rules-detailed)** (read Stages 1–2–3–4 in order; the island model is Stage 3). |
| **Code mapping** | `src/evolution.py`: `Population`, `Island`, `Population.add_rule`, `reset_worst_island`, `EvolutionEngine.step`, `apply_rules` |
| **Default experiment hyperparameters** (island reset period, Hybrid rule count) | `run_experiment.py`: `EVOLUTION_ISLAND_RESET_EVERY`, `HYBRID_RULE_TOP_K` |
| **Auto-generated Word pipeline report** | Run `python scripts/write_experiment_report_docx.py --exp-dir …` or `--aggregate-v1`, open the `.docx`. The script still emits **Chinese** section titles (e.g. reader guide, full pipeline by stage); see [Word report](#word-report-experiment_reportdocx). |

Repo-wide search for **`FunSearch island model`** also lands on that section.

## Install

```bash
pip install numpy pandas scikit-learn
# Optional: generate Word experiment reports
pip install python-docx
# Or install everything from the repo:
pip install -r requirements.txt
```

## Feature-search route

For the current `HybridSVM = LLM + SVM feature engineering` route, use:

- workflow doc: `FEATURE_SEARCH_WORKFLOW.md`
- tree-guidance note: `TREE_INSPIRED_FEATURE_HYPOTHESES.md`
- main search entry: `scripts/run_feature_search_agent.py`
- manual follow-up entry: `scripts/run_manual_feature_trial.py`

Recommended starting command:

```bash
python scripts/run_feature_search_agent.py \
  --llm-model glm-5.1 \
  --seed-trial \
  --n-iters 10 \
  --tree-guidance-path TREE_INSPIRED_FEATURE_HYPOTHESES.md
```

Current canonical cumulative GLM run:

```text
HybridSVM/experiments_feature_search/by_model/glm-5.1/exp_20260513_235323/
```

## Run

```bash
# SVM baseline only (no LLM)
python run_experiment.py --skip-evolution

# Ensemble baseline only (no LLM), full base-model set + linear combiner
python run_experiment.py --baseline-mode ensemble --skip-evolution

# TabTreeFormer(RF) baseline only (no LLM)
python run_experiment.py --baseline-mode tabtreeformer_rf --skip-evolution

# Full pipeline (default LLM: qwen3-max-2026-01-23; output under experiments_v2/by_model/<model>/exp_<time>/)
python run_experiment.py --n-evolution-iters 10

# Analyze whether SVM errors cluster near the decision boundary
python scripts/analyze_svm_boundary.py

# SVM-first boundary-gated Hybrid:
# only rows near the signed SVM hyperplane are sent through LLM heuristic rules
python run_experiment.py --baseline-mode svm --hybrid-boundary-margin 1.0 --evolve-boundary-only --n-evolution-iters 10

# Asymmetric gate: rescue negative-side FN near margin 1.0, veto positive-side FP up to margin 1.5
python run_experiment.py --baseline-mode svm --hybrid-fn-margin 1.0 --hybrid-fp-margin 1.5 --evolve-boundary-only --n-evolution-iters 10

# Recommended protocol: search/select rules on validation, report final Hybrid once on outer test
python run_experiment.py --baseline-mode svm --hybrid-fn-margin 1.0 --hybrid-fp-margin 1.5 --evolve-boundary-only --rule-selection-split val --rule-val-size 0.2 --n-evolution-iters 10

# Recommended (new) scoring mechanism for boundary-gated evolution:
# net_gain_v1 = (corrected_errors - harm_weight * harmed_correct) / n_hard, with hard-guard
python run_experiment.py --baseline-mode svm --hybrid-fn-margin 1.0 --hybrid-fp-margin 1.5 --evolve-boundary-only --rule-selection-split val --rule-val-size 0.2 --evolution-score-mode net_gain_v1 --evolution-harm-weight 3.0 --n-evolution-iters 10

# Legacy scoring replay (for A/B comparison with old runs)
python run_experiment.py --baseline-mode svm --hybrid-fn-margin 1.0 --hybrid-fp-margin 1.5 --evolve-boundary-only --rule-selection-split val --rule-val-size 0.2 --evolution-score-mode legacy_f1 --n-evolution-iters 10

# Build the boundary-agent prompt without calling an API
python scripts/run_boundary_heuristic_agent.py --dry-run --model-path models/linear_svm_20260413_165559.json

# Call the configured API once, extract apply_rule_patch, evaluate, and save artifacts
python scripts/run_boundary_heuristic_agent.py --model-path models/linear_svm_20260413_165559.json --n-calls 1

# Another DashScope-compatible model (each model gets its own subfolder)
python run_experiment.py --llm-model qwen3.6-plus --n-evolution-iters 10

# Full pipeline with ensemble baseline (Stage 1 baseline -> Stage 2/3/4 reuse)
python run_experiment.py --baseline-mode ensemble --n-evolution-iters 10

# Legacy output layout (same as older HybridSVM runs): experiments/by_model/...
python run_experiment.py --experiments-root experiments --n-evolution-iters 10

# Legacy flat layout: <experiments-root>/exp_<time>/ only
python run_experiment.py --flat-exp-dir --n-evolution-iters 10

# Continue evolution (point to the experiment folder, including under by_model/...)
python run_experiment.py --resume-exp-dir experiments/by_model/qwen3.5-plus/exp_20260412_140519 --n-evolution-iters 10

# Longer LLM completions (default 4096) if rules are cut off mid–code fence
python run_experiment.py --llm-max-tokens 4096 --n-evolution-iters 10

# Island population controls (defaults: round_robin placement; trim_top_k reset)
python run_experiment.py --rule-placement round_robin --island-reset-mode trim_top_k --island-reset-keep-top 3

# Ensemble tuning knobs
python run_experiment.py --baseline-mode ensemble --ensemble-base-models "svm,lr,rf,xgb" --ensemble-meta-model logreg --ensemble-cv-folds 5
```

For API calls, configure `.env` under `HybridSVM/`. If you are changing both URL and model, use the generic OpenAI-compatible form:

```bash
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-id
```

Provider aliases are also supported: `BAILIAN_API_KEY` / `BAILIAN_BASE_URL` / `BAILIAN_MODEL`, `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`, and `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`.

### Boundary heuristic agent workflow

This is the recommended flow for the SVM-first Hybrid path:

1. **Configure the API model** in `HybridSVM/.env`.

   For the current code-generation heuristic path, `qwen3coder` is the recommended first model:

   ```bash
   LLM_API_KEY=your_api_key_here
   LLM_BASE_URL=https://models.sjtu.edu.cn/api/v1
   LLM_MODEL=qwen3coder
   ```

2. **Analyze the SVM decision boundary**.

   ```bash
   python scripts/analyze_svm_boundary.py
   ```

   On the reference split, the useful gate is asymmetric:

   - `--hybrid-fn-margin 1.0`: rescue negative-side samples close to the SVM boundary.
   - `--hybrid-fp-margin 1.5`: veto positive-side samples close to the SVM boundary.

   This selects `605 / 2500` rows and covers `155 / 181` SVM errors. The corresponding **oracle accuracy** is a theoretical upper bound: if a perfect rule fixed every selected error and did not harm any correct row, Accuracy would reach `0.9896`. It is not a real model score.

3. **Dry-run the heuristic-agent prompt**.

   ```bash
   python scripts/run_boundary_heuristic_agent.py \
     --dry-run \
     --model-path models/linear_svm_20260413_165559.json \
     --hybrid-fn-margin 1.0 \
     --hybrid-fp-margin 1.5
   ```

   Output goes to `experiments_boundary_agent/boundary_agent_<timestamp>/` and includes:

   - `prompt.md`: the exact prompt sent to the model.
   - `metadata.json`: baseline accuracy, gate size, FN/FP counts, and split info.

4. **Generate heuristic candidates through the API**.

   ```bash
   python scripts/run_boundary_heuristic_agent.py \
     --model-path models/linear_svm_20260413_165559.json \
     --hybrid-fn-margin 1.0 \
     --hybrid-fp-margin 1.5 \
     --n-calls 3
   ```

   The harness extracts Python code blocks defining `apply_rule_patch(svm_prob, features) -> int`, scores each candidate, and writes:

   - `raw_response_*.md`: raw LLM replies.
   - `call*_cand*.py`: extracted candidate rules.
   - `candidate_reports.json`: candidate metrics, corrected errors, harmed correct rows, and override counts.
   - `candidate_rules.json`: serialized candidate pool.
   - `best_rule.py`: best candidate under the harness ranking.

5. **Interpret candidate metrics carefully**.

   The first harness run is diagnostic: the prompt and evaluation use the same boundary rows. A higher Accuracy there tells us the heuristic found a plausible pattern, but it is not a final holdout claim. For a reportable result, generate rules on a validation split, freeze the best rule, then evaluate once on a separate holdout split.

6. **Use the rule in the full Hybrid pipeline**.

   The main experiment entrypoint supports the same gate:

   ```bash
   python run_experiment.py \
     --baseline-mode svm \
     --hybrid-fn-margin 1.0 \
     --hybrid-fp-margin 1.5 \
     --evolve-boundary-only \
     --n-evolution-iters 10
   ```

   This route uses the FunSearch/ReEvo loop; the one-shot `run_boundary_heuristic_agent.py` harness is faster for testing prompt and model choices.

### Boundary-gated implementation walkthrough

This section maps the **SVM-first, boundary-gated Hybrid** path to the current code, so you can trace one sample from Stage 1 through rule override.

**1. Stage 1 still starts with a normal SVM forward pass**

- Entry: `run_experiment.py`
- The baseline SVM produces:
  - `svm_probs_test`: `P(y=1)` used for the base hard prediction.
  - `baseline_decision_test`: the signed linear decision score.
- The signed score is the key boundary signal:
  - `decision < 0`: SVM predicts class `0`; this is the side where an FN-rescue rule may flip to `1`.
  - `decision >= 0`: SVM predicts class `1`; this is the side where an FP-veto rule may flip to `0`.

**2. The boundary gate is defined on signed SVM margin, not only on probability**

- Code: `src/evolution.py` → `boundary_candidate_mask(...)`
- Supported gate forms:
  - `--hybrid-boundary-margin m`: symmetric gate, equivalent to `abs(decision_score) <= m`
  - `--hybrid-fn-margin m_fn`: negative side only, `-m_fn <= decision_score < 0`
  - `--hybrid-fp-margin m_fp`: positive side only, `0 <= decision_score <= m_fp`
- In the main experiment loop, the gate is applied right after SVM inference and logged as:
  - candidate row count,
  - covered SVM errors,
  - covered FN / FP counts.

**3. Boundary metadata is injected into the rule feature space**

- Code: `run_experiment.py`
- After SVM inference, the rule-side feature frame gets three synthetic fields:
  - `svm_decision_score`
  - `svm_abs_decision_score`
  - `svm_near_boundary`
- This means LLM-generated rules can use raw packing features together with the SVM's signed margin as a guardrail, for example "only rescue when `svm_decision_score` is slightly negative and geometry is favorable".

**4. `--evolve-boundary-only` changes the optimization set, not just final inference**

- Without `--evolve-boundary-only`:
  - evolution sees all mined FN / FP / easy TN / easy TP rows,
  - but the final Hybrid override can still be boundary-gated.
- With `--evolve-boundary-only`:
  - the mined FN / FP / easy TN / easy TP lists are filtered to gate-selected rows before prompt construction and rule scoring,
  - so the LLM is optimized only on the part of the test split that is actually eligible for rule intervention.

**5. The one-shot harness and the main evolution loop use the same boundary idea in two different ways**

- `scripts/analyze_svm_boundary.py`
  - diagnostic analysis only,
  - prints how many rows and errors are captured by candidate margin settings,
  - reports an **oracle accuracy** upper bound for each gate.
- `scripts/run_boundary_heuristic_agent.py`
  - builds one focused prompt for boundary rows,
  - asks the API for candidate `apply_rule_patch(...)` implementations,
  - scores candidates on the same boundary subset,
  - useful for prompt/model iteration, but its metric is diagnostic.
- `run_experiment.py`
  - runs the full FunSearch/ReEvo-style multi-iteration rule search,
  - uses the same signed-margin gate during final rule application.

**6. The boundary prompt is more than a hard-case dump**

- In `scripts/run_boundary_heuristic_agent.py`, the prompt includes:
  - boundary FN / FP examples,
  - boundary TP / TN contrast rows,
  - feature-scale tables inside the gate,
  - linear SVM coefficient summaries,
  - explicit instructions to abstain aggressively with `return -1`.
- The intent is to make the rule behave like a **narrow second review module**, not a global replacement classifier.

**7. Final Hybrid inference is "SVM first, rules second, boundary-only if configured"**

- Code: `src/evolution.py` → `apply_rules(...)`
- Inference order:
  1. Start from SVM hard predictions.
  2. If a boundary gate is active, skip every sample outside the gate.
  3. For each eligible sample, try the global top-k evolved rules in score order.
  4. The first rule that returns `0` or `1` wins; `-1` means "trust SVM and continue to the next rule".
  5. If no rule fires, keep the SVM prediction unchanged.

**8. Current evaluation caveat**

- The boundary-gated path is already implemented as a proper selective override mechanism.
- However, in the current main experiment flow, the same held-out split is often reused for:
  - mining boundary hard cases,
  - evolving / selecting rules,
  - reporting final Hybrid metrics.
- This is fine for exploratory development, but it is not the cleanest protocol for a reportable result.
- For a stricter experiment:
  - freeze the SVM first,
  - evolve rules on a validation split,
  - then evaluate the frozen rule set exactly once on a separate holdout split.

### Baseline modes (Stage 1)

- `--baseline-mode svm` (default): existing linear SVM baseline path.
- `--baseline-mode ensemble`: stacked ensemble baseline with:
- `--baseline-mode tabtreeformer_rf`: RF leaf-token + leaf-stats + RF-residual baseline.

  - base learners: `svm,lr,rf,xgb` (configurable),
  - OOF stacking (`StratifiedKFold`) to avoid leakage,
  - linear meta-combiner (`--ensemble-meta-model logreg` by default).

Boundary-gated Hybrid flags (SVM mode only):

- `--hybrid-boundary-margin`: symmetric signed-margin gate, equivalent to `abs(decision_score) <= margin`.
- `--hybrid-fn-margin`: negative-side gate for FN rescue, `-margin <= decision_score < 0`.
- `--hybrid-fp-margin`: positive-side gate for FP veto, `0 <= decision_score <= margin`.
- `--evolve-boundary-only`: the LLM evolution prompt and rule scoring use only rows inside the configured gate. Without this flag, only final rule application is gated.
- `--rule-selection-split test`: legacy behavior, where the same held-out split is used both for rule search and final Hybrid reporting.
- `--rule-selection-split val`: recommended behavior, where the outer train portion is split again; rules are searched/selected on validation, and the outer test split is kept for final reporting.
- `--rule-val-size`: when `--rule-selection-split val`, controls how much of the outer-train portion becomes rule-selection validation.
- `--llm-timeout-per-call`: timeout seconds for each OpenAI-compatible LLM call (default `180`).
- `--evolution-score-mode net_gain_v1` (default): optimize net gain with explicit harm penalty; `legacy_f1` keeps the previous rule ranking logic.
- `--evolution-harm-weight`: harm multiplier for `net_gain_v1` (default `3.0`).
- `--evolution-score-hard-guard` / `--no-evolution-score-hard-guard`: whether to clip candidates with `harmed_correct >= corrected_errors` to non-positive score.
- Boundary-gated rule features include `svm_decision_score`, `svm_abs_decision_score`, and `svm_near_boundary`.
- Rule prompts use raw physical features for the rule-search rows, including `total_skuvolume`, `vehicle_capacity`, and derived `fill_ratio`, even though the SVM itself may drop some of these columns before training.

Ensemble-specific flags:

- `--ensemble-base-models`: comma-separated subset/order of `svm,lr,rf,xgb`
- `--ensemble-meta-model`: combiner type (`logreg` default; `mlp` also supported)
- `--ensemble-cv-folds`: OOF folds
- `--ensemble-random-state`: random seed for ensemble learners/folds
- `--tabtree-rf-n-estimators`, `--tabtree-rf-max-depth`, `--tabtree-rf-min-samples-leaf`: RF tokenizer backbone
- `--tabtree-d-model`, `--tabtree-nhead`, `--tabtree-n-layers`, `--tabtree-ff-dim`, `--tabtree-dropout`: Transformer core
- `--tabtree-mlp-hidden`, `--tabtree-epochs`, `--tabtree-batch-size`, `--tabtree-lr`, `--tabtree-weight-decay`: training/head config

If `xgboost` is unavailable at runtime, the `xgb` slot falls back to sklearn GBDT and this is recorded in outputs (`xgb_fallback_used`).

`--n-evolution-iters` is the number of **additional** steps when resuming. Omit `--data-path` / `--model-path` on resume if `evolution_checkpoint.json` exists (it stores the CSV and model used). Old runs without that file still work: iteration index is inferred from `hard_case_summary_hybrid_iter_*.md`, and the SVM is found as `models/linear_svm_<same_suffix_as_exp_folder>.json` when the names align (`exp_YYYYMMDD_HHMMSS` → `linear_svm_YYYYMMDD_HHMMSS.json`). If you resume with different `--llm-model`, `--llm-max-tokens`, island options, or evolution defaults than those recorded in the checkpoint, a **WARNING** is printed (run continues).

When resuming with ensemble mode, the same consistency warnings apply for baseline settings (`baseline_mode`, `ensemble_base_models`, `ensemble_meta_model`, `ensemble_cv_folds`, `ensemble_random_state`).

## Acceptance criteria (ensemble project skeleton)

- `svm` mode remains behaviorally compatible with previous runs.
- `ensemble` mode runs end-to-end through Stage 1~4 and writes full artifacts.
- `results.json`/checkpoint include baseline metadata (`baseline_mode`, base models, meta model, fallback markers).
- On same split, `ensemble` should improve at least one target metric (commonly AUC or `TPR@FPR=1%`) over `svm`.

**LLM HTTP timeout** (`EvolutionConfig.timeout_per_call`, default 180s) is applied to **OpenAI-compatible** clients (Bailian, OpenAI, DeepSeek). The DashScope native `Generation.call` path does not use it; prefer the compatible API if you need a strict timeout. You can override with `--llm-timeout-per-call`.

**Code extraction** accepts common Markdown Python fences such as `` ```python `` and `` ```py `` (and unclosed/truncated fences per `src/evolution.extract_code_from_response`).

### Layout: one folder per LLM model

**New runs** default to output root **`experiments_v2/`** (sibling to legacy `experiments/`), so older artifacts are not mixed with new runs:

`experiments_v2/by_model/<llm_model_slug>/exp_<timestamp>/`

Use `--experiments-root experiments` to write under the original `experiments/` tree. `--resume-exp-dir` accepts any existing folder (old or new layout).

Each model’s runs stay grouped. `results.json` records `llm_model` and `n_evolution_iters`.

To prevent mixing old/new scoring experiments, new runs are bucketed by mechanism tag:

`experiments_v2/by_model/<llm_model_slug>/by_mechanism/<mechanism_tag>/exp_<timestamp>/`

Examples:

- `net_gain_v1_hw3_guard`
- `legacy_f1`

**Evolution prompt:** `hard_case_summary.md` includes FN/FP plus **easy TN/TP** (borderline-correct rows) and optional **§5** contrastive rows (easy TP near the FN feature cloud, easy TN near the FP cloud). Tune with `--n-easy-typical` and `--n-near-hard-easy` (set the latter to `0` to disable §5).

When `--rule-selection-split val` is used, `results.json` distinguishes the two roles:

- `n_rule_search_cases_total`: hard/easy case counts on the rule-search split (validation).
- `n_hard_cases`: hard/easy case counts on the final evaluation split (outer test).

## Architecture

```
Raw CSV (10k dispatches, 41 features)
        │
   Linear SVM (C=10, kernel=linear)
        │
   P(y=1) — soft probabilities
        │
   Optional signed-margin gate
   (only near-boundary rows go to rules)
        │
   ┌────┴────┐
   │ Hard case mining (FN / FP / EasyTN / EasyTP)
   └─────────┘
        │
   ┌────┴─────────────────────────────────────────┐
   │ FunSearch + ReEvo Evolution Loop              │
   │   • Islands model (4 islands, 50 rules each)   │
   │   • Best-shot prompting (k=2 seed rules)     │
   │   • ReEvo reflection → verbal gradients       │
   │   • FunSearch island reset every 5 iters      │
   │   • New rules: round-robin across islands (configurable) │
   │   • Worst-island reset: trim to top-k then clone (vs. clear all) │
   └─────────────────────────────────────────────────┘
        │
   Top-K evolved rule patches
        │
   ┌────┴────┐
   │ SVM + Rule Ensemble (Hybrid)                  │
   └─────────┘
        │
   Final predictions (SVM corrected by rules)
```

### FunSearch island model: evolution pipeline (detailed)

> **This is the full “island model” section**, together with [Stage 2](#stage-2-hard-case-mining-detailed) and [Stage 4](#stage-4-hybrid-evaluation-and-top-k-rules-detailed) below, it documents the core HybridSVM pipeline.

The following describes **Stage 3** as implemented in `src/evolution.py` / `run_experiment.py` (aligned with FunSearch ideas: skeleton + evolving the critical part; islands for diversity). **ReEvo** here shows up as *prompt design* (hard cases + seed rules + optional previous-iteration best). `build_reflection_prompt` provides a parent/child reflection template, but **`EvolutionEngine.step()` only issues one LLM call per iteration** by default — no second “reflection” call.

**0. Why multiple islands instead of one list?**

- If every rule sits in one list and you keep only global tops, the population can **collapse to similar code** quickly (local optimum).
- The **island model** splits the population across several `Island`s; new rules are placed by policy; periodically **reset the worst island** and **clone** a strong rule from a strong island — balancing **exploration** (each island evolves) and **exploitation** (inject good code into weak islands).
- Versus a single list: easier to keep **phenotypic/code diversity** (`diversity_score()` in logs is a coarse measure).

**Schematic (logical layout, not a memory diagram)**

```
Population
├── Island 0  [Rule, Rule, …]  best_score = max rule.score on this island
├── Island 1  [Rule, Rule, …]
├── Island 2  [Rule, Rule, …]
└── Island 3  [Rule, Rule, …]
         ↑
   add_rule() uses rule_placement to choose an island
         ↑
   Every island_reset_every steps: reset_worst_island()
         → pick islands with max vs min best_score; clone best rule from best island into worst
```

**1. Population structure**

- `Population` holds `n_islands` `Island`s (default **4**).
- Each `Island` stores `Rule`s (Python source strings defining `apply_rule_patch(svm_prob, features) -> {-1,0,1}`).
- Per-island cap `max_island_size` (default **50**): if exceeded, keep top rules by `rule_sort_key`: primary **score**; tie-break **FN+FP corrections**; then **fewer easy-case overrides** (`easy_overrides`).

**2. How new rules enter islands (`rule_placement`)**

- **round_robin** (default): round-robin placement across islands 0→1→…→(n−1)→0… so candidates do not all pile onto the “weakest” island.
- **min_best_score**: place on the island with smallest `best_score` (try to lift the weakest island).
- **min_rule_count**: place on the island with the fewest rules.

**3. What one `step()` does**

1. `iteration += 1`.
2. **Best-shot seeds**: `sample_prompt_rules(k=n_seed_rules_per_prompt)` (default **k=2**): **pick one island uniformly at random**, then sample `k` rules inside it with Boltzmann weights (`temperature=0.5`); if the library is empty, fall back to `RULE_SKELETON`.
3. **Build prompt** (`build_evolution_prompt`): optional blocks include **Feature scales** (test quantiles), **linear SVM coefficient summary**, full `hard_case_summary.md`, **k seed rules** with FN/FP/easy stats, **rule skeleton**, output format; if `include_prev_iter_in_prompt`, also **last iteration’s best candidate in this batch** (`format_prev_iter_best_md`) for cross-iteration refinement.
4. **One LLM call** → response string.
5. **Code extraction** (`extract_code_from_response`): extract zero or more code blocks from `` ```python `` / `` ```py ``, unclosed fences, or non-Python fences that still contain `apply_rule_patch`, etc.
6. **Per code block**: build a `Rule`, `fit` on **FN / FP / easy TN / easy TP** to get score and counts; `population.add_rule` (island placement + per-island trimming).
7. **Island reset (FunSearch-style)**: if `iteration % island_reset_every == 0`, run `reset_worst_island()`:
   - Find **best** and **worst** islands by each island’s `best_score`;
   - Take the globally best rule from the best island, **deep-copy** with a new id, add to the worst island;
   - **trim_top_k** (default): on the worst island, keep only the top `island_reset_keep_top` (default **3**) rules by score, then append the clone (avoids wiping the whole island at once);
   - **clear_and_clone**: clear the worst island, then add the clone.
8. Store the **best candidate in this iteration** in `_prev_iter_best_rule` for the next prompt.
9. Log prints global best score, this-iteration best score, and `diversity_score()` (approximate pairwise diversity of rule sources).

**4. Per-rule scoring (`Rule.fit`) — not island-specific, but decides who survives**

- On **FN** cases: the rule should **override to 1** where SVM is wrong; count `fn_corr` / `fn_miss`.
- On **FP** cases: **override to 0**; count `fp_corr` / `fp_over`.
- On **easy TN / easy TP**: any return other than `-1` counts as an erroneous intervention on a sample where SVM should be trusted → `easy_overrides` (TN/TP counted separately, penalty merged).
- **Raw score**: F1-style terms on FN and FP branches averaged (`compute_score`), then multiply by **easy penalty** `penalty = max(0, 1 - 0.1 * easy_overrides / N_easy)` → final `rule.score`.
- **Global ordering** (prompt seeds, island trimming, Hybrid top-k) uses `rule_sort_key`: **score → (fn_corr+fp_corr) → -easy_overrides**.

**5. Defaults vs `run_experiment.py`**

- `island_reset_every` is set from **`EVOLUTION_ISLAND_RESET_EVERY = 5`** in the experiment entrypoint (**worst-island reset every 5 iterations**; differs from `EvolutionConfig` default 10 — **trust `run_experiment`**).
- Other island CLI flags: `--rule-placement`, `--island-reset-mode`, `--island-reset-keep-top` (see Run section above).

### Stage 1: Linear SVM (brief)

- **Role**: Learn a linear decision boundary on the fixed train/validation split; output `P(y=1)` and hard labels per sample. Later rules **do not** update SVM weights; they only override at inference when triggered.
- **Details**: Kernel and regularization live in `src/svm_train.py` and `run_experiment` args; features are usually MinMax-scaled, etc.; SVM coefficient summaries go into the evolution prompt (`svm_linear_insights.md`, etc.).

### Stage 2: Hard case mining (detailed)

- **Inputs**: SVM hard predictions and `P(y=1)` on the test split, threshold 0.5.
- **Partitioning**: Against ground truth, **FN** (GT=1, pred=0) and **FP** (GT=0, pred=1); among **correct** SVM predictions, split **Easy TN / Easy TP** to constrain rules from breaking correct calls.
- **Outputs**: `hard_case_summary.md` (tables/excerpts in the evolution prompt), usually with borderline easy rows and optional **§5** contrastive rows (easy samples near FN/FP feature clouds) so the LLM sees the boundary.
- **Code**: `src/hard_cases.py` (`mine_hard_cases`, `render_hard_case_prompt`, etc.).

### Stage 4: Hybrid evaluation and top-k rules (detailed)

- **Goal**: On the test split, compare **SVM-only** vs **Hybrid** (SVM + rule overrides).
- **Where rules come from**: After evolution (or resumed checkpoint), pool all `Rule`s from all islands, take the global top **`HYBRID_RULE_TOP_K`** (default **3** in `run_experiment.py`) by `rule_sort_key` — **not** “one rule per island”.
- **Inference** (`src/evolution.py` → `apply_rules`):
  1. Start from SVM hard predictions `final_pred = (svm_prob >= 0.5)`.
  2. For **each sample**, walk the **top-k list in order** and call `apply_rule_patch(svm_prob, features)`:
     - **`-1`**: no override → **try the next rule**;
     - **`0` or `1`**: set prediction to that value and **stop** (short-circuit, first hit wins).
  3. If every rule returns `-1` or raises, keep the SVM prediction.
- **Probabilities for metrics**: `src/evaluate.py` may set hybrid-like scores to ~0.95 / ~0.05 where Hybrid disagrees with SVM (see `evaluate_pipeline`); interpret **Hybrid AUC** with care.
- **Outputs**: `results.json`, `hard_case_summary_hybrid.md` (mine FN/FP **against Hybrid predictions** to see remaining errors).

## Design principles

- **Linear SVM** gives a strong, interpretable baseline with well-calibrated probabilities
- **Hard case mining** finds where the linear boundary breaks down
- **Rule patches** target only the specific feature ranges where SVM fails — they don't retrain the SVM
- **Boundary-gated Hybrid** optionally invokes rules only near the SVM signed margin, which is safer than letting LLM rules override high-confidence easy rows
- **Islands model** (FunSearch) keeps diversity, avoids local optima
- **Best-shot prompting** + **ReEvo reflection** guides LLM toward incrementally better rules
- **Easy case protection** — rules that override TN/TP cases are penalized in scoring
- **Feature scales in prompt** — min/max/median quantiles for the test-split features are injected so the LLM uses correct thresholds (e.g. `fill_ratio` as a 0–1 fraction, not m³)
- **Linear SVM coefficients** — top |w_j| on MinMax-scaled features plus intercept are injected so the LLM sees which dimensions drive the baseline and in which direction; also saved as `<experiments-root>/.../svm_linear_insights.md`
- **Tie-break** — when scores tie, rules are ranked by total FN/FP corrections, then by fewer spurious overrides on easy TN/TP
- **Previous-iteration best** — each evolution step (after the first) includes the prior step’s best candidate in the prompt for refinement

## Experiment directory outputs

Under `experiments_v2/by_model/<model>/exp_<timestamp>/` (or `--experiments-root` / `--flat-exp-dir`), the main artifacts are:

| File | Meaning |
|------|---------|
| `hard_case_summary.md` | SVM error analysis (FN/FP) **plus** easy TN/TP and optional contrastive rows for the evolution LLM |
| `hard_case_summary_hybrid.md` | Same mining logic, but hard labels come from the **final Hybrid** (SVM + rules) on the test split; shows remaining FN/FP after rule correction |
| `hard_case_summary_hybrid_iter_NNN.md` | **One file per evolution iteration** when `save_hybrid_snapshot_each_iter` is on: if the step **succeeded**, Hybrid hard-case mining summary; if it **failed**, a short **diagnostic** (failure kind, message, optional truncated LLM reply) — not a mining summary |
| `evolution_checkpoint.json` | Written each evolution step: last `iteration`, paths / split settings, plus evolution fields (`llm_max_tokens`, `rule_placement`, island reset options, `timeout_per_call`, …) for reproducibility |
| `evolution_progress.jsonl` | **One JSON object per line per iteration** — includes `step_ok`, `best_score`, `n_rules`, and on failure `failure_kind`, `failure_message`, `failure_detail`, `response_preview` (truncated) |

**`failure_detail` when code extraction fails** (`failure_kind` = `no_extractable_code`): `no_python_fence` (no Python-tagged fence such as `` ```python `` / `` ```py ``), `has_open_python_fence_no_close` (opener but no well-formed closed block — often truncation), `python_blocks_without_apply_rule_patch` (closed Python-tagged blocks but none define `apply_rule_patch`), or `unknown` (unexpected).

`results.json` includes `n_hard_cases` (SVM mining) and `n_hard_cases_hybrid` (Hybrid mining counts). After a resumed run it also has `resumed` and `resume_start_iteration`. `evolved_rules.json` is still written **once** at the end of evolution (not each iteration).

## Pipeline overview (stages)

| Stage | What happens |
|-------|----------------|
| 1 | Train or load linear SVM; probabilities on the held-out test split |
| 2 | Mine FN / FP / easy TN / TP vs SVM; write `hard_case_summary.md` (rich prompt: errors + easy + optional §5 contrasts) |
| 3 | LLM evolution (islands, optional island reset); `evolved_rules.json`, per-iter hybrid snapshots |
| 4 | Evaluate SVM vs Hybrid; `results.json`, `hard_case_summary_hybrid.md` |

## Word report (`experiment_report.docx`)

[`scripts/write_experiment_report_docx.py`](scripts/write_experiment_report_docx.py) builds a `.docx` from `results.json`: pipeline text (reader guide, Stages 1–4 including FunSearch islands and Hybrid top-k), plus metric tables. **Body text in the generated file is still Chinese** (the script was not localized in this README pass). After a run:

```bash
pip install python-docx   # or: pip install -r requirements.txt
python scripts/write_experiment_report_docx.py --exp-dir experiments_v2/by_model/<model>/exp_<timestamp>
```

Default output: `<exp-dir>/experiment_report.docx`. Use `--output path/to/report.docx` to override.

**Legacy layout (`experiments/`):** aggregate every `results.json` under `experiments/by_model/*/exp_*/` into one combined report (overview table + per-run detail; Chinese body text from the same script):

```bash
python scripts/write_experiment_report_docx.py --aggregate-v1 --v1-root experiments
```

Default output: `experiments/v1_all_models_experiment_report.docx`.

## Example recorded run (reference)

| Item | Value |
|------|--------|
| Layout | `experiments_v2/by_model/qwen3-coder-480b-a35b-instruct/exp_20260413_165559/` |
| Model | `qwen3-coder-480b-a35b-instruct` (Bailian / DashScope compatible) |
| Evolution iters | 10 (fresh run, not resume) |
| Test SVM accuracy | 0.9276 |
| Hybrid accuracy | 0.6664 (global metrics often drop when rules override many easy rows) |
| Hard-case subset | SVM errors on FN+FP: 181 rows; Hybrid improves accuracy on that subset vs always-wrong SVM baseline there |

Figures above come from that folder’s `results.json`. Your own runs will differ; always cite the `experiment_dir` and `timestamp` fields in `results.json`.

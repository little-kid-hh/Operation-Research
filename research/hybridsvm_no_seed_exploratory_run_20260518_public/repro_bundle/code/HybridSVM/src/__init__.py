# -*- coding: utf-8 -*-
"""
HybridSVM: Linear SVM + LLM-Evolved Rule Heuristics for 3D Bin Packing Feasibility

Architecture:
  1. Linear SVM (baseline) → soft probability predictions
  2. Hard case mining → extract FP/FN error cases
  3. FunSearch/ReEvo loop → LLM generates rule corrections
  4. Rule ensemble → combine SVM + multiple rule layers
  5. Evaluation → compare SVM vs Hybrid on hard cases

Design rationale:
  - Linear SVM gives a strong, interpretable baseline with well-calibrated probabilities
  - FP/FN cases are where the linear boundary fails; these are the "holes" in SVM
  - Instead of retraining SVM (which is expensive), we patch it with targeted rules
  - Rules are evolved using an evolutionary prompt mechanism (ReEvo-style):
      LLM compares seed rules vs higher-quality ones → extracts evolution strategy
      → applies to new rules → evaluated → best kept
  - FunSearch contributes the "skeleton + evolve critical part" idea:
      we provide the rule skeleton (if-then structure with feature references)
      and let LLM evolve the threshold expressions
"""

from .svm_train import train_svm, load_svm_pipeline
from .hard_cases import mine_hard_cases, summarize_hard_cases, summarize_for_evolution_prompt
from .evolution import (
    EvolutionEngine,
    EvolutionConfig,
    Rule,
    RULE_SKELETON,
    load_rules,
    save_rules,
    apply_rules,
)
from .evaluate import evaluate_pipeline, compare_svm_vs_hybrid

__all__ = [
    "train_svm",
    "load_svm_pipeline",
    "mine_hard_cases",
    "summarize_hard_cases",
    "summarize_for_evolution_prompt",
    "EvolutionEngine",
    "EvolutionConfig",
    "Rule",
    "RULE_SKELETON",
    "load_rules",
    "save_rules",
    "apply_rules",
    "evaluate_pipeline",
    "compare_svm_vs_hybrid",
]
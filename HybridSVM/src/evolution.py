# -*- coding: utf-8 -*-
"""
Evolution module: LLM-driven rule generation for correcting SVM hard cases.

Combines ideas from:
  - FunSearch: skeleton + evolve critical logic; islands model for population diversity
  - ReEvo: reflective evolution — LLM compares pairs of rules and extracts improvement strategies
  - Best-shot prompting: feed top-k programs into the prompt

Architecture:
  Population = list of Rule objects
  Each Rule = {
      id: str
      code: str (Python function)
      fn_acc: float  (accuracy on FN cases)
      fp_acc: float  (accuracy on FP cases)
      score: float   (combined metric)
  }

  Iterative loop:
    1. Sample k rules from population (favor high score + diversity)
    2. Build prompt: skeleton + k seed rules + FN/FP summaries
    3. Call LLM → generate new rule candidates
    4. Evaluate candidates → compute score
    5. Add to population, trim to max size
    6. Periodically reset low-score islands (islands model)
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.hard_cases import (
    mine_hard_cases,
    render_hard_case_prompt,
    summarize_hard_cases,
)


# ─── Rule dataclass ──────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    A candidate rule as executable Python code.

    The rule must implement:
        apply_rule_patch(svm_prob: float, features: dict) -> int
    Returns:
        -1 : don't override SVM
        0  : override SVM → predict NO
        1  : override SVM → predict YES

    After `fit()`, the rule is assigned a score based on:
        score = (FN_corrections + FP_corrections) /
                (FN_misses + FP_overrides + epsilon)
    Where:
        FN_corrections  = FN cases correctly overridden to 1
        FP_corrections  = FP cases correctly overridden to 0
        FN_misses       = FN cases we failed to correct
        FP_overrides    = FP cases wrongly overridden to 1
    """

    id: str
    code: str
    fn_corr: int = 0
    fp_corr: int = 0
    fn_miss: int = 0
    fp_over: int = 0
    fn_acc: float = 0.0
    fp_acc: float = 0.0
    score: float = 0.0
    fitness_history: list[float] = field(default_factory=list)
    island_id: int = 0
    easy_overrides: int = 0  # overrides on easy TN/TP (for tie-break when score ties)

    def compute_score(self, eps: float = 1e-6) -> float:
        """Compute combined F1-style score for FN and FP correction."""
        p_fn = self.fn_corr / (self.fn_corr + self.fn_miss + eps)
        p_fp = self.fp_corr / (self.fp_corr + self.fp_over + eps)
        r_fn = self.fn_corr / (self.fn_corr + self.fn_miss + eps)
        r_fp = self.fp_corr / (self.fp_corr + self.fp_over + eps)
        f_fn = 2 * p_fn * r_fn / (p_fn + r_fn + eps)
        f_fp = 2 * p_fp * r_fp / (p_fp + r_fp + eps)
        self.score = (f_fn + f_fp) / 2
        self.fitness_history.append(self.score)
        return self.score

    def fit(
        self,
        fn_cases,
        fp_cases,
        easy_tn,
        easy_tp,
        default_pred: Callable,
    ) -> None:
        """
        Evaluate this rule on all case categories.

        Parameters
        ----------
        fn_cases  : list of Case objects that are FN
        fp_cases  : list of Case objects that are FP
        easy_tn   : list of Case objects that are easy TN (must NOT override)
        easy_tp   : list of Case objects that are easy TP (must NOT override)
        default_pred : callable(Case) → the SVM's default prediction
        """
        fn_corr = fn_miss = 0
        fp_corr = fp_over = 0

        apply_fn = self.get_apply_fn()
        if apply_fn is None:
            # code error: score = 0
            self.fn_corr = self.fp_corr = 0
            self.fn_miss = len(fn_cases)
            self.fp_over = len(fp_cases)
            self.easy_overrides = 0
            self.compute_score()
            return

        # ── FN cases: should override to 1 ──────────────────────────────────
        for case in fn_cases:
            try:
                override = apply_fn(case.svm_prob, case.features)
            except Exception:
                override = -1
            if override == 1:
                fn_corr += 1
            else:
                fn_miss += 1

        # ── FP cases: should override to 0 ──────────────────────────────────
        for case in fp_cases:
            try:
                override = apply_fn(case.svm_prob, case.features)
            except Exception:
                override = -1
            if override == 0:
                fp_corr += 1
            else:
                fp_over += 1

        # ── Easy cases: must NOT override (no false interventions) ─────────
        easy_tn_over = 0
        easy_tp_over = 0
        for case in easy_tn + easy_tp:
            try:
                override = apply_fn(case.svm_prob, case.features)
            except Exception:
                override = -1
            if override != -1:
                if case.gt == 0:
                    easy_tn_over += 1
                else:
                    easy_tp_over += 1

        # Penalize: overriding easy cases is bad
        penalty = 1.0 - 0.1 * (easy_tn_over + easy_tp_over) / max(len(easy_tn) + len(easy_tp), 1)
        penalty = max(penalty, 0.0)

        self.easy_overrides = easy_tn_over + easy_tp_over
        self.fn_corr = fn_corr
        self.fp_corr = fp_corr
        self.fn_miss = fn_miss
        self.fp_over = fp_over
        raw_score = self.compute_score()
        self.score = raw_score * penalty

    def get_apply_fn(self) -> Optional[Callable]:
        """Compile the code and return the apply_rule_patch function."""
        try:
            namespace: dict = {}
            exec(self.code, namespace)
            return namespace.get("apply_rule_patch")
        except Exception:
            return None

    def __hash__(self):
        return hash(self.id)


def rule_sort_key(r: Rule) -> tuple[float, float, float]:
    """
    Lexicographic ordering: larger tuple means better rule.
    Used when scores tie to prefer more corrections and fewer easy-case overrides.
    """
    total_corr = r.fn_corr + r.fp_corr
    return (float(r.score), float(total_corr), float(-r.easy_overrides))


# ─── Population & Islands ────────────────────────────────────────────────────

@dataclass
class Island:
    """One island in the island model."""

    id: int
    rules: list[Rule] = field(default_factory=list)
    best_score: float = 0.0

    def add(self, rule: Rule) -> None:
        rule.island_id = self.id
        self.rules.append(rule)
        if rule.score > self.best_score:
            self.best_score = rule.score

    def get_top(self, k: int = 1) -> list[Rule]:
        return sorted(self.rules, key=rule_sort_key, reverse=True)[:k]

    def sample(self, k: int = 2, temperature: float = 1.0) -> list[Rule]:
        """Sample k rules, favoring high score (Boltzmann selection)."""
        if not self.rules:
            return []
        scores = np.array([max(r.score, 0.0) for r in self.rules])
        weights = np.exp(scores / temperature)
        weights /= weights.sum()
        indices = np.random.choice(len(self.rules), size=min(k, len(self.rules)), replace=False, p=weights)
        return [self.rules[i] for i in indices]


@dataclass
class Population:
    """
    Island-based evolutionary population.
    """

    n_islands: int = 4
    max_island_size: int = 50
    rule_placement: str = "round_robin"
    island_reset_mode: str = "trim_top_k"
    island_reset_keep_top: int = 3
    islands: list[Island] = field(default_factory=list)
    _rr_counter: int = field(default=0, repr=False)

    def __post_init__(self):
        self.islands = [Island(id=i) for i in range(self.n_islands)]

    def add_rule(self, rule: Rule) -> None:
        """Add rule to an island per ``rule_placement`` (default: round-robin)."""
        if self.rule_placement == "min_best_score":
            island = min(self.islands, key=lambda i: i.best_score)
        elif self.rule_placement == "min_rule_count":
            island = min(self.islands, key=lambda i: len(i.rules))
        else:
            island = self.islands[self._rr_counter % self.n_islands]
            self._rr_counter += 1
        island.add(rule)

        # Trim island to max size (keep top performers)
        if len(island.rules) > self.max_island_size:
            island.rules = sorted(island.rules, key=rule_sort_key, reverse=True)[
                : self.max_island_size
            ]
            island.best_score = max((r.score for r in island.rules), default=0.0)

    def sample_prompt_rules(self, k: int = 2) -> list[Rule]:
        """Sample k rules across islands for prompt building (best-shot prompting)."""
        island = random.choice(self.islands)
        return island.sample(k=k, temperature=0.5)

    def get_best_overall(self) -> Rule:
        """Return the globally best rule (score, then corrections, then fewest easy overrides)."""
        all_rules = [r for island in self.islands for r in island.rules]
        return max(all_rules, key=rule_sort_key) if all_rules else None

    def reset_worst_island(self) -> None:
        """
        FunSearch-style island reset: kill the worst island,
        re-seed it with a copy of the best rule from the best island.
        """
        if len(self.islands) < 2:
            return
        best_island = max(self.islands, key=lambda i: i.best_score)
        worst_island = min(self.islands, key=lambda i: i.best_score)
        best_rule = best_island.get_top(1)
        if not best_rule:
            return
        if self.island_reset_mode == "clear_and_clone":
            worst_island.rules = []
            worst_island.best_score = 0.0
        else:
            # trim_top_k: keep best rules on the worst island, then re-seed with a clone
            sorted_rules = sorted(worst_island.rules, key=rule_sort_key, reverse=True)
            k = max(1, int(self.island_reset_keep_top))
            worst_island.rules = sorted_rules[:k]
            worst_island.best_score = (
                max((r.score for r in worst_island.rules), default=0.0)
            )
        clone = copy.deepcopy(best_rule[0])
        clone.id = f"{clone.id}_reset_{random.randint(1000,9999)}"
        worst_island.add(clone)

    def diversity_score(self) -> float:
        """Approximate population diversity: average pairwise code edit distance."""
        all_rules = [r for island in self.islands for r in island.rules]
        if len(all_rules) < 2:
            return 0.0
        codes = [r.code.encode() for r in all_rules]
        n = len(codes)
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = codes[i], codes[j]
                # Levenshtein-like: fraction of identical chars
                sim = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
                total += 1 - sim
        return total / (n * (n - 1) / 2) if n > 1 else 0.0


def population_from_flat_rules(
    rules: list[Rule],
    n_islands: int,
    max_island_size: int,
    rule_placement: str = "round_robin",
    island_reset_mode: str = "trim_top_k",
    island_reset_keep_top: int = 3,
) -> Population:
    """
    Rebuild island structure from a flat list (e.g. load_rules output).
    Each rule's island_id selects the island; invalid ids map to island 0.
    """
    pop = Population(
        n_islands=n_islands,
        max_island_size=max_island_size,
        rule_placement=rule_placement,
        island_reset_mode=island_reset_mode,
        island_reset_keep_top=island_reset_keep_top,
    )
    total = len(rules)
    pop._rr_counter = (total % n_islands) if n_islands else 0
    for r in rules:
        iid = int(r.island_id)
        if iid < 0 or iid >= n_islands:
            iid = 0
        r.island_id = iid
        pop.islands[iid].rules.append(r)
    for isl in pop.islands:
        if isl.rules:
            isl.best_score = max(r.score for r in isl.rules)
            if len(isl.rules) > max_island_size:
                isl.rules = sorted(isl.rules, key=rule_sort_key, reverse=True)[
                    :max_island_size
                ]
                isl.best_score = max((r.score for r in isl.rules), default=0.0)
        else:
            isl.best_score = 0.0
    return pop


# ─── Prompt building ─────────────────────────────────────────────────────────

RULE_SKELETON = textwrap.dedent("""\
    # -*- coding: utf-8 -*-
    '''
    3D-BPP Rule Patch — SVM Hard Case Corrector

    This function is called AFTER SVM prediction to correct hard cases.
    Parameters:
        svm_prob : float  — P(y=1) from SVM (already sigmoid of decision score)
        features : dict   — raw feature dict with all 41 feature values

    Returns:
        -1  : do not override SVM (rule not triggered)
        0   : override → predict NO (not feasible)
        1   : override → predict YES (feasible)

    IMPORTANT:
      - fill_ratio is a dimensionless fraction (often ~0–1): loaded volume / vehicle capacity.
        It is NOT cubic meters; use the Feature scales table in the prompt for thresholds.
      - Base your decision on physical 3D packing constraints:
          * sku_counts: number of items
          * sku_average_volume: avg item volume
          * fill_ratio = total_skuvolume / vehicle_capacity (same as fraction of capacity used)
          * sku_min_length/max_length, sku_min_width/max_width, sku_min_height/max_height
          * aspect_ratio (max_asr): longest / shortest dimension of any item
          * spare_capacity: remaining vehicle volume
          * sku_length_var, sku_width_var, sku_height_var: variance in item sizes
          * l_to_L_ratio_avg/min/max: SKU length / vehicle length ratio
          * etc.
      - Rules should only fire in specific feature ranges, not globally
      - Do NOT just return 0 or 1 unconditionally
      - Test your rule mentally on these patterns before writing code:
          * High fill_ratio + many items → likely infeasible
          * Low fill_ratio → likely feasible
          * Very high aspect ratio items → hard to pack
    '''
    import math

    def apply_rule_patch(svm_prob: float, features: dict) -> int:
        # ─── Your rule logic here ────────────────────────────────────────────
        sku_counts = features.get('sku_counts', 0)
        fill_ratio = features.get('fill_ratio', 0)
        max_asr    = features.get('max_asr', 1.0)
        spare_cap   = features.get('spare_capacity', 0)

        # Example skeleton (REPLACE with your evolved logic):
        # if fill_ratio > 0.95 and sku_counts > 10:
        #     return 0  # too many items for the remaining space
        return -1
    """)


def build_evolution_prompt(
    seed_rules: list[Rule],
    hard_case_md: str,
    skeleton: str = RULE_SKELETON,
    feature_stats_md: str = "",
    svm_insights_md: str = "",
    prev_iter_best_md: str = "",
) -> str:
    """
    Build a ReEvo-style evolution prompt:
      1. FN/FP analysis summary
      2. k seed rules (best-shot prompting)
      3. Skeleton for new rule
      4. ReEvo reflection instruction

    Parameters
    ----------
    seed_rules : list of Rule objects (k=1 or k=2)
    hard_case_md : Markdown string from render_hard_case_prompt()
    skeleton : the rule skeleton to include
    feature_stats_md : numeric min/max/median table for the current feature split
    svm_insights_md : linear SVM coef / intercept summary (scaled feature space)
    prev_iter_best_md : optional markdown block with last iteration's best candidate
    """
    lines = [
        "# 3D-BPP Rule Evolution Prompt",
        "",
        "## Problem: Improving SVM Hard Case Predictions",
        "",
        "The linear SVM achieves good overall accuracy but fails on specific hard cases:",
        "- FN (False Negative): SVM predicts NO but the order CAN be loaded (under-confident)",
        "- FP (False Positive): SVM predicts YES but the order CANNOT be loaded (over-confident)",
        "",
        "Your task: Design rule patches that correct these hard cases WITHOUT hurting easy cases.",
        "",
    ]

    if feature_stats_md.strip():
        lines += [
            "## Feature scales (test split — do not guess units)",
            "",
            "Use these statistics when choosing numeric thresholds. Do not confuse ratios with volumes.",
            "",
            feature_stats_md.strip(),
            "",
        ]

    if svm_insights_md.strip():
        lines += [
            "## Linear SVM weights (which features drive the baseline, scaled space)",
            "",
            "These coefficients are for **MinMax-scaled** inputs used inside the SVM. "
            "Use them to see which raw features matter most to the linear model and in which direction.",
            "",
            svm_insights_md.strip(),
            "",
        ]

    if prev_iter_best_md.strip():
        lines += [
            "## Previous iteration best candidate (refine or beat this)",
            "",
            prev_iter_best_md.strip(),
            "",
        ]

    lines += [
        "## Hard Case Analysis",
        "",
        hard_case_md,
        "",
        "## Seed Rules (best-performing so far)",
        "",
    ]

    for i, rule in enumerate(seed_rules, 1):
        lines.append(f"### Seed Rule {i} (score={rule.score:.4f})")
        lines.append("```python")
        lines.append(rule.code)
        lines.append("```")
        lines.append(f"- FN corrections: {rule.fn_corr}, FP corrections: {rule.fp_corr}")
        lines.append(f"- FN misses: {rule.fn_miss}, FP over: {rule.fp_over}")
        lines.append("")

    lines += [
        "## Your Task",
        "",
        "Study the above FN/FP patterns and seed rules. Then generate a NEW rule that:",
        "1. Corrects more FN cases (feels NO when should be YES)",
        "   OR corrects more FP cases (feels YES when should be NO)",
        "2. Does NOT override SVM on easy TN/TP cases (minimal false interventions)",
        "3. Has clear physical / geometric justification",
        "",
        "## Rule Skeleton",
        "",
        "```python",
        skeleton,
        "```",
        "",
        "## Output Format",
        "",
        "Return ONLY a Python code block with the `apply_rule_patch` function.",
        "Do not include any other text.",
    ]

    return "\n".join(lines)


def format_prev_iter_best_md(rule: Rule | None) -> str:
    """Markdown block for the previous iteration's best candidate (cross-iteration context)."""
    if rule is None:
        return ""
    return (
        f"- Rule id: `{rule.id}`\n"
        f"- score={rule.score:.6f}, fn_corr={rule.fn_corr}, fp_corr={rule.fp_corr}, "
        f"easy_overrides={rule.easy_overrides}\n\n"
        "```python\n"
        f"{rule.code}\n"
        "```\n"
    )


def build_reflection_prompt(
    parent_rule: Rule,
    new_rule: Rule,
    fn_cases: list,
    fp_cases: list,
) -> str:
    """
    ReEvo-style reflection prompt:
      LLM reflects on WHY parent_rule works and new_rule is better/worse,
      then extracts a "verbal gradient" — an insight about what to improve.
    """
    lines = [
        "## ReEvo Reflection: Understanding Rule Improvements",
        "",
        "### Parent Rule (previous best)",
        f"- Score: {parent_rule.score:.4f}",
        f"- FN corrections: {parent_rule.fn_corr}, misses: {parent_rule.fn_miss}",
        f"- FP corrections: {parent_rule.fp_corr}, over: {parent_rule.fp_over}",
        "```python",
        parent_rule.code,
        "```",
        "",
        "### New Candidate Rule",
        f"- Score: {new_rule.score:.4f}",
        f"- FN corrections: {new_rule.fn_corr}, misses: {new_rule.fn_miss}",
        f"- FP corrections: {new_rule.fp_corr}, over: {new_rule.fp_over}",
        "```python",
        new_rule.code,
        "```",
        "",
        "### Reflection Questions",
        "1. What specific FN cases does the new rule fix that the parent missed?",
        "2. What FP cases does it correctly reject?",
        "3. What easy cases does it wrongly override (false interventions)?",
        "4. What is the key physical insight driving the improvement?",
        "5. If you were to evolve this rule further, what threshold or condition would you tweak?",
        "",
        "Please respond with a concise verbal gradient (2-3 sentences) summarizing the key insight.",
    ]
    return "\n".join(lines)


# ─── Evolution driver ────────────────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """
    Configuration for the evolution loop.

    ``timeout_per_call`` is applied when building the LLM callable for **OpenAI-compatible**
    HTTP APIs (Bailian, OpenAI, DeepSeek). The DashScope ``Generation.call`` path may not
    honor it; see README.
    """

    n_iterations: int = 20
    n_islands: int = 4
    max_island_size: int = 50
    n_seed_rules_per_prompt: int = 2
    island_reset_every: int = 10      # reset worst island every N iterations
    llm_model: str = "qwen3-max-2026-01-23"  # Bailian/DashScope compatible when using that backend
    timeout_per_call: int = 60        # seconds
    llm_temperature: float = 0.7
    include_prev_iter_in_prompt: bool = True  # feed last iter best into next prompt
    save_hybrid_snapshot_each_iter: bool = True  # write hybrid iter MD each iteration (success or failure diag)
    save_failed_iter_response_chars: int = 1500  # max chars of LLM reply in failed-iter MD; 0 = omit
    save_failed_iter_jsonl_preview_chars: int = 400  # response snippet in evolution_progress.jsonl
    rule_placement: str = "round_robin"  # "round_robin" | "min_best_score" | "min_rule_count"
    island_reset_mode: str = "trim_top_k"  # "trim_top_k" | "clear_and_clone"
    island_reset_keep_top: int = 3  # when trim_top_k: keep this many best rules before cloning
    n_easy_typical: int = 3  # rows of easy TN/TP shown in evolution prompt (borderline-correct)
    n_near_hard_easy: int = 3  # easy TP/TN rows similar to FN/FP clouds; 0 disables §5


# Markdown code fences: ```python / ```py / ```Python3 … (case-insensitive)
PYTHON_FENCE_OPEN_RE = re.compile(r"```(?i:(?:python3?|py))\b\s*")


def _header_line_is_python_fence(header: str) -> bool:
    """Whether the first line after ``` is a Python-tagged fence (python, py, python3, …)."""
    h = (header or "").strip().lower()
    if not h:
        return False
    first = h.split()[0]
    return bool(re.match(r"(?:python3?|py)\b", first))


def _closed_python_fence_bodies(text: str) -> list[str]:
    """Bodies of closed ```python|py|… … ``` blocks (same rules as extraction)."""
    return re.findall(
        r"```(?i:(?:python3?|py))\b\s*(.*?)\s*```",
        text,
        re.DOTALL,
    )


def _extract_python_fence_open_to_close_or_eof(response: str) -> str | None:
    """
    After a Python-tagged fence, take content until the next ``` or end of string.
    """
    m = PYTHON_FENCE_OPEN_RE.search(response)
    if not m:
        return None
    rest = response[m.end() :]
    close = re.search(r"```", rest)
    chunk = rest[: close.start()] if close else rest
    return chunk.strip() or None


def _extract_non_python_fenced_blocks_with_apply(response: str) -> list[str]:
    """Fenced ``` blocks that are not Python-tagged but body contains apply_rule_patch."""
    out: list[str] = []
    for m in re.finditer(r"```([^\n]*)\n(.*?)```", response, re.DOTALL):
        header = m.group(1) or ""
        if _header_line_is_python_fence(header):
            continue
        body = m.group(2).strip()
        if "apply_rule_patch" in body:
            out.append(body)
    return out


def extract_code_from_response(response: str) -> list[str]:
    """
    Extract Python code defining ``apply_rule_patch`` from LLM response.

    Strategy 1: closed ```python|py|… … ``` blocks.
    Strategy 2: open fence to next ``` or EOF (unclosed / truncated).
    Strategy 3: non-python-tagged ``` ... ``` fences that still contain ``apply_rule_patch``.
    """
    if not response:
        return []
    matches = _closed_python_fence_bodies(response)
    out = [m.strip() for m in matches if "apply_rule_patch" in m]
    if out:
        return out

    chunk = _extract_python_fence_open_to_close_or_eof(response)
    if chunk and "apply_rule_patch" in chunk:
        return [chunk]

    for body in _extract_non_python_fenced_blocks_with_apply(response):
        if "def apply_rule_patch" in body or "apply_rule_patch" in body:
            return [body]

    return []


def _truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0 or not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n… [truncated]"


def build_extraction_failure_info(response: str | None) -> dict:
    """
    When extract_code_from_response returns [], explain why (for logging / MD).

    Keys: failure_kind ('no_extractable_code' | 'llm_exception'), message, detail, response_preview

    ``detail`` may be:
    ``no_python_fence``, ``has_open_python_fence_no_close``, ``python_blocks_without_apply_rule_patch``.
    """
    text = response if isinstance(response, str) else ""
    preview = _truncate_text(text, 8000)
    closed_python_blocks = _closed_python_fence_bodies(text)
    has_python_open = PYTHON_FENCE_OPEN_RE.search(text) is not None

    if closed_python_blocks:
        if not any("apply_rule_patch" in b for b in closed_python_blocks):
            return {
                "failure_kind": "no_extractable_code",
                "message": (
                    f"Found {len(closed_python_blocks)} closed ```python``` block(s) "
                    "but none define apply_rule_patch."
                ),
                "detail": "python_blocks_without_apply_rule_patch",
                "response_preview": preview,
            }
        return {
            "failure_kind": "no_extractable_code",
            "message": "Could not extract valid apply_rule_patch code (unexpected).",
            "detail": "unknown",
            "response_preview": preview,
        }

    if has_python_open:
        return {
            "failure_kind": "no_extractable_code",
            "message": (
                "Found a Python-tagged code fence (```python / ```py / …) but no closed block "
                "(unclosed fence or truncated response); could not extract apply_rule_patch."
            ),
            "detail": "has_open_python_fence_no_close",
            "response_preview": preview,
        }

    return {
        "failure_kind": "no_extractable_code",
        "message": "No Python-tagged fenced code block (```python, ```py, …) in LLM response.",
        "detail": "no_python_fence",
        "response_preview": preview,
    }


class EvolutionEngine:
    """
    Main evolution engine that drives the FunSearch/ReEvo loop.

    Usage:
        engine = EvolutionEngine(config)
        engine.seed_initial_rules(seed_code_list)
        engine.run(llm_callable, fn_cases, fp_cases, easy_tn, easy_tp, hard_case_md, ...)
        best = engine.population.get_best_overall()

        step() returns (Rule | None, failure_dict | None).
    """

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()
        self.population = Population(
            n_islands=self.config.n_islands,
            max_island_size=self.config.max_island_size,
            rule_placement=self.config.rule_placement,
            island_reset_mode=self.config.island_reset_mode,
            island_reset_keep_top=self.config.island_reset_keep_top,
        )
        self.iteration: int = 0
        self.history: list[Rule] = []
        self._seeded: bool = False
        self._prev_iter_best_rule: Rule | None = None

    def seed_initial_rules(self, seed_codes: list[str]) -> None:
        """Seed the population with initial rules (can be trivial or hand-designed)."""
        for i, code in enumerate(seed_codes):
            rid = f"seed_{i}"
            rule = Rule(id=rid, code=code)
            rule.score = 0.0
            self.population.add_rule(rule)
        self._seeded = True

    def _fit_rule(self, rule: Rule, fn_cases, fp_cases, easy_tn, easy_tp) -> Rule:
        rule.fit(
            fn_cases=fn_cases,
            fp_cases=fp_cases,
            easy_tn=easy_tn,
            easy_tp=easy_tp,
            default_pred=lambda c: c.svm_pred,
        )
        return rule

    def step(
        self,
        llm_callable: Callable[[str], str],
        fn_cases: list,
        fp_cases: list,
        easy_tn: list,
        easy_tp: list,
        hard_case_md: str,
        feature_stats_md: str = "",
        svm_insights_md: str = "",
    ) -> tuple[Rule | None, dict | None]:
        """
        One iteration of the evolution loop.

        Returns
        -------
        (rule, None) on success; (None, failure_info) on failure.
        failure_info keys: failure_kind, message, detail, response_preview
        """
        self.iteration += 1

        prev_md = ""
        if self.config.include_prev_iter_in_prompt and self._prev_iter_best_rule is not None:
            prev_md = format_prev_iter_best_md(self._prev_iter_best_rule)

        # ── 1. Sample seed rules (best-shot prompting) ─────────────────────
        seed_rules = self.population.sample_prompt_rules(
            k=self.config.n_seed_rules_per_prompt
        )
        if not seed_rules:
            # Fallback: use default skeleton
            seed_rules = [
                Rule(id="fallback", code=RULE_SKELETON)
            ]

        # ── 2. Build prompt ──────────────────────────────────────────────────
        prompt = build_evolution_prompt(
            seed_rules=seed_rules,
            hard_case_md=hard_case_md,
            skeleton=RULE_SKELETON,
            feature_stats_md=feature_stats_md,
            svm_insights_md=svm_insights_md,
            prev_iter_best_md=prev_md,
        )

        # ── 3. Call LLM ─────────────────────────────────────────────────────
        response: str | None = None
        try:
            response = llm_callable(prompt)
        except Exception as e:
            print(f"[Evolution step {self.iteration}] LLM call failed: {e}", flush=True)
            return None, {
                "failure_kind": "llm_exception",
                "message": f"{type(e).__name__}: {e}",
                "detail": type(e).__name__,
                "response_preview": "",
            }

        if not isinstance(response, str):
            response = "" if response is None else str(response)

        # ── 4. Extract code candidates ──────────────────────────────────────
        code_blocks = extract_code_from_response(response)
        if not code_blocks:
            print(
                f"[Evolution step {self.iteration}] No valid code found in LLM response",
                flush=True,
            )
            return None, build_extraction_failure_info(response)

        # ── 5. Evaluate candidates ─────────────────────────────────────────
        best_rule_in_iter: Rule | None = None
        for i, code in enumerate(code_blocks):
            rid = f"iter{self.iteration}_cand{i}_{hashlib.md5(code.encode()).hexdigest()[:6]}"
            rule = Rule(id=rid, code=code)
            self._fit_rule(rule, fn_cases, fp_cases, easy_tn, easy_tp)
            self.history.append(rule)
            self.population.add_rule(rule)

            if best_rule_in_iter is None or rule_sort_key(rule) > rule_sort_key(best_rule_in_iter):
                best_rule_in_iter = rule

        # ── 6. Island reset (FunSearch style) ───────────────────────────────
        if self.iteration % self.config.island_reset_every == 0:
            self.population.reset_worst_island()

        top = self.population.get_best_overall()
        diversity = self.population.diversity_score()
        print(
            f"[Iter {self.iteration:3d}] "
            f"Best score={top.score if top else 0:.4f}  "
            f"This iter best={best_rule_in_iter.score if best_rule_in_iter else 0:.4f}  "
            f"Diversity={diversity:.4f}",
            flush=True,
        )

        self._prev_iter_best_rule = best_rule_in_iter
        return best_rule_in_iter, None

    def _write_hybrid_iter_snapshot(
        self,
        snapshot_dir: Path,
        snapshot_ctx: tuple,
    ) -> None:
        """Write hybrid hard-case summary for current population (after a successful step)."""
        X_test_df, y_test, svm_probs_test, dispatch_ids_test, top_k = snapshot_ctx
        all_rules = [r for island in self.population.islands for r in island.rules]
        if not all_rules:
            return
        hybrid_pred = apply_rules(
            all_rules,
            X_test_df,
            svm_probs_test,
            top_k=top_k,
        )
        fn_h, fp_h, _, _ = mine_hard_cases(
            X_test_df,
            y_test,
            svm_probs_test,
            dispatch_ids=dispatch_ids_test,
            y_pred=hybrid_pred,
        )
        summary_h = summarize_hard_cases(
            fn_h,
            fp_h,
            y_test,
            list(X_test_df.columns),
        )
        md_h = render_hard_case_prompt(
            summary_h,
            analysis_title=f"## Hybrid after iteration {self.iteration}",
            predictor_name="Hybrid (SVM + rules)",
            prob_column_caption="SVM P(y=1) (reference)",
            include_llm_instruction=False,
        )
        out = snapshot_dir / f"hard_case_summary_hybrid_iter_{self.iteration:03d}.md"
        out.write_text(md_h, encoding="utf-8")

    def _write_failed_hybrid_iter_snapshot(self, snapshot_dir: Path, failure_info: dict) -> None:
        """Diagnostic markdown when step() did not add a rule (same filename pattern as success)."""
        max_c = self.config.save_failed_iter_response_chars
        raw = failure_info.get("response_preview") or ""
        body_preview = _truncate_text(raw, max_c) if max_c > 0 else ""

        lines = [
            f"## Hybrid iteration {self.iteration} — **FAILED** (no new rule added)",
            "",
            "This file records why this evolution step did not produce parseable "
            "`apply_rule_patch` code. It is **not** a Hybrid hard-case mining summary.",
            "",
            f"- **failure_kind**: `{failure_info.get('failure_kind', '?')}`",
            f"- **detail**: `{failure_info.get('detail', '')}`",
            f"- **message**: {failure_info.get('message', '')}",
            "",
        ]
        if body_preview:
            lines.extend(
                [
                    "### LLM response (truncated)",
                    "",
                    "```text",
                    body_preview,
                    "```",
                    "",
                ]
            )

        out = snapshot_dir / f"hard_case_summary_hybrid_iter_{self.iteration:03d}.md"
        out.write_text("\n".join(lines), encoding="utf-8")

    def run(
        self,
        llm_callable: Callable[[str], str],
        fn_cases: list,
        fp_cases: list,
        easy_tn: list,
        easy_tp: list,
        hard_case_md: str,
        n_iterations: int | None = None,
        feature_stats_md: str = "",
        svm_insights_md: str = "",
        snapshot_dir: Path | None = None,
        snapshot_ctx: tuple | None = None,
        warm_start: bool = False,
        checkpoint_path: Path | None = None,
        checkpoint_meta: dict | None = None,
    ) -> Rule:
        """
        Run the full evolution loop for n_iterations.

        Parameters
        ----------
        warm_start : If True, do not clear _prev_iter_best_rule; seed it from
            current population best (used when resuming from saved rules).
        checkpoint_path / checkpoint_meta : If set, write evolution_checkpoint.json
            after each iteration (meta merged with current iteration index).
        """
        if not warm_start:
            self._prev_iter_best_rule = None
        elif self.population.get_best_overall() is not None:
            self._prev_iter_best_rule = self.population.get_best_overall()

        n = n_iterations or self.config.n_iterations
        jc = self.config.save_failed_iter_jsonl_preview_chars
        for _ in range(n):
            br, fail_info = self.step(
                llm_callable=llm_callable,
                fn_cases=fn_cases,
                fp_cases=fp_cases,
                easy_tn=easy_tn,
                easy_tp=easy_tp,
                hard_case_md=hard_case_md,
                feature_stats_md=feature_stats_md,
                svm_insights_md=svm_insights_md,
            )
            if (
                self.config.save_hybrid_snapshot_each_iter
                and snapshot_dir is not None
                and snapshot_ctx is not None
            ):
                if br is not None:
                    self._write_hybrid_iter_snapshot(snapshot_dir, snapshot_ctx)
                elif fail_info is not None:
                    self._write_failed_hybrid_iter_snapshot(snapshot_dir, fail_info)
            if checkpoint_path is not None and checkpoint_meta is not None:
                payload = {**checkpoint_meta, "iteration": self.iteration}
                checkpoint_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if snapshot_dir is not None:
                top = self.population.get_best_overall()
                rp_raw = (fail_info or {}).get("response_preview") or ""
                rp_json = (
                    _truncate_text(rp_raw, jc) if (fail_info and jc > 0 and rp_raw) else None
                )
                prog = {
                    "iteration": self.iteration,
                    "step_ok": br is not None,
                    "best_score": float(top.score) if top else None,
                    "n_rules": sum(len(isl.rules) for isl in self.population.islands),
                    "failure_kind": fail_info.get("failure_kind") if fail_info else None,
                    "failure_message": fail_info.get("message") if fail_info else None,
                    "failure_detail": fail_info.get("detail") if fail_info else None,
                    "response_preview": rp_json,
                }
                prog_path = snapshot_dir / "evolution_progress.jsonl"
                with prog_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(prog, ensure_ascii=False) + "\n")
        return self.population.get_best_overall()


# ─── Serialization ───────────────────────────────────────────────────────────

def save_rules(rules: list[Rule], path: pathlib.Path) -> None:
    data = []
    for r in rules:
        data.append({
            "id": r.id,
            "code": r.code,
            "fn_corr": r.fn_corr,
            "fp_corr": r.fp_corr,
            "fn_miss": r.fn_miss,
            "fp_over": r.fp_over,
            "score": r.score,
            "fitness_history": r.fitness_history,
            "island_id": r.island_id,
            "easy_overrides": r.easy_overrides,
        })
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def load_rules(path: pathlib.Path) -> list[Rule]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = []
    for item in data:
        r = Rule(
            id=item["id"],
            code=item["code"],
            fn_corr=item["fn_corr"],
            fp_corr=item["fp_corr"],
            fn_miss=item["fn_miss"],
            fp_over=item["fp_over"],
            score=item["score"],
            fitness_history=item.get("fitness_history", []),
            island_id=item.get("island_id", 0),
            easy_overrides=int(item.get("easy_overrides", 0)),
        )
        rules.append(r)
    return rules


def apply_rules(
    rules: list[Rule],
    X_df: pd.DataFrame,
    svm_probs: np.ndarray,
    top_k: int = 3,
) -> np.ndarray:
    """
    Apply top-k evolved rules to override SVM predictions.

    The ensemble applies rules in descending quality order (score, then corrections, then fewest easy overrides).
    A rule can:
      - Override to 0 (FP correction)
      - Override to 1 (FN correction)
      - Return -1 (pass through to next rule or SVM)
    """
    top_rules = sorted(rules, key=rule_sort_key, reverse=True)[:top_k]

    apply_fns = []
    for rule in top_rules:
        fn = rule.get_apply_fn()
        if fn is not None:
            apply_fns.append((rule, fn))

    if not apply_fns:
        # Fallback: return SVM predictions
        return (svm_probs >= 0.5).astype(int)

    final_pred = (svm_probs >= 0.5).astype(int)

    for i in range(len(X_df)):
        feat = X_df.iloc[i].to_dict()
        p = svm_probs[i]

        for rule, apply_fn in apply_fns:
            try:
                override = apply_fn(p, feat)
                if override != -1:
                    final_pred[i] = override
                    break
            except Exception:
                continue

    return final_pred

# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic task graders producing scores in [0.0, 1.0]."""

from __future__ import annotations

from typing import Any, Mapping


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_str(s: str) -> str:
    return s.strip().lower()


def _norm_tid(t: str) -> str:
    t = t.strip().upper()
    if t.startswith("T") and len(t) >= 2:
        return t
    return t


def _f1_sets(pred: set[str], gold: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not pred and gold:
        return 0.0
    if not gold and pred:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return 2.0 * p * r / (p + r)


def grade_easy(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    """Verdict + optional primary technique + consistency."""
    gv = _norm_str(str(gold["gold_verdict"]))
    sv = _norm_str(str(submission.get("verdict", "")))
    if sv not in ("true_positive", "false_positive", "benign"):
        return 0.0

    verdict_score = 1.0 if sv == gv else 0.0

    gold_primary = gold.get("gold_primary_technique")
    pred_primary = submission.get("primary_technique")
    if gold_primary is None or str(gold_primary).strip() == "":
        tech_ok = pred_primary is None or str(pred_primary).strip() == ""
    else:
        tech_ok = _norm_tid(str(pred_primary or "")) == _norm_tid(str(gold_primary))
    technique_score = 1.0 if tech_ok else 0.0

    gold_set = {_norm_tid(x) for x in gold.get("gold_technique_ids", [])}
    pred_set = {_norm_tid(x) for x in submission.get("technique_ids", [])}
    set_score = 1.0 if pred_set == gold_set else 0.5 if pred_set <= gold_set else 0.0

    contradiction = (sv in ("benign", "false_positive")) and len(pred_set) > 0
    if contradiction:
        verdict_score *= 0.3

    return _clamp(0.55 * verdict_score + 0.25 * technique_score + 0.20 * set_score)


def grade_medium(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    """Correlation fields + verdict + technique alignment."""
    corr = submission.get("correlation") or {}
    fields = [
        (_norm_str(str(corr.get("service_a", ""))), _norm_str(str(gold["gold_service_a"]))),
        (_norm_str(str(corr.get("service_b", ""))), _norm_str(str(gold["gold_service_b"]))),
        (_norm_str(str(corr.get("link_key", ""))), _norm_str(str(gold["gold_link_key"]))),
        (_norm_str(str(corr.get("link_value", ""))), _norm_str(str(gold["gold_link_value"]))),
    ]
    partial = sum(1.0 for a, b in fields if a == b) / 4.0

    gv = _norm_str(str(gold["gold_verdict"]))
    sv = _norm_str(str(submission.get("verdict", "")))
    verdict_score = 1.0 if sv == gv else 0.0

    gold_set = {_norm_tid(x) for x in gold.get("gold_technique_ids", [])}
    pred_set = {_norm_tid(x) for x in submission.get("technique_ids", [])}
    f1 = _f1_sets(pred_set, gold_set)

    fp_penalty = 1.0
    if gv in ("benign", "false_positive") and sv == "true_positive":
        fp_penalty = 0.4

    return _clamp((0.45 * partial + 0.25 * verdict_score + 0.30 * f1) * fp_penalty)


def grade_hard(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    """Noisy APT: technique F1, ordered remediation, anti-hallucination guard."""
    gv = _norm_str(str(gold["gold_verdict"]))
    sv = _norm_str(str(submission.get("verdict", "")))
    verdict_score = 1.0 if sv == gv else 0.0

    gold_set = {_norm_tid(x) for x in gold.get("gold_technique_ids", [])}
    pred_set = {_norm_tid(x) for x in submission.get("technique_ids", [])}
    f1 = _f1_sets(pred_set, gold_set)

    gold_steps = [_norm_str(x) for x in gold.get("gold_remediation", [])]
    pred_steps = [_norm_str(x) for x in submission.get("remediation_steps", [])]

    prefix = 0
    for i, g in enumerate(gold_steps):
        if i < len(pred_steps) and pred_steps[i] == g:
            prefix += 1
        else:
            break
    order_score = prefix / len(gold_steps) if gold_steps else 1.0

    noise_penalty = 1.0
    if len(pred_set) > len(gold_set) + 4:
        noise_penalty *= 0.55
    if sv == "true_positive" and len(pred_set) >= 12:
        noise_penalty *= 0.5

    base = 0.30 * verdict_score + 0.45 * f1 + 0.25 * order_score
    return _clamp(base * noise_penalty)

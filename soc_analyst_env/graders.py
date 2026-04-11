"""Deterministic task graders — scores strictly in (0, 1) exclusive."""

from typing import Any, Mapping


def grade_identify_malicious_ip(
    gold: Mapping[str, Any], submission: Mapping[str, Any]
) -> float:
    try:
        expected = str(gold.get("gold_malicious_ip", "")).strip()
        actual = str(
            (submission.get("metadata") or {}).get("malicious_ip", "")
        ).strip()
        raw_score = 0.99 if (expected and expected == actual) else 0.01
        final_score = max(0.01, min(0.99, float(raw_score)))
        return final_score
    except Exception:
        return 0.01


def grade_find_compromised_account(
    gold: Mapping[str, Any], submission: Mapping[str, Any]
) -> float:
    try:
        expected = str(gold.get("gold_compromised_account", "")).strip().lower()
        actual = str(
            (submission.get("metadata") or {}).get("compromised_account", "")
        ).strip().lower()
        raw_score = 0.99 if (expected and expected == actual) else 0.01
        final_score = max(0.01, min(0.99, float(raw_score)))
        return final_score
    except Exception:
        return 0.01


def grade_recommend_firewall_rule(
    gold: Mapping[str, Any], submission: Mapping[str, Any]
) -> float:
    try:
        expected_rules = gold.get("gold_firewall_rules", [])
        if not expected_rules or not isinstance(expected_rules, list):
            return 0.01
        remediation = submission.get("remediation_steps", [])
        if isinstance(remediation, list):
            joined = " ".join(str(s).lower().replace("-", "_") for s in remediation)
        else:
            joined = str(remediation).lower().replace("-", "_")
        norm_rules = [str(r).strip().lower().replace("-", "_") for r in expected_rules if str(r).strip()]
        if not norm_rules:
            return 0.01
        hits = sum(1 for r in norm_rules if r in joined)
        raw_score = float(hits) / float(len(norm_rules))
        final_score = max(0.01, min(0.99, raw_score))
        return final_score
    except Exception:
        return 0.01


grade_easy = grade_identify_malicious_ip
grade_medium = grade_find_compromised_account
grade_hard = grade_recommend_firewall_rule

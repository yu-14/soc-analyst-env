from typing import Mapping, Any

_FLOOR = 0.01
_CEIL = 0.99


def _clamp(x: float) -> float:
    return max(_FLOOR, min(_CEIL, x))


def grade_identify_malicious_ip(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    try:
        expected = str(gold.get("gold_malicious_ip", ""))
        actual = str(submission.get("metadata", {}).get("malicious_ip", ""))
        if expected and expected == actual:
            return _CEIL
        return _FLOOR
    except Exception:
        return _FLOOR


def grade_find_compromised_account(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    try:
        expected = str(gold.get("gold_compromised_account", ""))
        actual = str(submission.get("metadata", {}).get("compromised_account", ""))
        if expected and expected == actual:
            return _CEIL
        return _FLOOR
    except Exception:
        return _FLOOR


def grade_recommend_firewall_rule(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    try:
        expected_rules = gold.get("gold_firewall_rules", [])
        actual_remediation = str(submission.get("remediation_steps", ""))
        if not expected_rules:
            return _FLOOR
        matches = 0
        for rule in expected_rules:
            if str(rule).lower() in actual_remediation.lower():
                matches += 1
        return _clamp(float(matches) / len(expected_rules))
    except Exception:
        return _FLOOR

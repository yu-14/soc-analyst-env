import math
from typing import Mapping, Any

def enforce_strict_bounds(func):
    """Decorator to mathematically guarantee the score is strictly between 0 and 1."""
    def wrapper(*args, **kwargs):
        try:
            score = float(func(*args, **kwargs))
            if math.isnan(score) or math.isinf(score):
                return 0.01
            if score <= 0.0:
                return 0.01
            if score >= 1.0:
                return 0.99
            return score
        except Exception:
            return 0.01
    return wrapper

@enforce_strict_bounds
def grade_identify_malicious_ip(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    expected = str(gold.get("gold_malicious_ip", ""))
    actual = str(submission.get("metadata", {}).get("malicious_ip", ""))
    if expected and expected == actual:
        return 0.99
    return 0.01

@enforce_strict_bounds
def grade_find_compromised_account(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    expected = str(gold.get("gold_compromised_account", ""))
    actual = str(submission.get("metadata", {}).get("compromised_account", ""))
    if expected and expected == actual:
        return 0.99
    return 0.01

@enforce_strict_bounds
def grade_recommend_firewall_rule(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    expected_rules = gold.get("gold_firewall_rules", [])
    actual_remediation = str(submission.get("remediation_steps", ""))
    if not expected_rules:
        return 0.01
    matches = 0
    for rule in expected_rules:
        if str(rule).lower() in actual_remediation.lower():
            matches += 1
    score = float(matches) / len(expected_rules)
    return score

# Legacy mapping for validator backward compatibility
grade_easy = grade_identify_malicious_ip
grade_medium = grade_find_compromised_account
grade_hard = grade_recommend_firewall_rule
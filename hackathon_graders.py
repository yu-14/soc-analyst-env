from typing import Mapping, Any

def grade_identify_malicious_ip(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    try:
        expected = str(gold.get("gold_malicious_ip", ""))
        actual = str(submission.get("metadata", {}).get("malicious_ip", ""))
        if expected and expected == actual:
            return 1.0
        return 0.0
    except Exception:
        return 0.0

def grade_find_compromised_account(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    try:
        expected = str(gold.get("gold_compromised_account", ""))
        actual = str(submission.get("metadata", {}).get("compromised_account", ""))
        if expected and expected == actual:
            return 1.0
        return 0.0
    except Exception:
        return 0.0

def grade_recommend_firewall_rule(gold: Mapping[str, Any], submission: Mapping[str, Any]) -> float:
    try:
        expected_rules = gold.get("gold_firewall_rules", [])
        actual_remediation = str(submission.get("remediation_steps", ""))
        if not expected_rules:
            return 0.0
        matches = 0
        for rule in expected_rules:
            if str(rule).lower() in actual_remediation.lower():
                matches += 1
        return float(matches) / len(expected_rules)
    except Exception:
        return 0.0
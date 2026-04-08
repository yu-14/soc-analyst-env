# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
Registered SOC analyst tasks for OpenEnv / hackathon graders.

Each task has a stable id, human-readable title, data file stem, and grader function name.
Legacy aliases: easy -> identify_malicious_ip, medium -> find_compromised_account,
hard -> recommend_firewall_rule.
"""

from __future__ import annotations

from typing import Any, Callable, Final, Mapping

try:
    from .graders import (
        grade_find_compromised_account,
        grade_identify_malicious_ip,
        grade_recommend_firewall_rule,
    )
except ImportError:
    from graders import (
        grade_find_compromised_account,
        grade_identify_malicious_ip,
        grade_recommend_firewall_rule,
    )

GraderFn = Callable[[Mapping[str, Any], Mapping[str, Any]], float]

# Canonical task ids (use these in reset(task=...) for explicit naming)
TASK_IDENTIFY_MALICIOUS_IP: Final = "identify_malicious_ip"
TASK_FIND_COMPROMISED_ACCOUNT: Final = "find_compromised_account"
TASK_RECOMMEND_FIREWALL_RULE: Final = "recommend_firewall_rule"

CANONICAL_TASK_IDS: Final[tuple[str, ...]] = (
    TASK_IDENTIFY_MALICIOUS_IP,
    TASK_FIND_COMPROMISED_ACCOUNT,
    TASK_RECOMMEND_FIREWALL_RULE,
)

TASK_ALIASES: Final[dict[str, str]] = {
    "easy": TASK_IDENTIFY_MALICIOUS_IP,
    "medium": TASK_FIND_COMPROMISED_ACCOUNT,
    "hard": TASK_RECOMMEND_FIREWALL_RULE,
}

GRADERS_BY_TASK_ID: Final[dict[str, GraderFn]] = {
    TASK_IDENTIFY_MALICIOUS_IP: grade_identify_malicious_ip,
    TASK_FIND_COMPROMISED_ACCOUNT: grade_find_compromised_account,
    TASK_RECOMMEND_FIREWALL_RULE: grade_recommend_firewall_rule,
}

SOC_ANALYST_TASKS: Final[list[dict[str, str]]] = [
    {
        "id": TASK_IDENTIFY_MALICIOUS_IP,
        "title": "Task 1: Identify the malicious source IP",
        "grader": "grade_identify_malicious_ip",
        "data_file": f"{TASK_IDENTIFY_MALICIOUS_IP}.json",
        "description": (
            "From SSH/auth logs, determine verdict and the attacking IP. "
            "Pass malicious_ip in action.metadata on finalize_triage."
        ),
    },
    {
        "id": TASK_FIND_COMPROMISED_ACCOUNT,
        "title": "Task 2: Find the compromised user account",
        "grader": "grade_find_compromised_account",
        "data_file": f"{TASK_FIND_COMPROMISED_ACCOUNT}.json",
        "description": (
            "Correlate services via trace_id and name the compromised account used on "
            "suspicious requests. Pass compromised_account in action.metadata on finalize."
        ),
    },
    {
        "id": TASK_RECOMMEND_FIREWALL_RULE,
        "title": "Task 3: Recommend firewall / WAF-style controls",
        "grader": "grade_recommend_firewall_rule",
        "data_file": f"{TASK_RECOMMEND_FIREWALL_RULE}.json",
        "description": (
            "Noisy enterprise logs: multi-stage intrusion, MITRE mapping, ordered remediation, "
            "and firewall rule ids that block C2 / isolate / restore (see gold_firewall_rules)."
        ),
    },
]


def resolve_task_id(raw: str) -> str:
    """Map legacy easy/medium/hard (and case variants) to canonical ids."""
    key = (raw or "easy").strip().lower()
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    if key in GRADERS_BY_TASK_ID:
        return key
    return TASK_IDENTIFY_MALICIOUS_IP


def get_grader_for_task(task_id: str) -> GraderFn:
    canonical = resolve_task_id(task_id)
    return GRADERS_BY_TASK_ID[canonical]

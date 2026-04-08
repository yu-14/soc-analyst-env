# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""Pydantic models for the SOC analyst OpenEnv environment."""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


class SocReward(BaseModel):
    """Structured reward breakdown (also surfaced via observation metadata)."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    total: float = Field(..., description="Scalar reward for this transition")
    components: dict[str, float] = Field(default_factory=dict)


class SocAction(Action):
    """Agent action: incremental investigation or final triage submission."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    kind: Literal[
        "noop",
        "submit_hypothesis",
        "submit_correlation",
        "finalize_triage",
        "destructive_network_block",
    ] = Field(..., description="Action type")

    technique_ids: list[str] = Field(default_factory=list)
    rationale_tag: str = ""

    service_a: str = ""
    service_b: str = ""
    link_key: str = ""
    link_value: str = ""

    verdict: Optional[Literal["true_positive", "false_positive", "benign"]] = None
    primary_technique: Optional[str] = None
    remediation_steps: list[str] = Field(default_factory=list)

    correlation_service_a: str = ""
    correlation_service_b: str = ""
    correlation_link_key: str = ""
    correlation_link_value: str = ""

    @model_validator(mode="after")
    def _validate_finalize(self) -> SocAction:
        if self.kind == "finalize_triage" and self.verdict is None:
            raise ValueError("finalize_triage requires verdict")
        return self


class SocObservation(Observation):
    """What the analyst sees: alert context and bounded log window."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    task: str = Field(
        ...,
        description=(
            "Task id: identify_malicious_ip | find_compromised_account | "
            "recommend_firewall_rule (or legacy aliases easy, medium, hard)"
        ),
    )
    instruction: str = Field(..., description="Task prompt without hidden answers")
    alert_id: str = ""
    alert_rule: str = ""
    alert_severity: str = ""
    log_view: str = Field(..., description="Newline-separated synthetic log lines")
    max_steps: int = Field(default=30, ge=1)
    available_commands: list[str] = Field(default_factory=list)
    feedback: str = Field(default="", description="Short environment feedback")
    final_grader_score: Optional[float] = Field(
        default=None,
        description="Raw grader score in [0,1] after finalize (not shaped)",
    )
    episode_success: bool = Field(
        default=False,
        description="True when final_grader_score >= 0.85 at episode end",
    )


class SocState(State):
    """Server-side session state (serializable)."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    task: str = Field(default="identify_malicious_ip")
    shaped_score: float = Field(default=0.0)
    episode_complete: bool = Field(default=False)
    last_total_reward: float = Field(default=0.0)

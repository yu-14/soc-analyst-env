# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""WebSocket client for SocAnalystEnvironment."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SocAction, SocObservation, SocState


class SocAnalystEnv(EnvClient[SocAction, SocObservation, SocState]):
    """Typed client for the SOC analyst environment."""

    def _step_payload(self, action: SocAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SocObservation]:
        obs_data = payload.get("observation", {})
        observation = SocObservation(
            task=obs_data.get("task", "easy"),
            instruction=obs_data.get("instruction", ""),
            alert_id=obs_data.get("alert_id", ""),
            alert_rule=obs_data.get("alert_rule", ""),
            alert_severity=obs_data.get("alert_severity", ""),
            log_view=obs_data.get("log_view", ""),
            max_steps=obs_data.get("max_steps", 30),
            available_commands=obs_data.get("available_commands", []),
            feedback=obs_data.get("feedback", ""),
            final_grader_score=obs_data.get("final_grader_score"),
            episode_success=bool(obs_data.get("episode_success", False)),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SocState:
        return SocState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            shaped_score=float(payload.get("shaped_score", 0.0)),
            episode_complete=bool(payload.get("episode_complete", False)),
            last_total_reward=float(payload.get("last_total_reward", 0.0)),
        )

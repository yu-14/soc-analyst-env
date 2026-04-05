# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""SOC analyst environment: synthetic logs, shaped rewards, deterministic graders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional, cast
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from graders import grade_easy, grade_hard, grade_medium
    from models import SocAction, SocObservation, SocReward, SocState
except ImportError:
    from ..graders import grade_easy, grade_hard, grade_medium
    from ..models import SocAction, SocObservation, SocReward, SocState

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_COMMANDS = [
    "noop",
    "submit_hypothesis",
    "submit_correlation",
    "finalize_triage",
    "destructive_network_block",
]


def _load_gold(task: str) -> dict[str, Any]:
    path = _DATA_DIR / f"{task}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _correlation_from_action(action: SocAction) -> dict[str, str]:
    return {
        "service_a": action.service_a.strip(),
        "service_b": action.service_b.strip(),
        "link_key": action.link_key.strip(),
        "link_value": action.link_value.strip(),
    }


def _correlation_partial(gold: dict[str, Any], corr: dict[str, str]) -> float:
    if "gold_service_a" not in gold:
        return 0.0
    fields = [
        (corr.get("service_a", "").lower(), str(gold["gold_service_a"]).lower()),
        (corr.get("service_b", "").lower(), str(gold["gold_service_b"]).lower()),
        (corr.get("link_key", "").lower(), str(gold["gold_link_key"]).lower()),
        (corr.get("link_value", "").lower(), str(gold["gold_link_value"]).lower()),
    ]
    return sum(1.0 for a, b in fields if a == b) / 4.0


class SocAnalystEnvironment(Environment[SocAction, SocObservation, SocState]):
    """Synthetic alert triage with easy / medium / hard scenarios."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state: SocState = SocState(episode_id=str(uuid4()), step_count=0)
        self._gold: dict[str, Any] = {}
        self._task: Literal["easy", "medium", "hard"] = "easy"
        self._shaped: float = 0.5
        self._last_sig: Optional[str] = None
        self._last_correlation: Optional[dict[str, str]] = None
        self._best_correlation_partial: float = -1.0
        self._final_score: Optional[float] = None

    def _sig(self, action: SocAction) -> str:
        payload = action.model_dump(exclude={"metadata"})
        return json.dumps(payload, sort_keys=True)

    def _overlap_bonus(self, predicted: list[str]) -> float:
        gold = {t.strip().upper() for t in self._gold.get("gold_technique_ids", [])}
        if not gold:
            return 0.0
        pred = {t.strip().upper() for t in predicted}
        inter = len(gold & pred)
        return 0.12 * (inter / max(len(gold), 1))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SocObservation:
        self._reset_rubric()
        raw_task = kwargs.get("task", "easy")
        if raw_task not in ("easy", "medium", "hard"):
            self._task = "easy"
        else:
            self._task = cast(Literal["easy", "medium", "hard"], raw_task)
        self._gold = _load_gold(self._task)
        self._state = SocState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=self._task,
            shaped_score=0.5,
            episode_complete=False,
            last_total_reward=0.0,
        )
        self._shaped = 0.5
        self._last_sig = None
        self._last_correlation = None
        self._best_correlation_partial = -1.0
        self._final_score = None

        alert = self._gold.get("alert", {})
        logs = "\n".join(self._gold.get("log_lines", []))
        return SocObservation(
            task=self._task,
            instruction=str(self._gold.get("instruction", "")),
            alert_id=str(alert.get("id", "")),
            alert_rule=str(alert.get("rule", "")),
            alert_severity=str(alert.get("severity", "")),
            log_view=logs,
            max_steps=int(self._gold.get("max_steps", 30)),
            available_commands=list(_COMMANDS),
            feedback="Episode started. Investigate logs; call finalize_triage when ready.",
            done=False,
            reward=0.0,
            final_grader_score=None,
            episode_success=False,
            metadata={"seed": seed, "episode_id": self._state.episode_id},
        )

    def _build_submission(self, action: SocAction) -> dict[str, Any]:
        corr: dict[str, str] = {}
        if any(
            [
                action.correlation_service_a,
                action.correlation_service_b,
                action.correlation_link_key,
                action.correlation_link_value,
            ]
        ):
            corr = {
                "service_a": action.correlation_service_a.strip(),
                "service_b": action.correlation_service_b.strip(),
                "link_key": action.correlation_link_key.strip(),
                "link_value": action.correlation_link_value.strip(),
            }
        elif any([action.service_a, action.service_b, action.link_key, action.link_value]):
            corr = _correlation_from_action(action)
        elif self._last_correlation:
            corr = dict(self._last_correlation)

        return {
            "verdict": action.verdict,
            "primary_technique": action.primary_technique,
            "technique_ids": list(action.technique_ids),
            "remediation_steps": list(action.remediation_steps),
            "correlation": corr,
        }

    def _grade(self, submission: dict[str, Any]) -> float:
        if self._task == "easy":
            return grade_easy(self._gold, submission)
        if self._task == "medium":
            return grade_medium(self._gold, submission)
        return grade_hard(self._gold, submission)

    def step(
        self,
        action: SocAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SocObservation:
        del timeout_s, kwargs
        if self._state.episode_complete:
            mx = int(self._gold.get("max_steps", 30)) if self._gold else 30
            return SocObservation(
                task=self._task,
                instruction=str(self._gold.get("instruction", "")) if self._gold else "",
                log_view="",
                feedback="Episode already finished.",
                done=True,
                reward=0.0,
                max_steps=mx,
                final_grader_score=self._final_score,
                episode_success=bool((self._final_score or 0.0) >= 0.85),
                metadata={"error": "episode_complete"},
            )

        self._state.step_count += 1
        sig = self._sig(action)
        loop_penalty = 0.0
        if sig == self._last_sig and action.kind != "noop":
            loop_penalty = 0.08
        self._last_sig = sig

        reward_info = SocReward(total=0.0, components={})
        feedback = ""
        done = False
        terminal_score: Optional[float] = None

        max_steps = int(self._gold.get("max_steps", 30))
        over_limit = self._state.step_count >= max_steps

        if action.kind == "noop":
            self._shaped = max(0.0, self._shaped - 0.03)
            reward_info = SocReward(total=-0.02, components={"noop_cost": -0.02})
            feedback = "No-op recorded (small penalty)."

        elif action.kind == "destructive_network_block":
            self._shaped = max(0.0, self._shaped - 0.35)
            reward_info = SocReward(
                total=-0.45,
                components={"destructive": -0.45},
            )
            feedback = "Destructive wide block — high operational risk."

        elif action.kind == "submit_hypothesis":
            bonus = self._overlap_bonus(action.technique_ids)
            self._shaped = min(1.0, self._shaped + bonus - loop_penalty)
            reward_info = SocReward(
                total=0.04 + bonus - loop_penalty,
                components={"hypothesis_overlap": bonus, "loop": -loop_penalty},
            )
            feedback = "Hypothesis noted; overlap with ground truth partially credited."

        elif action.kind == "submit_correlation":
            corr = _correlation_from_action(action)
            self._last_correlation = corr
            partial = _correlation_partial(self._gold, corr)
            if partial > self._best_correlation_partial:
                self._best_correlation_partial = partial
            delta = 0.06 * partial - loop_penalty
            self._shaped = min(1.0, self._shaped + max(0.0, delta))
            reward_info = SocReward(
                total=delta,
                components={"correlation_partial": partial, "loop": -loop_penalty},
            )
            feedback = f"Correlation fields match ratio={partial:.2f}."

        elif action.kind == "finalize_triage":
            submission = self._build_submission(action)
            terminal_score = self._grade(submission)
            blend = 0.5 * self._shaped + 0.5 * terminal_score
            reward_info = SocReward(
                total=blend,
                components={
                    "shaped": self._shaped,
                    "grader": terminal_score,
                    "loop": -loop_penalty,
                },
            )
            self._final_score = terminal_score
            self._state.episode_complete = True
            self._state.shaped_score = self._shaped
            done = True
            feedback = f"Final grader score={terminal_score:.2f} (blended into reward)."

        if over_limit and not done:
            submission = self._build_submission(
                SocAction(
                    kind="finalize_triage",
                    verdict="benign",
                    technique_ids=[],
                    remediation_steps=[],
                )
            )
            terminal_score = self._grade(submission)
            reward_info = SocReward(
                total=0.25 * terminal_score,
                components={"timeout": 1.0, "grader": terminal_score},
            )
            self._final_score = terminal_score
            self._state.episode_complete = True
            done = True
            feedback = "Max steps exceeded; auto-submitted empty triage."

        self._state.shaped_score = self._shaped
        self._state.last_total_reward = reward_info.total

        alert = self._gold.get("alert", {})
        logs = "\n".join(self._gold.get("log_lines", []))
        ep_ok = bool(done and (self._final_score or 0.0) >= 0.85)
        meta = {
            "reward_breakdown": reward_info.model_dump(),
            "final_grader_score": self._final_score,
            "success": ep_ok,
        }

        return SocObservation(
            task=self._task,
            instruction=str(self._gold.get("instruction", "")),
            alert_id=str(alert.get("id", "")),
            alert_rule=str(alert.get("rule", "")),
            alert_severity=str(alert.get("severity", "")),
            log_view=logs,
            max_steps=max_steps,
            available_commands=list(_COMMANDS),
            feedback=feedback,
            done=done,
            reward=reward_info.total,
            final_grader_score=self._final_score,
            episode_success=ep_ok,
            metadata=meta,
        )

    @property
    def state(self) -> SocState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="soc_analyst_env",
            description=(
                "Synthetic SOC alert triage with bounded logs, MITRE ATT&CK-style technique IDs, "
                "and deterministic graders (easy / medium / hard)."
            ),
            version="0.1.0",
        )

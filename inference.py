#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
LLM-driven rollouts with strict stdout logging for hackathon evaluation.

Environment variables:
  API_BASE_URL  - OpenAI-compatible API base URL
  MODEL_NAME    - Model id
  HF_TOKEN      - API key (or set OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

# Repo layout: Round_One/soc_analyst_env/
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI

from soc_analyst_env.client import SocAnalystEnv
from soc_analyst_env.models import SocAction


def _fmt_reward(r: Optional[float]) -> str:
    if r is None:
        return "0.00"
    return f"{float(r):.2f}"


def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def _build_system_prompt() -> str:
    return """You are a SOC analyst agent in a simulated environment.
You must respond with a single JSON object (no markdown fences) matching this schema:
{
  "kind": one of "noop", "submit_hypothesis", "submit_correlation", "finalize_triage", "destructive_network_block",
  "technique_ids": string[] (MITRE IDs like T1190),
  "rationale_tag": string,
  "service_a", "service_b", "link_key", "link_value": strings for correlation,
  "verdict": "true_positive" | "false_positive" | "benign" (required only when kind is finalize_triage),
  "primary_technique": string or null,
  "remediation_steps": string[] (ordered playbook ids, lowercase_snake_case),
  "correlation_service_a", "correlation_service_b", "correlation_link_key", "correlation_link_value": optional strings on finalize for medium task,
  "metadata": {}
}
Use submit_hypothesis / submit_correlation for partial credit, then finalize_triage when confident.
Never use destructive_network_block unless the scenario explicitly requires blocking internal RFC1918 /24 (penalized)."""


def run_episode(
    *,
    task_name: str,
    benchmark: str,
    base_url: str,
    model: str,
    api_key: str,
    max_llm_steps: int,
) -> tuple[bool, int, float, List[float]]:
    client_ai = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    rewards: List[float] = []

    print(
        f"[START] task={task_name} env={benchmark} model={model}",
        flush=True,
    )

    final_score = 0.0
    success = False
    steps = 0

    openenv_url = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000")
    with SocAnalystEnv(base_url=openenv_url).sync() as env:
        result = env.reset(task=task_name)
        obs = result.observation
        err: Optional[str] = None

        while max_llm_steps > 0:
            max_llm_steps -= 1
            if result.done:
                break

            user = json.dumps(
                {
                    "instruction": obs.instruction,
                    "alert": {
                        "id": obs.alert_id,
                        "rule": obs.alert_rule,
                        "severity": obs.alert_severity,
                    },
                    "log_view": obs.log_view,
                    "max_steps": obs.max_steps,
                    "feedback": obs.feedback,
                    "available_commands": obs.available_commands,
                },
                ensure_ascii=False,
            )

            try:
                comp = client_ai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _build_system_prompt()},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                )
                raw = comp.choices[0].message.content or ""
                data = _extract_json(raw)
                if "metadata" not in data:
                    data["metadata"] = {}
                action = SocAction.model_validate(data)
                err = None
            except Exception as e:
                action = SocAction(kind="noop", metadata={})
                err = str(e).replace("\n", " ")[:200]

            result = env.step(action)
            obs = result.observation
            steps += 1
            rw = result.reward if result.reward is not None else obs.reward
            rwf = float(rw) if rw is not None else 0.0
            rewards.append(rwf)

            esc_err = "null" if err is None else err
            print(
                f"[STEP] step={steps} action={json.dumps(action.model_dump(), separators=(',', ':'))} "
                f"reward={_fmt_reward(rwf)} done={_fmt_bool(result.done)} error={esc_err}",
                flush=True,
            )

            if obs.final_grader_score is not None:
                final_score = float(obs.final_grader_score)
            success = bool(obs.episode_success)

            if result.done:
                break

    if not rewards:
        rewards.append(0.0)

    print(
        f"[END] success={_fmt_bool(success)} steps={steps} score={final_score:.2f} rewards={','.join(_fmt_reward(x) for x in rewards)}",
        flush=True,
    )
    return success, steps, final_score, rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK", "easy"),
        choices=("easy", "medium", "hard"),
    )
    parser.add_argument(
        "--benchmark",
        default=os.environ.get("BENCHMARK", "soc_analyst_env"),
    )
    parser.add_argument(
        "--max-llm-steps",
        type=int,
        default=int(os.environ.get("MAX_LLM_STEPS", "24")),
    )
    args = parser.parse_args()

    base_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""

    if not api_key:
        noop = json.dumps({"kind": "noop", "metadata": {}}, separators=(",", ":"))
        print(
            f"[STEP] step=0 action={noop} reward=0.00 done=true error=missing API key (HF_TOKEN or OPENAI_API_KEY)",
            flush=True,
        )
        print(
            "[END] success=false steps=0 score=0.00 rewards=0.00",
            flush=True,
        )
        sys.exit(1)

    run_episode(
        task_name=args.task,
        benchmark=args.benchmark,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_llm_steps=args.max_llm_steps,
    )


if __name__ == "__main__":
    main()

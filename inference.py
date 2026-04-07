#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
LLM-driven rollouts with strict stdout logging for hackathon evaluation.

Environment variables:
  API_BASE_URL      - OpenAI-compatible API base URL
  MODEL_NAME        - Model id
  HF_TOKEN          - API key (or set OPENAI_API_KEY)
  OPENENV_BASE_URL  - Full URL of the OpenEnv HTTP server (overrides host/port)
  OPENENV_HOST      - Host for OpenEnv (default 127.0.0.1)
  OPENENV_PORT      - Port for OpenEnv (default 7860; aligns with many HF Space Dockerfiles)
  PORT              - If set (e.g. on HF Spaces) and OPENENV_PORT unset, used as OpenEnv port
  ENV_WAIT_TIMEOUT  - Seconds to wait for /health (default 60)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
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


def _noop_action_json() -> str:
    return json.dumps({"kind": "noop", "metadata": {}}, separators=(",", ":"))


def _escape_step_error(msg: Optional[str]) -> str:
    if msg is None:
        return "null"
    # Keep stdout parseable: collapse whitespace, trim length
    s = str(msg).replace("\n", " ").replace("\r", " ").strip()
    if len(s) > 240:
        s = s[:237] + "..."
    return s


def resolve_openenv_base_url() -> str:
    """OPENENV_BASE_URL wins; else http://OPENENV_HOST:OPENENV_PORT (default port 7860)."""
    explicit = os.environ.get("OPENENV_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("OPENENV_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = (
        os.environ.get("OPENENV_PORT", "").strip()
        or os.environ.get("PORT", "").strip()
        or "7860"
    )
    return f"http://{host}:{port}".rstrip("/")


def wait_for_env_ready(
    base_url: str,
    *,
    timeout_s: float = 60.0,
    initial_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
) -> tuple[bool, Optional[str]]:
    """
    Poll GET {base_url}/health until 200 or timeout.
    Returns (ok, last_error_message).
    """
    health_url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_s
    delay = initial_delay_s
    last_err: Optional[str] = None

    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=min(10.0, max(1.0, deadline - time.monotonic()))) as resp:
                if resp.status == 200:
                    return True, None
                last_err = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URLError {e.reason!r}"
        except TimeoutError:
            last_err = "timeout"
        except OSError as e:
            last_err = f"OSError {e}"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, max_delay_s)

    return False, last_err or "environment not reachable"


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
    llm_base_url: str,
    openenv_url: str,
    model: str,
    api_key: str,
    max_llm_steps: int,
    env_wait_timeout_s: float,
) -> None:
    rewards: List[float] = []
    final_score = 0.0
    success = False
    steps = 0

    print(
        f"[START] task={task_name} env={benchmark} model={model}",
        flush=True,
    )

    ready, wait_err = wait_for_env_ready(
        openenv_url,
        timeout_s=env_wait_timeout_s,
    )
    if not ready:
        print(
            f"[STEP] step=0 action={_noop_action_json()} reward=0.00 done=true "
            f"error={_escape_step_error(f'env wait failed: {wait_err}')}",
            flush=True,
        )
        print(
            "[END] success=false steps=0 score=0.00 rewards=0.00",
            flush=True,
        )
        return

    try:
        client_ai = OpenAI(base_url=llm_base_url.rstrip("/"), api_key=api_key)
    except Exception as e:
        print(
            f"[STEP] step=0 action={_noop_action_json()} reward=0.00 done=true "
            f"error={_escape_step_error(f'OpenAI client init: {e}')}",
            flush=True,
        )
        print(
            "[END] success=false steps=0 score=0.00 rewards=0.00",
            flush=True,
        )
        return

    try:
        with SocAnalystEnv(base_url=openenv_url).sync() as env:
            try:
                result = env.reset(task=task_name)
            except Exception as e:
                print(
                    f"[STEP] step=0 action={_noop_action_json()} reward=0.00 done=true "
                    f"error={_escape_step_error(f'env reset: {e}')}",
                    flush=True,
                )
                if not rewards:
                    rewards.append(0.0)
                print(
                    f"[END] success=false steps=0 score=0.00 rewards={','.join(_fmt_reward(x) for x in rewards)}",
                    flush=True,
                )
                return

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

                action: SocAction
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
                    err = f"llm: {e}"

                try:
                    result = env.step(action)
                    obs = result.observation
                except Exception as e:
                    steps += 1
                    rwf = 0.0
                    rewards.append(rwf)
                    step_err = _escape_step_error(f"env step: {e}")
                    print(
                        f"[STEP] step={steps} action={json.dumps(action.model_dump(), separators=(',', ':'))} "
                        f"reward={_fmt_reward(rwf)} done=true error={step_err}",
                        flush=True,
                    )
                    break

                steps += 1
                rw = result.reward if result.reward is not None else obs.reward
                rwf = float(rw) if rw is not None else 0.0
                rewards.append(rwf)

                esc_err = _escape_step_error(err)
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

    except Exception as e:
        if not rewards:
            rewards.append(0.0)
        conn_step = steps + 1
        print(
            f"[STEP] step={conn_step} action={_noop_action_json()} reward=0.00 done=true "
            f"error={_escape_step_error(f'env connection: {e}')}",
            flush=True,
        )
        print(
            f"[END] success=false steps={conn_step} score={final_score:.2f} rewards={','.join(_fmt_reward(x) for x in rewards)}",
            flush=True,
        )
        return

    if not rewards:
        rewards.append(0.0)

    print(
        f"[END] success={_fmt_bool(success)} steps={steps} score={final_score:.2f} rewards={','.join(_fmt_reward(x) for x in rewards)}",
        flush=True,
    )


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
    parser.add_argument(
        "--openenv-url",
        default="",
        help="Override OpenEnv base URL (else env vars / defaults)",
    )
    args = parser.parse_args()

    llm_base_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""

    openenv_url = (args.openenv_url or "").strip() or resolve_openenv_base_url()
    env_wait_timeout_s = float(os.environ.get("ENV_WAIT_TIMEOUT", "60"))

    if not api_key:
        print(
            f"[START] task={args.task} env={args.benchmark} model={model}",
            flush=True,
        )
        print(
            f"[STEP] step=0 action={_noop_action_json()} reward=0.00 done=true "
            f"error={_escape_step_error('missing API key (HF_TOKEN or OPENAI_API_KEY)')}",
            flush=True,
        )
        print(
            "[END] success=false steps=0 score=0.00 rewards=0.00",
            flush=True,
        )
        sys.exit(0)

    run_episode(
        task_name=args.task,
        benchmark=args.benchmark,
        llm_base_url=llm_base_url,
        openenv_url=openenv_url,
        model=model,
        api_key=api_key,
        max_llm_steps=args.max_llm_steps,
        env_wait_timeout_s=env_wait_timeout_s,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
